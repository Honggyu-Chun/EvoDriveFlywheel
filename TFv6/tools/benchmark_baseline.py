"""Time the fixed official sample on CUDA, without hooks or simulator I/O.

Reports are local JSON. The optional figure contains measured latency aggregates.
See docs/08_INFERENCE_BENCHMARK.md for the timing boundaries.
"""

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from verify_baseline import (
    DATA_REVISION,
    DEVICE,
    LEAD_SHA,
    LOCAL,
    MODEL_REVISION,
    OBSERVATION,
    load_runner,
    outputs,
    postprocess,
    scene_loader,
)


def gpu_status():
    def query(option):
        return subprocess.check_output(
            ["nvidia-smi", option, "--format=csv,noheader"], text=True
        ).strip()

    return {
        "gpu": query(
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu"
        ),
        "compute_processes": query("--query-compute-apps=pid,process_name,used_memory"),
    }


def summary(samples):
    values = np.asarray(samples)
    p50, p95, p99 = np.percentile(values, (50, 95, 99))
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "max": float(values.max()),
        "samples_per_second": float(1000 / values.mean()),
        "misses_over_100ms": int((values > 100).sum()),
        "miss_rate_pct": float(100 * (values > 100).mean()),
    }


def forward_for(runner, mode):
    optimization = runner.lead_config.training.optimization
    optimization.use_mixed_precision_training = mode in ("bf16", "official")
    torch.backends.cuda.matmul.allow_tf32 = mode == "official"
    torch.backends.cudnn.allow_tf32 = mode == "official"
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if mode != "fp16":
        return runner.forward

    def forward(batch):
        # Keep the backbone's explicit cast to float32 and let eligible ops
        # select fp16. The official mixed-precision flag would instead cast to bf16.
        with torch.autocast("cuda", dtype=torch.float16):
            return runner.policy(batch)

    return forward


def timed_forward(forward, batch, begin, end):
    started = time.perf_counter()
    begin.record()
    result = forward(batch)
    end.record()
    end.synchronize()
    return result, begin.elapsed_time(end), 1000 * (time.perf_counter() - started)


def delta(actual, baseline):
    result = {}
    for name in ("route", "future_waypoints"):
        distance = np.linalg.norm(
            np.asarray(actual[name]) - np.asarray(baseline[name]), axis=-1
        )
        result[name + "_m"] = {
            "mean": float(distance.mean()),
            "max": float(distance.max()),
        }
    result["target_speed_scalar_mps"] = abs(
        actual["target_speed_scalar"] - baseline["target_speed_scalar"]
    )
    result["postprocessed_speed_mps"] = abs(
        actual["postprocessed_speed_mps"] - baseline["postprocessed_speed_mps"]
    )
    result["speed_probability_max_abs"] = float(
        np.max(
            np.abs(
                np.asarray(actual["probabilities"])
                - np.asarray(baseline["probabilities"])
            )
        )
    )
    return result


def benchmark_mode(runner, forward, batch, loader, anchor, args):
    begin, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    torch.cuda.synchronize(DEVICE)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    for _ in range(args.warmup):
        forward(batch)
    torch.cuda.synchronize(DEVICE)
    result = forward(batch)
    outputs(result)
    verified = postprocess(result, runner.lead_config)
    del result

    repeats = []
    gpu_times, wall_times = [], []
    for _ in range(args.repeats):
        per_gpu, per_wall = [], []
        for _ in range(args.iterations):
            result, gpu, wall = timed_forward(forward, batch, begin, end)
            per_gpu.append(gpu)
            per_wall.append(wall)
            del result
        repeats.append({"gpu": summary(per_gpu), "wall": summary(per_wall)})
        gpu_times.extend(per_gpu)
        wall_times.extend(per_wall)
    forward_memory = (
        torch.cuda.max_memory_allocated(DEVICE),
        torch.cuda.max_memory_reserved(DEVICE),
    )

    pipeline = {
        key: []
        for key in (
            "read_decode",
            "features_cpu_batch",
            "host_to_device",
            "forward_gpu",
            "forward_wall",
            "postprocess",
            "total",
        )
    }
    for index in range(anchor, anchor + args.frames):
        marks = [time.perf_counter()]
        scene = loader.read(index, False, OBSERVATION)
        marks.append(time.perf_counter())
        cpu = runner.policy.features_to_batch(
            runner.policy.build_features(scene), torch.device("cpu")
        )
        marks.append(time.perf_counter())
        gpu_batch = {
            k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
            for k, v in cpu.items()
        }
        torch.cuda.synchronize(DEVICE)
        marks.append(time.perf_counter())
        result, gpu, wall = timed_forward(forward, gpu_batch, begin, end)
        marks.append(time.perf_counter())
        postprocess(result, runner.lead_config)
        marks.append(time.perf_counter())
        for key, start, finish in zip(
            (
                "read_decode",
                "features_cpu_batch",
                "host_to_device",
                "forward_wall",
                "postprocess",
            ),
            marks,
            marks[1:],
        ):
            pipeline[key].append(1000 * (finish - start))
        pipeline["forward_gpu"].append(gpu)
        pipeline["total"].append(1000 * (marks[-1] - marks[0]))
        del gpu_batch, result
    return {
        "status": "PASS",
        "forward_gpu_ms": summary(gpu_times),
        "forward_wall_ms": summary(wall_times),
        "repeat_summaries": repeats,
        "output": verified,
        "pipeline_ms": {k: summary(v) for k, v in pipeline.items()},
        "forward_peak_allocated_mib": forward_memory[0] / 2**20,
        "forward_peak_reserved_mib": forward_memory[1] / 2**20,
        "pipeline_peak_allocated_mib": torch.cuda.max_memory_allocated(DEVICE) / 2**20,
        "pipeline_peak_reserved_mib": torch.cuda.max_memory_reserved(DEVICE) / 2**20,
    }


def plot_aggregate(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = {k: v for k, v in report["modes"].items() if v["status"] == "PASS"}
    fig, ax = plt.subplots(figsize=(8, 4.0), layout="constrained")
    x = np.arange(len(modes))
    for offset, metric, label in (
        (-0.25, "p50", "Forward p50"),
        (0, "p99", "Forward p99"),
        (0.25, "p99", "Offline pipeline p99"),
    ):
        field = "forward_wall_ms" if offset < 0.25 else None
        ys = [
            m[field][metric] if field else m["pipeline_ms"]["total"][metric]
            for m in modes.values()
        ]
        ax.bar(x + offset, ys, width=0.24, label=label)
    ax.axhline(100, color="#aa2727", linestyle="--", label="100 ms budget")
    ax.set(
        xticks=x,
        xticklabels=list(modes),
        ylabel="Latency (ms)",
        title="RTX 4080 SUPER | batch size 1",
    )
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=int, default=40)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report", type=Path, default=LOCAL / "benchmark.json")
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.repeats, args.frames) < 1:
        parser.error("all iteration counts must be positive")

    before = gpu_status()
    if before["compute_processes"]:
        print(
            "Other CUDA compute processes were present; see gpu_before in the report",
            flush=True,
        )
    torch.set_num_threads(1)
    runner, load_ms = load_runner()
    loader = scene_loader(runner.lead_config)
    if not 0 <= args.anchor <= len(loader) - args.frames:
        parser.error(f"requested frames must fit the {len(loader)}-scene official log")
    scene = loader.read(args.anchor, False, OBSERVATION)
    batch = runner.policy.features_to_batch(runner.policy.build_features(scene), DEVICE)
    report = {
        "lead_commit": LEAD_SHA,
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATA_REVISION,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_before": before,
        "load_ms": load_ms,
        "anchor": args.anchor,
        "frames": args.frames,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "modes": {},
    }
    for mode in ("fp32", "fp16", "bf16", "official"):
        print("Benchmark:", mode, flush=True)
        try:
            result = benchmark_mode(
                runner, forward_for(runner, mode), batch, loader, args.anchor, args
            )
        except (RuntimeError, ValueError) as error:
            if mode == "fp32":
                raise
            result = {"status": "FAIL", "error": str(error)}
        if mode != "fp32" and result["status"] == "PASS":
            result["vs_fp32"] = delta(
                result["output"], report["modes"]["fp32"]["output"]
            )
        report["modes"][mode] = result
        print(mode, result.get("forward_wall_ms", result.get("error")), flush=True)
    report["gpu_after"] = gpu_status()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.figure is not None:
        plot_aggregate(report, args.figure)
    print(f"Saved {args.report}", flush=True)
    if any(mode["status"] == "FAIL" for mode in report["modes"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
