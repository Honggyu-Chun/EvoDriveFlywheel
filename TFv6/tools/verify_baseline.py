"""Verify LEAD v1.5.1 Vision-only seed 0 on an official log; write local evidence.

Run with the Phase 1 Python after fetch_baseline.py. --figures writes the small
figures derived from that public log. No model source or training state is changed.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import timm
import torch
import yaml

import lead
from lead.api.scene_loading_spec import SceneLoadingSpec
from lead.api.py123d_log_api import LEAD_INDEX_BY_CAMERA_ID
from lead.common.image_decoders import decode_jpeg, decode_png
from lead.config.lead_config import load_lead_config
from lead.evaluation.inference.policy_runner import PolicyRunner
from lead.log_reader.scene_loader import SceneLoader
from lead.policy.transfuser.dataloader.dataset import TransfuserDataset
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.io.camera.jpeg_camera_io import set_jpeg_decoder
from py123d.common.io.camera.png_camera_io import set_png_decoder

ROOT = Path(__file__).resolve().parents[2]
LEAD_SHA = "3764f79698922896638638df8062f2e064034610"
MODEL_REVISION = "0a7a540df6a9dff6e0f3a64eb03222216b3f430c"
DATA_REVISION = "49565fcf35fd02b03c322a5ab560631b9c706a53"
MODEL_DIR = ROOT / "TFv6/checkpoints/visiononly_resnet34_v1.5.1/seed0"
LOCAL = ROOT / "TFv6/phase2-local"
DATA = LOCAL / "data"
LOG = "Town01_Rep0_route_002516_route0_08_02_07_01_23"
LOG_DIR = DATA / "logs/normal_view/SignalizedJunctionLeftTurn" / LOG
FILES = {
    "config.yaml": (
        28483,
        "2c01fa4193125200424c276a2c300b0fd9b3ee18cc09e439442ae99bd78ab1dd",
    ),
    "model_0030.pth": (
        256565012,
        "807839b53d82347da7911a94a44b7d9dd23afbbb0080ca6c04360d91713c71d3",
    ),
}
DEVICE = torch.device("cuda:0")
OBSERVATION = SceneLoadingSpec(rgb_tick_ages=(0,))
PLANNING = (
    "route",
    "future_waypoints",
    "target_speed_distribution",
    "target_speed_scalar",
)
OUTPUT_SHAPES = {
    "route": (1, 10, 2),
    "future_waypoints": (1, 8, 2),
    "target_speed_distribution": (1, 8),
    "target_speed_scalar": (1,),
    "semantic": (1, 10, 384, 2304),
    "depth": (1, 384, 2304),
    "bev_semantic": (1, 12, 320, 384),
    "box.center_heatmap_logit_pred": (1, 7, 80, 96),
    "box.center_heatmap_pred": (1, 7, 80, 96),
    "box.wh_pred": (1, 2, 80, 96),
    "box.offset_pred": (1, 2, 80, 96),
    "box.yaw_class_pred": (1, 12, 80, 96),
    "box.yaw_res_pred": (1, 1, 80, 96),
    "box.velocity_pred": (1, 1, 80, 96),
}


def checkpoint_identity():
    for name, (size, expected) in FILES.items():
        path = MODEL_DIR / name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != size or digest.hexdigest() != expected:
            raise ValueError(f"Official artifact identity mismatch for {name}")


def load_runner():
    """Use official config/loader and require every checkpoint key and shape."""
    if os.environ.get("LEAD_CONFIG"):
        raise RuntimeError("Unset LEAD_CONFIG to verify the published configuration")
    checkout = ROOT / "TFv6/lead"
    if Path(lead.__file__).resolve().parent != checkout / "src/lead":
        raise RuntimeError("Run with the Phase 1 editable LEAD installation")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    if revision != LEAD_SHA:
        raise RuntimeError(f"Expected LEAD {LEAD_SHA}, found {revision}")
    checkpoint_identity()
    started = time.perf_counter()
    config = load_lead_config(yaml.safe_load((MODEL_DIR / "config.yaml").read_text()))
    policy = config.policy.transfuser
    if not (
        config.evaluation.inference.strict_weight_load
        and policy.LTF
        and not policy.use_radar_detection
        and not config.expert.sensor_rig.use_radars
    ):
        raise ValueError("Expected strict Vision-only ResNet34 release configuration")

    create_model = timm.create_model

    def offline_init(*args, **kwargs):
        # PolicyRunner strictly overwrites the entire state immediately. The
        # intermediate ImageNet initialization otherwise fetches unrelated weights.
        if kwargs.get("pretrained"):
            if args != ("resnet34",):
                raise ValueError("Unexpected pretrained model constructor")
            kwargs["pretrained"] = False
        return create_model(*args, **kwargs)

    with patch.object(timm, "create_model", side_effect=offline_init):
        runner = PolicyRunner(config, str(MODEL_DIR), DEVICE)
    torch.cuda.synchronize(DEVICE)
    return runner, (time.perf_counter() - started) * 1000


def scene_loader(config):
    set_jpeg_decoder(decode_jpeg)
    set_png_decoder(decode_png)
    return SceneLoader(
        DATA,
        SceneFilter(
            split_names=["normal_view"],
            log_names=[LOG],
            history_num_iterations=0,
            future_num_iterations=40,
        ),
        camera_ids=config.policy.transfuser.input_cameras,
        target_point_pop_distance=config.policy.transfuser.target_point_pop_distance,
        tick_duration_us=50_000,
    )


def describe(value):
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device) if isinstance(value, torch.Tensor) else "cpu",
        }
    if isinstance(value, (list, tuple)):
        return [describe(item) for item in value]
    if isinstance(value, dict):
        return {key: describe(item) for key, item in value.items()}
    return value


def cpu_array(tensor):
    return tensor.detach().float().cpu().numpy()


def outputs(prediction):
    raw = {name: getattr(prediction, name) for name in PLANNING}
    for name in ("semantic", "depth", "bev_semantic"):
        raw[name] = getattr(prediction, name)
    for name in (
        "center_heatmap_logit_pred",
        "center_heatmap_pred",
        "wh_pred",
        "offset_pred",
        "yaw_class_pred",
        "yaw_res_pred",
        "velocity_pred",
    ):
        raw[f"box.{name}"] = getattr(prediction.bounding_box, name)
    for name, tensor in raw.items():
        if tensor is None or not torch.isfinite(tensor).all().item():
            raise ValueError(f"Missing or non-finite output {name}")
        if tuple(tensor.shape) != OUTPUT_SHAPES[name] or tensor.device != DEVICE:
            raise ValueError(
                f"Unexpected {name} shape/device: {tensor.shape} on {tensor.device}"
            )
    if (
        prediction.radar_features is not None
        or prediction.radar_predictions is not None
    ):
        raise ValueError("Radar branch unexpectedly active")
    return raw


def postprocess(prediction, config):
    probabilities = prediction.target_speed_distribution.float().softmax(dim=-1)
    target_speed = prediction.target_speed_scalar
    inference = config.evaluation.inference
    target_speed = torch.where(
        probabilities[:, 0] > inference.brake_threshold, 0, target_speed
    )
    if inference.lower_target_speed:
        target_speed = target_speed * inference.lower_target_speed_factor
    return {
        name: cpu_array(getattr(prediction, name))[0].tolist() for name in PLANNING
    } | {
        "probabilities": cpu_array(probabilities)[0].tolist(),
        "postprocessed_speed_mps": cpu_array(target_speed)[0].item(),
    }


def inspect_state(model):
    state = torch.load(
        MODEL_DIR / "model_0030.pth", weights_only=True, map_location="cpu"
    )
    actual = model.state_dict()
    if state.keys() != actual.keys():
        raise ValueError("Checkpoint keys differ from model")
    for key, expected in state.items():
        if actual[key].shape != expected.shape or not torch.equal(
            actual[key].cpu(), expected
        ):
            raise ValueError(f"Checkpoint not exactly installed at {key}")
    counts = {}
    for name, param in model.named_parameters():
        group = name.split(".")[0]
        if name.startswith(("backbone.image_encoder.", "backbone.lidar_encoder.")):
            group = ".".join(name.split(".")[:2])
        counts.setdefault(group, [0, 0])
        counts[group][0] += param.numel()
        if param.requires_grad:
            counts[group][1] += param.numel()
    return {
        "keys": len(state),
        "param_counts": counts,
        "buffers": sum(t.numel() for t in model.buffers()),
    }


def trace_forward(runner, batch):
    traced = {}
    names = {
        "backbone",
        "planning_decoder",
        "planning_decoder.planning_context_encoder",
        "planning_decoder.transformer_decoder",
        "backbone.c5_conv",
        "backbone.up_conv4",
        "semantic_decoder",
        "depth_decoder",
        "bev_semantic_decoder",
    }

    def hook(name):
        def record(_module, args, kwargs, result):
            traced[name] = {
                "input": describe(args[:2]),
                "kwargs": describe(kwargs),
                "output": describe(result),
            }

        return record

    hooks = [
        module.register_forward_hook(hook(name), with_kwargs=True)
        for name, module in runner.policy.named_modules()
        if name in names
        or name.startswith("backbone.transformers.")
        and name.count(".") == 2
        or name.startswith(
            ("backbone.image_encoder.layer", "backbone.lidar_encoder.layer")
        )
        and name.count(".") == 2
    ]
    try:
        prediction = runner.forward(batch)
        shape = {name: describe(tensor) for name, tensor in outputs(prediction).items()}
    finally:
        for handle in hooks:
            handle.remove()
    return prediction, traced, shape


def make_figures(folder, scene, batch, values, sample, pred, config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    fig, rig = plt.subplots(figsize=(6.0, 6.0), layout="constrained")
    for camera in scene.cameras:
        camera_id = camera.metadata.camera_id
        mount = config.expert.sensor_rig.cameras[LEAD_INDEX_BY_CAMERA_ID[camera_id] - 1]
        x, y, _ = mount["pos"]
        yaw = mount["rot"][2]
        angles = np.deg2rad(
            np.linspace(yaw - mount["fov"] / 2, yaw + mount["fov"] / 2, 32)
        )
        rig.fill(
            np.r_[y, y + 3.5 * np.sin(angles), y],
            np.r_[x, x + 3.5 * np.cos(angles), x],
            alpha=0.2,
        )
        point = np.deg2rad(yaw)
        rig.plot(
            [y, y + 3.5 * np.sin(point)], [x, x + 3.5 * np.cos(point)], linewidth=1
        )
        rig.annotate(
            camera_id.name.removeprefix("PCAM_"),
            (y + 3.9 * np.sin(point), x + 3.9 * np.cos(point)),
            ha="center",
        )
        rig.scatter(y, x, s=18)
    rig.scatter(0, 0, marker="^", color="black", label="CARLA actor origin")
    rig.set(
        xlim=(-5, 5),
        ylim=(-5, 5),
        aspect="equal",
        xlabel="CARLA +Y right (m)",
        ylabel="CARLA +X forward (m)",
        title="Seed 0 camera mounts and horizontal FOV",
    )
    rig.grid(alpha=0.2)
    rig.legend(loc="lower right")
    fig.savefig(folder / "camera-rig.png", dpi=135)
    plt.close(fig)

    mosaic = np.concatenate([camera.image for camera in scene.cameras], axis=1)
    actual = cpu_array(batch["rgb"])[0].transpose(1, 2, 0).astype(np.uint8)
    np.testing.assert_array_equal(mosaic, actual)
    fig = plt.figure(figsize=(14, 5.0), layout="constrained")
    grid = fig.add_gridspec(2, 6)
    for i, camera in enumerate(scene.cameras):
        axis = fig.add_subplot(grid[0, i])
        axis.imshow(camera.image)
        axis.set_title(camera.metadata.camera_id.name)
        axis.axis("off")
    axis = fig.add_subplot(grid[1, :])
    axis.imshow(actual)
    axis.set_title(
        "Official stitched tensor before backbone normalization | RGB 384 × 2304"
    )
    axis.axis("off")
    fig.savefig(folder / "official-camera-input.jpg", dpi=120)
    plt.close(fig)

    fig, (bev, speed) = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for points, style, label in (
        (values["route"], "o-", "Predicted spatial route"),
        (sample.route, "o--", "Expert spatial label"),
        (values["future_waypoints"], "s-", "Predicted 0.25 s waypoints"),
        (sample.future_waypoints, "s--", "Expert future ego"),
    ):
        xy = np.asarray(points)
        bev.plot(xy[:, 1], xy[:, 0], style, label=label, markersize=4)
    target_points = np.array(
        [scene.previous_target_point, scene.target_point, scene.next_target_point]
    )
    bev.scatter(
        target_points[:, 1], target_points[:, 0], marker="x", label="Navigation targets"
    )
    bev.scatter([0], [0], marker="^", color="black", label="Ego at anchor")
    for name, (x, y) in zip(("Prev", "Target", "Next"), target_points):
        bev.annotate(name, (y, x), xytext=(4, 4), textcoords="offset points")
    bev.set(xlabel="Ego +Y right (m)", ylabel="Ego +X forward (m)", aspect="equal")
    bev.grid(alpha=0.25)
    bev.legend(fontsize=7)
    classes = config.policy.transfuser.target_speed_classes
    speed.bar(range(len(classes)), values["probabilities"])
    speed.set(
        xticks=range(len(classes)),
        xticklabels=[f"{x:.1f}" for x in classes],
        xlabel="Class speed (m/s)",
        ylabel="Probability",
    )
    speed.set_title(
        f"Prediction {values['target_speed_scalar']:.2f} m/s | "
        f"tracked {values['postprocessed_speed_mps']:.2f} m/s"
    )
    fig.savefig(folder / "official-planning.png", dpi=125)
    plt.close(fig)

    semantic = cpu_array(pred.semantic)[0].argmax(0)
    depth = cpu_array(pred.depth)[0].squeeze()
    bev_semantic = cpu_array(pred.bev_semantic)[0].argmax(0)
    centers = cpu_array(pred.bounding_box.center_heatmap_pred)[0].max(0)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    for axis, image, title, categorical in zip(
        axes.flat,
        (semantic, depth, bev_semantic, centers),
        (
            "Perspective semantic class",
            "Perspective depth (network output)",
            "BEV semantic class",
            "Max class center probability",
        ),
        (True, False, True, False),
    ):
        artist = axis.imshow(
            image, cmap="tab20" if categorical else "viridis", interpolation="nearest"
        )
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(artist, ax=axis, shrink=0.66)
    fig.savefig(folder / "official-perception.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor",
        type=int,
        default=50,
        help="Scene index in the fixed normal-view log",
    )
    parser.add_argument("--report", type=Path, default=LOCAL / "verification.json")
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    if Path(os.environ.get("PY123D_DATA_ROOT", "")).resolve() != DATA:
        parser.error(
            "Set PY123D_DATA_ROOT to TFv6/phase2-local/data (absolute path) for the official map lookup"
        )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    runner, load_ms = load_runner()
    print("Official strict load passed", flush=True)
    model, config = runner.policy, runner.lead_config
    state = inspect_state(model)
    loader = scene_loader(config)
    if not 0 <= args.anchor < len(loader) - 2:
        raise ValueError(
            f"Anchor must leave two consecutive frames in {len(loader)} scenes"
        )
    scene = loader.read(args.anchor, False, OBSERVATION)
    features = model.build_features(scene)
    expected = {
        "rgb",
        "speed",
        "previous_target_point",
        "target_point",
        "next_target_point",
        "town",
    }
    if set(features) != expected or any(
        (scene.lidar_sweeps, scene.radar_sweeps, scene.future_ego_states)
    ):
        raise ValueError(
            f"Unexpected model input keys or offline-only modality: {features.keys()}"
        )
    online_fields = replace(
        scene, driving_meta=None, box_detections=None, traffic_lights=None
    )
    online_features = model.build_features(online_fields)
    for name in expected:
        np.testing.assert_array_equal(features[name], online_features[name])
    batch = model.features_to_batch(features, DEVICE)
    fake = {
        key: tensor.clone() if isinstance(tensor, torch.Tensor) else tensor
        for key, tensor in batch.items()
    }
    fake["rgb"].fill_(128)
    fake["previous_target_point"][:] = torch.tensor([2.0, 0.0], device=DEVICE)
    fake["target_point"][:] = torch.tensor([8.0, 0.0], device=DEVICE)
    fake["next_target_point"][:] = torch.tensor([14.0, 0.0], device=DEVICE)
    fake["speed"].fill_(4)
    outputs(runner.forward(fake))
    print("Synthetic camera-only forward passed", flush=True)
    prediction, hooks, output_shapes = trace_forward(runner, batch)
    values = postprocess(prediction, config)
    nearby = []
    for index in range(args.anchor, args.anchor + 3):
        point = loader.read(index, False, OBSERVATION)
        current = model.features_to_batch(model.build_features(point), DEVICE)
        result = runner.forward(current)
        outputs(result)
        nearby.append(
            {
                "index": index,
                "tick": point.scene_metadata.initial_idx,
                "camera_timestamp_us": int(point.cameras[0].timestamp.time_us),
                "speed_mps": current["speed"].item(),
                "prediction": postprocess(result, config),
            }
        )
    print("Official RGB forward passed on three consecutive observations", flush=True)

    config.training.data.read_from_cache_store = False
    dataset = TransfuserDataset(config, loader)
    sample = dataset[args.anchor]
    labels = dataset.collate_fn([sample])
    labels = {
        key: val.to(DEVICE) if isinstance(val, torch.Tensor) else val
        for key, val in labels.items()
    }
    with torch.inference_mode():
        losses, _ = model.compute_loss(prediction, labels)
        loss_values = {name: float(loss) for name, loss in losses.items()}
    if not all(np.isfinite(list(loss_values.values()))):
        raise ValueError("Non-finite official evaluation loss")
    print("Official label builders and loss evaluation passed", flush=True)
    schemas = {}
    for path in sorted(LOG_DIR.glob("*.arrow")):
        with pa.memory_map(str(path), "r") as stream:
            reader = pa.ipc.open_file(stream)
            schemas[path.name] = {
                "bytes": path.stat().st_size,
                "fields": [(field.name, str(field.type)) for field in reader.schema],
            }
    metadata = [camera.metadata.to_dict() for camera in scene.cameras]
    report = {
        "lead_commit": LEAD_SHA,
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATA_REVISION,
        "strict": state,
        "load_ms": load_ms,
        "scenes": len(loader),
        "anchor": args.anchor,
        "inputs": {name: describe(value) for name, value in batch.items()},
        "rgb_range": [int(features["rgb"].min()), int(features["rgb"].max())],
        "features": {
            name: np.asarray(features[name]).tolist() for name in expected - {"rgb"}
        },
        "cameras": metadata,
        "ego": {"pose": scene.ego_state.imu_se3.tolist()},
        "hooks": hooks,
        "outputs": output_shapes,
        "frames": nearby,
        "prediction": values,
        "training_fields": {
            f.name: describe(getattr(sample, f.name))
            for f in fields(sample)
            if isinstance(getattr(sample, f.name), np.ndarray)
        },
        "labels": {
            "route": sample.route.tolist(),
            "future_waypoints": sample.future_waypoints.tolist(),
            "future_yaws": sample.future_yaws.tolist(),
            "target_speed": scene.driving_meta["target_speed"],
            "brake": scene.driving_meta["brake"],
        },
        "losses": loss_values,
        "schemas": schemas,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, default=str, allow_nan=False) + "\n"
    )
    if args.figures is not None:
        make_figures(args.figures, scene, batch, values, sample, prediction, config)
    print(f"Saved {args.report}", flush=True)


if __name__ == "__main__":
    main()
