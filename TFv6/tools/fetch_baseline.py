"""Fetch the two public model files and one pinned, normal-view LEAD Arrow log."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from verify_baseline import (
    DATA,
    DATA_REVISION,
    FILES,
    LOG_DIR,
    MODEL_DIR,
    MODEL_REVISION,
    ROOT,
    checkpoint_identity,
)


def main():
    model_repo = "ln2697/transfuser-carla-123d"
    data_repo = "ln2697/lead-123d"
    checkpoints = ROOT / "TFv6/checkpoints"
    log = LOG_DIR.relative_to(DATA).as_posix()
    streams = (
        "camera.",
        "camera_depth.",
        "camera_instance.",
        "camera_semantic.",
        "box_detections_se3.",
        "custom.driving_meta.",
        "ego_state_se3.",
        "lidar.",
        "sync.",
        "traffic_light_detections.",
    )
    entries = HfApi().list_repo_tree(
        data_repo, log, revision=DATA_REVISION, repo_type="dataset", token=False
    )
    paths = [
        entry.path for entry in entries if Path(entry.path).name.startswith(streams)
    ]
    paths.extend(("config.yaml", "maps/carla/carla_town01.arrow"))
    if len(paths) != 32:
        raise ValueError(f"Expected 32 official sample files, got {len(paths)}")

    def model(name):
        hf_hub_download(
            model_repo,
            (MODEL_DIR / name).relative_to(checkpoints).as_posix(),
            revision=MODEL_REVISION,
            local_dir=checkpoints,
            token=False,
        )

    def data(name):
        return (
            Path(
                hf_hub_download(
                    data_repo,
                    name,
                    revision=DATA_REVISION,
                    repo_type="dataset",
                    local_dir=DATA,
                    token=False,
                )
            )
            .stat()
            .st_size
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(model, FILES))
        checkpoint_identity()
        sizes = list(pool.map(data, paths))
    print(
        f"Verified model identity; fetched {len(paths)} official sample files ({sum(sizes)} bytes)"
    )


if __name__ == "__main__":
    main()
