# LEAD upstream branch comparison

Review date: 2026-11-26. Scope: Camera-only TFv6 ResNet34 for EvoDriveFlywheel on MORAI. Static review of official GitHub source, upstream docs and Hugging Face model cards/configs, not runtime testing. Branch selection remains pending researcher approval.

## 1. Relationship and compatibility

Upstream main explicitly says the two branches are independent and their datasets/checkpoints incompatible. GitHub compare(cvpr2026...main) returned HTTP 404 "No common ancestor between cvpr2026 and main". This is not a conventional upgrade path. Use branch-specific code, dataset, config and checkpoint together; do not mix weights or caches. Exact branch-tip commit hashes were not obtained in this review; pin them and HF revision before implementation. main pyproject identifies version 1.5.1; branch HEAD is not necessarily identical to release tag.

Sources: [main README](https://github.com/kesai-labs/lead/blob/main/README.md), [cvpr2026 README](https://github.com/kesai-labs/lead/blob/cvpr2026/README.md), [v1.5.1 release](https://github.com/kesai-labs/lead/releases/tag/v1.5.1).

## 2. Comparison

| Item | cvpr2026 | main (v1.5.1 generation) |
|---|---|---|
| Purpose | Paper reproduction, earlier carla_garage lineage | Maintained research infrastructure, upstream recommended for future work |
| Package | lead/ | src/lead/ |
| Runtime | Python 3.10, Torch 2.5, torchvision 0.20, cu124 index | Python 3.10–3.12, Torch 2.8, torchvision 0.23, cu128 index, Lightning 2.6.1 |
| Simulator reference | CARLA 0.9.15 | CARLA 0.9.16 collection; 0.9.15 also supported for evaluation |
| Dataset | ln2697/lead, route files and frame metadata, persistent/session caches, buckets | ln2697/lead-123d, py123d Arrow streams, log/scene filtering, LMDB dependency, explicit prebuilt cache |
| Training config | Python config_training.py, checkpoint config.json | LeadConfig nested policy/training config, checkpoint config.yaml |
| Policy interface | OpenLoopInference / ClosedLoopInference; latter adds PID and ensembling | AbstractPolicy: build_features, features_to_batch, forward, compute_loss; policy selected by config |
| Evaluation | CARLA route harness; infractions.json, metric_info.json, videos, webapp | checkpoint_endpoint.json, configurable videos, per-route resume and SLURM result aggregation |
| Vision-only checkpoint | ln2697/tfv6/visiononly_resnet34 | ln2697/transfuser-carla-123d/visiononly_resnet34_v1.5.1 |
| Reported Bench2Drive DS | 91.60 | 87.6 ± 1.2 |
| Reported Longest6 v2 DS | 43.00 | 43.5 ± 4.1 |

Sources: [old dependencies](https://github.com/kesai-labs/lead/blob/cvpr2026/pyproject.toml), [new dependencies](https://github.com/kesai-labs/lead/blob/main/pyproject.toml), [old config](https://github.com/kesai-labs/lead/blob/cvpr2026/lead/training/config_training.py), [old loader](https://github.com/kesai-labs/lead/blob/cvpr2026/lead/data_loader/carla_dataset.py), [old inference](https://github.com/kesai-labs/lead/blob/cvpr2026/lead/inference/closed_loop_inference.py), [new policy interface](https://github.com/kesai-labs/lead/blob/main/src/lead/api/abstract_policy.py), [new evaluation](https://github.com/kesai-labs/lead/blob/main/docs/eval.md).

## 3. Camera-only checkpoint details

- cvpr2026 [config.json](https://huggingface.co/ln2697/tfv6/blob/main/visiononly_resnet34/config.json): num_cameras=6, stitched image 2304x384, image_architecture=resnet34, LTF=true, use_radar_detection=false, planning decoder enabled, 10 route points and 8 waypoints, waypoint spacing=5, previous/current/next target points and velocity used. Historical lidar/radar config fields remain present; trace actual forward inputs and label builders instead of inferring sensor requirements from field names alone. Paper README requests Kalman/creeping/stop-sign heuristics for reported scores and describes potential ensemble loading of three seeds; do not conflate published score with a single seed.
- main [model card](https://huggingface.co/ln2697/transfuser-carla-123d/blob/main/visiononly_resnet34_v1.5.1/README.md): six camera order L1,L0,F0,R0,R1,B0, ResNet34, no lidar/radar. Each seed has config.yaml and model_0030.pth (seed0 ~257 MB). Planning initialized from perception pretrain; 31 epochs, roughly 30 h per seed on 4 GPUs (GPU type not specified on model card). Scores 86.2/88.6/88.0; all 220 routes included, 10–16 per seed counted zero because of tick deadlines, crashes or agent setup failures rather than driving failures. Hence the 4-point difference is not attributable solely to model architecture or data quality. Exact camera extrinsics/FOV and all config values still need extraction and validation against runtime code.
- Neither checkpoint family is VLA. No verified MORAI speed or fine-tuning result.

## 4. Dataset and flywheel

main [data access](https://github.com/kesai-labs/lead/blob/main/docs/data_access.md) specifies normal_view and perturbated_view splits, route logs grouped by scenario type, modality-specific Arrow IPC streams and separate maps. CARLA simulation time step is 50 ms (20 Hz); RGB/depth/semantic/instance streams saved every fifth tick (4 Hz), ego/boxes/lights and point-cloud modalities 20 Hz. This is storage frequency, not online planning frequency. py123d SceneFilter can constrain log names, split, location and temporal margins; SceneLoader produces SceneData. Its raw sensor and calibration data follow ISO 8855/OpenCV conventions while navigation and view-local values retain CARLA conventions. Inspect [SceneData](https://github.com/kesai-labs/lead/blob/main/src/lead/api/scene_data.py) for exact contract. Training-only future states and driving_meta are separate from online feature construction; do not expose privileged future information to deployed policy.

Official main README reports 8,930 routes/43 scenario types, ~1 TB storage, up to 1000x fewer files, ~10 data-generation steps/s on GTX 1080 Ti, optional ~60 GB cache. These are upstream full-pipeline claims, not camera-only MORAI throughput. Cache fingerprints prevent silently using stale geometry/label configurations. cvpr2026 CARLAData builds samples from frame metadata, buckets and caches; future/past waypoints and target-point augmentation live in loader. Older branch also has experimental 123D collection and NAVSIM extension, but that does not establish compatibility with main's training format.

MORAI adaptation: preserve raw observation/label separation, record synchronized camera streams, calibration, ego motion and route targets, ground truth for training/evaluation only, and expert trajectories. Need an adapter for map geometry, object classes, timestamps, coordinate transforms, and route-target generation. CARLA expert is not a drop-in MORAI expert. A camera-only policy can still need depth, segmentation, boxes or lidar-derived visibility labels for auxiliary supervision; inspect requirements before collecting data, disable/replace only with controlled experiments. Do not feed synthetic zero lidar/radar to a fusion checkpoint as a substitute for camera-only.

## 5. Training and hardware

main [training](https://github.com/kesai-labs/lead/blob/main/docs/training.md) uses perception pretrain then planning posttrain. [posttrain.sh](https://github.com/kesai-labs/lead/blob/main/scripts/common/posttrain.sh) sets training.experiment.initial_weights_file and policy.transfuser.use_planning_decoder=true. For flywheel use final planning checkpoint as initialization, verify strict load and optimizer/reset semantics. [speed guide](https://github.com/kesai-labs/lead/blob/main/docs/speed.md) reports global batch 64 and H100 x4; optimized baseline 23.9 h total vs ~42 h baseline; freezing backbone reduced Longest6 score substantially in the tested setup. These experiments are not camera-only A5000 estimates. Choose batch divisible across 3 GPUs, measure peak allocated/reserved VRAM, use accumulation if supported, smoke test checkpoint save/reload and evaluation. Compare training throughput with identical data and settings; no time estimate until measured.

cvpr2026 checkpoint config records mixed precision, batch 64 and 4 L40S GPUs. Old README reports training instability with some newer Torch versions. main torch.compile defaults and cache can benefit repeated runs but compilation warmup costs matter for short experiments.

Nuvo driver 560.35.03 reports CUDA 12.6. cvpr2026 cu124 dependency index is more conservative; main cu128 index needs explicit runtime/driver compatibility verification (minor-version compatibility is conditional, not a guarantee, and uv pip vs uv sync can resolve differently). Do not blindly upgrade drivers on the school simulator PC; isolate environment and test CUDA ops, Torch compile and any extensions; use compatible wheels or authorized driver changes only. Training server driver and CUDA remain unknown.

## 6. Inference and MORAI port

main AbstractPolicy shares build_features between training and online simulation; features_to_batch creates batch-one tensors. That is a concrete reuse point for MORAI without duplicating preprocessing. cvpr2026 also separates OpenLoopInference and ClosedLoopInference, with PID controllers for route+speed or waypoint execution. Neither package has verified MORAI support. Keep the simulator transport and vehicle controller independent of the policy; preserve checkpoint sensor layout, image order/color, navigation target-point conventions, timestamps, units and output conventions. Unit-test coordinate transforms with straight/left/right examples. Confirm no privileged boxes/maps/labels enter the camera-only inference path; navigation map route is distinct from privileged perception.

10 Hz means fresh observations produce new plans, not merely replaying controller commands. Benchmark one seed first, headless/no visualization for timed test, then actual MORAI concurrent execution; measure capture-to-plan latency, p50/p95/p99, miss rate, VRAM and sustained frequency. 4 Hz training image storage does not prove or prohibit 10 Hz runtime. No measured latency comparison between branches is available.

## 7. Decision considerations (not a decision)

- main: strongest fit for long-term data system research because of standardized log/scene interfaces, shared featurization, configurable policy and explicit cache management; higher dependency migration risk and lower reported vision-only DS.
- cvpr2026: strongest fit for exact paper replication, published score and older Torch/cu124 environment; older data format and more migration work if adopting main later. Existing failure webapp and NAVSIM extensions may help.
- Proposed gate before locking either: resolve exact upstream SHA + HF revision/seed; validate six-camera configuration; load checkpoint and compare forward outputs on a known upstream sample; profile on 4080 SUPER with MORAI; run small 3x A5000 fine-tune; verify new checkpoint reload, and one data round-trip from capture to training. Compare models on identical MORAI scenarios, not CARLA leaderboard numbers.

This review did not execute code, download weight binaries, measure latency, inspect every source file or obtain exact branch-tip hashes. Branch selection and sensor rig remain unapproved; no implementation is authorized by this document.
