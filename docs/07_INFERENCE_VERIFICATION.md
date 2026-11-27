# 07 · 공식 가중치와 실제 로그의 독립 추론 검증

## 검증 대상과 출처

Planning까지 학습한 **Vision-only ResNet34 v1.5.1 seed 0**을 기준으로 고정했다. seed 0은 공개 모델 카드에서 사용 가능한 첫 재현 기준이다. 카드의 Bench2Drive Driving Score는 seed 0/1/2 순서로 86.2/88.6/88.0이며, Phase 2에서는 CARLA 주행 점수를 다시 산출하지 않았다. 모델 카드와 데이터 카드가 선언한 라이선스는 MIT다. [고정 모델 카드](https://huggingface.co/ln2697/transfuser-carla-123d/blob/0a7a540df6a9dff6e0f3a64eb03222216b3f430c/visiononly_resnet34_v1.5.1/README.md) · [공식 데이터 카드](https://huggingface.co/datasets/ln2697/lead-123d)

| 검증 대상 | 고정 버전 또는 값 |
|---|---|
| LEAD | `3764f79698922896638638df8062f2e064034610`, 태그 `v.1.5.1` |
| py123d | 0.7.0, `4d20a6cfd50f7966647448084f6482c0b65effff` |
| Hugging Face 모델 `ln2697/transfuser-carla-123d` | `0a7a540df6a9dff6e0f3a64eb03222216b3f430c` |
| `visiononly_resnet34_v1.5.1/seed0/config.yaml` | 28,483 bytes · SHA-256 `2c01fa4193125200424c276a2c300b0fd9b3ee18cc09e439442ae99bd78ab1dd` |
| `visiononly_resnet34_v1.5.1/seed0/model_0030.pth` | 256,565,012 bytes · SHA-256 `807839b53d82347da7911a94a44b7d9dd23afbbb0080ca6c04360d91713c71d3` |
| Hugging Face 데이터 `ln2697/lead-123d` | `49565fcf35fd02b03c322a5ab560631b9c706a53` |
| 샘플 | `logs/normal_view/SignalizedJunctionLeftTurn/Town01_Rep0_route_002516_route0_08_02_07_01_23`, Arrow 30개 + Town01 map + dataset 설정, 총 30,383,784 bytes |

직접 읽은 Arrow 중 LiDAR 파일은 스키마 조사에만 사용했다. 객체의 visibility에 필요한 값은 사전에 박스 metadata에 포함되어 있다. 추론 입력은 6개 RGB를 현재 anchor에서만 읽고, speed와 세 navigation 지점으로 제한했다. [공식 모델 파일](https://huggingface.co/ln2697/transfuser-carla-123d/tree/0a7a540df6a9dff6e0f3a64eb03222216b3f430c/visiononly_resnet34_v1.5.1/seed0) · [고정 데이터 경로](https://huggingface.co/datasets/ln2697/lead-123d/tree/49565fcf35fd02b03c322a5ab560631b9c706a53/logs/normal_view/SignalizedJunctionLeftTurn/Town01_Rep0_route_002516_route0_08_02_07_01_23)

## 실측 결과

| 검사 | 결과 | 관측 근거 |
|---|---|---|
| Config / Model | PASS | 공식 `load_lead_config`, `PolicyRunner`, LTF Vision-only, CUDA `eval()` |
| Strict load | PASS | `torch.load(weights_only=True)`와 `strict=True` 사용. 771개 key의 이름·shape·값을 GPU 적재 상태와 전부 비교. missing=0, unexpected=0, 불일치=0 |
| 초기화와 파라미터 | PASS | 생성 시 `timm`의 별도 ImageNet 초기화 다운로드만 검증 프로세스에서 건너뛰었다. 바로 이어진 strict load가 전체 파라미터·버퍼 값을 덮으며, 완전 일치를 검사했다. 총 64,044,368개 파라미터, 전부 `requires_grad=True`, buffer 원소 34,130개 |
| Synthetic | PASS | RGB 6개 회색값 128, 전진 4 m/s, `(2,0)/(8,0)/(14,0)`의 3점. 모든 활성 출력의 shape와 유한값 확인 |
| 공식 real RGB | PASS | 일반 뷰의 연속 scene 50–52, 모든 활성 출력 유한. `driving_meta`, 객체 및 신호를 제거한 SceneData에서 입력 feature byte-for-byte 동일. future ego/LiDAR/Radar 미사용 |
| 공식 정답·loss | PASS | 별도 `TransfuserDataset`으로 동일 anchor의 정답 생성. 미래 궤적·전문가 경로·속도·인지 head loss가 모두 유한 |
| CARLA/MORAI 폐루프, Bench2Drive 재현 | NOT TESTED | 공개 Arrow를 사용한 offline 검증 |

공식 `PolicyRunner.forward`는 공식 training precision에서 결정한 BF16 autocast로 모델을 실행한다. 모델의 속도 후처리는 공식 agent의 로직에 맞게 같은 설정의 braking threshold를 적용했다. GPU에 적재된 상태와 출력 shape, 수치·플롯은 [실제 관측 Walkthrough](10_END_TO_END_WALKTHROUGH.md)에 기록했다. [고정 PolicyRunner](https://github.com/kesai-labs/lead/blob/3764f79698922896638638df8062f2e064034610/src/lead/evaluation/inference/policy_runner.py) · [검증 스크립트](../TFv6/tools/verify_baseline.py)

## 재현 명령

Phase 1 설치를 마친 저장소 루트에서 시작한다. 스크립트는 기본적으로 `TFv6/checkpoints/`와 `TFv6/phase2-local/`을 사용한다. 환경에 설정값 덮어쓰기가 있다면 `LEAD_CONFIG`를 해제한다. 다운로더는 고정 리비전의 파일만 받아 모델 바이트 수와 SHA-256을 강제한다. 공개 파일 다운로드에 토큰이 필요하지 않았다. GitHub 실행 환경이 아닌 이 시뮬레이션 PC에서 검증했다. [환경 설치](../TFv6/README.md) · [다운로더](../TFv6/tools/fetch_baseline.py)

```bash
unset LEAD_CONFIG
HF_HUB_DISABLE_XET=1 TFv6/.venv/bin/python -Bu TFv6/tools/fetch_baseline.py
export PY123D_DATA_ROOT="$PWD/TFv6/phase2-local/data"
TFv6/.venv/bin/python -Bu TFv6/tools/verify_baseline.py
```

이 호스트의 libjpeg-turbo 2.0.3은 Phase 1의 PyTurboJPEG 2.5와 호환되지 않았다. 공식 libjpeg-turbo 3.2.1 배포 파일을 로컬 디렉터리에 풀고 프로세스에만 라이브러리 경로를 지정한 후 위 검증과 벤치마크가 통과했다. 같은 오류가 발생하는 Ubuntu x86-64 환경에서 적용한다. 3.2.1 설치 패키지의 SHA-256은 아래에서 확인한다. `PY123D_DATA_ROOT`는 pinned py123d의 전역 map resolution에도 사용되므로 `SceneLoader` 인자로만 경로를 주면 BEV 정답 생성에서 Town01 지도를 찾지 못할 수 있다. [libjpeg-turbo 공식 릴리스](https://github.com/libjpeg-turbo/libjpeg-turbo/releases/tag/3.2.1)

```bash
mkdir -p TFv6/phase2-local/libjpeg
(
  cd TFv6/phase2-local/libjpeg
  curl -fL -O https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.2.1/libjpeg-turbo-official_3.2.1_amd64.deb
  echo '21c7af6e505386dda0fa05e9eab5a8d288947f2e09ade8d7a48ff393ed08182d  libjpeg-turbo-official_3.2.1_amd64.deb' | sha256sum -c - &&
    dpkg-deb -x libjpeg-turbo-official_3.2.1_amd64.deb .
)
export LD_LIBRARY_PATH="$PWD/TFv6/phase2-local/libjpeg/opt/libjpeg-turbo/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PY123D_DATA_ROOT="$PWD/TFv6/phase2-local/data"
TFv6/.venv/bin/python -Bu TFv6/tools/verify_baseline.py --figures docs/assets
TFv6/.venv/bin/python -Bu TFv6/tools/benchmark_baseline.py --figure docs/assets/latency.png
```

`verify_baseline.py`는 기본 anchor 50의 입출력·정답·스키마를 `TFv6/phase2-local/verification.json`에 기록한다. 기본 실행에는 그림 생성이나 학습이 없다. `--figures`를 지정하면 공식 공개 로그에서 파생된 카메라·계획·인지 그림과 rig를 생성한다. 벤치마크 수치는 [측정 문서](08_INFERENCE_BENCHMARK.md)에 있다.
