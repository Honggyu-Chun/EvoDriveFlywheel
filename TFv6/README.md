# TFv6 · Phase 1 개발 환경

`TFv6/`는 EvoDriveFlywheel의 camera-only TransFuser v6 Vision-only ResNet34를 개발할 LEAD 환경이다. `lead/`에 공식 소스를 두고, `.venv/`에 이 프로젝트 전용 Python과 의존성을 설치한다. 이 단계의 검증 범위는 패키지 설치, 주요 모듈 import, 대상 GPU의 기본 CUDA 연산이다.

## Upstream

| 항목 | 값 |
|---|---|
| 저장소 | `https://github.com/kesai-labs/lead.git` |
| 계열 | `main` |
| 고정 커밋 | `3764f79698922896638638df8062f2e064034610` |
| 커밋 일시 | 2026-09-14 09:09:02 UTC |
| 릴리스 | v1.5.1, 실제 Git 태그 `v.1.5.1` |
| LEAD 패키지 | `1.5.1`, MIT 라이선스 |
| py123d | `0.7.0`, Git 커밋 `4d20a6cfd50f7966647448084f6482c0b65effff` |

클론 시점에 `origin/main`과 `v.1.5.1` 태그가 동일한 커밋을 가리켰다. Vision-only ResNet34 체크포인트의 릴리스 계열에 맞춰 이 커밋을 선택했다. 체크포인트 바이너리와의 strict-load 호환성은 다음 단계에서 확인한다. [LEAD 공식 릴리스](https://github.com/kesai-labs/lead/releases/tag/v.1.5.1) · [Vision-only 모델 공개 페이지](https://huggingface.co/ln2697/transfuser-carla-123d/tree/main/visiononly_resnet34_v1.5.1)

## 시스템 조사

2026-11-26 기준 시뮬레이션 PC에서 조사했다. 학습 서버용 환경은 별도로 검증해야 한다.

| 항목 | 실측 |
|---|---|
| OS 및 커널 | Ubuntu 20.04.6 LTS, `5.15.0-139-generic`, x86-64, glibc 2.31 |
| CPU·메모리 | Intel Core i7-13700, 16코어·24스레드, RAM 31 GiB |
| 프로젝트 파일시스템 | 916 GiB, 설치 전 가용 약 450 GiB |
| GPU | NVIDIA GeForce RTX 4080 SUPER 1장, 16,376 MiB |
| NVIDIA 드라이버 | `560.35.03`, `nvidia-smi` 표시 CUDA `12.6` |
| 시스템 CUDA Toolkit | 11.1 및 11.8 설치 확인. `/usr/local/cuda`는 11.1을 가리키고, PATH의 `nvcc`는 11.8.89 |
| GPU 사용 상태 | 초기 조사에서 Xorg와 GNOME Shell이 그래픽 메모리를 사용 중이었고 CUDA compute 프로세스는 0개 |
| 시스템 Python / pip | `/usr/bin/python3` 3.8.10, `python3 -m pip` 20.0.2 |
| 기존 환경 도구 | Conda 4.7.12와 기존 LEAD·perception 가상환경이 있었음. 초기 PATH에서 uv와 Micromamba를 찾을 수 없었음 |
| TFv6 도구 | CPython 3.10.22, uv 0.13.10. CPython도 `.venv/.python`에 보관 |

| 설치 패키지 | 확인한 버전 |
|---|---|
| PyTorch / TorchVision | `2.8.0+cu128` / `0.23.0+cu128` |
| PyTorch CUDA 빌드 / Runtime 패키지 | `12.8` / `nvidia-cuda-runtime-cu12==12.8.90` |
| Lightning | `lightning==2.6.1` 및 간접 의존성 `pytorch-lightning==2.6.6` |
| py123d / PyArrow | `0.7.0` / `25.0.1` |
| CARLA Python API | `0.9.16` |

## 의존성 근거

- **Python 및 빌드.** LEAD `pyproject.toml`의 Python 범위는 `>=3.10,<3.13`이고 빌드 백엔드는 `setuptools==81`이다. 공식 개발 안내와 CI의 기본 Python은 3.10이다. 고정 커밋에 `uv.lock`은 없다. [LEAD 설정](https://github.com/kesai-labs/lead/blob/3764f79698922896638638df8062f2e064034610/pyproject.toml) · [CI 환경 설정](https://github.com/kesai-labs/lead/blob/3764f79698922896638638df8062f2e064034610/.github/actions/python_venv/action.yml)
- **런타임과 개발 도구.** 공식 패키지는 Torch 2.8, TorchVision 0.23, Lightning 2.6.1, CARLA Python API 0.9.16을 요구한다. PyTorch와 TorchVision의 공식 인덱스는 `cu128`이다. 테스트·정적 분석 도구인 pytest, Ruff, basedpyright 등도 같은 공식 의존성 목록에 선언되어 있다.
- **데이터 계층.** py123d는 LEAD가 선언한 Git 커밋을 설치한다. PyArrow와 Apache Arrow IPC를 사용해 모달리티별 로그를 읽는다. 학습은 LEAD의 PyTorch Lightning 구현을 사용한다. [py123d 고정 소스](https://github.com/kesai-labs/py123d/tree/4d20a6cfd50f7966647448084f6482c0b65effff) · [LEAD 데이터 접근](https://github.com/kesai-labs/lead/blob/3764f79698922896638638df8062f2e064034610/docs/data_access.md) · [LEAD 학습](https://github.com/kesai-labs/lead/blob/3764f79698922896638638df8062f2e064034610/docs/training.md)
- **시스템 라이브러리.** 호스트의 glibc, `libGL`, GLib, `libgomp`, `libxcb-xinerama`, `libturbojpeg.so.0`을 확인했다. LEAD는 카메라 JPEG를 디코딩할 때 `libturbojpeg`를 지연 로딩한다. 설치 범위의 CARLA는 Python API 패키지다.

## 설치 및 활성화

uv 0.13.10 실행 파일을 준비한다. 이 호스트에서는 공식 PyPI wheel의 SHA-256을 확인한 뒤 uv 실행 파일을 추출했다. 아래 명령은 EvoDriveFlywheel 루트에서 시작한다. [uv 0.13.10](https://pypi.org/project/uv/0.13.10/) · [uv 명령 참조](https://docs.astral.sh/uv/reference/cli/)

```bash
mkdir -p TFv6
git clone --branch main --single-branch https://github.com/kesai-labs/lead.git TFv6/lead
git -C TFv6/lead switch --detach 3764f79698922896638638df8062f2e064034610

cd TFv6
uv python install --install-dir "$PWD/.venv/.python" --no-bin 3.10.22
phase_python=$(UV_PYTHON_INSTALL_DIR="$PWD/.venv/.python" uv python find --managed-python 3.10.22)
uv venv --allow-existing --python "$phase_python" .venv
cp "$(command -v uv)" .venv/bin/uv

(
    unset CONDA_PREFIX PYTHONPATH PYTHONHOME UV_SYSTEM_PYTHON
    source .venv/bin/activate
    test "$VIRTUAL_ENV" = "$(realpath .venv)" || exit 1
    cd lead
    "$VIRTUAL_ENV/bin/uv" pip install --python "$VIRTUAL_ENV/bin/python" --torch-backend cu128 --strict -c ../requirements.lock -e .
    "$VIRTUAL_ENV/bin/uv" pip check --python "$VIRTUAL_ENV/bin/python"
)
```

이미 구축한 환경은 프로젝트 루트에서 다음과 같이 사용한다.

```bash
source TFv6/.venv/bin/activate
python --version
```

설치 환경은 Linux x86-64, Python 3.10.22, cu128 기준이다. 처음에는 LEAD의 공식 editable 설치 명령에 `--torch-backend cu128`을 명시해 설치했다. 그 결과에서 editable LEAD를 제외한 283개 패키지를 [requirements.lock](requirements.lock)에 고정했다. 별도 가상환경에서 위 고정 목록과 이전에 받은 패키지 캐시로 설치를 재실행했고, 283개 항목이 모두 일치했다. 위 명령에서 LEAD는 지정 커밋의 소스에서 editable로 설치된다.

Upstream 설치 과정에서 로컬 `lead/.env`와 패키지 메타데이터가 생성됐다. 상위 프로젝트 Git은 `TFv6/lead/`와 `TFv6/.venv/`를 제외한다. 현재는 소스 커밋에 고정한 독립 클론을 사용한다. 향후 공동 배포에서 소스 포인터를 Git으로 관리해야 한다면 Submodule을 검토하고, Upstream 수정이 생겨 별도 이력을 유지해야 한다면 Fork를 검토한다.

## 검증 결과

| 확인 항목 | 결과 | 근거 |
|---|---|---|
| LEAD 클론·버전 확인 | PASS | 공식 `main`에서 클론. 작업 사본의 HEAD와 `v.1.5.1` 태그가 위 SHA에 일치하고 추적 파일 변경 없음 |
| Python 환경·활성화 | PASS | 독립 CPython 3.10.22의 `sys.executable`이 `TFv6/.venv/bin/python`. 시스템 및 사용자 site-packages 제외 |
| 공식 의존성 설치·일관성 | PASS | 284개 설치. `uv pip check`에서 충돌 0건. 별도 환경에서 같은 버전으로 재설치 |
| PyTorch·TorchVision·Lightning·CARLA API import | PASS | 위에서 확인한 설치 버전으로 모두 import 성공 |
| CUDA 인식 | PASS | `torch.cuda.is_available() == True`. RTX 4080 SUPER 1장, compute capability 8.9 |
| CUDA 연산 | PASS | `cuda:0`에서 FP32 `512×512` 행렬 곱셈 후 동기화 및 전체 결과 유한값 확인 |
| py123d·PyArrow import | PASS | 설치 메타데이터의 py123d 커밋이 고정값과 일치. 메모리 내 Arrow IPC 쓰기·읽기 성공 |
| LEAD import | PASS | `TFv6/lead/src/lead`의 editable 소스에서 import |
| 핵심 및 데이터 로딩 모듈 import | PASS | `AbstractPolicy`, `SceneData`, `Transfuser`, `LeadLightningModule`, `LeadConfig`, `SceneLoader`, `TransfuserDataset`, `ArrowSceneBuilder` |

## 호환성 및 검증 범위

드라이버 `560.35.03`에서 공식 cu128 wheel이 위 CUDA 연산을 수행했다. `nvidia-smi`의 12.6은 드라이버가 표시하는 버전이고, PyTorch Runtime은 wheel 의존성으로 설치된 12.8이다. CUDA minor-version compatibility에서 PTX와 드라이버가 더 필요한 기능에는 제한이 있으므로 `torch.compile`과 CUDA 확장 빌드는 `NOT TESTED`로 남긴다. [NVIDIA CUDA Minor Version Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)

체크포인트 로딩·모델 forward, MORAI 실행, 학습 서버 및 실제 학습 성능은 이번 환경 검증 범위 밖이다.

## 다음 단계

TFv6 Vision-only ResNet34 체크포인트 확보 및 모델 구조 분석.
