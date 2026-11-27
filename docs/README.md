# Phase 2 · TFv6 베이스라인 기술 문서

Phase 2의 기준은 LEAD `main` v1.5.1, Vision-only ResNet34 seed 0이다. 독자는 앞에서부터 모델의 목적과 구조를 익히고, 공식 CARLA 로그의 실제 한 관측을 따라 입력에서 계획 결과와 지도 학습 정답까지 확인할 수 있다.

| 순서 | 문서 | 확인할 내용 |
|---:|---|---|
| 1 | [TFv6 개요](01_TFV6_OVERVIEW.md) | LEAD와 TFv6의 관계, 체크포인트 출처 |
| 2 | [모델 아키텍처](02_MODEL_ARCHITECTURE.md) | ResNet34, latent BEV, attention, 계획·인지 head |
| 3 | [입력·출력 계약](03_MODEL_INPUT_OUTPUT.md) | 카메라 보정, 좌표계, Navigation, Route·Waypoint·Speed |
| 4 | [데이터 흐름](04_DATA_FLOW.md) | 추론·학습 실행 경로, shape와 device, GT 분리 |
| 5 | [학습 데이터와 정답](05_TRAINING_DATA_AND_LABELS.md) | Arrow 스키마, 속도·궤적·인지 정답, 수집 요구사항 |
| 6 | [학습과 Loss](06_TRAINING_AND_LOSS.md) | 사전·후속 학습, loss 수식과 ablation 한계 |
| 7 | [독립 추론 검증](07_INFERENCE_VERIFICATION.md) | 엄격 로딩, 공개 데이터, 재현 명령, 시각화 |
| 8 | [RTX 4080 SUPER 벤치마크](08_INFERENCE_BENCHMARK.md) | 정밀도, 지연·VRAM, 수치 차이와 100 ms 예산 |
| 9 | [MORAI 연동 요구사항](09_MORAI_INTEGRATION_REQUIREMENTS.md) | 기존 Expert·MPC·DAgger 재사용에 필요한 데이터 검증 |
| 10 | [공식 관측 Walkthrough](10_END_TO_END_WALKTHROUGH.md) | 실제 RGB, 입력 Tensor, 계획, 정답과 loss를 한 번에 추적 |

처음 읽을 때는 **개요 → 아키텍처 → 입출력 → Walkthrough** 순서가 좋다. 수집·학습 설계에는 04–06과 09, 실행 재현에는 07–08을 참고한다.

| 문서에서 사용하는 근거 | 의미 |
|---|---|
| Source Verified | 고정 LEAD 또는 py123d 소스에서 동작 확인 |
| Config Verified | 공개된 seed 0 YAML과 해석된 설정에서 확인 |
| Runtime Verified | 시뮬레이션 PC에서 실제 실행 결과 확인 |
| Official Report | 공개 논문·릴리스·모델/데이터 카드에 보고 |
| Interpretation | 코드나 실측값을 해설한 내용 |
| Proposal | 향후 MORAI에서 시험할 요구사항·가설 |
| Unverified | 이 단계에서 실측하지 못한 항목 |

공식 가중치 771개 key의 strict load와 연속 세 프레임의 독립 추론을 통과했다. RTX 4080 SUPER의 공식 BF16 설정에서 모델 단독 p99 14.36 ms, 20개 관측 offline p99 23.31 ms를 기록했다. 벤치마크와 시각화의 범위는 한 개의 공식 normal-view CARLA 로그다. MORAI 센서 경로 지연, 차량 제어와 10 Hz 폐루프 성능, A5000 학습은 별도 검증이 필요하다. 실행 환경은 [Phase 1 기록](../TFv6/README.md)에서 확인할 수 있다.

| 완료 기준 | 판정 |
|---|---|
| 공식 모델 리비전·해시, 구조·입출력·파라미터, 공개 샘플 strict load와 forward | PASS |
| 공식 Arrow 및 GT label, loss와 학습 코드 분석, Expert/DAgger 계약 대조 | PASS |
| RTX 4080 SUPER의 FP32·FP16·BF16·공식 설정, 수치 비교 및 100 ms 예산 | PASS |
| 11개 기술 문서와 다이어그램, 실제 관측 수치·그림, 두 README 및 재현 도구 | PASS |
| `torch.compile` 선택 검사, Bench2Drive 재실행, A5000 학습 및 MORAI 폐루프 | NOT TESTED |
