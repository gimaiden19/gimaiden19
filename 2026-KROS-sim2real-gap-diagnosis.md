# 다중 지표 기반 Sim-to-Real 갭 진단: 시뮬레이터 보정 효과의 예측 구간 및 구동 조건 의존성 분석

*Multi-Metric Diagnosis of the Sim-to-Real Gap: Prediction-Horizon and Excitation-Condition Dependence of Simulator Calibration Effects*

| | |
|---|---|
| **Authors** | 김종훈 (Jonghun Gim), 민정익† (Jeongik Min) — 전주대학교 |
| **Venue** | 로봇학회논문지 (Journal of Korea Robotics Society), KCI |
| **Status** | **Under review** (2026) |
| **Role** | First author — replay protocol, metric design, factorial experiment, controls, writing |
| **Related** | [SIVP (under review)](2026-SIVP-fourier-component-selection.md) — same diagnostic stance applied to image appearance |

## 한 줄 요약

"시뮬레이터를 보정했더니 오차가 줄었다"는 보고가 **실제 동역학 충실도 향상인지, 한 지표만 좋아진 상충(trade-off)인지**를 실기 로그와 리플레이만으로 가려내는 진단 절차.

## 문제

로그 기반 시뮬레이터 보정은 대개 **위치 궤적 오차 하나**를 최소화하고 그 감소를 "갭 감소"로 보고합니다. 그러나 그 보정이 속도·주파수응답·시간지연 같은 다른 특성까지 개선한다는 보장은 없습니다. 추가 실기 실험 없이, 어떤 개선이 기준모델과 통제를 통과하는지 확인할 방법이 필요했습니다.

## 설정

- 공개된 **토크제어 3자유도 핑거**(ODRI)의 1 kHz 실기 로그 + Isaac Sim
- **리플레이 프로토콜**: 실기 토크를 시뮬레이터에 그대로 인가하고 1초마다 상태를 실기값으로 재초기화 (순수 개루프는 저질량 링크가 1초 내 발산하여 평가 불가)
- 구동 조건: 정현파(sinusoidal) / 가우시안 프로세스(GP)

## 방법

1. 같은 리플레이 위에서 **5계열 지표** 동시 측정 — 점유(occupancy), 속도, 주파수응답, 시간지연, 예측구간별 발산
2. 관절경계·입력지연·마찰·감쇠 **4개 보정 요인을 2⁴ = 16셀 요인실험**으로 분해, **Shapley 값**으로 지표별 기여 분리
3. **위치유지·등속외삽 두 기준모델**과 비교하여 시뮬레이터가 실제로 우위인 예측구간을 판정
4. 식별에 쓰지 않은 **보류 시퀀스 12개**에서 사전 지정 12개 대비 재평가, 교차 조건 전이, 사후 통제 6종(경계 밖 초기상태·창 정의·다중비교·집계규칙·리셋주기·해 다중성)

## 결과

- 위치 발산 목적함수로 식별한 마찰·감쇠 보정은 **마찰↑·감쇠↓ 대체(substitution)**가 만드는 상충: 점유·장기 발산은 줄지만
  - 속도 변동이 실기의 **18–37 %**로 과억제
  - 정현파 조건 100 ms 발산 **22 % 악화**, 주파수응답 크기 오차 증가
- 단순 **위치유지 기준모델**의 점유 오차가 최선 보정셀보다 **4–11배 낮음** → 점유 오차 감소는 동역학 충실도의 충분조건이 아님
- 시뮬레이터가 두 기준모델을 모두 이기는 구간은 **정현파 100·200 ms뿐**, GP 조건에서는 없음
- 사전 지정 **12개 대비 전부 보류 시퀀스에서 재현** (Bonferroni 기준 9개); 상충은 리셋주기 1·2·4 s와 근최적 대안해에서 유지
- 감쇠·지연은 위치 목적함수로는 **비식별(non-identifiable)**

## 산출물

상충 판정 · 구간 판정 · 결정성 판정의 3단계 출력과 후속 행동 규칙(게이트 통과 셀 선택, 사용 구간 명시, 미결정 파라미터 재식별)으로 정리한 **재사용 가능한 진단 절차**.

## 한계

단일 로봇·두 구동 조건; 식별된 지연은 유효 지연이지 물리 지연의 추정치가 아님; 물리 엔진 간 비교는 범위 밖.

## Keywords

Sim-to-Real Gap · Simulator Calibration · System Identification · Rollout Horizon · Factorial Design · Shapley Value · Torque-Controlled Robot

## Cite

```bibtex
@unpublished{gim2026gapdiagnosis,
  author = {Gim, Jonghun and Min, Jeongik},
  title  = {Multi-Metric Diagnosis of the Sim-to-Real Gap: Prediction-Horizon and Excitation-Condition Dependence of Simulator Calibration Effects},
  note   = {Under review, Journal of Korea Robotics Society (in Korean)},
  year   = {2026}
}
```
