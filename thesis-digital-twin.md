# 소량 시험자료 기반 디지털 트윈 보완과 물량 확대 시 도입 의사결정 평가 (가제)

*Augmenting a manufacturing-process digital twin from small-batch trial records and evaluating technology-adoption decisions under scaled-up volume* (working title)

| | |
|---|---|
| **Type** | M.S. thesis, Department of Artificial Intelligence, Jeonju University |
| **Status** | **In progress** — comparison experiments computed and numerically verified; interpretation and contribution framing under way |
| **Expected** | February 2027 (thesis) · journal submission planned |
| **Stack** | Python · NumPy · SciPy (Powell optimisation) · scikit-learn · joblib · event-based process simulation |

_The title, results and contribution below are provisional and will change. Nothing here is a submitted or accepted claim._

## 한 줄 요약

작은 물량의 시험에서 얻은 작업 기록으로 제조공정 디지털 트윈을 보완할 때, **최종 성과를 직접 고치는 것**과 **내부 작업시간을 고친 뒤 공정을 다시 계산하는 것** 중 어느 쪽이 더 큰 물량에서의 성과 예측과 기술 도입 판단에 유리한지 — 그리고 공정 규칙이 일부 틀렸을 때도 그 이점이 유지되는지 — 비교하는 연구.

## The decision problem

A manufacturer trying a new piece of equipment or a new work method runs a small trial first. The question is whether the benefit seen in that trial holds at full production volume. A task that ran 10 s slower in the trial does not simply add 10 s × repetitions at scale: whether the delay is absorbed or amplified depends on repetition count, concurrency, shared workers and pickup timing. So the thesis asks whether extrapolating the trial's *total* time difference is enough, or whether the process structure has to be re-simulated.

## Setup

- **Process**: a synthetic partial process with batch work, transport, pickup waiting and shared workers; two alternative configurations handling the same volume
- **Engineering model**: event-based simulation of each configuration → completion time and labour input; *benefit* = completion-time difference between configurations
- **Data**: computer-generated trial records of both configurations (synthetic, not field data — stated explicitly)
- **AI's role**: learn the residual error the engineering model leaves, while keeping the process relations the model already encodes (engineering + data-driven hybrid)

## Two correction strategies compared

| | Direct benefit correction | In-process correction |
|---|---|---|
| Where the correction is applied | Final completion-time difference | Unit task times of each configuration |
| Extrapolation to larger volume | Add learned correction to engineering benefit | Re-simulate the process with corrected task times |
| Model | Linear correction, 4 features (closed form) | Linear task-time correction, 8 coefficients (Powell search) |
| Key question | Does the learned trend survive beyond the trial range? | Does the process structure propagate the correction usefully, even when rules are partly wrong? |

Both use the same records, same engineering base and same benefit-fitting objective; the comparison is not a perfectly isolated test of correction *location* alone (coefficient count and optimiser differ), and this is stated in the thesis.

## Evaluation design

- Training volumes 27 / 36 / 45 / 54 boxes; **extrapolation** evaluated at 63–81 boxes (15 conditions) plus 12 in-range conditions
- 16 training-data bundles × 4 model/rule conditions = **64 fit–evaluate bundles** (not 64 factories)
- Axes varied: correction location · whether task-time generation is expressible by the engineering model · whether the pickup-waiting rule is correct · which optimiser iterate is kept (final vs best-loss)
- Metrics kept separate: **benefit prediction error (MAE)** · **adoption-choice loss** (benefit forgone by choosing the wrong configuration) · **constraint misjudgement** (due-date / labour)
- Adoption threshold: +2 min of benefit (0 and 5 min as fixed comparisons); thresholds are research settings, not measured installation costs

## Provisional findings (as of 2026-10-12)

Extrapolated-volume benefit MAE (minutes; lower is better; in-process column keeps the best-loss iterate):

| Task-time relation · waiting rule | Engineering base | Workload variant | Direct correction | In-process correction |
|---|---:|---:|---:|---:|
| Expressible · rule correct | 0.2078 | 1.7184 | 0.3725 | 0.6339 |
| Expressible · rule wrong | 3.9873 | 3.7988 | 3.2269 | **1.7968** |
| Hard to express · rule correct *(pre-registered main condition)* | 1.9368 | 2.4431 | 1.7571 | **1.6517** |
| Hard to express · rule wrong | 5.7611 | 2.5926 | **2.0384** | 2.0926 |

- Main condition: in-process correction is lower by ~0.105 min, but the paired conditional SE is 0.085 min → **not claimed as a stable advantage**
- When the engineering model and the waiting rule are both right, the plain engineering base is best — **adding AI correction is not uniformly beneficial**
- At the 2-min threshold on extrapolated volumes, **all methods have zero choice loss**: prediction differences did not change the decision there; differences appear only in some in-range conditions, where the workload engineering variant is sometimes better

What this currently supports: *a comparison of the conditions under which each strategy helps or hurts.* It does not support claims about field productivity gains, cost savings or ROI.

## Intended contribution (to be confirmed)

1. When does using the process's repetition / waiting / resource-sharing structure add value for predictions **beyond the trial range**?
2. Separate **prediction error** from **decision loss** — under what conditions does better prediction actually change the adoption decision?
3. How do **wrong process rules** and **optimiser-iterate selection** change the evaluation, and therefore the method's applicability?

## Engineering practices carried over from earlier work

Seed control · train/eval separation with leakage checks · saved inputs, models, predictions and scoring records for every bundle · numeric verification before interpretation.
