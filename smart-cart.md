# Carrot-washing line transport automation (agricultural processing facility)

| | |
|---|---|
| **Period** | 2026 (ongoing) |
| **Programme** | JST 공유대학 문제해결형 프로젝트 |
| **Partner** | A regional agricultural corporation's carrot-washing facility |
| **Role** | **Team lead** (M.S.) — 2 undergraduate members; supervising professor; company contact |
| **Deliverables so far** | Field survey & floor plan · process-flow analysis · performance targets · manufacturer specification · monthly activity reports |

## Process as observed

Forklift brings pallets of unwashed carrots → worker manually loads a non-powered roller conveyor into the washer → washer runs a batch in a rotating drum → worker transfers to plastic crates and palletises → forklift moves pallets to trimming. Workspace 7.40 × 7.00 m, ceiling 2.97 m.

## What changed because of the analysis

The project started as a *worker-following cart*. After measuring the room and mapping the flow, the bottleneck turned out to be **loading and palletising, not transport**. The plan was re-scoped at the mid-term review to **two devices**:

1. **Fixed-route autonomous cart** (wash → trimming transfer); the original person-following / detection stack is retained as a safety-perception function
2. **Lifting & rotating input lift** placed next to the washer, feeding carrots through a gentle chute (dropping from height bruises carrots)

Layout kept close to the existing floor plan so the facility's routine does not change.

## Targets (draft, to be fixed at the next stage)

| Item | Target |
|---|---|
| Cart payload | 100 kg |
| Stop accuracy | ± 10 cm |
| Stop after obstacle detection | ≤ 1 s |
| Speed / range | 4 km/h · 5 km per charge |
| Lift capacity | 100 kg class |

## My part

Field measurement and floor plan, process-flow analysis and bottleneck identification, re-scoping proposal at the mid-term review, numeric targets, text specification for the manufacturer meeting (3D rendering to follow once the spec is fixed), task split and review with the undergraduates, monthly reports to the company contact and supervisor.

## Status

Manufacturer consultation stage. Build / test results and productivity figures are not yet available. <!-- TODO: 진행 상황 갱신 -->
