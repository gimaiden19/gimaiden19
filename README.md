# Jonghun Gim (김종훈)

M.S. candidate · Department of Artificial Intelligence, Jeonju University (Jeonju, Korea)  

**Research question I keep returning to:** *before a simulation result or an AI judgement is sent to a real machine or a real person, under which conditions can it be trusted — and how do we measure that?*

- 📧 gimaiden19@jj.ac.kr
- 📍 Jeonbuk, Korea

---

## Research threads

| Thread | What it means | Work |
|---|---|---|
| **Sim-to-real gap as a measured quantity** | Distributional distances (Wasserstein, sliced-Wasserstein, JSD) used as *predict → verify → control* signals before deployment, not as post-hoc scores | [KROS 2026 (under review)](publications/2026-KROS-sim2real-gap-diagnosis.md) · [SIVP (under review)](publications/2026-SIVP-fourier-component-selection.md) · [INTCOM 2026](publications/2026-INTCOM-synthetic-layouts.md) |
| **Selective prediction under uncertainty** | Separate what can be decided from what must be abstained, and expose the abstention rate instead of hiding it | [JIIS 2026](publications/2026-JIIS-topn-serving-policy.md) · [KBS (under review)](publications/2026-KBS-nutriheal.md) · [Patents](publications/patents.md) |
| **Digital-twin process models from small data** | Engineering model + data-driven residual correction, evaluated on *decisions*, not fit alone | [M.S. thesis (in progress)](publications/thesis-digital-twin.md) |

---

## Publications at a glance

| # | Title | Venue | Status |
|---|---|---|---|
| J1 | 결측 영양 정보 환경에서 제약 기반 레시피 추천의 Top-N 반환 보장을 위한 운영 정책 | 지능정보연구 32(1) · KCI | **Published** 2026.03 |
| C1 | Designing Synthetic Training Layouts from Real Hand–Product Priors for Industrial Assembly Vision | KSII INTCOM 2026 | **Presented** |
| J2 | Measurement-guided selection of Fourier amplitude components for training-free sim-to-real correction | Signal, Image and Video Processing · SCI | Under review |
| J3 | NutriHeal: Graph-Derived APWIS Evidence with Goal-Conditioned Selective Verification | Knowledge-Based Systems · SCI | Under review |
| J4 | 다중 지표 기반 Sim-to-Real 갭 진단: 시뮬레이터 보정 효과의 예측 구간 및 구동 조건 의존성 분석 | 로봇학회논문지 · KCI | Under review |
| P1–P2 | Top-N 레시피 추천 방법 및 장치 (×2) | KR 10-2026-0166621 / 0166622 | Filed 2026.09 |
| T | 소량 시험자료 기반 디지털 트윈 보완과 물량 확대 시 도입 의사결정 평가 (가제) | M.S. thesis | In progress, Feb 2027 |

Full list with abstracts → [`publications/`](publications/README.md)

---

## Projects

- **Operations-management LLM Agent app** (2026.09 –) — Next.js 15 + SQLite/Drizzle, role-based agents (accounting / content / HR) on the Claude Messages API with an Ollama-swappable adapter, Electron installer. → [projects/theone-agent-app.md](projects/theone-agent-app.md)
- **Carrot-washing line transport automation** (2026, team lead) — field measurement, bottleneck analysis, spec for an autonomous cart + lift. → [projects/smart-cart.md](projects/smart-cart.md)
- **Teaching** — Isaac Lab RL labs, synthetic-data & sim-to-real gap module, ROS2/Gazebo SLAM lab (bootcamp / undergraduate TA). → [projects/teaching.md](projects/teaching.md)

---

## Tools

Isaac Sim 6 · Isaac Lab · Python (NumPy / SciPy / pandas / scikit-learn / PyTorch) · SimPy-style event simulation · Next.js · SQLite · Claude API · Linux GPU server (RTX 5090 ×2)

> How I work: I design the experiment, the data split and the acceptance criteria myself; implementation is done in close collaboration with AI coding agents, and I review and verify the resulting code. I say this up front because it is how the work above was actually produced.

_Last updated 2026-10-12_
