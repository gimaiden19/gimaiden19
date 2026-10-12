# NutriHeal: Graph-Derived APWIS Evidence with Goal-Conditioned Selective Verification under Incomplete Nutrition Data

| | |
|---|---|
| **Authors** | Jonghun Gim, Sunwoo Ko — Department of Artificial Intelligence, Jeonju University |
| **Venue** | *Knowledge-Based Systems* (Elsevier, SCI), manuscript KNOSYS-D-26-11029 |
| **Status** | **Under review** (2026) |
| **Preprint** | SSRN (Elsevier First Look) <!-- TODO: link --> |
| **Code & data** | [Zenodo 10.5281/zenodo.20152417](https://doi.org/10.5281/zenodo.20152417) |
| **Funding** | 2026-RISE-13-JJU |
| **Role** | First author — framework design, evidence encoding, gate calibration, experiments, writing |
| **Related** | Precursor: [JIIS 2026](2026-JIIS-topn-serving-policy.md) · Patents [P1, P2](patents.md) |

## One-line summary

Instead of labelling every dish, **endorse only the few whose evidence is strong enough to meet a confidence target, and stay silent on the rest** — with the gate calibrated on validation data only, never on test.

## Problem

Someone managing a health condition cannot rely on recipe platforms to verify whether a dish meets a nutritional goal, because nutrient information for many ingredients is missing or incomplete. A system that labels every dish under these conditions presents unreliable recommendations as if they were trustworthy. In a health context, **a false endorsement may be more harmful than no endorsement at all.**

## Method

- **NUTRIHEAL-GRAPH** — an ingredient–nutrient knowledge graph that links recipes to nutritional evidence
- **APWIS** (weighted pooling) — encodes the evidence for each *dish–goal pair* as a compact structured signal
- **HGB integration** — gradient-boosted classifier over the evidence signal
- **Per-goal confidence gate**, calibrated solely on a held-out validation set (validation-locked thresholds) to a target precision of **0.90**

## Results (dish–goal pairs from a Korean recipe platform)

| Metric | Value |
|---|---|
| Per-goal test precision (4 goals) | 0.806 – 0.920 |
| Pooled test precision | **0.827**, Wilson 95 % CI [0.782, 0.864] |

- Ablations show a clear **precision–coverage trade-off** as the evidence representation is expanded
- A baseline that ignores nutritional evidence **fails to reach the target precision** — the nutritional signal is what drives the selectivity of the gate

## Why it matters beyond nutrition

The same structure — *evidence → calibrated confidence gate → endorse / abstain* — is what an AI agent needs before it is allowed to act on a physical system: an explicit, auditable boundary between "execute automatically" and "ask a human".

## Keywords

selective prediction · goal-conditioned classification · knowledge graph · incomplete data · calibrated abstention · nutrition recommendation

## Cite (preprint)

```bibtex
@unpublished{gim2026nutriheal,
  author = {Gim, Jonghun and Ko, Sunwoo},
  title  = {{NutriHeal}: Graph-Derived {APWIS} Evidence with Goal-Conditioned Selective Verification under Incomplete Nutrition Data},
  note   = {Under review, Knowledge-Based Systems (KNOSYS-D-26-11029). Preprint on SSRN},
  year   = {2026}
}
```
