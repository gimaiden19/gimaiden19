# Designing Synthetic Training Layouts from Real Hand–Product Priors for Industrial Assembly Vision

| | |
|---|---|
| **Authors** | Jonghun Gim, Jeongik Min* — Department of Artificial Intelligence, Jeonju University |
| **Venue** | KSII The 21st International Conference on Intelligent Computing (INTCOM 2026) |
| **Status** | **Presented** |
| **PDF** | [(Final) INTCOM2026.pdf](../INTCOM2026.pdf) <!-- TODO: 경로 확인 --> |
| **Role** | First author — formulation, generator design, evaluation, writing |
| **Related** | [SIVP (under review)](2026-SIVP-fourier-component-selection.md) — same "measure the gap first" perspective, applied to appearance instead of layout |

## One-line summary

Synthetic training layouts are generated **to match the structural distribution of the real process** (where hands enter, how close they get to products, how they overlap) instead of being randomly sampled — and the alignment is verified with a distance metric that was *not* used for generation.

## Problem

Synthetic-data pipelines usually rely on random placement or broad domain randomization. That increases appearance diversity but does not reproduce the *structure* of real scenes. In industrial assembly, a vision model needs to know not just whether a hand or product appears, but where products concentrate, how hands enter, and how interactions evolve. Under a fixed generation budget, random layouts spend most of the budget on frames that are visually plausible yet structurally irrelevant.

## Method

- Formulate synthetic layout generation as a **distribution-alignment problem**
- Extract two kinds of priors from real annotation:
  - **Interaction-aware** — hand entry direction, nearest hand–product distance, overlap (IoU), relative offset
  - **Workspace-aware** — product / hand centre heatmaps and workspace mask
- Generate frame-level layouts from these priors and compare against a **count-matched random baseline** with the same budget (5,000 frames)

## Data

KAMP industrial assembly dataset — 5,020 annotated frames, 68,699 bounding boxes.

## Results

| Statistic | Random baseline → Prior-guided |
|---|---|
| Distance JSD | 0.279 → 0.0086 |
| IoU JSD | 0.435 → 0.0163 |
| Entry-direction JSD | 0.185 → 0.0067 |
| Held-out Wasserstein distance (not used in generation) | 99.3 px → 26.5 px (**−73 %**) |

Ablations show that interaction-aware priors alone are not enough; workspace-aware priors add alignment beyond them under the same budget.

## Limitations

Single dataset and single process; downstream detector gains are reported as supporting evidence, not the primary claim. The claim is structural alignment under a fixed budget.

## Keywords

synthetic data generation · distribution alignment · layout generation · industrial assembly

## Cite

```bibtex
@inproceedings{gim2026layouts,
  author    = {Gim, Jonghun and Min, Jeongik},
  title     = {Designing Synthetic Training Layouts from Real Hand--Product Priors for Industrial Assembly Vision},
  booktitle = {KSII 21st International Conference on Intelligent Computing (INTCOM)},
  year      = {2026}
}
```
