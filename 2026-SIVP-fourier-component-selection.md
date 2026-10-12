# Measurement-guided selection of Fourier amplitude components for training-free sim-to-real correction in low-label semantic segmentation

| | |
|---|---|
| **Authors** | Jonghun Gim*, Jeongik Min — Department of Artificial Intelligence, Jeonju University |
| **Venue** | *Signal, Image and Video Processing* (Springer, SCI) |
| **Status** | **Under review** (2026) |
| **Preprint** | Research Square <!-- TODO: link --> |
| **Role** | First / corresponding author — protocol design, measurements, experiments, writing |
| **Related** | [INTCOM 2026](2026-INTCOM-synthetic-layouts.md) (layout-level gap) · [KROS (under review)](2026-KROS-sim2real-gap-diagnosis.md) (dynamics-level gap) |

## One-line summary

Before correcting the appearance gap between simulator and real images, **measure which Fourier amplitude component actually causes the downstream error** — then correct only that one, with two FFTs per frame and no retraining.

## Problem

Fourier domain adaptation (FDA) is a cheap, training-free way to transfer appearance from real to synthetic images. But existing methods swap or perturb the low-frequency amplitude **wholesale**, without asking which component — global colour/exposure or texture — is responsible for the error. Over-correcting can destroy useful structure; under-correcting leaves the error in place.

## Method — a measurement-guided protocol

1. **Decompose** the low-frequency amplitude into a global colour/exposure (**DC**) component and a texture-related (**AC**) component
2. **Quantify** each mismatch with sliced-Wasserstein distances over pixel intensities and frozen deep features
3. **Verify** the induced error pattern with confusion-matrix analysis and **negative controls** (apply the same correction to an unrelated region and confirm no effect)
4. **Correct only the selected component** in a single phase-preserving FDA pass

Cost: two FFT operations per frame (forward + inverse). No generator, no retraining, no inference-time change — deployable on embedded on-site hardware.

## Case study

Excavator-worksite semantic segmentation, trained with **1 % labeled real frames** plus simulator-rendered synthetic data (Isaac Sim). The measurements localize the correctable mismatch to **insufficient texture variation in simulated soil** (within-frame brightness variation ≈ 1/6 of real).

## Results

| | Before | After (selected-component FDA) |
|---|---|---|
| Rare-class **Dump IoU** | 0.465 | **0.565** (n = 6 matched seeds, p < 0.001) |
| Dump precision | 0.52 | 0.63 |
| mIoU | 0.755 | 0.778 |

- Significantly outperforms generative translation with **ControlNet** (+0.036, p = 0.022) without any generator
- Matches far costlier per-image diffusion restyling
- Surpasses classical colour and augmentation corrections
- Reported honestly: pixel-level gap closes ~53 %, deep-feature gap only 15–30 % — the protocol tells you **how far a two-FFT correction can go before heavier generative adaptation is justified**

## Limitations

Single dataset / class pair; gains shrink with a pre-trained backbone (+0.022) and beyond ~2 % real labels.

## Keywords

Fourier domain adaptation · amplitude spectrum · sim-to-real domain gap · semantic segmentation · sliced-Wasserstein distance · training-free correction

## Cite (preprint)

```bibtex
@unpublished{gim2026fourier,
  author = {Gim, Jonghun and Min, Jeongik},
  title  = {Measurement-guided selection of {Fourier} amplitude components for training-free sim-to-real correction in low-label semantic segmentation},
  note   = {Under review, Signal, Image and Video Processing},
  year   = {2026}
}
```
