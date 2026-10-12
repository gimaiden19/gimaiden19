# Teaching & lab design (TA)

Bootcamp and undergraduate course TA, 2025–2026.

## Labs I designed or ran

| Lab / module | Content | Notes |
|---|---|---|
| **Isaac Lab reinforcement learning** | Cartpole training, reward-design experiments, Go2 quadruped locomotion | Hands-on sessions for undergraduates on the lab GPU server |
| **ND2 Module 16 — Synthetic data & sim-to-real gap measurement** | Cityscapes gtFine **Cut-Paste** pipeline; students generate synthetic data, train a detector and measure the gap on real data | The first PIL-based pipeline produced an almost-zero gap, which hid the lesson; redesigned so the gap is real and measurable — structured-DR sim val **mAP50 0.877 vs. real test 0.211** (gap 0.666) |
| **ND3 Module 15 — ROS2 / Gazebo** | TurtleBot3 SLAM | |

## Materials

- Six notebooks refactored with Korean comments and cross-platform fixes
- Activity logs produced in HWP-compatible format for programme reporting

## What I took from it

Explaining sim-to-real to people who have never trained a model forced the vocabulary I now use in papers: *the gap is something you measure, then decide what to do about* — not something you hope goes away with more randomisation.
