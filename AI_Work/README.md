# DroneAI AI Experiments

This folder contains the current AI/ML experiments for the DroneAI project. The goal is to classify drone flight events from labeled video clips, including:

- takeoff
- land
- minor-crash
- severe-crash

The current research direction is focused on moving from static image classification toward motion-based video understanding.

---

## 1. Dataset Overview

The labeled clips were created using the DroneAI labeling GUI. Each clip is a short video segment corresponding to a drone event label. Frames were extracted from these clips and organized into a structured frame dataset.

Current clip count:

| Label | Clips |
|---|---:|
| land | 12 |
| minor-crash | 25 |
| severe-crash | 24 |
| takeoff | 22 |
| **Total** | **83** |

The dataset is still small and imbalanced, especially for the landing class.

---

## 2. ViT Baseline

The first baseline used a pretrained Vision Transformer model on extracted video frames.

Pipeline:

```text
clip -> extracted frames -> ViT frame classification -> averaged clip prediction## DroneAI Progress Visualizations
### Accuracy Table
The table below tracks model accuracy and keeps earlier baseline results for paper writing.
| Method | Split | Accuracy | Macro F1 | Weighted F1 | Run | Notes |
|---|---|---:|---:|---:|---|---|
| Static ViT baseline | clip | 31.58% |  |  |  | Static frame baseline; weak because events are motion-based. |
| Farneback motion RF | clip | 42.86% |  |  |  | Classical motion features with Random Forest. |
| Farneback optical-flow LSTM | clip | 52.38% |  |  |  | Previous optical-flow sequence baseline. |
| Farneback optical-flow LSTM | session | 56.25% |  |  |  | Previous optical-flow sequence baseline. |

![DroneAI event-classification accuracy](AI_Work/results/readme_assets/accuracy_bar.png)

### Object-Centered Optical Flow Visualization

Object-flow examples were not found. Run DPFlow extraction with debug images enabled.

### Time-Series Motion Data
These plots show the motion signal over time instead of treating each frame independently. This is important because takeoff, landing, and crash are motion events.
