## DroneAI Progress Visualizations
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
