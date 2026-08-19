| Method | Split | Accuracy | Macro F1 | Weighted F1 | Run | Notes |
|---|---|---:|---:|---:|---|---|
| Static ViT baseline | clip | 31.58% |  |  |  | Static frame baseline; weak because events are motion-based. |
| Farneback motion RF | clip | 42.86% |  |  |  | Classical motion features with Random Forest. |
| Farneback optical-flow LSTM | clip | 52.38% |  |  |  | Previous optical-flow sequence baseline. |
| Farneback optical-flow LSTM | session | 56.25% |  |  |  | Previous optical-flow sequence baseline. |
