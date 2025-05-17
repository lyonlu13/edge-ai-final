# Edge AI
## 可嘗試的方案
- [x] HQQ 加速/減少精準度
- [ ] other generating implement 加速
- [ ] PEFT 提升精準度
- [ ] (?)ONNX/TensorRT 加速

## 目前嘗試過的方案
1. 僅套用 Lab2 的 HQQ，可達到strong baseline: 
    - PPL: 11.46
    - throughput: 58.9
    - 下一步: 可以再砍的用力一點，然後用PEFT提升精準度
