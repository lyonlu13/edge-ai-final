# Edge AI
## 可嘗試的方案
- [x] HQQ 加速/減少精準度
- [ ] other generating implement 加速
- [x] PEFT 提升精準度
- [ ] (?)ONNX/TensorRT 加速

## 目前嘗試過的方案
1. 僅套用 Lab2 的 HQQ，可達到strong baseline: 
    - PPL: 11.46
    - throughput: 58.9
    - 下一步: 可以再砍的用力一點，然後用PEFT提升精準度
2. HQQ砍用力一點，用PEFT降低PPL
    - PPL: 10.52
    - throughput: 62.2
    - 效果不錯，只要PPL不要壞的太誇張，lora都就救得回來
    - 應該還可以更好，但參數不是很好調，還沒找到比較好的實驗方法
    - 使用方式：
        - 調整 quant_cfg 量化參數
        - 調整 peft_train 的 lora 跟訓練參數
        - 跑 python peft_train.py && python peft_test.py 訓練之後看結果
