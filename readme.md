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
    - PPL: 11.46
    - throughput: 72.4
    - 效果不錯，只要PPL不要壞的太誇張，lora都就救得回來
    - 應該還可以更好，但參數不是很好調，還沒找到比較好的實驗方法
    - 使用方式：
        - 調整 quant_cfg 量化參數
        - 調整 peft_train 的 lora 跟訓練參數
        - 跑 python peft_train.py && python peft_test.py 訓練之後看結果

# 環境設置
建立conda環境
```bash
mkdir -p ~/miniconda3

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

bash ~/miniconda3/miniconda.sh -b -u -p ~miniconda3

rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init --all

conda create --name edge--no-default-packages python=3.1111=he870216_0

conda activate edge
pip install torch torchvision torchaudio
conda env update -f environment.yml
conda activate edge
```

# 復現方法1 - lora weight
1. 先至 https://huggingface.co/lyonlu13/edge-ai-final 下載 peft_model目錄 至專案根目錄
2. 執行 `python result.py`
3. 結果位於 `result.csv`

# 復現方法2 - merged model
在一些環境上測試時效果有點差，無法確認原因
1. 先至 https://huggingface.co/lyonlu13/edge-ai-final 下載 merged_model.pth 至專案根目錄
2. 執行 `python result_merge.py`
3. 結果位於 `result.csv`

# merged_model 的產生方式
1. 使用peft_train.py，依照quant_cfg.py的quantize設定建立model，並加上lora weight
2. 直接將模型匯出
