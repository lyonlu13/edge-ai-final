import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np
from peft import PeftModel

from hqq_utils import AutoHQQHFModel
from hqq.utils.patching import prepare_for_inference, recommended_inductor_config_setter
from quant_cfg import get_quant_config_slm


model_name = "meta-llama/Llama-3.2-3B-Instruct"
device= "cuda:0"
torch.manual_seed(0)
random.seed(0)
recommended_inductor_config_setter()
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=device,
)
quant_config = get_quant_config_slm(model)
AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float16, device=device)
fine_tuned_model = "./peft_model" 
model = PeftModel.from_pretrained(model, fine_tuned_model).merge_and_unload()

# save merged model
torch.save(model, "merged_model.pth")