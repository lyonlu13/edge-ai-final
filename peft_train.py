# ref:
#   https://medium.com/@dataoilst.info/breakdown-of-hugging-face-peft-776539e45231
#   https://github.com/mobiusml/hqq/blob/master/examples/hqq_plus.py
#   https://github.com/unslothai/unsloth/issues/1264
#   https://stackoverflow.com/questions/79546910/typeerror-in-sfttrainer-initialization-unexpected-keyword-argument-tokenizer

import torch
import random
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from hqq_utils import AutoHQQHFModel, get_size_of_model
from hqq.utils.patching import recommended_inductor_config_setter
from quant_cfg import get_quant_config_slm
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# === Argument Parser ===
parser = argparse.ArgumentParser(description="Train HQQ + LoRA with FlashAttention")
parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device to use (default: cuda:0)")
args = parser.parse_args()

# Set seeds for reproducibility
torch.manual_seed(0)
random.seed(0)
recommended_inductor_config_setter()

# Parameters
device = args.device
max_new_tokens = 256
model_name = "meta-llama/Llama-3.2-3B-Instruct"

# === Load Model with FlashAttention 2 ===
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=device
    # attn_implementation="flash_attention_2"
)


# === HQQ Quantization ===
print("Start quantization...")
print("Model size before quantization:", get_size_of_model(model) / 1e6, "MB")
quant_config = get_quant_config_slm(model)
AutoHQQHFModel.quantize_model(
    model, quant_config=quant_config, compute_dtype=torch.float16, device=device
)
print("Model size after quantization:", get_size_of_model(model) / 1e6, "MB")


# === Prepare for LoRA Training ===
print("Start LoRA setup...")
model = prepare_model_for_kbit_training(model)

# lora參數
peft_config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    # target_modules=["k_proj", "o_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# === Load Dataset ===
print("Loading dataset...")
train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
tokenizer = AutoTokenizer.from_pretrained(model_name)


# === Training Configuration ===
print("Start training...")
# Training parameters
training_arguments = SFTConfig(
    output_dir=".",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    logging_steps=1,
    num_train_epochs=4,
    max_steps=-1,
    remove_unused_columns=False,
    bf16=True,
    max_grad_norm=1.0,
    save_steps=500,
    lr_scheduler_type="cosine",
    packing=True,                   # allows multiple shorter sequences to be packed into a single training example, maximizing the use of the model's context window
    max_seq_length=1024,            # it determines the maximum length of input sequences during fine-tuning
    dataset_text_field="text",      # pointing to the 'text' column in the dataset
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,    # the training dataset
    eval_dataset=eval_dataset,      # the evaluation dataset
    peft_config=peft_config,        # from LoRA Configuration
    processing_class=tokenizer,     # model tokenizer
    args=training_arguments,        # the training parameters
)

model.train()
trainer.train()


# === Save PEFT model ===
print("Saving model...")
trainer.model.save_pretrained("peft_model")