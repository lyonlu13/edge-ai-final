# ref:
#   https://medium.com/@dataoilst.info/breakdown-of-hugging-face-peft-776539e45231
#   https://github.com/mobiusml/hqq/blob/master/examples/hqq_plus.py
#   https://github.com/unslothai/unsloth/issues/1264
#   https://stackoverflow.com/questions/79546910/typeerror-in-sfttrainer-initialization-unexpected-keyword-argument-tokenizer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random
from hqq_utils import AutoHQQHFModel, get_size_of_model
from hqq.utils.patching import recommended_inductor_config_setter
from quant_cfg import get_quant_config_slm
from peft import get_peft_model, LoraConfig
from peft import prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

torch.manual_seed(0)
random.seed(0)
recommended_inductor_config_setter()

max_new_tokens = 256
device = "cuda:0"

model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=device,
)

print("Start quantization...")
print("Model size before quantization:", get_size_of_model(model) / 1e6, "MB")
quant_config = get_quant_config_slm(model)
AutoHQQHFModel.quantize_model(
    model, quant_config=quant_config, compute_dtype=torch.float16, device=device
)
print("Model size after quantization:", get_size_of_model(model) / 1e6, "MB")

print("Start LoRA...")
model = prepare_model_for_kbit_training(model)


# lora參數
peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    # target_modules=["k_proj", "o_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


print("Loading dataset...")
train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Start training...")
# 訓練參數
grad_acc = 1
logging_st = 1
max_steps = -1
lr = 1e-5
batch_size = 1
n_epochs = 2
max_tokens = 1024
training_arguments = SFTConfig(
    output_dir=".",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_acc,
    learning_rate=lr,
    logging_steps=logging_st,
    num_train_epochs=n_epochs,
    max_steps=max_steps,
    remove_unused_columns=False,
    bf16=True,
    max_grad_norm=1.0,
    save_steps=500,
    lr_scheduler_type="cosine",
    packing=True,  # allows multiple shorter sequences to be packed into a single training example, maximizing the use of the model's context window
    max_seq_length=max_tokens,  # it determines the maximum length of input sequences during fine-tuning
    dataset_text_field="text",  # pointing to the 'text' column in the dataset
)

trainer = SFTTrainer(
    model=model,  # model to train
    train_dataset=train_dataset,  # the training dataset
    eval_dataset=eval_dataset,  # the evaluation dataset
    peft_config=peft_config,  # from LoRA Configuration
    processing_class=tokenizer,  # model tokenizer
    args=training_arguments,  # the training parameters
)

model.train()
trainer.train()

new_model = "peft_model"
print("Saving model...")
trainer.model.save_pretrained(new_model)
