import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np
import argparse
from hqq_utils import AutoHQQHFModel, get_size_of_model
from hqq.utils.patching import prepare_for_inference, recommended_inductor_config_setter
from quant_cfg import get_quant_config_slm

#####################################################################
# === SPEC NOTICE ===
# Only "load model" and "generate" function selection can be modified.
# DO NOT change PPL calculation, timing, or throughput logic.
#####################################################################

# === (Optional) Define your own custom generate function. ===
# This is useful if you want full control over KV cache and generation steps.
# You can modify this function to suit your needs.
# By default, we use model.generate() for simplicity and general use.
def generate(model, input_ids, past_key_values, max_new_tokens):
    input_ids = input_ids.clone()
    with torch.no_grad():
        # Prefill
        assert input_ids.device.type == 'cuda'

        outputs = model.prefill_forward(input_ids, past_key_values=past_key_values, position_ids=None, attention_mask=None, cache_position=None, logits_to_keep=1)
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # Token-by-token Decoding
        for _ in range(max_new_tokens):
            pos = input_ids.shape[1]
            cache_position = torch.arange(pos, pos + 1, device=input_ids.device, dtype=torch.long)

            outputs = model(
                next_token,
                past_key_values=past_key_values,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position
            )
            logits = outputs.logits
            next_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values

    return input_ids

def evaluate_ppl(model, tokenizer, device):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    model.seqlen = 2048
    test_enc = test_enc.input_ids.to(device)

    nsamples = test_enc.numel() // model.seqlen
    nlls = []  
    for i in tqdm(range(nsamples), desc="Evaluating..."):
        batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]

        # with torch.no_grad():
        #     lm_logits = model(batch).logits
        with torch.no_grad():
            lm_logits = model.prefill_forward(batch).logits

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    return ppl.item()

def main():
    ############## Set Up ##############
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--nbits", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use: cuda:0, cuda:1, etc.")
    args = parser.parse_args()

    group_size = args.group_size
    nbits = args.nbits
    device = args.device

    torch.manual_seed(0)
    random.seed(0)
    recommended_inductor_config_setter()
    max_new_tokens = 256    # Number of new tokens to generate
    

    ### === TODO: Load your model (you may change this part) ===
    model_name = "meta-llama/Llama-3.2-3B-Instruct"   
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    #####################################
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.prefill_forward = model.forward
    model.forward = torch.compile(model.forward, mode='max-autotune', dynamic=False, fullgraph=True)
    # === (Optional) Uncomment the following lines if using the custom generate() function. ===
    # quant_config = get_quant_config_slm(model)
    quant_config = get_quant_config_slm(model, nbits, group_size)
    AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float16, device=device)

    backend = 'gemlite'
    prepare_for_inference(model, backend=backend)
    torch.cuda.empty_cache()

    warmup_prompt = "Explain what AI is."
    inputs = tokenizer(warmup_prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # === (Optional) Set up StaticCache for manual KV cache management ===
    from transformers import StaticCache
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_new_tokens + 16,
        device=device,
        dtype=torch.float16
    )
    ####################################################################
    for _ in tqdm(range(5), desc="Warm Up..."):
        #  === Default: use model.generate() for end-to-end warm-up === 
        # _ = model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     pad_token_id=tokenizer.eos_token_id,
        # )
        
        # === (Optional) Use custom generate() if uncommented ===
        _ = generate(model, input_ids, past_key_values, max_new_tokens)
        past_key_values.reset()

    prompt = "How to learn a new language?"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    tputs = []
    time_record = []
    for _ in tqdm(range(10), desc="Test Inference"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # === Default: Use model.generate() for end-to-end timing === 
        # generated = model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     pad_token_id=tokenizer.eos_token_id,
        # )
        
        # === Optional: Use custom generate() if uncommented ===
        generated = generate(model, input_ids, past_key_values, max_new_tokens)
        past_key_values.reset()

        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        tput = max_new_tokens / (elapsed_ms / 1000)
        time_record.append(elapsed_ms / 1000)
        tputs.append(tput)

    response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    sorted_tputs = np.sort(tputs)[2:-2]
    org_tput = np.mean(sorted_tputs)
    print(f'Prompt: {prompt}\nResponse: {response}\n')
    print(f'Time Record: {time_record}')
    print(f'Throughput Record: {tputs} toks/s\n')
    ### Your final throughput result ###
    print(f'Throughput: {org_tput} toks/s')

    ppl = evaluate_ppl(model, tokenizer, device)
    print(f"Perplexity (PPL): {ppl}")

    # Save results to CSV
    import csv
    with open("result.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "value"])
        writer.writerow([0, round(ppl, 2)])
        writer.writerow([1, round(org_tput, 1)])

if __name__ == '__main__':
    main()
