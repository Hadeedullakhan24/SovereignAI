"""Standalone Generation Test with Qwen2.5-1.5B-Instruct (Without RAG).

Validates 100% offline local model loading with local_files_only=True and
measures exact model load time, generation latency, and output token counts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import time

# Prevent OpenMP deadlocks on Windows CPU
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import torch
torch.set_num_threads(4)

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = Path("models/llm/qwen2.5-1.5b-instruct").resolve()
PROMPT = "What is a pressure vessel?"

def main() -> None:
    print(f"=== STANDALONE GENERATION TEST (PHASE 4) ===", flush=True)
    print(f"Model Path: {MODEL_PATH}", flush=True)
    print(f"Prompt: '{PROMPT}'", flush=True)
    print(f"PyTorch Version: {torch.__version__}", flush=True)
    print(f"CUDA Available: {torch.cuda.is_available()}", flush=True)

    # 1. Measure Tokenizer & Model Load Time
    print("\n[1/3] Loading tokenizer with local_files_only=True...", flush=True)
    t0 = time.perf_counter()
    
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        trust_remote_code=False,
    )
    print(f"Tokenizer loaded in {time.perf_counter() - t0:.2f} s. Vocab: {len(tokenizer)}", flush=True)

    print("\nLoading model with local_files_only=True...", flush=True)
    t_mod = time.perf_counter()
    
    # Test float32 on CPU for maximum CPU kernel compatibility and speed
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()
    
    load_time_s = time.perf_counter() - t_mod
    load_time_ms = load_time_s * 1000.0
    print(f"Model loaded successfully in: {load_time_s:.2f} s ({load_time_ms:.1f} ms)", flush=True)
    
    # 2. Format Prompt with Chat Template
    messages = [
        {"role": "system", "content": "You are an expert refinery mechanical engineer. Answer concisely."},
        {"role": "user", "content": PROMPT}
    ]
    formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Formatted input:\n{formatted_input}\n", flush=True)
    
    inputs = tokenizer(formatted_input, return_tensors="pt")
    prompt_tokens = inputs.input_ids.shape[1]
    print(f"Prompt Tokens: {prompt_tokens}", flush=True)

    # 3. Measure Generation Latency
    print("\n[2/3] Generating response (max_new_tokens=64)...", flush=True)
    t_gen_start = time.perf_counter()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    gen_time_s = time.perf_counter() - t_gen_start
    gen_time_ms = gen_time_s * 1000.0

    generated_ids = outputs[0][prompt_tokens:]
    completion_tokens = len(generated_ids)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    tok_per_sec = completion_tokens / gen_time_s if gen_time_s > 0 else 0.0

    print(f"\n[3/3] Results:", flush=True)
    print(f"Generation Time: {gen_time_s:.2f} s ({gen_time_ms:.1f} ms)", flush=True)
    print(f"Completion Tokens: {completion_tokens}", flush=True)
    print(f"Throughput: {tok_per_sec:.2f} tokens/s", flush=True)
    print(f"\n--- Generated Text ---", flush=True)
    print(text, flush=True)
    print(f"----------------------\n", flush=True)

    assert len(text) > 0, "Generated text is empty!"
    print("[PHASE 4] STANDALONE GENERATION TEST: PASSED", flush=True)

if __name__ == "__main__":
    main()
