# Praman In-House Model — Phase 1: LoRA Fine-Tune
#
# Fine-tunes Qwen2.5-7B on Indian financial data using LoRA (Low-Rank Adaptation).
# The base model stays frozen — only a small adapter (~20MB) is trained.
#
# Output: brain-v2-7b-indian-finance.gguf (loadable by Ollama)
#
# Requirements:
#   pip install unsloth torch datasets accelerate
#   GPU: A100 80GB (preferred) or T4 16GB (with 4-bit quantization)
#
# Training time: ~4-8 hours on A100, ~12-24 hours on T4
# Training cost: ~$20 on RunPod A100, ~$12 on T4

import json
import os
import sys

# === Configuration ===
BASE_MODEL = "unsloth/Qwen2.5-7B"          # Base model (4-bit quantized)
OUTPUT_DIR = "../models/brain-v2-7b"        # Where to save the adapter
GGUF_DIR = "../models/gguf"                 # Where to save the GGUF file
DATA_FILE = "../data/training_pairs.jsonl"  # Training data (instruction-response pairs)
EVAL_FILE = "../data/eval_pairs.jsonl"      # Evaluation data
MAX_STEPS = 1000                            # Training steps
LEARNING_RATE = 2e-4                        # LoRA learning rate
LORA_RANK = 64                              # Higher = more capacity, slower training
LORA_ALPHA = 128                            # Usually 2x LoRA rank
BATCH_SIZE = 4                              # Per-device batch size
GRADIENT_ACCUMULATION = 4                   # Effective batch = 4 x 4 = 16
MAX_SEQ_LENGTH = 8192                       # Max context window

# === Step 1: Load the base model ===
print("=" * 60)
print("  Praman In-House Model — Phase 1: LoRA Fine-Tune")
print("=" * 60)
print()

try:
    from unsloth import FastLanguageModel
    import torch
except ImportError:
    print("ERROR: Unsloth not installed.")
    print("Run: pip install unsloth torch datasets accelerate")
    sys.exit(1)

print(f"[1/6] Loading base model: {BASE_MODEL}")
print(f"      Quantization: 4-bit (QLoRA — fits on single T4 GPU)")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# === Step 2: Add LoRA adapter ===
print(f"\n[2/6] Adding LoRA adapter (rank={LORA_RANK}, alpha={LORA_ALPHA})")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# === Step 3: Load training data ===
print(f"\n[3/6] Loading training data: {DATA_FILE}")

def load_jsonl(filepath):
    """Load JSONL file (one JSON object per line)."""
    data = []
    if not os.path.exists(filepath):
        print(f"  WARNING: {filepath} not found. Using empty dataset.")
        return data
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"  Loaded {len(data)} training pairs")
    return data

train_raw = load_jsonl(DATA_FILE)
eval_raw = load_jsonl(EVAL_FILE)

if len(train_raw) == 0:
    print("\n  ERROR: No training data found!")
    print("  Run: python scripts/01_collect_training_data.py")
    print("  This will scrape financial data and generate Q&A pairs.")
    sys.exit(1)

# === Step 4: Format as instruction-response pairs ===
print(f"\n[4/6] Formatting training data")

# System prompt — teaches the model it's an Indian financial analyst
SYSTEM_PROMPT = (
    "You are Praman Brain v2, an AI financial analyst specializing in Indian markets. "
    "You understand NSE/BSE listed companies, Indian accounting standards (Ind AS), "
    "SEBI regulations, and Indian financial formatting (crore/lakh). "
    "Always cite the source filing when quoting numbers. "
    "If you don't have the data, say so explicitly — never hallucinate."
)

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

def format_training_data(raw_data):
    """Convert raw Q&A pairs to Alpaca instruction format."""
    formatted = []
    for item in raw_data:
        instruction = item.get("instruction", "")
        context_input = item.get("input", "")
        output = item.get("output", "")

        text = ALPACA_TEMPLATE.format(
            instruction=instruction,
            input=context_input,
            output=output,
        )

        formatted.append({
            "text": text,
            "instruction": instruction,
            "input": context_input,
            "output": output,
        })
    return formatted

train_data = format_training_data(train_data) if 'train_data' in dir() else format_training_data(train_raw)
eval_data = format_training_data(eval_raw)

print(f"  Train: {len(train_data)} pairs")
print(f"  Eval:  {len(eval_data)} pairs")

# === Step 5: Train ===
print(f"\n[5/6] Starting training ({MAX_STEPS} steps, lr={LEARNING_RATE})")
print(f"      Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"      Estimated time: 4-8 hours on A100, 12-24 hours on T4")
print()

from trl import SFTTrainer, SFTConfig

training_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    warmup_steps=50,
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    save_steps=200,
    save_total_limit=3,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=True,  # Pack multiple short samples into one sequence for efficiency
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=eval_data if eval_data else None,
    args=training_config,
)

# Show GPU memory before training
if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print()

trainer.train()

# === Step 6: Export to GGUF (Ollama format) ===
print(f"\n[6/6] Exporting to GGUF format for Ollama")
os.makedirs(GGUF_DIR, exist_ok=True)

gguf_path = os.path.join(GGUF_DIR, "brain-v2-7b-indian-finance")
model.save_pretrained_gguf(
    gguf_path,
    quantization_method="q4_k_m",  # 4-bit quantization (same as Ollama default)
)

print()
print("=" * 60)
print("  Training complete!")
print("=" * 60)
print()
print(f"  LoRA adapter saved to: {OUTPUT_DIR}")
print(f"  GGUF model saved to:  {gguf_path}.gguf")
print()
print("  Next steps:")
print("  1. Copy the .gguf file to your Ollama models directory")
print("  2. Create a Modelfile:")
print("     FROM ./brain-v2-7b-indian-finance.gguf")
print("     PARAMETER temperature 0.1")
print("     PARAMETER num_ctx 8192")
print(f'     SYSTEM "{SYSTEM_PROMPT}"')
print("  3. Load into Ollama:")
print("     ollama create brain-v2-7b -f Modelfile")
print("  4. Update coordinator.ts:")
print('     MODELS.worker = "brain-v2-7b" as ModelId;')
print("  5. Test: type 'What is RELIANCE ROE?' in the terminal")
print()
print("  To evaluate the model, run:")
print("     python scripts/03_evaluate.py")
