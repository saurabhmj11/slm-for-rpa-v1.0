import os
import random
import re
from minbpe import RegexTokenizer

# 1. Load the logistics dataset we generated
data_path = "logistics_dataset.txt"
print(f"Loading {data_path}...")
with open(data_path, "r", encoding="utf-8") as f:
    raw = f.read()

# 2. Split into 90% train / 10% validation (at example boundaries)
# Each example ends with "<end>", so we split on that to keep examples intact.
END_TOKEN = "<end>"
examples = [ex.strip() + END_TOKEN for ex in raw.split(END_TOKEN) if ex.strip()]
print(f"Found {len(examples):,} examples.")

random.seed(42)
random.shuffle(examples)

split_idx = int(0.9 * len(examples))
train_examples = examples[:split_idx]
val_examples = examples[split_idx:]

with open("train.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(train_examples) + "\n")
with open("val.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(val_examples) + "\n")

print(f"Train: {len(train_examples):,} examples -> train.txt")
print(f"Val:   {len(val_examples):,} examples -> val.txt")

# 3. Special tokens: the agent DSL tags must stay atomic (one token each)
# so the model learns structure, not memorised template strings.
SPECIAL_TOKENS_LIST = [
    "<|im_start|>", "<|im_end|>",
    "<thought>", "<route>", "<tool_call>", "<observation>", "<action>",
    "<end>", "<pad>",
]

# Sanity check: make sure we haven't missed any tag present in the data
found_tags = set(re.findall(r"<\|?[a-z_]+\|?>", raw))
missing = found_tags - set(SPECIAL_TOKENS_LIST)
assert not missing, f"Tags in data not registered as special tokens: {missing}"

# Tokenizer is trained ONLY on the training split (no leakage from val).
# Special tokens are stripped out of the training text so BPE never merges them.
special_pattern = "|".join(re.escape(t) for t in SPECIAL_TOKENS_LIST)
text = re.sub(special_pattern, " ", "\n".join(train_examples))

# Subsample for speed: BPE only needs a representative sample.
# Our data is synthetic & repetitive — 500KB captures all patterns.
MAX_CHARS = 500_000
if len(text) > MAX_CHARS:
    original_len = len(text)
    text = text[:MAX_CHARS]
    print(f"Subsampled to {MAX_CHARS:,} chars (was {original_len:,}) for faster training.")

# 4. Initialize the BPE RegexTokenizer (GPT-4 split pattern).
# The regex stops merges from crossing word/punctuation boundaries.
tokenizer = RegexTokenizer()

# 5. Vocab: 256 base bytes + merges. With regex splitting, merges stay
# inside words, so ~600 is plenty for this small logistics vocabulary.
vocab_size = 600

print(f"Training logistics tokenizer (target vocab size: {vocab_size} + {len(SPECIAL_TOKENS_LIST)} special)...")
tokenizer.train(text, vocab_size, verbose=False)

# 6. Register special tokens after the merged vocab
special_tokens = {tok: vocab_size + i for i, tok in enumerate(SPECIAL_TOKENS_LIST)}
tokenizer.register_special_tokens(special_tokens)

# 7. Quick round-trip check + compression stats
sample = val_examples[0]
ids = tokenizer.encode(sample, allowed_special="all")
assert tokenizer.decode(ids) == sample, "round-trip encode/decode failed"
lengths = [len(tokenizer.encode(ex, allowed_special="all")) for ex in val_examples]
print(f"Sample: {len(sample)} chars -> {len(ids)} tokens")
print(f"Val example token lengths: max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

# 8. Save the trained tokenizer
os.makedirs("models", exist_ok=True)
prefix = os.path.join("models", "logistics_regex_bpe")
tokenizer.save(prefix)

print(f"Total vocab size: {vocab_size + len(SPECIAL_TOKENS_LIST)}")
print(f"Tokenizer trained and saved to {prefix}.model and {prefix}.vocab")
