"""
Praman Brain Mini — Layer 1: News Intelligence Inference.

Tests all 3 Layer 1 skills:
  - sentiment_agent  (bullish / bearish / neutral)
  - impact_agent     (positive/negative + high/medium/low)
  - signal_agent     (incline / decline / sideways)
"""
import torch

from model import GPT
from tokenizer import AgentTokenizer

tok  = AgentTokenizer()
ckpt = torch.load("dpo_model_v2.pt", map_location="cpu", weights_only=False)
cfg  = ckpt["config"]
model = GPT(cfg)
model.load_state_dict(ckpt["model"])
model.eval()

print(f"loaded model: {model.num_params():,} params")
print(f"layer 1      : News Intelligence (sentiment / impact / signal)\n")

PROMPTS = [
# ── Skill A: Sentiment ─────────────────────────────────────────────
    "<|im_start|>user\nwhat is the sentiment of reliance quarterly_profit news ?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nwhat is the sentiment of wipro quarterly_loss news ?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nwhat is the sentiment of tcs revenue_beat news ?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nwhat is the sentiment of adani regulatory_fine news ?<|im_end|>\n<|im_start|>assistant\n",
    # ── Skill B: Impact ────────────────────────────────────────────────
    "<|im_start|>user\nwhat is the market impact of hdfc dividend news ?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nwhat is the market impact of sbi credit_downgrade news ?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nwhat is the market impact of infosys contract_win news ?<|im_end|>\n<|im_start|>assistant\n",
    # ── Skill C: Signal ────────────────────────────────────────────────
    "<|im_start|>user\nwill tcs incline or decline due to merger news ?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nwill bajaj incline or decline due to rate_hike news ?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nwill ongc incline or decline due to global_rally news ?<|im_end|>\n<|im_start|>assistant\n",
]

end_id = tok.end_id

for p in PROMPTS:
    prompt_ids = tok.encode(p)
    generated  = prompt_ids[:]

    temperature    = 0.7
    top_k          = 50
    max_new_tokens = 55
    ngram_n        = 3

    for _ in range(max_new_tokens):
        window   = generated[-cfg.block_size:]
        idx_cond = torch.tensor([window], dtype=torch.long)
        logits, _ = model(idx_cond)
        logits     = logits[:, -1, :] / temperature

        if len(generated) >= ngram_n - 1:
            ctx    = tuple(generated[-(ngram_n - 1):])
            banned = {
                generated[i + ngram_n - 1]
                for i in range(len(generated) - ngram_n + 1)
                if tuple(generated[i : i + ngram_n - 1]) == ctx
            }
            for token in banned:
                logits[0, token] = -float("inf")

        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")

        probs   = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)

        if next_id == end_id:
            break

    text = tok.decode(generated)
    print(f"OUTPUT:  {text}")
    print("-" * 70)
