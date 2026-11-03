import torch
from tokenizer import AgentTokenizer
from model import GPT

tok = AgentTokenizer()
ckpt = torch.load("best_model.pt", map_location="cpu", weights_only=False)
model = GPT(ckpt["config"])
model.load_state_dict(ckpt["model"])
model.eval()

prompt = "<bos> <user> what is the sentiment of reliance quarterly_profit news ?"
generated = tok.encode(prompt)

for _ in range(60):
    idx_cond = torch.tensor([generated[-ckpt["config"].block_size:]], dtype=torch.long)
    logits, _ = model(idx_cond)
    logits = logits[:, -1, :] / 0.9
    v, _ = torch.topk(logits, min(20, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).item()
    generated.append(next_id)
    if next_id == tok.stoi["<end>"]: break

print("RAW GENERATED:")
print(repr(tok.decode(generated)))
