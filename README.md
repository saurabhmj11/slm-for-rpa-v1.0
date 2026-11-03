# 100K-Param Agent-Brain Prototype

A from-scratch, ~100K-parameter GPT-style transformer trained to do three
things at once: **route** a query to the right agent, emit a well-formed
**tool call**, and produce a final **reasoned answer** from the tool's
observation — the core loop your agent force needs, at toy scale.

This is a **proof-of-pipeline model**, not a capability model. The point
is to validate the full stack — custom tokenizer → synthetic agent data →
architecture → training → inference — cheaply on CPU, before spending
real compute scaling the same recipe to a model that can actually reason
over open-ended tasks.

## Results

Trained for 3000 iterations (~6.5 min on CPU): train/val loss dropped
from 6.15 → 0.14. Sample outputs (see `infer.py`):

```
PROMPT:  what is the weather in nagpur ?
OUTPUT:  <thought> user wants weather info <route> weather_agent
         <tool_call> {"tool":"get_weather","args":{"city":"nagpur"}}
         <observation> 27 celsius cloudy
         <action> the weather in nagpur is 27 degrees celsius and cloudy

PROMPT:  what is 9 plus 6 ?
OUTPUT:  <route> math_agent
         <tool_call> {"tool":"calculator","args":{"expr":"9+6"}}
         <observation> 13
         <action> the sum is 13          <- correct
```

Routing accuracy, tool-call formatting, and simple arithmetic are all
correctly learned at 99,880 parameters. This confirms the DSL design and
training loop are sound.

## Files

| File | Purpose |
|---|---|
| `tokenizer.py` | Fixed ~460-token word-level vocab for the agent DSL (routes, JSON syntax, digits, toy-domain words) |
| `data_gen.py` | Synthetic trajectory generator for 3 toy skills: weather lookup, math, fact search |
| `model.py` | GPT-style decoder-only transformer (`vocab=460, n_embd=40, n_layer=4, n_head=2` → 99,920 params) |
| `train.py` | Training loop with checkpoint/resume (`--iters N --total_iters M`) |
| `infer.py` | Loads `best_model.pt` and runs sample prompts |
| `best_model.pt` | Trained checkpoint (best val loss: 0.141) |

## Known limitation

Numbers must be digit-spaced in prompts (`1 4` not `14`) — the tokenizer
has no token for multi-digit numbers by design, to keep the vocab (and
therefore the model) tiny. `data_gen.py` already does this correctly for
training data; it only matters if you hand-type a prompt for `infer.py`.

## Scaling this up

This same recipe (custom control tokens + routing/tool-call/reasoning
mix in the training data) is exactly what the full-scale plan needs —
just with:
- a much larger, more diverse synthetic + real dataset per skill
- a real subword tokenizer once vocab needs exceed toy scale
- a bigger model (350M–1B+) and GPU time for pretraining
- real tool integrations instead of templated toy tools

Re-run `train.py --iters N --total_iters M` to keep training this
checkpoint further, or bump `GPTConfig` in `model.py` to size up.
