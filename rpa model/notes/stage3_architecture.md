# Stage 3: nanoGPT (GPT-2) vs your model.py vs nanochat

Read from source: `nanoGPT/model.py`, root `model.py` ("Praman Brain Mini"), `nanochat/nanochat/gpt.py`.
The nanochat demo (`runs/runcpu.sh`) has NOT been run yet: it needs downloads that are waiting on approval (see end).

| Piece | nanoGPT (GPT-2 style) | Your `model.py` | nanochat `gpt.py` |
|---|---|---|---|
| Positions | learned `wpe` table, fixed to block_size | RoPE | RoPE (base 100,000), no position table |
| Normalization | LayerNorm (optional bias), pre-norm | RMSNorm with learnable weight | RMSNorm with **no** learnable parameters; also norms the embedding, and q and k (QK norm) |
| MLP | Linear, GELU, Linear (4x wide) | SwiGLU (8/3 x) | Linear, **relu squared**, Linear (4x wide) |
| Attention | MHA, fused SDPA | MHA, manual softmax | GQA support, Flash Attention 3 with SDPA fallback, **sliding-window layers** (pattern `SSSL`: 3 short, 1 full) |
| Embeddings | `wte` tied with `lm_head` | tied | **untied**; plus per-layer **value embeddings** (alternating layers) |
| Bias in Linears | yes (default) | no | no |
| Dropout | optional | 0.1 default | none |
| Residual | plain `x + f(norm(x))` | plain | per-layer learnable scales `resid_lambdas`, `x0_lambdas` (blend the original embedding back in at every layer) |
| Extras | none | shared block looped 4 times ("share_layers") | smear (mix previous token's embedding), backout (subtract mid-layer residual), logit soft-cap at 15 |
| Init | N(0, 0.02) | N(0, 0.02) | uniform for q, k, v, c_fc; **zeros** for output projections; embedding std 0.8 |
| Optimizer | AdamW | AdamW | **Muon** for matrices, AdamW for embeddings and scalars, separate learning rates per group |

## What to take from this
1. **The core is identical in all three:** token embedding, a stack of (norm, causal attention, residual) and
   (norm, MLP, residual) blocks, a final norm, and a linear head trained with cross-entropy. Everything else is a refinement.
2. **nanochat is "GPT plus tricks from the last few years"** (RoPE, RMSNorm, QK norm, relu squared, GQA, sliding windows, Muon).
   Each is a small, separately testable change. Several can be removed one at a time to see their effect.
3. **Your `model.py` is closer to Llama** (SwiGLU, RMSNorm, RoPE). Its distinctive choices are the shared block and tied weights.
   Sharing one block 4 times is why the "100K" model had only about 36K parameters.
4. **Sliding-window attention is relevant to the RPA project.** Short-window layers are cheaper over long sequences,
   which is how nanochat can afford long contexts. It matters because flows are long (Stage 2 finding).
5. **Context cost is quadratic in attention.** nanochat's `SSSL` pattern cuts that. The KV-cache helpers
   (`kv_bytes_per_token`, `kv_read_bytes`) show how inference memory grows with context.
6. **Parameter counts hide where the weights are.** `num_scaling_params()` splits them into embeddings, value
   embeddings, head, transformer matrices and scalars. With a 4,096-token vocab the embedding share is small
   for larger depths but large for tiny models.

## Exercises
1. Count the parameters of nanochat `GPT(GPTConfig(n_layer=6, n_embd=384, n_head=6, n_kv_head=6, vocab_size=4096))`
   with `num_scaling_params()`. How much is embeddings vs matrices?
2. In nanoGPT, replace LayerNorm with RMSNorm and GELU with relu squared. Compare the Stage 1 val loss (1.886 at 2000 steps).
3. Write one paragraph on why zero-initialising `c_proj` makes every block start as the identity.
4. Draw the data flow of one nanochat Block (norm, attention, add, norm, MLP, add).

## Pending: nanochat CPU demo
`runs/runcpu.sh` (depth 6, 512 context) would need: ~8 dataset shards from huggingface.co (climbmix-400b-shuffle),
extra packages (pyarrow, rustbpe, wandb, kernels, psutil), and SFT datasets downloaded at the SFT step.
Not started.
