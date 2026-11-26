"""Catalog-locked decoding: while the model writes an action id, an input name (inside "args") or an output name
(inside "out"), it may only pick tokens that keep the text on a name that really exists in the catalog.

Why: the most common failure on plain requests is a made-up name ("ooa.folder/list_files", "ooa.file/download",
"target_folder" for ooa.file/copy's "target"). The model usually knows WHAT to do; it misremembers the exact name.
Everything else (descriptions, values, structure) is left exactly as the model writes it.

Pieces:
  JsonTracker      incremental JSON state over the generated text (which container, which key, inside a string?)
  NameConstraint   catalog names + "which tokens are allowed right now?" (prefix check with bisect)
  FlowGate         finds where the flow JSON starts (at once, or after "Flow:" when the reply opens with a plan)
  generate()       nanochat decode loop (same sampling as nanochat's Engine) that applies the mask when a name is
                   being written. When no name is being written the logits are untouched.
Pure Python + torch; no change to nanochat itself.
"""
import bisect
import json
import re


class JsonTracker:
    """Feed text; know whether we are inside a string, whether it is a key or a value, and the container path."""

    def __init__(self):
        self.stack = []          # frames: {"t": "o"|"a", "key", "expect", "vals", "keys", "pkey"}
        self.in_str = False
        self.esc = False
        self.buf = ""
        self.role = None         # "key" | "val"

    def feed(self, text):
        for c in text:
            self._char(c)

    def _top(self):
        return self.stack[-1] if self.stack else None

    def _char(self, c):
        top = self._top()
        if self.in_str:
            if self.esc:
                self.esc = False
                self.buf += c
            elif c == "\\":
                self.esc = True
                self.buf += c
            elif c == '"':
                self.in_str = False
                if top is not None and top["t"] == "o":
                    if self.role == "key":
                        top["key"] = self.buf
                        top["keys"].add(self.buf)
                        top["expect"] = "colon"
                    else:
                        top["vals"][top["key"]] = self.buf
                        top["expect"] = "comma"
            else:
                self.buf += c
            return
        if c in " \t\r\n":
            return
        if c in "{[":
            pkey = top["key"] if top is not None and top["t"] == "o" else None
            self.stack.append({"t": "o" if c == "{" else "a", "key": None, "expect": "key" if c == "{" else "value",
                               "vals": {}, "keys": set(), "pkey": pkey})
        elif c in "}]":
            if self.stack:
                self.stack.pop()
            if self._top() is not None:
                self._top()["expect"] = "comma"
        elif c == '"':
            self.in_str, self.esc, self.buf = True, False, ""
            self.role = "key" if top is not None and top["t"] == "o" and top["expect"] == "key" else "val"
        elif c == ":":
            if top is not None:
                top["expect"] = "value"
        elif c == ",":
            if top is not None:
                top["expect"] = "key" if top["t"] == "o" else "value"
        # numbers, true/false/null: nothing to track

    def name_slot(self):
        """(kind, action_id or None, used_keys, typed_prefix) when a constrained name is being written, else None.
        kind: "action" (value of "action"), "args" (a key in args), "out" (a key in out)."""
        if not self.in_str or not self.stack:
            return None
        top = self.stack[-1]
        if top["t"] != "o":
            return None
        if self.role == "val" and top["key"] == "action":
            return ("action", None, (), self.buf)
        if self.role == "key" and top["pkey"] in ("args", "out") and len(self.stack) >= 2:
            step = self.stack[-2]
            if step["t"] == "o" and isinstance(step["vals"].get("action"), str):
                return (top["pkey"], step["vals"]["action"], top["keys"], self.buf)
        return None


_STRUCTURAL = re.compile(r'^[\s:,{}\[\]"]*$')


class NameConstraint:
    """Which next tokens are allowed while a name is being written."""

    def __init__(self, catalog_entries, token_strings, lock_core_args=False, lock_keys=False):
        # lock_keys: also lock input/output names. Off by default: measured on plain requests, forcing a key the model
        # did not mean ("target" when it was writing "target_folder") derails the rest of the line, while the repair
        # step (training/repair.py) maps such keys afterwards without that risk. Action ids are always locked.
        self.lock_keys = lock_keys
        self.tokens = token_strings                       # list: token id -> decoded text
        self.actions = sorted({e["id"] for e in catalog_entries})
        self.inputs, self.outputs = {}, {}
        for e in catalog_entries:
            if e["id"].startswith("core/") and not lock_core_args:
                continue                                   # control steps: the catalog lists fewer keys than flows use
            self.inputs[e["id"]] = sorted({i["name"] for i in e.get("inputs", [])})
            self.outputs[e["id"]] = sorted({o["name"] for o in e.get("outputs", [])})

    def names_for(self, slot):
        kind, action, used, _ = slot
        if kind == "action":
            return self.actions
        if not self.lock_keys:
            return None
        table = self.inputs if kind == "args" else self.outputs
        names = table.get(action)
        if not names:
            return None                                    # unknown action or core step: do not constrain
        free = [n for n in names if n not in used]
        return free or None

    @staticmethod
    def _has_prefix(names, p):
        i = bisect.bisect_left(names, p)
        return i < len(names) and names[i].startswith(p)

    def ok(self, names, prefix, tok_text):
        q = tok_text.find('"')
        part = tok_text if q < 0 else tok_text[:q]
        if "\\" in part or "\n" in part:
            return False
        cand = prefix + part
        if q < 0:
            return self._has_prefix(names, cand)
        if not _STRUCTURAL.match(tok_text[q + 1:]):
            return False                                     # after the closing quote only JSON punctuation may follow
        i = bisect.bisect_left(names, cand)
        return i < len(names) and names[i] == cand          # the name ends here and must be complete

    def allowed(self, slot, candidate_ids):
        names = self.names_for(slot)
        if names is None:
            return None
        prefix = slot[3]
        return [t for t in candidate_ids if self.tokens[t] and self.ok(names, prefix, self.tokens[t])]


def token_strings(tok, vocab_size):
    """Decoded text of every token id (special tokens -> '' so they are never allowed inside a name)."""
    out = []
    for t in range(vocab_size):
        try:
            s = tok.decode([t])
        except Exception:
            s = ""
        if s.startswith("<|") and s.endswith("|>"):
            s = ""
        out.append(s)
    return out


# A reply may open with the flow JSON (the format before plans), with a plan ("Plan:" ... "Flow:" + JSON), with a
# restatement of the request before the plan ("Clear request: ..."), with a thinking block ("Understand: ...", then
# Facts / Missing / Steps / Check before "Flow:" or "Question:"), or with a clarifying question; anything else is
# chat (see training/ooa_format.py, ANSWER_STARTS).
REPLY_STARTS = ("{", "Plan:", "Question:", "Flow:", "Clear request:", "Understand:")   # Understand: a thinking answer


def not_a_flow(out, decode):
    """True once the reply's start is known to be none of REPLY_STARTS (a chat reply, not a flow or a question)."""
    head = "".join(decode(t) for t in out[:8]).lstrip()
    if not head:
        return False
    return not any(p.startswith(head) or head.startswith(p) for p in REPLY_STARTS)


class FlowGate:
    """Which part of the reply is flow JSON, so the catalog lock applies there only: all of it when the reply opens
    with '{' (old format), else the text after a line starting with 'Flow:' (plan-first format). The plan before it
    is free text in plain words and is never constrained; a question never reaches the JSON."""

    def __init__(self):
        self.text, self.open = "", False

    def feed(self, piece):
        """Add decoded text; return the part of it that belongs to the flow JSON ('' before the JSON starts)."""
        if self.open:
            return piece
        self.text += piece
        head = self.text.lstrip()
        if head.startswith("{"):
            self.open = True
            return head          # only white space came before, and nothing of it was returned
        i = self.text.find("Flow:")
        if i >= 0 and (i == 0 or self.text[i - 1] == "\n") and len(self.text) > i + len("Flow:"):
            self.open = True
            return self.text[i + len("Flow:"):]
        return ""


def generate(model, prompt_ids, constraint, max_tokens, temperature, top_k, seed, stop_ids, decode, guard=None,
             first_k=256, flow=False):
    """Decode one reply with the catalog lock. Returns (ids, stop_reason, n_constrained_steps, n_changed).
    decode(token_id) -> text (used to keep the JSON tracker in step). Sampling matches nanochat's Engine.
    flow=True: stop at once (reason "not a flow") when the reply opens with none of REPLY_STARTS (a request it read as
    chat). The first token is NOT forced to '{': that makes the model invent an unrelated flow for a non-task.
    The lock applies inside the flow JSON only (FlowGate): a plan before "Flow:" is written freely."""
    import torch
    from nanochat.engine import KVCache, sample_next_token
    from nanochat.common import COMPUTE_DTYPE
    device = model.get_device()
    m = model.config
    cache = KVCache(batch_size=1, seq_len=len(prompt_ids) + max_tokens + 1, device=device, dtype=COMPUTE_DTYPE,
                    num_heads=m.n_kv_head, head_dim=m.n_embd // m.n_head, num_layers=m.n_layer)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    tracker, gate = JsonTracker(), FlowGate()
    out, reason, n_con, n_changed = [], "max_tokens", 0, 0
    with torch.inference_mode():
        logits = model.forward(torch.tensor([prompt_ids], dtype=torch.long, device=device), kv_cache=cache)[:, -1, :]
        for _ in range(max_tokens):
            slot = tracker.name_slot() if constraint is not None else None
            if slot is not None:
                top = torch.topk(logits[0], min(first_k, logits.size(-1))).indices.tolist()
                allowed = constraint.allowed(slot, top)
                if allowed == []:
                    allowed = constraint.allowed(slot, range(logits.size(-1)))
                if allowed:                         # None = this slot is not constrained; [] = nothing fits, leave it
                    n_con += 1
                    mask = torch.full_like(logits, float("-inf"))
                    idx = torch.tensor(allowed, device=device)
                    mask[0, idx] = logits[0, idx]
                    if int(torch.argmax(logits[0])) not in allowed:
                        n_changed += 1
                    logits = mask
            nxt = int(sample_next_token(logits, rng, temperature, top_k)[0, 0])
            if nxt in stop_ids:
                reason = "end"
                break
            out.append(nxt)
            tracker.feed(gate.feed(decode(nxt)))
            if flow and not_a_flow(out, decode):
                reason = "not a flow"
                break
            if guard is not None and guard.push(nxt):
                reason = "loop: " + guard.reason
                break
            logits = model.forward(torch.tensor([[nxt]], dtype=torch.long, device=device), kv_cache=cache)[:, -1, :]
    return out, reason, n_con, n_changed


# ---------------------------------------------------------------------------------------------- batched decoding
# nanochat's Engine batches samples of ONE prompt only (generate(tokens, num_samples)); its KV cache keeps one position
# for the whole batch (the SDPA fallback reads cache_seqlens[0]). The agent team's builders need several DIFFERENT
# prompts (one per part of a plan) decoded together, so this is a small batched decoder over the same weights: each
# prompt is prefilled on its own (nanochat's forward, batch 1), its keys and values are copied into one cache of
# B rows, and every decode step runs the B rows at once with per-row positions (rotary angles, cache slot, causal +
# sliding-window mask per row). The step mirrors GPT.forward at nanochat commit 92d63d4 (smear, value embeddings,
# x0 / residual lambdas, backout, softcap 15); test_constrained.py checks it against the one-prompt path.


def _norm(x):
    import torch.nn.functional as TF
    return TF.rms_norm(x, (x.size(-1),))


def _rotary(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    import torch
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


def _decode_step(model, idx, K, V, pos, prev):
    """One decode step for B rows. idx (B, 1) token ids; K, V (n_layer, B, T, H, D) caches; pos (B,) long tensor: the
    position each row's token takes; prev (B, 1, C) the previous token's normalized embedding (smear). Writes the new
    keys / values into the caches and returns (logits (B, vocab), new prev)."""
    import torch
    import torch.nn.functional as TF
    B = idx.size(0)
    x = _norm(model.transformer.wte(idx).to(model.cos.dtype))
    new_prev = x
    gate = model.smear_lambda.to(x.dtype) * torch.sigmoid(model.smear_gate(x[:, :, :24]))
    x = x + gate * prev
    cos = model.cos[0, pos].view(B, 1, 1, -1)
    sin = model.sin[0, pos].view(B, 1, 1, -1)
    tcur = int(pos.max()) + 1
    cols = torch.arange(tcur, device=idx.device).view(1, tcur)
    causal = cols <= pos.view(B, 1)                                    # (B, tcur)
    masks = {}
    x0, n_layer = x, model.config.n_layer
    rows = torch.arange(B, device=idx.device)
    x_back = None
    for i, block in enumerate(model.transformer.h):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        ve = model.value_embeds[str(i)](idx).to(x.dtype) if str(i) in model.value_embeds else None
        a = block.attn
        h = _norm(x)
        q = a.c_q(h).view(B, 1, a.n_head, a.head_dim)
        k = a.c_k(h).view(B, 1, a.n_kv_head, a.head_dim)
        v = a.c_v(h).view(B, 1, a.n_kv_head, a.head_dim)
        if ve is not None:
            g = 3 * torch.sigmoid(a.ve_gate(h[..., :a.ve_gate_channels]))
            v = v + g.unsqueeze(-1) * ve.view(B, 1, a.n_kv_head, a.head_dim)
        q, k = _rotary(q, cos, sin), _rotary(k, cos, sin)
        q, k = _norm(q) * 1.2, _norm(k) * 1.2
        K[i, rows, pos] = k[:, 0]
        V[i, rows, pos] = v[:, 0]
        window = model.window_sizes[i][0]
        if window not in masks:
            m = causal
            if 0 <= window < tcur:
                m = m & ((pos.view(B, 1) - cols) <= window)
            masks[window] = m.view(B, 1, 1, tcur)
        y = TF.scaled_dot_product_attention(q.transpose(1, 2), K[i, :, :tcur].transpose(1, 2),
                                            V[i, :, :tcur].transpose(1, 2), attn_mask=masks[window],
                                            enable_gqa=a.n_head != a.n_kv_head)
        x = x + a.c_proj(y.transpose(1, 2).contiguous().view(B, 1, -1))
        x = x + block.mlp(_norm(x))
        if i == n_layer // 2:
            x_back = x
    if x_back is not None:
        x = x - model.backout_lambda.to(x.dtype) * x_back
    x = _norm(x)
    logits = model.lm_head(x)[..., :model.config.vocab_size].float()
    return (15 * torch.tanh(logits / 15))[:, -1, :], new_prev


class BatchRow:
    """One prompt of a batched generation: its own sampling (temperature, seed), catalog lock, guards and limits."""

    def __init__(self, prompt_ids, max_tokens, temperature=0.2, seed=1000, constraint=None, guard=None, flow=False,
                 top_k=50):
        self.prompt, self.max_tokens, self.temperature, self.seed = list(prompt_ids), int(max_tokens), temperature, seed
        self.constraint, self.guard, self.flow, self.top_k = constraint, guard, flow, top_k
        self.out, self.reason, self.n_con, self.n_changed, self.done = [], "max_tokens", 0, 0, False


def generate_batch(model, rows, stop_ids, decode, first_k=256, deadline=None):
    """Decode several prompts at once (see the note above). rows: BatchRow list; each row stops on its own (a stop
    token, its max_tokens, its guard, "not a flow"); finished rows leave the batch. deadline: time.time() after which
    every row still running stops (reason "time"). Fills each row's out / reason / n_con / n_changed and returns rows.
    Rows with the same prompt share one prefill (best-of-N samples)."""
    import time
    import torch
    from nanochat.engine import KVCache, sample_next_token
    if not rows:
        return rows
    device = model.get_device()
    m = model.config
    hd = m.n_embd // m.n_head
    dtype = model.cos.dtype
    B = len(rows)
    T = max(len(r.prompt) + r.max_tokens for r in rows) + 2
    K = torch.zeros(m.n_layer, B, T, m.n_kv_head, hd, device=device, dtype=dtype)
    V = torch.zeros_like(K)
    prev = torch.zeros(B, 1, m.n_embd, device=device, dtype=dtype)
    logits = [None] * B
    done_prefill = {}
    with torch.inference_mode():
        for b, r in enumerate(rows):
            key = tuple(r.prompt)
            if key not in done_prefill:
                c = KVCache(batch_size=1, seq_len=len(r.prompt) + 1, device=device, dtype=dtype,
                            num_heads=m.n_kv_head, head_dim=hd, num_layers=m.n_layer)
                lg = model.forward(torch.tensor([r.prompt], dtype=torch.long, device=device), kv_cache=c)[:, -1, :]
                done_prefill[key] = (c, lg, b)
            c, lg, src = done_prefill[key]
            n = len(r.prompt)
            K[:, b, :n] = c.k_cache[:, 0, :n]
            V[:, b, :n] = c.v_cache[:, 0, :n]
            prev[b] = c.prev_embedding[0]
            logits[b] = lg.clone()
        rngs = []
        for r in rows:
            g = torch.Generator(device=device)
            g.manual_seed(r.seed)
            rngs.append(g)
        trackers = [JsonTracker() for _ in rows]
        gates = [FlowGate() for _ in rows]
        pos = [len(r.prompt) for r in rows]
        active = list(range(B))          # row numbers still decoding, in cache order
        lg_all = torch.cat([logits[b] for b in active], 0)
        while active:
            nxt, keep = [], []
            late = deadline is not None and time.time() >= deadline
            for j, b in enumerate(active):
                r = rows[b]
                if late:
                    r.reason, r.done = "time", True
                    continue
                lgt = lg_all[j:j + 1]
                slot = trackers[b].name_slot() if r.constraint is not None else None
                if slot is not None:
                    top = torch.topk(lgt[0], min(first_k, lgt.size(-1))).indices.tolist()
                    allowed = r.constraint.allowed(slot, top)
                    if allowed == []:
                        allowed = r.constraint.allowed(slot, range(lgt.size(-1)))
                    if allowed:
                        r.n_con += 1
                        mask = torch.full_like(lgt, float("-inf"))
                        ix = torch.tensor(allowed, device=device)
                        mask[0, ix] = lgt[0, ix]
                        if int(torch.argmax(lgt[0])) not in allowed:
                            r.n_changed += 1
                        lgt = mask
                t = int(sample_next_token(lgt, rngs[b], r.temperature, r.top_k)[0, 0])
                if t in stop_ids:
                    r.reason, r.done = "end", True
                    continue
                r.out.append(t)
                trackers[b].feed(gates[b].feed(decode(t)))
                if r.flow and not_a_flow(r.out, decode):
                    r.reason, r.done = "not a flow", True
                    continue
                if r.guard is not None and r.guard.push(t):
                    r.reason, r.done = "loop: " + r.guard.reason, True
                    continue
                if len(r.out) >= r.max_tokens:
                    r.reason, r.done = "max_tokens", True
                    continue
                nxt.append(t)
                keep.append(j)
            if not keep:
                break
            if len(keep) < len(active):            # finished rows leave the cache
                ix = torch.tensor(keep, device=device)
                K, V, prev = K.index_select(1, ix), V.index_select(1, ix), prev.index_select(0, ix)
                active = [active[j] for j in keep]
            p = torch.tensor([pos[b] for b in active], dtype=torch.long, device=device)
            lg_all, prev = _decode_step(model, torch.tensor(nxt, dtype=torch.long, device=device).view(-1, 1), K, V,
                                        p, prev)
            for b in active:
                pos[b] += 1
    return rows
