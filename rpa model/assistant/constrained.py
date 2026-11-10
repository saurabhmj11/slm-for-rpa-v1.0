"""Catalog-locked decoding: while the model writes an action id, an input name (inside "args") or an output name
(inside "out"), it may only pick tokens that keep the text on a name that really exists in the catalog.

Why: the most common failure on plain requests is a made-up name ("ooa.folder/list_files", "ooa.file/download",
"target_folder" for ooa.file/copy's "target"). The model usually knows WHAT to do; it misremembers the exact name.
Everything else (descriptions, values, structure) is left exactly as the model writes it.

Pieces:
  JsonTracker      incremental JSON state over the generated text (which container, which key, inside a string?)
  NameConstraint   catalog names + "which tokens are allowed right now?" (prefix check with bisect)
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


def generate(model, prompt_ids, constraint, max_tokens, temperature, top_k, seed, stop_ids, decode, guard=None,
             first_k=256):
    """Decode one reply with the catalog lock. Returns (ids, stop_reason, n_constrained_steps, n_changed).
    decode(token_id) -> text (used to keep the JSON tracker in step). Sampling matches nanochat's Engine."""
    import torch
    from nanochat.engine import KVCache, sample_next_token
    from nanochat.common import COMPUTE_DTYPE
    device = model.get_device()
    m = model.config
    cache = KVCache(batch_size=1, seq_len=len(prompt_ids) + max_tokens + 1, device=device, dtype=COMPUTE_DTYPE,
                    num_heads=m.n_kv_head, head_dim=m.n_embd // m.n_head, num_layers=m.n_layer)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    tracker = JsonTracker()
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
            tracker.feed(decode(nxt))
            if guard is not None and guard.push(nxt):
                reason = "loop: " + guard.reason
                break
            logits = model.forward(torch.tensor([[nxt]], dtype=torch.long, device=device), kv_cache=cache)[:, -1, :]
    return out, reason, n_con, n_changed
