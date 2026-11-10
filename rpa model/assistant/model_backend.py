"""Model backends for the OOA assistant.

NanochatLocal runs our trained nanochat checkpoint on this PC (CPU is enough). It streams tokens with nanochat's
Engine.generate so it can stop early: at <|assistant_end|> / bos, at max_tokens, or when the RepetitionGuard sees the
output looping (the main failure today; a loop used to run all the way to max_tokens).

FakeModel is a tiny stand-in for tests: canned valid flows for known prompts, garbage otherwise.

Both expose generate(request, attempt=0) -> str, which is what serving/ooa_server.Router calls.
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path


class RepetitionGuard:
    """Cheap loop detector over the generated token ids.

    Rule (checked every `every` tokens, early exit on the first mismatch, so it costs microseconds next to a forward
    pass): the output ends in one n-gram repeated back to back, i.e. tok[i] == tok[i-n] over the last `span` tokens, with
      n <= short_n  -> span = short_span  (e.g. "`` and `` and ..." or the same token over and over)
      n <= max_n    -> span = long_span
    Tuned on the real model and on every answer in the SFT data (12k flows, 13M tokens):
      - in valid flows the longest exact periodic stretch is 18 tokens for n <= 8 and 225 tokens for n <= 40 (a description
        like "read and parse a JSON file, then read and parse a JSON file, ..."); real flows repeat structures such as
        {"name":"x","type":"String","direction":"local"} but the name always differs, which breaks the period.
      - the loops seen on plain requests are exact periodic stretches of hundreds to thousands of tokens.
    An optional window rule (`window` > 0: the same `window`-token stretch seen `max_seen` times anywhere) is off by
    default: with window 30 / 3 times it also cut 5 of 14 outputs that ended on their own (flows legitimately repeat
    args such as {"path":"${folder}/data/orders.csv"} in several steps).
    """

    def __init__(self, short_n=8, short_span=48, max_n=40, long_span=320, window=0, max_seen=3, every=8):
        self.short_n, self.short_span, self.max_n, self.long_span = short_n, short_span, max_n, long_span
        self.window, self.max_seen, self.every = window, max_seen, every
        self.reset()

    def reset(self):
        self.toks, self.seen, self.reason = [], defaultdict(int), None

    def push(self, tok):
        """Add one token; return True when the output is looping (self.reason then says why)."""
        t = self.toks
        t.append(tok)
        w = self.window
        if w and len(t) >= w:
            key = hash(tuple(t[-w:]))
            self.seen[key] += 1
            if self.seen[key] >= self.max_seen:
                self.reason = f"the same {w}-token stretch appeared {self.max_seen} times"
                return True
        if len(t) % self.every == 0 and self._periodic():
            return True
        return False

    def _periodic(self):
        t, end = self.toks, len(self.toks)
        for n in range(1, self.max_n + 1):
            span = self.short_span if n <= self.short_n else self.long_span
            if span > end:
                continue
            if all(t[i] == t[i - n] for i in range(end - span + n, end)):
                self.reason = f"a {n}-token pattern repeated {span // n}+ times in a row"
                return True
        return False


class NanochatLocal:
    def __init__(self, model_dir, nanochat_dir, source="sft", device=None, guard=True, catalog=None):
        os.environ["NANOCHAT_BASE_DIR"] = str(Path(model_dir).resolve())
        dev_type = device
        if dev_type is None:
            import torch
            dev_type = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        if dev_type == "cpu":
            os.environ["NANOCHAT_DTYPE"] = "float32"   # must be set before nanochat.common is imported
        nd = str(Path(nanochat_dir).resolve())
        if nd not in sys.path:
            sys.path.insert(0, nd)
        import torch
        from nanochat.checkpoint_manager import load_model
        from nanochat.engine import Engine
        torch.set_grad_enabled(False)
        self.device = torch.device(dev_type)
        self.model, self.tok, meta = load_model(source, self.device, phase="eval")
        self.engine = Engine(self.model, self.tok)
        self.guard_enabled = guard
        enc = self.tok.encode_special
        self.bos = self.tok.get_bos_token_id()
        self.user_start, self.user_end = enc("<|user_start|>"), enc("<|user_end|>")
        self.assistant_start, self.assistant_end = enc("<|assistant_start|>"), enc("<|assistant_end|>")
        params = sum(p.numel() for p in self.model.parameters())
        self.info = {"device": dev_type, "params_M": round(params / 1e6, 1), "step": meta.get("step"), "source": source,
                     "catalog_lock": False}
        self.last_stats = {}
        # catalog lock (constrained.py): action ids and input/output names can only be real catalog names
        self.constraint = None
        if catalog:
            import constrained as C
            entries = [json.loads(l) for l in open(catalog, encoding="utf-8") if l.strip()]
            self._tok_text = C.token_strings(self.tok, self.model.config.vocab_size)
            # attempt 0 locks action ids AND input/output names; the retry locks only action ids and leaves names to
            # the repair step. Each fixes cases the other misses (measured on eval/plain_requests.jsonl).
            self.constraints = [C.NameConstraint(entries, self._tok_text, lock_keys=True),
                                C.NameConstraint(entries, self._tok_text, lock_keys=False)]
            self.constraint = self.constraints[0]
            self.info["catalog_lock"] = True

    def _prompt(self, text):
        return [self.bos, self.user_start] + self.tok.encode(text) + [self.user_end, self.assistant_start]

    def _run(self, text, max_tokens, temperature, seed, guard=None, on_token=None, attempt=0):
        """Stream one reply. Returns the decoded text; details go to self.last_stats."""
        use_guard = self.guard_enabled if guard is None else guard
        g = RepetitionGuard() if use_guard else None
        out, reason, t0 = [], "max_tokens", time.time()
        if self.constraint is not None and on_token is None:
            import constrained as C
            con = self.constraints[attempt % len(self.constraints)]
            ids, reason, ncon, nch = C.generate(
                self.model, self._prompt(text), con, max_tokens, temperature, 50, seed,
                {self.assistant_end, self.bos}, lambda t: self._tok_text[t] or self.tok.decode([t]), g)
            self.last_stats = {"tokens": len(ids), "stop": reason, "seconds": round(time.time() - t0, 2), "ids": ids,
                               "names_locked": ncon, "names_corrected": nch}
            return self.tok.decode(ids)
        stream = self.engine.generate(self._prompt(text), num_samples=1, max_tokens=max_tokens,
                                      temperature=temperature, top_k=50, seed=seed)
        for column, _ in stream:
            tok = column[0]
            if tok == self.assistant_end or tok == self.bos:
                reason = "end"
                break
            out.append(tok)
            if on_token is not None:
                on_token(tok)
            if g is not None and g.push(tok):
                reason = "loop: " + g.reason
                break
        stream.close()
        self.last_stats = {"tokens": len(out), "stop": reason, "seconds": round(time.time() - t0, 2), "ids": out}
        return self.tok.decode(out)

    def generate(self, request, attempt=0, max_tokens=2500):
        return self._run(request, max_tokens, 0.2 + 0.3 * attempt, 1000 + attempt, attempt=attempt)

    def ask(self, question):
        return self._run(question, 300, 0.2, 1000).strip()


class FakeModel:
    """Test double: valid canned flows for known prompts, garbage (not JSON) otherwise."""

    FLOWS = {
        "read a csv file into a table, then write the table to an excel sheet": {
            "kind": "TaskBot", "id": "csv_to_excel", "name": "CSV to Excel",
            "description": "Read a CSV file into a table, then write the table to an Excel sheet",
            "variables": [
                {"name": "csv_path", "type": "String", "direction": "input", "required": True},
                {"name": "xlsx_path", "type": "String", "direction": "input", "required": True},
                {"name": "table", "type": "Table", "direction": "local"}],
            "steps": [
                {"id": "s1", "action": "ooa.csv/read", "description": "Read the CSV file into a table",
                 "args": {"path": "$csv_path"}, "out": {"table": "table"}},
                {"id": "s2", "action": "ooa.excel/open", "description": "Open the workbook",
                 "args": {"path": "$xlsx_path"}},
                {"id": "s3", "action": "ooa.excel/write_table", "description": "Write the table to the sheet",
                 "args": {"table": "$table"}},
                {"id": "s4", "action": "ooa.excel/close", "description": "Save and close the workbook",
                 "args": {"save": True}}]},
        "list the files in a folder": {
            "kind": "TaskBot", "id": "list_files", "name": "List files", "description": "List the files in a folder",
            "variables": [
                {"name": "folder", "type": "String", "direction": "input", "required": True},
                {"name": "files", "type": "List", "direction": "output"}],
            "steps": [{"id": "s1", "action": "ooa.file/list", "description": "List the files in the folder",
                       "args": {"folder": "$folder"}, "out": {"files": "files"}}]},
    }
    FLOWS["list files in a folder matching a glob pattern. the flow takes folder as input. it returns files"] = \
        FLOWS["list the files in a folder"]      # the first example chip in the page
    FLOWS["read a text file. the flow takes path as input. it returns text"] = {      # the third example chip
        "kind": "TaskBot", "id": "read_text", "name": "Read text", "description": "Read a text file",
        "variables": [
            {"name": "path", "type": "String", "direction": "input", "required": True},
            {"name": "text", "type": "String", "direction": "output"}],
        "steps": [{"id": "s1", "action": "ooa.file/read_text", "description": "Read a text file",
                   "args": {"path": "$path"}, "out": {"text": "text"}}]}
    ANSWERS = {"ooa.csv/read": "Use `ooa.csv/read` (CSV/TXT: Read to table)."}

    def __init__(self, *a, **kw):
        self.info = {"device": "fake", "params_M": 0.0, "step": 0, "source": "fake"}
        self.calls = []
        self.last_stats = {}

    def generate(self, request, attempt=0, max_tokens=2500):
        self.calls.append((request, attempt))
        flow = self.FLOWS.get(" ".join(str(request).lower().split()).rstrip("."))
        text = json.dumps(flow, separators=(",", ":")) if flow else "garbage garbage garbage {not json"
        self.last_stats = {"tokens": len(text) // 3, "stop": "end", "seconds": 0.0}
        return text

    def ask(self, question):
        q = str(question).lower()
        if "csv" in q:
            return self.ANSWERS["ooa.csv/read"]
        return "I am a fake model used for tests; I do not know."
