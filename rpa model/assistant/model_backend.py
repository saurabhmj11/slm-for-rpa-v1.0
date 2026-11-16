"""Model backends for the OOA assistant.

NanochatLocal runs our trained nanochat checkpoint on this PC (CPU is enough). It streams tokens with nanochat's
Engine.generate so it can stop early: at <|assistant_end|> / bos, at max_tokens, or when the RepetitionGuard sees the
output looping (the main failure today; a loop used to run all the way to max_tokens).

FakeModel is a tiny stand-in for tests: canned valid flows for known prompts (as flow JSON only, or plan + flow, or
a clarifying question), garbage otherwise.

Both expose generate(request, attempt=0) -> str, which is what serving/ooa_server.Router calls.
"""
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path


MAX_ANSWER = 2500        # tokens for a whole answer (plan + flow)
MIN_ANSWER = 256         # the window cut never leaves less than this


class RepetitionGuard:
    """Cheap loop detector over the generated token ids.

    Rule (checked every `every` tokens, early exit on the first mismatch, so it costs microseconds next to a forward
    pass): the output ends in one n-gram repeated back to back, i.e. tok[i] == tok[i-n] over the last `span` tokens, with
      n <= short_n  -> span = short_span  (e.g. "`` and `` and ..." or the same token over and over)
      n <= max_n    -> span = long_span
      n <= huge_n   -> span = max(4n, 240) up to n = 63, 3n beyond (a whole step or sentence written again and again)
    Tuned on the real model and on every answer in the SFT data (12k flows, 13M tokens):
      - in valid flows the longest exact periodic stretch is 18 tokens for n <= 8 and 225 tokens for n <= 40 (a description
        like "read and parse a JSON file, then read and parse a JSON file, ..."); real flows repeat structures such as
        {"name":"x","type":"String","direction":"local"} but the name always differs, which breaks the period.
      - for 41 <= n <= 256 (11k answers, the 4096-token data) the longest is 185 tokens (n = 46, a description sentence 4
        times) and for n >= 51 never more than 2.1 copies: 4 copies (3 from n = 64) never happen in a valid answer.
      - the loops seen on plain requests are exact periodic stretches of hundreds to thousands of tokens.
    An optional window rule (`window` > 0: the same `window`-token stretch seen `max_seen` times anywhere) is off by
    default: with window 30 / 3 times it also cut 5 of 14 outputs that ended on their own (flows legitimately repeat
    args such as {"path":"${folder}/data/orders.csv"} in several steps).
    """

    def __init__(self, short_n=8, short_span=48, max_n=40, long_span=320, window=0, max_seen=3, every=8, huge_n=200):
        self.short_n, self.short_span, self.max_n, self.long_span = short_n, short_span, max_n, long_span
        self.window, self.max_seen, self.every, self.huge_n = window, max_seen, every, huge_n
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

    def span(self, n):
        if n <= self.short_n:
            return self.short_span
        if n <= self.max_n:
            return self.long_span
        return max(4 * n, 240) if n < 64 else 3 * n

    def _periodic(self):
        t, end = self.toks, len(self.toks)
        for n in range(1, max(self.max_n, self.huge_n) + 1):
            span = self.span(n)
            if span > end:
                if n > self.max_n:
                    break                     # spans only grow from here
                continue
            if all(t[i] == t[i - n] for i in range(end - span + n, end)):
                self.reason = f"a {n}-token pattern repeated {span // n}+ times in a row"
                return True
        return False


_STEP_ID = re.compile(r'"id"\s*:\s*"([^"\\$][^"\\]{0,79})"')    # step ids (and the flow's own id), not "$var$" args
_PLAN_NUM = re.compile(r"^\s*\d+[.)]\s*")


class AnswerWatch:
    """Loop detector over the decoded text of a plan-first answer ("Plan: ... Flow: {...}", ooa_format.py), for loops
    that are not exact token repeats (RepetitionGuard): the model keeps writing steps with small changes, past the end
    of its own plan, and never closes the JSON. Stops when
      - the plan repeats a line (same words, number aside) max_same_line times, or runs past max_plan_lines lines
        (in the 11k training answers a plan has under 20 lines and repeats a line at most 10 times: "Set text" for
        10 fields);
      - the flow has more steps than the plan has lines, plus slack (max(3, a third of the plan)): the plan has one
        line per step (ooa_format.plan_from_flow), so a trained answer never does this;
      - one "id" value is written max_same_id times (ids are unique in every training flow; a looping answer often
        writes {"id":"x", ...} fragments that are not whole steps, so any "id" counts, not only one before "action").
    on_plan(plan_lines) -> bool is called once, when the plan is complete ("Flow:" starts): True stops the answer there
    (reason "plan": the caller builds the flow in parts from the plan instead). self.reason says why it stopped;
    self.kind is "loop" or "plan". prefix: text the answer was started with (given, not generated)."""

    def __init__(self, decode, on_plan=None, max_plan_lines=40, max_same_line=12, max_same_id=3, every=8, prefix=""):
        self.decode, self.on_plan = decode, on_plan
        self.max_plan_lines, self.max_same_line, self.max_same_id, self.every = (max_plan_lines, max_same_line,
                                                                                  max_same_id, every)
        self.text, self.flow_at, self.plan_lines, self.n = prefix or "", -1, None, 0
        self.reason = self.kind = None
        self._lines_seen = defaultdict(int)
        self._done_lines = 0

    def _plan(self, text):
        body = text.split("Plan:", 1)[1] if text.lstrip().startswith("Plan:") else ""
        return [l.rstrip() for l in body.split("\n") if l.strip()]

    def push(self, tok):
        self.text += self.decode(tok)
        self.n += 1
        t = self.text
        if self.flow_at < 0:
            head = t.lstrip()
            if head.startswith("{"):                                    # flow JSON only (the format before plans)
                self.flow_at, self.plan_lines = len(t) - len(head), 0
                return False
            if not any(head.startswith(p) or p.startswith(head) for p in ("Plan:", "Flow:")):
                self.flow_at, self.plan_lines = 10 ** 9, 0               # a question or a chat reply: not watched
                return False
            i = t.find("Flow:")
            if i >= 0 and (i == 0 or t[i - 1] == "\n"):
                self.flow_at = i + len("Flow:")
                plan = self._plan(t[:i])
                self.plan_lines = len(plan)
                if self.on_plan is not None and plan and self.on_plan(plan):
                    self.reason, self.kind = f"the plan has {len(plan)} lines: build the flow in parts", "plan"
                    return True
                return False
            lines = t.split("\n")
            for line in lines[self._done_lines:-1]:                      # complete lines not seen yet
                key = _PLAN_NUM.sub("", line).strip().lower()
                if key and key != "plan:":
                    self._lines_seen[key] += 1
                    if self._lines_seen[key] >= self.max_same_line:
                        self.reason, self.kind = f"the plan repeats the line '{key[:60]}'", "loop"
                        return True
            self._done_lines = max(self._done_lines, len(lines) - 1)
            if self._done_lines > self.max_plan_lines + 1:
                self.reason, self.kind = f"the plan runs past {self.max_plan_lines} lines", "loop"
                return True
            return False
        if self.n % self.every:
            return False
        flow = t[self.flow_at:]
        if self.plan_lines:
            steps = flow.count('"action"')
            limit = self.plan_lines + max(3, self.plan_lines // 3)
            if steps > limit:
                self.reason, self.kind = (f"the flow has {steps} steps for a plan of {self.plan_lines} lines", "loop")
                return True
        ids = defaultdict(int)
        for sid in _STEP_ID.findall(flow):
            ids[sid] += 1
            if ids[sid] >= self.max_same_id:
                self.reason, self.kind = f"the step id '{sid}' is written {ids[sid]} times", "loop"
                return True
        return False


class Guards:
    """Several guards pushed together (each has push(tok) -> bool and reason); reason / kind come from the one that
    stopped. kind: "loop" (RepetitionGuard always) or what the guard says (AnswerWatch: "loop" or "plan")."""

    def __init__(self, *guards):
        self.guards = [g for g in guards if g is not None]
        self.reason = self.kind = None

    def push(self, tok):
        for g in self.guards:
            if g.push(tok):
                self.reason, self.kind = g.reason, getattr(g, "kind", None) or "loop"
                return True
        return False


def pack_version(model_dir):
    """The model pack's version: the first line of VERSION in the model folder or the folder above it (the pack
    folder, see DEPLOY.md), else None."""
    for d in (Path(model_dir), Path(model_dir).parent):
        f = d / "VERSION"
        if f.is_file():
            line = f.read_text(encoding="utf-8").strip().splitlines()
            if line and line[0].strip():
                return line[0].strip()[:100]
    return None


def base_window(model_dir):
    """sequence_len of the newest base checkpoint's meta under model_dir/base_checkpoints (the pretraining window),
    or None when the base checkpoint is not in the pack."""
    best = None
    for f in Path(model_dir).glob("base_checkpoints/*/meta_*.json"):
        try:
            step = int(f.stem.split("_")[-1])
            n = json.loads(f.read_text(encoding="utf-8"))["model_config"]["sequence_len"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if isinstance(n, int) and (best is None or step > best[0]):
            best = (step, n)
    return best[1] if best else None


def window_check(meta, base_seq=None):
    """(trained window or None, warning or None) for a chat checkpoint's meta. The served attention window is
    model_config.sequence_len. SFT made with the notebook's window fix saves "trained_window"; SFT made before it
    kept the BASE model's window (nanochat builds the SFT model from the base config) while saving --max-seq-len, so a
    model pretrained at 1024 and fine-tuned "at 4096" is served with a 4096 window it never trained with (its JSON
    breaks past about 2,000 tokens). base_seq: the base checkpoint's sequence_len when known (base_window)."""
    mc = meta.get("model_config") if isinstance(meta, dict) else None
    served = mc.get("sequence_len") if isinstance(mc, dict) else None
    trained = meta.get("trained_window") if isinstance(meta, dict) else None
    if not isinstance(served, int):
        return (trained if isinstance(trained, int) else None), None
    if isinstance(trained, int):
        if trained != served:
            return trained, (f"window mismatch: this model trained with an attention window of {trained} tokens but is "
                             f"served with {served}")
        return trained, None
    if isinstance(base_seq, int):
        if base_seq < served:
            return base_seq, (f"window mismatch: SFT kept the base model's window of {base_seq} tokens (trained before "
                              f"the SFT window fix) but the model is served with {served}; long answers break past "
                              f"about {2 * base_seq} tokens. Retrain SFT with the current notebook.")
        return served, None
    if served > 2048:
        return None, (f"unknown trained window: this checkpoint was trained before the SFT window fix and its base "
                      f"checkpoint is not here, so it may have trained with the base model's shorter window (1024 on "
                      f"the t4 profile) while it is served with {served}")
    return None, None


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
        seq = getattr(self.model.config, "sequence_len", None)
        self.sequence_len = seq if isinstance(seq, int) and seq > 0 else None
        self.info = {"device": dev_type, "params_M": round(params / 1e6, 1), "step": meta.get("step"), "source": source,
                     "catalog_lock": False, "version": pack_version(model_dir), "sequence_len": self.sequence_len}
        # the window the model trained with vs the one it is served with (see window_check)
        trained, warning = window_check(meta, base_window(model_dir))
        self.info["trained_window"] = trained
        if warning:
            self.info["window_warning"] = warning
            print("WARNING (model):", warning, file=sys.stderr)
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

    def _decode1(self, t):
        txt = getattr(self, "_tok_text", None)
        return (txt[t] if txt is not None and txt[t] else None) or self.tok.decode([t])

    def token_budget(self, prompt_ids, max_tokens):
        """max_tokens, cut so prompt + answer stay inside the model's window (when known), never under MIN_ANSWER."""
        if self.sequence_len:
            return max(MIN_ANSWER, min(max_tokens, self.sequence_len - len(prompt_ids) - 1))
        return max_tokens

    def _run(self, text, max_tokens, temperature, seed, guard=None, on_token=None, attempt=0, flow=False, on_plan=None,
             prefix=None):
        """Stream one reply. Returns the decoded text; details go to self.last_stats (stop: "end", "max_tokens",
        "loop: <why>", "plan: <why>" when on_plan asked to stop after the plan, "not a flow"). prefix: the start of the
        reply, given (e.g. a plan and "Flow:"): the model continues it, and the text returned starts with it."""
        use_guard = self.guard_enabled if guard is None else guard
        prefix = prefix or ""
        g = Guards(RepetitionGuard(), AnswerWatch(self._decode1, on_plan, prefix=prefix)) if use_guard else (
            Guards(AnswerWatch(self._decode1, on_plan, prefix=prefix, max_plan_lines=10 ** 6, max_same_line=10 ** 6,
                               max_same_id=10 ** 6)) if on_plan is not None else None)
        prompt = self._prompt(text) + (self.tok.encode(prefix) if prefix else [])
        max_tokens = self.token_budget(prompt, max_tokens)
        out, reason, t0 = [], "max_tokens", time.time()
        if self.constraint is not None and on_token is None:
            import constrained as C
            con = self.constraints[attempt % len(self.constraints)]
            ids, reason, ncon, nch = C.generate(
                self.model, prompt, con, max_tokens, temperature, 50, seed,
                {self.assistant_end, self.bos}, lambda t: self._tok_text[t] or self.tok.decode([t]), g, flow=flow)
            if reason.startswith("loop: ") and g is not None and g.kind == "plan":
                reason = "plan: " + g.reason
            self.last_stats = {"tokens": len(ids), "stop": reason, "seconds": round(time.time() - t0, 2), "ids": ids,
                               "names_locked": ncon, "names_corrected": nch, "max_tokens": max_tokens}
            return prefix + self.tok.decode(ids)
        stream = self.engine.generate(prompt, num_samples=1, max_tokens=max_tokens,
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
                reason = ("plan: " if g.kind == "plan" else "loop: ") + g.reason
                break
            if flow:
                import constrained as C
                if C.not_a_flow(out, lambda t: self.tok.decode([t])):
                    reason = "not a flow"
                    break
        stream.close()
        self.last_stats = {"tokens": len(out), "stop": reason, "seconds": round(time.time() - t0, 2), "ids": out,
                           "max_tokens": max_tokens}
        return prefix + self.tok.decode(out)

    def count_tokens(self, text):
        """Tokens in text with the model's own tokenizer (for the usage numbers of the OpenAI-style answer)."""
        return len(self.tok.encode(text))

    def generate(self, request, attempt=0, max_tokens=MAX_ANSWER, on_plan=None, prefix=None):
        """One answer. max_tokens is cut to what the window leaves (token_budget); on_plan(plan_lines) -> True stops
        the answer when its plan is complete (last_stats["stop"] then starts with "plan:"); prefix: the answer's
        start, given (the server gives a part its plan lines and "Flow:", so only the flow JSON is written)."""
        return self._run(request, max_tokens, 0.2 + 0.3 * attempt, 1000 + attempt, attempt=attempt, flow=True,
                         on_plan=on_plan, prefix=prefix)

    def ask(self, question):
        return self._run(question, 300, 0.2, 1000).strip()


def _training_module(name):
    """Import a module from training/ (ooa_format) next to this folder."""
    d = str(Path(__file__).resolve().parent.parent / "training")
    if d not in sys.path:
        sys.path.insert(0, d)
    return __import__(name)


class FakeModel:
    """Test double: valid canned flows for known prompts, garbage (not JSON) otherwise.

    style "json": answers with the flow JSON only (the format before plans); "plan": plan + flow, and a clarifying
    question for the QUESTIONS prompts (training/ooa_format.py). The prompt may be a plain request or a user turn with
    context (request + "### Context ..."): the request part picks the answer. script: answers returned in order
    (the last one repeats) whatever the prompt, for tests of the retry layers."""

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
    QUESTIONS = {"automate my work": "Which application should the bot work in, and what should it do there?",
                 "do the usual thing": "What is the usual thing: which files or emails, and what should happen to them?"}

    def __init__(self, *a, style="json", script=None, catalog=None, **kw):
        self.info = {"device": "fake", "params_M": 0.0, "step": 0, "source": "fake"}
        self.calls = []
        self.last_stats = {}
        self.style, self.script, self.catalog = style, list(script or []), catalog

    def answer_for(self, request):
        F = _training_module("ooa_format")
        key = " ".join(F.request_of(request).lower().split()).rstrip(".")
        if self.style == "plan" and key in self.QUESTIONS:
            return F.QUESTION_HEAD + " " + self.QUESTIONS[key]
        flow = self.FLOWS.get(key)
        if not flow:
            return "garbage garbage garbage {not json"
        if self.style == "plan":
            return F.render_flow_answer(flow, self.catalog)
        return json.dumps(flow, separators=(",", ":"))

    def generate(self, request, attempt=0, max_tokens=2500, on_plan=None, prefix=None):
        """on_plan as NanochatLocal: a plan-first answer whose plan it accepts is cut after the plan (stop "plan").
        prefix: the answer starts with it and goes on with the flow JSON of its canned answer."""
        self.calls.append((request, attempt))
        if self.script:
            text = self.script[min(len(self.calls), len(self.script)) - 1]
        else:
            text = self.answer_for(request)
        self.last_stats = {"tokens": len(text) // 3, "stop": "end", "seconds": 0.0}
        F = _training_module("ooa_format")
        ans = F.parse_answer(text)
        if on_plan is not None and ans["plan"] and ans["flow_text"] is not None and on_plan(ans["plan"]):
            text = text[:text.index("\n" + F.FLOW_HEAD) + 1] + F.FLOW_HEAD
            self.last_stats = {"tokens": len(text) // 3, "stop": "plan: build the flow in parts", "seconds": 0.0}
        if prefix:
            text = prefix + (ans["flow_text"] if ans["flow_text"] is not None else text)
        return text

    def ask(self, question):
        q = str(question).lower()
        if "csv" in q:
            return self.ANSWERS["ooa.csv/read"]
        return "I am a fake model used for tests; I do not know."
