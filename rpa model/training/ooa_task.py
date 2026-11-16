"""nanochat Task that serves OOA dialogues from a JSONL file.

Copy into nanochat/tasks/ (or add to PYTHONPATH) and use in scripts/chat_sft.py's train_tasks list, e.g.
    from tasks.ooa import OOADialogues
    train_tasks = [OOADialogues("data/dialogues/train.jsonl"), ...]
Each JSONL line: {"messages": [{"role": "user"|"assistant", "content": str}, ...]}
 - optional first message with role "system" (nanochat merges it into the first user turn)
 - roles must alternate user, assistant, ...; assistant content is the minified Flow JSON or a short reply
Format checked against nanochat tasks/smoltalk.py and Tokenizer.render_conversation (commit 92d63d4).
"""
import importlib.util
import json
import os
from tasks.common import Task

_FIXERS = {}


def _session_fixer(data_path):
    """repair.session_fix + the catalog next to the data file, or None. Training flows that use Excel / Word / a
    browser without opening it first, or change a file without saving it, cannot run; they are corrected on load
    so older data files teach the runnable form too."""
    cat_path = os.path.join(os.path.dirname(os.path.abspath(data_path)), "catalog.jsonl")
    if cat_path in _FIXERS:
        return _FIXERS[cat_path]
    fixer = None
    dirs = [os.environ.get("OOA_TRAINING_DIR"), "/content/ooa", os.path.dirname(os.path.abspath(__file__))]
    rp = next((os.path.join(d, "repair.py") for d in dirs if d and os.path.exists(os.path.join(d, "repair.py"))), None)
    if rp and os.path.exists(cat_path):
        try:
            spec = importlib.util.spec_from_file_location("ooa_repair_for_data", rp)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            with open(cat_path, encoding="utf-8") as f:
                cat = {e["id"]: e for e in (json.loads(l) for l in f if l.strip())}
            if hasattr(mod, "session_fix"):
                fixer = lambda flow: mod.session_fix(flow, cat)
        except Exception as ex:          # never stop training over this
            print(f"note: training flows not session-checked ({ex})")
    _FIXERS[cat_path] = fixer
    return fixer


class OOADialogues(Task):
    """views: how many times the file is read per pass of the mixture (a float: 2.4 = every row twice, 40% of them a
    third time, chosen by a seeded shuffle); None = once, in file order (as before). See sft_setup.plan_mixture."""
    def __init__(self, path, views=None, seed=1, **kwargs):
        super().__init__(**kwargs)
        with open(path, encoding="utf-8") as f:
            self.rows = [json.loads(l) for l in f if l.strip()]
        self.num_rows = len(self.rows)
        self.order = None
        if views is not None and self.rows:
            import random
            n = max(1, int(round(float(views) * len(self.rows))))
            order, c = [], 0
            while len(order) < n:
                perm = list(range(len(self.rows)))
                random.Random(f"{seed}:{os.path.basename(path)}:{c}").shuffle(perm)
                order += perm
                c += 1
            self.order = order[:n]
        fix, fixed = _session_fixer(path), 0
        for row in self.rows if fix else []:
            m = row["messages"][-1]
            if m.get("role") != "assistant" or not m.get("content", "").lstrip().startswith("{"):
                continue
            try:
                flow = json.loads(m["content"])
                if isinstance(flow, dict) and isinstance(flow.get("steps"), list) and fix(flow):
                    m["content"] = json.dumps(flow, separators=(",", ":"), ensure_ascii=False)
                    fixed += 1
            except Exception:
                continue
        if fixed:
            print(f"{os.path.basename(path)}: {fixed} flows given their missing open / save step")

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.order) if self.order is not None else len(self.rows)

    def get_example(self, index):
        if self.order is not None:
            index = self.order[index]
        messages = self.rows[index]["messages"]
        rest = messages[1:] if messages[0]["role"] == "system" else messages
        assert len(rest) >= 2
        for i, m in enumerate(rest):
            assert m["role"] == ("user" if i % 2 == 0 else "assistant")
            assert isinstance(m["content"], str)
        return {"messages": messages}
