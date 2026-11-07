"""Action retrieval + prompt format. Pure standard library (BM25 over the catalog), works anywhere, also in a browser worker.

Why: the model invents action and input names when it has to recall ~1,300 of them. Instead each request carries the
most likely candidate actions (id, required inputs, optional inputs), so the model copies names instead of remembering them.

    idx = ActionIndex(catalog)            # catalog = list of entries from catalog.jsonl
    idx.search("read invoices from csv", k=10)
    make_prompt(request, idx.search(request, 10))  ->  text the model is trained and run on
"""
import math, re
from collections import Counter

STOP = set("a an the and or of to in on for with from by is are be it its this that as at into then than each every all any "
           "me my your our their can will should would please make create build flow automation automations do does".split())


def tokens(text):
    out = []
    for w in re.split(r"[^A-Za-z0-9]+", text.lower()):
        if not w or w in STOP:
            continue
        out.append(w)
        if w.endswith("s") and len(w) > 3:      # crude plural folding: invoices -> invoice
            out.append(w[:-1])
    return out


def _doc_tokens(e):
    t = tokens(e["id"]) * 2 + tokens(e["name"]) * 2 + tokens(e["toolkit"]) + tokens(e["description"])
    t += tokens(" ".join(i["name"] for i in e["inputs"]))
    t += tokens(" ".join(o["name"] for o in e["outputs"]))
    return t


class ActionIndex:
    def __init__(self, catalog, kinds=("action",), k1=1.4, b=0.75):
        self.entries = [e for e in catalog if e["kind"] in kinds]
        self.k1, self.b = k1, b
        self.docs = [Counter(_doc_tokens(e)) for e in self.entries]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.inv = {}
        for i, d in enumerate(self.docs):
            for w in d:
                self.inv.setdefault(w, []).append(i)
        self.by_id = {e["id"]: e for e in self.entries}

    def search(self, query, k=10):
        q = Counter(tokens(query))
        scores = {}
        for w in q:
            if w not in self.inv:
                continue
            for i in self.inv[w]:
                f = self.docs[i][w]
                s = self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
                scores[i] = scores.get(i, 0.0) + s
        top = sorted(scores, key=lambda i: -scores[i])[:k]
        return [self.entries[i] for i in top]


def candidate_line(e):
    req = [i["name"] for i in e["inputs"] if i["required"]]
    opt = [i["name"] for i in e["inputs"] if not i["required"]][:6]
    outs = [o["name"] for o in e["outputs"]][:3]
    s = f"{e['id']} | in: {','.join(req) or '-'}"
    if opt:
        s += f" | opt: {','.join(opt)}"
    if outs:
        s += f" | out: {','.join(outs)}"
    return s


def make_prompt(request, candidates):
    lines = "\n".join(candidate_line(e) for e in candidates)
    return f"Actions:\n{lines}\n\nRequest: {request}"


def with_gold(candidates, gold_ids, by_id, k):
    """Training only: make sure the gold actions are in the candidate list (replace the lowest-ranked extras)."""
    ids = [e["id"] for e in candidates]
    need = [g for g in dict.fromkeys(gold_ids) if g not in ids and g in by_id]
    keep = candidates[: max(0, k - len(need))]
    return keep + [by_id[g] for g in need]
