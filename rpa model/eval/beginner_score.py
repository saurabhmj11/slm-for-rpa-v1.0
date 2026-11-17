"""Beginner test set (eval/beginner_requests.jsonl): checks and scoring.

Each row: {"prompt", "kind": "flow"|"question", "must": [action ids in order], "ask_about": [terms], "style": [...]}.
A must entry is an action id or a list of ids that do the same job (any one of them counts at that position):
["ooa.excel/save", "ooa.excel/close"] (close saves when asked), ["ooa.network/ping", "ooa.network/host_reachable"].
  - flow rows (clear enough to build): right = a verified flow (the full assistant: repair + platform rules; the bare
    model: ooa_reward.score(...)["valid"]) whose actions contain `must` in order (other steps may come in between),
    the same rule as the plain_requests sets.
  - question rows (a key detail is truly missing: which folder, file, app, person or condition): right = the answer
    is a Question that mentions at least one ask_about term (case-insensitive). A flow built anyway is NOT right,
    even a verified one: the assistant has no channel that states an assumption clearly enough for a beginner to
    notice it, so guessing the missing folder / person / condition is the failure this set measures. It is counted
    separately as "assumed" so a reader sees how often the model guesses rather than fails.
Styles: vague, business, casual, short, typo, hinglish, indian_english, voice, story, missing_detail.

    python beginner_score.py check CATALOG.jsonl [TRAIN_FILE_OR_DIR ...]   ids exist; no prompt close to training data
    python beginner_score.py summary RESULTS.jsonl                          right per kind and per style

Closeness (too_close): word overlap = Jaccard of the lower-cased word sets ([a-z0-9]+) above MAX_OVERLAP, or content-
word containment = the share of the prompt's content words (no stop or politeness words, plural "s" dropped) found in
a training request of similar length (2/3 to 3/2 as many content words; prompts of MIN_CONTENT or more) at MAX_CONTAIN
or more; containment
catches a close paraphrase in the same style that Jaccard misses ("Kindly save the attachments of all unread mails in
inbox to D:\\Mails and mark them as read also" ~ "As discussed, kindly save the attachments of each new mail in Inbox
to Exports and mark them read pls": overlap 0.54, containment 0.86). The set was written so that no prompt has a
word overlap above MAX_OVERLAP with any training request (smart/data neutral records and the chat rows' user turns);
`check` lists containment matches as notes (the plain data teaches the same tasks on purpose), and the beginner
generator drops every generated request that is too close by either measure (gen_beginner.Guard).
Standard library only (also copied into the Colab notebook).
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

MAX_OVERLAP = 0.6
MAX_CONTAIN = 0.7
MIN_CONTENT = 5
# words that carry no content for the containment check (function words, politeness, fillers)
STOP = set("""a an the and or but if then so to of in on at by for from with into onto as is are was were be been it its
this that these those them they their there here i me my we our us you your he she his her him do does did done
please pls plz kindly also just all each every any some new now only itself very really can could would will shall
should may might must need want like ok okay hey hi um uh yeah so basically thanks thx ty asap sir team how many
have has does called named""".split())
KINDS = ("flow", "question")
STYLES = ("vague", "business", "casual", "short", "typo", "hinglish", "indian_english", "voice", "story", "missing_detail")
_WORD = re.compile(r"[a-z0-9]+")


def words(text):
    return set(_WORD.findall(str(text).lower()))


def word_overlap(a, b):
    """Jaccard of the word sets (0..1)."""
    x, y = words(a), words(b)
    return len(x & y) / len(x | y) if x and y else 0.0


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def validate(rows, catalog_ids):
    """Problems with the set itself (empty list = fine)."""
    out = []
    for i, r in enumerate(rows):
        if r.get("kind") not in KINDS:
            out.append(f"row {i}: kind {r.get('kind')!r}")
        if not str(r.get("prompt") or "").strip():
            out.append(f"row {i}: no prompt")
        for s in r.get("style") or []:
            if s not in STYLES:
                out.append(f"row {i}: unknown style {s}")
        if r.get("kind") == "flow":
            if not r.get("must"):
                out.append(f"row {i}: a flow row needs must")
            for a in r.get("must") or []:
                for x in (a if isinstance(a, list) else [a]):
                    if x not in catalog_ids:
                        out.append(f"row {i}: unknown action {x}")
        if r.get("kind") == "question" and not r.get("ask_about"):
            out.append(f"row {i}: a question row needs ask_about")
    seen = set()
    for r in rows:
        k = r.get("prompt", "").strip().lower()
        if k in seen:
            out.append(f"duplicate prompt: {k[:60]}")
        seen.add(k)
    return out


def content_words(text):
    """Content words of a text: no STOP words, no one-letter words, a plural "s" dropped (mails -> mail)."""
    out = set()
    for w in words(text):
        if w in STOP or len(w) < 2:
            continue
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w)
    return out


def containment(prompt, request):
    """The share of the prompt's content words found in the request (0..1); 0 when the prompt has fewer than
    MIN_CONTENT content words (short prompts: the Jaccard overlap and difflib checks cover them) or the request has
    fewer than 2/3 or more than 3/2 as many."""
    p, r = content_words(prompt), content_words(request)
    if len(p) < MIN_CONTENT or not (len(p) * 2 / 3 <= len(r) <= len(p) * 3 / 2):
        return 0.0
    return len(p & r) / len(p)


def too_close(prompt, request):
    """(True, why) when a training request is too close to a test prompt (Jaccard overlap or containment)."""
    o = word_overlap(prompt, request)
    if o > MAX_OVERLAP:
        return True, f"overlap {o:.2f}"
    c = containment(prompt, request)
    if c >= MAX_CONTAIN:
        return True, f"containment {c:.2f}"
    return False, ""


def closest_contained(prompt, requests, index=None):
    """(containment, request) of the training request that contains most of the prompt's content words."""
    best, who = 0.0, ""
    cand = requests
    if index is not None:
        ids = set()
        for w in words(prompt):
            ids.update(index.get(w, ()))
        cand = [requests[i] for i in ids]
    for r in cand:
        c = containment(prompt, r)
        if c > best:
            best, who = c, r
    return best, who


def closest(prompt, requests, index=None):
    """(overlap, request) of the most word-overlapping request. index: from overlap_index(requests), much faster."""
    best, who = 0.0, ""
    pw = words(prompt)
    cand = requests
    if index is not None:
        ids = set()
        for w in pw:
            ids.update(index.get(w, ()))
        cand = [requests[i] for i in ids]
    for r in cand:
        o = word_overlap(prompt, r)
        if o > best:
            best, who = o, r
    return best, who


def overlap_index(requests):
    idx = defaultdict(list)
    for i, r in enumerate(requests):
        for w in words(r):
            idx[w].append(i)
    return idx


def training_requests(paths):
    """Every request in neutral-record files ({"request"}) and chat files (first user turn, before '### Context')."""
    files = []
    for p in paths:
        p = Path(p)
        files += sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    out = []
    for f in files:
        if f.name in ("catalog.jsonl", "examples.jsonl", "train_catalog.jsonl"):
            continue
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and isinstance(r.get("request"), str):
                out.append(r["request"])
            elif isinstance(r, dict) and isinstance(r.get("messages"), list):
                u = next((m for m in r["messages"] if isinstance(m, dict) and m.get("role") == "user"), None)
                if u and isinstance(u.get("content"), str):
                    out.append(u["content"].split("\n### Context")[0])
    return out


def in_order(need, got):
    """True when got holds need in order (other actions may come in between); a need entry that is a list accepts
    any one of its ids."""
    it = iter(got)
    return all(any((g in a) if isinstance(a, list) else (a == g) for g in it) for a in need)


def mentions(question, terms):
    q = str(question or "").lower()
    return any(str(t).lower() in q for t in terms)


def score(row, kind, valid=False, actions=(), question=None):
    """One answer -> {"right", "outcome"}. kind: "flow" | "question" | "reply" (what the answer was); valid: the flow
    passed the checker; actions: the flow's actions in order; question: the question text."""
    if row["kind"] == "flow":
        if kind == "flow" and valid:
            ok = in_order(row["must"], list(actions))
            return {"right": ok, "outcome": "right" if ok else "wrong_actions"}
        return {"right": False, "outcome": {"flow": "invalid_flow", "question": "asked"}.get(kind, "no_flow")}
    if kind == "question":
        ok = mentions(question, row.get("ask_about") or [])
        return {"right": ok, "outcome": "asked_right" if ok else "asked_off_topic"}
    if kind == "flow" and valid:
        return {"right": False, "outcome": "assumed"}
    return {"right": False, "outcome": "no_answer"}


def score_router(row, res, actions_of):
    """Router.run result -> score. actions_of(flow_text) -> action list (ooa_reward.actions_of(json.loads(...)))."""
    if res.get("verified"):
        return score(row, "flow", True, actions_of(res["flow"]))
    if res.get("question"):
        return score(row, "question", question=res["question"])
    return score(row, "reply")


def summarize(rows, results):
    """rows and their scores (same order) -> {"all": [right, n], "kind": {k: [right, n]}, "style": {s: [right, n]},
    "outcomes": {o: n}}."""
    s = {"all": [0, 0], "kind": defaultdict(lambda: [0, 0]), "style": defaultdict(lambda: [0, 0]), "outcomes": defaultdict(int)}
    for row, res in zip(rows, results):
        r = int(bool(res["right"]))
        for key in [s["all"], s["kind"][row["kind"]]] + [s["style"][x] for x in row.get("style") or []]:
            key[0] += r
            key[1] += 1
        s["outcomes"][res["outcome"]] += 1
    return {"all": s["all"], "kind": dict(s["kind"]), "style": dict(s["style"]), "outcomes": dict(s["outcomes"])}


def summary_text(summ):
    f = lambda v: f"{v[0]}/{v[1]}"
    return (f"right {f(summ['all'])} | " + " ".join(f"{k} {f(v)}" for k, v in summ["kind"].items()) + " | styles: "
            + ", ".join(f"{k} {f(v)}" for k, v in sorted(summ["style"].items())) + " | outcomes: "
            + ", ".join(f"{k} {v}" for k, v in sorted(summ["outcomes"].items())))


def main(argv):
    here = Path(__file__).resolve().parent
    if argv and argv[0] == "check":
        rows = load(here / "beginner_requests.jsonl")
        ids = {json.loads(l)["id"] for l in open(argv[1], encoding="utf-8") if l.strip()}
        probs = validate(rows, ids)
        reqs = training_requests(argv[2:]) if len(argv) > 2 else []
        idx = overlap_index(reqs)
        worst, near = 0.0, []
        for r in rows:
            o, who = closest(r["prompt"], reqs, idx)
            worst = max(worst, o)
            if o > MAX_OVERLAP:
                probs.append(f"too close ({o:.2f}): {r['prompt'][:60]} ~ {who[:60]}")
            c, who = closest_contained(r["prompt"], reqs, idx)
            if c >= MAX_CONTAIN:
                near.append(f"same task (containment {c:.2f}): {r['prompt'][:60]} ~ {who[:60]}")
        kinds = defaultdict(int)
        for r in rows:
            kinds[r["kind"]] += 1
        print(f"{len(rows)} rows {dict(kinds)} | {len(reqs)} training requests | highest word overlap {worst:.2f}")
        if near:     # the plain training data covers the same tasks on purpose: shown, not a failure
            print("note - training requests with most of a prompt's content words (fine for plain wording; the "
                  "beginner generator drops such requests, gen_beginner.Guard):\n  " + "\n  ".join(near))
        print("\n".join(probs) or "OK")
        return 1 if probs else 0
    if argv and argv[0] == "summary":
        res = load(argv[1])
        rows = [r["row"] for r in res]
        print(summary_text(summarize(rows, res)))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
