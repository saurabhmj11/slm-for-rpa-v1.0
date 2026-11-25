"""Reinforcement learning with the platform checker as the reward, in the format the model is trained and served with
(plan-first answers with compact flows, retrieval context in the prompt). Used by Colab cell 9 (RL_V5) and the PC
notebook's MODE "rl" (training/local/rl_pc.py); tested on a CPU by test_rl_setup.py.

1. The prompt pool (build_pool): requests with a known-good answer that the last SFT never trained on and that are not
   test prompts. Fresh records come from the data generators with new seeds (gen_targeted: everyday tasks, gen_long:
   whole business processes, gen_clarify: vague requests whose right answer is a question + short clear ones,
   gen_flows: short catalog flows), and a share of the flow records are reworded the way beginners write
   (gen_beginner.beginner_versions: vague, business words, casual, typos, Hinglish, voice, story; same flow), and
   two records are joined into one request asking for both jobs (joined_records; the model's measured weak spot: it
   writes one of the two jobs). Dropped:
   a request already in the SFT data (exact match after normalising, against examples.jsonl and any other training
   file given), and any request too close to a test prompt (eval/*.jsonl and the generated eval sets of cell 6:
   eval/beginner_score.too_close(prompt, request)[0]). Each request's user turn is built by the caller's prompt
   function: the serving Router.prompt (ooa_format.context_actions + build_user with the example index of sft_render),
   so the model learns on exactly the prompts it gets in use.
2. The reward (reward): the answer is read like serving reads it (ooa_format.parse_answer(expand=True): a compact
   flow is expanded into the full flow) and the expanded flow is scored by the platform checker (ooa_reward.score).
   See REWARD_DOC.
3. The patch of nanochat scripts/chat_rl.py (pinned commit 92d63d4; patch_chat_rl): asserted anchors, our task, the
   attention window the SFT checkpoint trained with (trained_window, 4096), max new tokens = what the window has left,
   the end token trained when the model ended on its own, requests whose answers all score the same skipped (they
   carry no signal; requests every answer solves are not sampled again), fp16 loss scaling (T4), our evaluation
   (mean reward / valid / solved on the held-out RL rows), resumable checkpoints (model + optimizer every
   --save-every steps; a new SFT model starts a new RL run and the old one is kept aside).
"""
import json
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import ooa_format as F   # noqa: E402
import ooa_reward as R   # noqa: E402

REWARD_DOC = """Reward (0..1) for one sampled answer, from the platform checker on the EXPANDED flow:
  flow prompt (the gold answer is a flow), answer is a flow:
    0.25 valid      the checker's rule parts (JSON, schema, actions exist, inputs / outputs, variables), weighted as
                    in ooa_reward.W without its sequence part, scaled to 0..1
    0.25 recall     gold actions present in order: longest common subsequence with the gold sequence / gold length
    0.10 precision  the same subsequence / the answer's length (no padding with extra steps)
    0.10 plan       the plan names the gold actions in order (same measure on the plan's action ids)
    0.10 coverage   the plan is there and every catalog action it names is in the flow (ooa_format.plan_coverage)
    0.10 ended      the model ended the answer itself (not cut at the token cap)
    0.10 solved     valid and the exact gold action sequence
    - up to 0.20 for repetition (the same plan line / step written again and again)
    - up to 0.10 for length past 2 x the gold answer + 128 tokens
  flow prompt, answer is a question or a chat reply: 0 (questions earn only on clarify prompts)
  clarify prompt (the gold answer is a question): a question 0.8 (+0.2 when it ended on its own, -0.3 when it is over
    300 characters); a flow or a reply 0 (a flow built on a guess is the failure the beginner set measures)
Clipped to 0..1. solved / valid are reported for evaluation."""

W_FLOW = {"valid": 0.25, "recall": 0.25, "precision": 0.10, "plan": 0.10, "coverage": 0.10, "ended": 0.10,
          "solved": 0.10}
REPEAT_PENALTY = 0.20
LENGTH_PENALTY = 0.10
QUESTION_MAX_CHARS = 300
_RULE_PARTS = [k for k in R.W if k != "match"]

_CAT = None          # {id: entry} for parse_answer / plan_coverage (set by configure)


def configure(catalog_path=None, schema_path=None):
    """Load the catalog and the flow schema for the checker (ooa_reward) and the answer reader (ooa_format)."""
    global _CAT
    catalog_path = catalog_path or os.environ["OOA_CATALOG"]
    schema_path = schema_path or os.environ["OOA_SCHEMA"]
    R.configure(catalog_path, schema_path)
    _CAT = {}
    with open(catalog_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                _CAT[e["id"]] = e
    return _CAT


def lcs(a, b):
    """Length of the longest common subsequence of two sequences."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


_NUM = re.compile(r"^\s*(?:\d+[.)]\s*)?")


def repetition(units):
    """Share of repeated units (0 when fewer than 4): 1 - distinct / all."""
    units = [u for u in units if u]
    if len(units) < 4:
        return 0.0
    return 1.0 - len(set(units)) / len(units)


def _units(ans, flow):
    """What repetition is measured on: the plan lines (without their numbers) and the steps (without id and
    description); for an answer that is not a flow, its non-empty lines."""
    units = [_NUM.sub("", l).strip().lower() for l in ans.get("plan") or []]
    if isinstance(flow, dict):
        for s in R.walk(flow.get("steps")):
            units.append(F.compact({k: v for k, v in s.items() if k not in ("id", "description", "children", "else",
                                                                             "elseIf", "catch", "finally")}))
    if not units:
        units = [l.strip() for l in str(ans.get("text") or "").split("\n") if l.strip()]
    return units


def _penalties(ans, flow, tokens, gold_tokens):
    rep = repetition(_units(ans, flow))
    pen_rep = REPEAT_PENALTY * min(1.0, max(0.0, rep - 0.2) / 0.6)
    pen_len = 0.0
    if gold_tokens and tokens:
        cap = 2 * gold_tokens + 128
        if tokens > cap:
            pen_len = LENGTH_PENALTY * min(1.0, (tokens - cap) / cap)
    return pen_rep, pen_len, rep


def reward(gold, text, ended=True, tokens=0):
    """Score one sampled answer. gold: {"kind": "flow"|"question", "actions": [gold action sequence], "request": str,
    "tokens": tokens of the gold answer}. text: the model's answer (decoded, without the end token); ended: it ended on
    its own; tokens: how many tokens it wrote. Returns {"total", "kind" (of the answer), "valid", "solved", parts}."""
    if _CAT is None:
        configure()
    ans = F.parse_answer(text, expand=True, catalog=_CAT, request=gold.get("request"))
    out = {"kind": ans["kind"], "valid": False, "solved": False, "ended": bool(ended)}
    if gold.get("kind") == "question":
        q = ans.get("question") or ""
        if ans["kind"] == "question" and q.strip():
            t = 0.8 + (0.2 if ended else 0.0) - (0.3 if len(q) > QUESTION_MAX_CHARS else 0.0)
            out.update(solved=bool(ended) and len(q) <= QUESTION_MAX_CHARS)
        else:
            t = 0.0
        out["total"] = round(max(0.0, min(1.0, t)), 4)
        return out
    if ans["kind"] != "flow":
        out["total"] = 0.0
        return out
    gold_acts = list(gold.get("actions") or [])
    flow = F.flow_object(ans.get("flow_text")) if ans.get("flow_text") else None
    sc = R.score(ans["flow_text"], gold_acts) if ans.get("flow_text") else {k: 0.0 for k in R.W}
    acts = R.actions_of(flow) if isinstance(flow, dict) else []
    plan_acts = F.plan_actions(ans.get("plan"))
    p = {"valid": sum(R.W[k] * sc.get(k, 0.0) for k in _RULE_PARTS) / sum(R.W[k] for k in _RULE_PARTS)}
    common = lcs(acts, gold_acts)
    p["recall"] = common / len(gold_acts) if gold_acts else 0.0
    p["precision"] = common / len(acts) if acts else 0.0
    p["plan"] = lcs(plan_acts, gold_acts) / len(gold_acts) if gold_acts else 0.0
    p["coverage"] = 1.0 if ans.get("plan") and isinstance(flow, dict) and not F.plan_coverage(
        ans["plan"], flow, _CAT) else 0.0
    p["ended"] = 1.0 if ended else 0.0
    p["solved"] = 1.0 if sc.get("solved") else 0.0
    pen_rep, pen_len, rep = _penalties(ans, flow, tokens, gold.get("tokens"))
    total = sum(W_FLOW[k] * p[k] for k in W_FLOW) - pen_rep - pen_len
    out.update({k: round(v, 4) for k, v in p.items()})
    out.update(valid=bool(sc.get("valid")), solved=bool(sc.get("solved")), repetition=round(rep, 3),
               penalty=round(pen_rep + pen_len, 4), total=round(max(0.0, min(1.0, total)), 4))
    return out


# ---------------------------------------------------------------------------------------------- the prompt pool
# Shares of the pool. Measured on the first 4096-window model (v4, 3 answers each at temperature 1): it solved 99% of
# the plain generator requests (targeted, long, beginner rewordings, short clarify flows) exactly, and all answers to
# such a request get the same reward, so they teach nothing (advantage 0). The pool therefore leans on what the model
# still gets wrong: two requests joined into one ("joined": longer, two jobs in one sentence), whole processes and
# vague requests that need a question; the chat_rl patch also skips requests whose answers all score the same.
POOL_SHARES = {"targeted": 0.12, "long": 0.18, "clarify_clear": 0.04, "synth": 0.03, "joined": 0.30}  # of n
BEGINNER_SHARE = 0.20          # flow records reworded the way beginners write (same flow), of n
QUESTION_SHARE = 0.13          # clarify prompts (the right answer is a question), of n
JOIN_WORDS = ["After that, ", "Then ", "Also ", "Once that is done, ", "And after that ", "Next, ", "phir "]
N_VAL = 40                     # held-out RL rows: the evaluation every --eval-every steps
SEED = 1123                    # new seeds: the SFT data used gen_targeted 11, gen_long 23, gen_clarify 31, gen_flows 1/2


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(text).lower())).strip()


def eval_prompts(eval_dir=None, extra_files=()):
    """Every test prompt: eval/*.jsonl ("prompt") and extra files (Drive sft/eval*.jsonl of cell 6: "prompt")."""
    ev = eval_dir or os.path.join(os.path.dirname(HERE), "eval")
    files = [os.path.join(ev, n) for n in sorted(os.listdir(ev)) if n.endswith(".jsonl")] if os.path.isdir(ev) else []
    out = []
    for p in list(files) + [x for x in extra_files if x and os.path.exists(x)]:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(r, dict) and str(r.get("prompt") or "").strip():
                        out.append(str(r["prompt"]))
    return out


def training_requests(files):
    """Normalised requests of the training data: examples.jsonl ("request"), neutral records ("request") and chat rows
    (the user turn's request part)."""
    seen = set()
    for p in files:
        if not p or not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(r, dict):
                    continue
                if r.get("request"):
                    seen.add(norm(r["request"]))
                for m in r.get("messages") or []:
                    if isinstance(m, dict) and m.get("role") == "user":
                        seen.add(norm(F.split_user(m.get("content", ""))[0]))
    seen.discard("")
    return seen


class LeakCheck:
    """A request is out when it is in the training data or too close to a test prompt (beginner_score.too_close)."""

    def __init__(self, train_requests, prompts):
        ev = os.path.join(os.path.dirname(HERE), "eval")       # the repo; in Colab beginner_score.py is in /content/ooa
        if os.path.isdir(ev) and ev not in sys.path:
            sys.path.insert(0, ev)
        import beginner_score as B
        self.B, self.train, self.prompts = B, set(train_requests), list(dict.fromkeys(prompts))
        self.words = [B.words(p) for p in self.prompts]
        self.dropped = {"in_training": 0, "close_to_test": 0}

    def ok(self, request):
        if norm(request) in self.train:
            self.dropped["in_training"] += 1
            return False
        rw = self.B.words(request)
        for p, pw in zip(self.prompts, self.words):
            if not (pw & rw):
                continue
            if self.B.too_close(p, request)[0]:
                self.dropped["close_to_test"] += 1
                return False
        return True


def _read_records(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def generate_records(catalog_path, schema_path, work_dir, n, seed=SEED, platform=None, python=None, log=print):
    """Fresh neutral records from the generators with new seeds, about n flow records + n x QUESTION_SHARE clarify
    records: [{"kind": "flow"|"clarify", "request", "flow"|"question", "source"}]."""
    python = python or sys.executable
    os.makedirs(work_dir, exist_ok=True)
    need = {k: max(1, int(round(n * s * 1.6))) for k, s in POOL_SHARES.items()}    # 1.6: room for the leak filter
    need["targeted"] += need.pop("joined")             # the joined requests are made from targeted + long records
    n_vague = max(1, int(round(n * QUESTION_SHARE * 1.6)))
    jobs = [("targeted", ["gen_targeted.py", catalog_path, schema_path, "targeted.jsonl", str(need["targeted"]),
                          str(seed)]),
            ("long", ["gen_long.py", catalog_path, schema_path, "long.jsonl", str(need["long"] + need["targeted"] // 2),
                      str(seed + 1)]),
            ("clarify", ["gen_clarify.py", catalog_path, schema_path, "clarify.jsonl", str(n_vague),
                         str(need["clarify_clear"]), str(seed + 2)])]
    recs = []
    for name, cmd in jobs:
        out = os.path.join(work_dir, cmd[3])
        cmd = [python, os.path.join(HERE, cmd[0])] + cmd[1:3] + [out] + cmd[4:]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
        if r.returncode != 0:
            raise RuntimeError(f"{name} generator failed: {r.stderr[-800:]}")
        log(f"  {name}: {(r.stdout.strip().splitlines() or [''])[-1][:160]}")
        recs += _read_records(out)
    if platform and need["synth"] > 0:
        import gen_flows as G
        cat_list = list(_read_jsonl_list(catalog_path))
        for q, flow, _ in G.generate(cat_list, platform, need["synth"], seed + 3):
            recs.append({"kind": "flow", "request": q, "flow": flow, "source": "synth"})
        log(f"  synth: {need['synth']} short catalog flows")
    return recs


def _read_jsonl_list(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def beginner_records(records, n, seed=SEED):
    """n beginner-worded copies of flow records (one style each, the same flow)."""
    import gen_beginner as GB
    rng = random.Random(seed + 7)
    flows = [r for r in records if r.get("kind") == "flow" and isinstance(r.get("flow"), dict)
             and not str(r.get("source", "")).startswith(("synth", "beginner"))]
    rng.shuffle(flows)
    out = []
    for r in flows:
        if len(out) >= n:
            break
        try:
            vs = GB.beginner_versions(r["request"], rng, 1, flow=r["flow"], origin=str(r.get("source", "")))
        except Exception:
            vs = []
        for tags, text in vs:
            out.append({"kind": "flow", "request": text, "flow": r["flow"],
                        "source": "beginner:" + "+".join(tags) + ":" + str(r.get("source", ""))})
    return out


def _lower_first(t):
    return t[:1].lower() + t[1:] if t[:2] != t[:2].upper() else t


def joined_records(records, n, seed=SEED):
    """n requests that ask for two jobs in one message: two flow records (not two long processes) joined as
    "<first>. After that, <second>", gold = the first flow's steps, then the second's (ooa_format.merge_flows; a
    variable name both use is renamed in the second, so the two jobs keep their own folders / files). Kept only when
    the joined flow passes the checker (ooa_reward.problems)."""
    rng = random.Random(seed + 11)
    flows = [r for r in records if r.get("kind") == "flow" and isinstance(r.get("flow"), dict)
             and str(r.get("source", "")).split(":")[0] in ("targeted", "long", "clarify_counter")]
    out, tries = [], 0
    while len(out) < n and len(flows) >= 2 and tries < n * 6:
        tries += 1
        a, b = rng.sample(flows, 2)
        if str(a.get("source", "")).startswith("long") and str(b.get("source", "")).startswith("long"):
            continue
        fa, fb = a["flow"], b["flow"]
        names_a = {v.get("name") for v in fa.get("variables") or [] if isinstance(v, dict)}
        for v in list(fb.get("variables") or []):
            nm = v.get("name") if isinstance(v, dict) else None
            if nm in names_a:
                fb = F.rename_variable(fb, nm, nm + "_2")
        merged = F.merge_flows([fa, fb])
        ra, rb = " ".join(str(a["request"]).split()).rstrip(" .!"), " ".join(str(b["request"]).split())
        req = ra + ". " + rng.choice(JOIN_WORDS) + _lower_first(rb)
        merged["description"] = req
        if R.problems(F.compact(merged)):
            continue
        out.append({"kind": "flow", "request": req, "flow": merged,
                    "source": "joined:" + str(a.get("source", "")).split(":")[0] + "+" + str(b.get("source", "")).split(":")[0]})
    return out


def gold_of(record, catalog):
    """(gold dict, gold answer text) for a neutral record, or (None, None). The flow gets repair.session_fix first,
    like render_dataset, so the gold matches the flows the model was trained on."""
    if record.get("kind") == "clarify":
        q = " ".join(str(record.get("question") or "").split())
        if not q:
            return None, None
        return {"kind": "question", "actions": [], "request": record["request"], "tokens": F.approx_tokens(q) + 2,
                "source": record.get("source", "")}, F.QUESTION_HEAD + " " + q
    flow = record.get("flow")
    if isinstance(flow, str):
        try:
            flow = json.loads(flow)
        except ValueError:
            return None, None
    if not isinstance(flow, dict) or not flow.get("steps"):
        return None, None
    try:
        import repair as RP
        flow = json.loads(json.dumps(flow))
        RP.session_fix(flow, catalog)
    except Exception:
        pass
    acts = R.actions_of(flow)
    if not acts:
        return None, None
    answer = F.render_flow_answer(flow, catalog, F.ANSWER_COMPACT)
    return {"kind": "flow", "actions": acts, "request": record["request"], "tokens": F.approx_tokens(answer),
            "source": record.get("source", "")}, answer


def build_pool(records, prompt_fn, catalog, leak, n, n_val=N_VAL, window=F.LONG_WINDOW, min_new=256, seed=SEED,
               log=print):
    """(train rows, val rows, report). records: neutral records; prompt_fn(request) -> the user turn (the serving
    Router.prompt); leak: LeakCheck. A row is kept when its prompt plus its gold answer (+30%, + min_new spare) fits
    the window by ooa_format.approx_tokens. Row: {"messages": [user, assistant (gold answer)], "gold": {...}}."""
    rng = random.Random(seed)
    seen, rows = set(), []
    stats = {"records": len(records), "no_gold": 0, "leak": 0, "duplicate": 0, "too_long": 0}
    order = list(records)
    rng.shuffle(order)
    for r in order:
        req = " ".join(str(r.get("request") or "").split())
        if not req:
            continue
        k = norm(req)
        if k in seen:
            stats["duplicate"] += 1
            continue
        seen.add(k)
        if not leak.ok(req):
            stats["leak"] += 1
            continue
        gold, answer = gold_of(r, catalog)
        if gold is None:
            stats["no_gold"] += 1
            continue
        user = prompt_fn(req)
        need = F.approx_tokens(user) + 8 + max(min_new, int(gold["tokens"] * 1.3) + 32)
        if need > window:
            stats["too_long"] += 1
            continue
        rows.append({"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": answer}],
                     "gold": gold})
    keep = pick_shares(rows, n + n_val)
    rng.shuffle(keep)
    val, train = keep[:n_val], keep[n_val:]
    stats.update(train=len(train), val=len(val), questions=sum(x["gold"]["kind"] == "question" for x in keep),
                 leak_detail=dict(leak.dropped))
    by = {}
    for x in keep:
        g = source_group(x["gold"])
        by[g] = by.get(g, 0) + 1
    stats["sources"] = by
    return train, val, stats


def source_group(gold):
    """The pool share a row counts toward: question, beginner, targeted, long, clarify_clear or synth."""
    if gold.get("kind") == "question":
        return "question"
    src = str(gold.get("source", ""))
    if src.startswith("beginner"):
        return "beginner"
    if src.startswith("clarify_counter"):
        return "clarify_clear"
    return src.split(":")[0] or "other"


def pick_shares(rows, total):
    """Up to total rows in the pool's shares (POOL_SHARES, BEGINNER_SHARE, QUESTION_SHARE; rows in their given order).
    A group with too few rows leaves its places to the others, in proportion to their shares."""
    shares = dict(POOL_SHARES, beginner=BEGINNER_SHARE, question=QUESTION_SHARE)
    groups = {}
    for x in rows:
        groups.setdefault(source_group(x["gold"]), []).append(x)
    taken = {g: 0 for g in groups}
    left = total
    while left > 0:
        open_ = [g for g in groups if taken[g] < len(groups[g])]
        if not open_:
            break
        w = {g: shares.get(g, 0.01) for g in open_}
        ws = sum(w.values())
        got = 0
        for g in open_:
            q = min(len(groups[g]) - taken[g], max(1, int(left * w[g] / ws)))
            q = min(q, left - got)
            taken[g] += q
            got += q
            if got >= left:
                break
        left -= got
    return [x for g in groups for x in groups[g][:taken[g]]]


def write_rows(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def router_prompter(catalog_path, platform, examples_path, serving_dir=None):
    """The serving Router's prompt function (serving/ooa_server.Router.prompt) for this model pack: the same retrieval
    context, examples and context budget (render_report.json beside examples.jsonl) as in use."""
    sd = serving_dir or os.path.join(os.path.dirname(HERE), "serving")
    if sd not in sys.path:
        sys.path.insert(0, sd)
    import ooa_server as S
    router = S.Router(catalog_path, platform, S.Policy(allow_escalation=False), None, None, examples_path, "context")
    return router.prompt


def prepare(out_dir, catalog_path, schema_path, platform, examples_path, n, train_files=(), eval_files=(),
            seed=SEED, n_val=None, window=F.LONG_WINDOW, python=None, eval_dir=None, log=print):
    """Everything cell 9 needs before training: fresh records -> beginner rewordings -> leak filter -> prompts ->
    out_dir/rl_train.jsonl + rl_val.jsonl + rl_pool_report.json. Returns the report."""
    cat = configure(catalog_path, schema_path)
    n_val = min(N_VAL, max(4, n // 10)) if n_val is None else n_val
    work = os.path.join(out_dir, "rl_records")
    log(f"generating fresh records (seed {seed}) ...")
    recs = generate_records(catalog_path, schema_path, work, n + n_val, seed, platform, python, log)
    recs += joined_records(recs, int(round((n + n_val) * POOL_SHARES["joined"] * 1.5)), seed)
    recs += beginner_records(recs, int(round((n + n_val) * BEGINNER_SHARE * 1.5)), seed)
    leak = LeakCheck(training_requests([examples_path] + list(train_files)), eval_prompts(eval_dir, eval_files))
    log(f"leak check: {len(leak.train)} training requests, {len(leak.prompts)} test prompts")
    prompt_fn = router_prompter(catalog_path, platform, examples_path)
    train, val, rep = build_pool(recs, prompt_fn, cat, leak, n, n_val, window, seed=seed, log=log)
    write_rows(os.path.join(out_dir, "rl_train.jsonl"), train)
    write_rows(os.path.join(out_dir, "rl_val.jsonl"), val)
    rep.update(seed=seed, examples=examples_path, window=window)
    with open(os.path.join(out_dir, "rl_pool_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    return rep


# ---------------------------------------------------------------------------------------------- chat_rl.py patch
IMPORT_OLD = "from tasks.gsm8k import GSM8K"
IMPORT_NEW = "from tasks.ooa_rl import OOARL"
ARGS_ANCHOR = 'parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")\n'
ARGS_NEW = ARGS_ANCHOR + '''# OOA: our task, the step budget, sampling in smaller batches than the samples, resume
parser.add_argument("--train-path", type=str, required=True, help="rl_train.jsonl (rl_setup.prepare)")
parser.add_argument("--val-path", type=str, required=True, help="rl_val.jsonl (rl_setup.prepare)")
parser.add_argument("--num-steps", type=int, default=0, help="optimizer steps (0: one pass over the requests)")
parser.add_argument("--sample-batch-size", type=int, default=0, help="answers sampled at once (0: all --num-samples)")
parser.add_argument("--min-new-tokens", type=int, default=256, help="skip a request when the window leaves fewer")
parser.add_argument("--eval-temperature", type=float, default=0.0, help="sampling temperature of the evaluation")
parser.add_argument("--keep-checkpoints", type=int, default=2, help="RL model files kept (the newest ones)")
parser.add_argument("--resume", type=int, default=1, help="1: continue the RL run of this SFT model when there is one")
parser.add_argument("--skip-same", type=int, default=1, help="1: a request whose answers all score the same teaches "
                    "nothing (advantage 0): sample the next one instead (requests all answers solve are not tried again)")
parser.add_argument("--max-skips", type=int, default=0, help="skips in a row before such a group is used anyway "
                    "(0: 3 x --examples-per-step), so a step's time stays bounded")
'''
LOAD_OLD = ('model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag=args.model_tag, '
            'step=args.model_step)\nengine = Engine(model, tokenizer) # for sampling rollouts\n')
LOAD_NEW = '''# OOA: start from the newest SFT model, or continue the RL run that started from it (resume)
import glob as _glob, json as _json, re as _re, shutil as _shutil, time as _time
from nanochat.checkpoint_manager import find_largest_model as _largest, find_last_step as _last_step
_sft_root = os.path.join(get_base_dir(), "chatsft_checkpoints")
_sft_tag = args.model_tag or _largest(_sft_root)
_sft_step = args.model_step if args.model_step is not None else _last_step(os.path.join(_sft_root, _sft_tag))
with open(os.path.join(_sft_root, _sft_tag, f"meta_{_sft_step:06d}.json"), encoding="utf-8") as _f:
    _sft_meta = _json.load(_f)
SFT_ID = {"tag": _sft_tag, "step": _sft_step, "val_bpb": _sft_meta.get("val_bpb")}
_rl_root = os.path.join(get_base_dir(), "chatrl_checkpoints")
_rl_dir = os.path.join(_rl_root, _sft_tag)
start_step, _resume_meta = 0, None
if args.resume and _glob.glob(os.path.join(_rl_dir, "model_*.pt")):
    _st = _last_step(_rl_dir)
    with open(os.path.join(_rl_dir, f"meta_{_st:06d}.json"), encoding="utf-8") as _f:
        _m = _json.load(_f)
    if _m.get("sft") == SFT_ID:
        _resume_meta, start_step = _m, _st + 1
    elif master_process:      # an RL run of another SFT model: keep it aside, start a new run
        _keep = _rl_root + "_before_" + _time.strftime("%Y%m%d_%H%M%S")
        _shutil.move(_rl_root, _keep)
        print0(f"RL checkpoints of another SFT model moved to {_keep}")
if _resume_meta is not None:
    model, tokenizer, meta = load_model("rl", device, phase="eval", model_tag=_sft_tag, step=start_step - 1)
    print0(f"RL resumes after step {start_step - 1} (SFT {SFT_ID})")
else:
    model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag=_sft_tag, step=_sft_step)
    print0(f"RL starts from the SFT model {SFT_ID}")
# OOA: sample and train with the attention window the SFT model trained with (its "trained_window", 4096), the
# window the model is served with. A checkpoint without it (SFT before the window fix) keeps its config's window.
WINDOW = int(meta.get("trained_window") or _sft_meta.get("trained_window") or model.config.sequence_len)
if model.config.sequence_len != WINDOW:
    model.config.sequence_len = WINDOW
    model.window_sizes = model._compute_window_sizes(model.config)
    if model.rotary_seq_len < WINDOW * 10:
        _hd = model.config.n_embd // model.config.n_head
        model.rotary_seq_len = WINDOW * 10
        model.cos, model.sin = model._precompute_rotary_embeddings(model.rotary_seq_len, _hd, device=model.cos.device)
assert model.window_sizes[-1][0] == WINDOW, (model.window_sizes, WINDOW)
print0(f"RL attention window: {WINDOW} (window per layer {[w for w, _ in model.window_sizes]})")
engine = Engine(model, tokenizer) # for sampling rollouts
'''
TASKS_OLD = ('train_task = GSM8K(subset="main", split="train")\nval_task = GSM8K(subset="main", split="test")\n'
             'num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs\n')
TASKS_NEW = '''train_task = OOARL(args.train_path)
val_task = OOARL(args.val_path)
num_steps = args.num_steps if args.num_steps > 0 else (len(train_task) // args.examples_per_step) * args.num_epochs
'''
BATCH_START = "@torch.no_grad()\ndef get_batch():\n"
BATCH_END = "# -----------------------------------------------------------------------------\n# Simple evaluation loop"
BATCH_NEW = '''RL_STATS = {"answers": 0, "ended": 0, "valid": 0, "solved": 0, "skipped_long": 0, "tokens": 0, "groups": 0,
            "no_signal": 0}
MASTERED = set()                  # requests every answer solved: not sampled again in this run


def ooa_prompt_tokens(conversation):
    """The prompt exactly as served (assistant/model_backend, Colab cell 8): bos, user turn, assistant start. Not
    tokenizer.render_for_completion: it cuts a conversation at 2048 tokens."""
    user = conversation["messages"][0]["content"]
    return ([tokenizer.get_bos_token_id(), tokenizer.encode_special("<|user_start|>")] + tokenizer.encode(user)
            + [tokenizer.encode_special("<|user_end|>"), tokenizer.encode_special("<|assistant_start|>")])


@torch.no_grad()
def get_batch(start=0):
    """Like nanochat's, plus: max new tokens = what the window leaves (requests leaving fewer than --min-new-tokens
    are skipped), sampling in batches of --sample-batch-size, the end token is trained when the answer ended on its
    own (generate_batch drops it), and the reward knows whether the answer ended and how long it was. start: the
    request to begin with (resume)."""
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    rank_indices = list(range(ddp_rank, len(train_task), ddp_world_size))
    pos = start // ddp_world_size
    skips = too_long = 0
    max_skips = args.max_skips or 3 * args.examples_per_step
    while True:
        example_idx = rank_indices[pos % len(rank_indices)]
        pos += 1
        if example_idx in MASTERED:
            if len(MASTERED) >= len(rank_indices):
                print0("every RL request is solved by all its answers: nothing left to learn from this pool")
                MASTERED.clear()
            continue
        conversation = train_task[example_idx]
        tokens = ooa_prompt_tokens(conversation)
        prefix_length = len(tokens)
        max_new = min(args.max_new_tokens, WINDOW - prefix_length)
        if max_new < min(args.min_new_tokens, args.max_new_tokens):
            RL_STATS["skipped_long"] += 1
            too_long += 1
            assert too_long <= len(rank_indices), "every RL request is too long for the window"
            continue
        too_long = 0
        model.eval()
        generated_token_sequences, masks = [], []
        per = args.sample_batch_size or args.num_samples
        sampling_step = 0
        while len(generated_token_sequences) < args.num_samples:
            n = min(per, args.num_samples - len(generated_token_sequences))
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
            seqs, msk = engine.generate_batch(tokens, num_samples=n, max_tokens=max_new,
                                              temperature=args.temperature, top_k=args.top_k, seed=seed)
            generated_token_sequences.extend(seqs)
            masks.extend(msk)
            sampling_step += 1
        rewards = []
        for i, sample_tokens in enumerate(generated_token_sequences):
            generated_tokens = sample_tokens[prefix_length:]
            ended = len(generated_tokens) < max_new
            if ended:                     # learn to stop: the end token joins the trained tokens
                generated_token_sequences[i] = sample_tokens + [assistant_end]
                masks[i] = masks[i] + [1]
            info = train_task.score(conversation, tokenizer.decode(generated_tokens), ended=ended,
                                    tokens=len(generated_tokens))
            rewards.append(info["total"])
            RL_STATS["answers"] += 1
            RL_STATS["ended"] += int(ended)
            RL_STATS["valid"] += int(info.get("valid", False))
            RL_STATS["solved"] += int(info.get("solved", False))
            RL_STATS["tokens"] += len(generated_tokens)
        RL_STATS["groups"] += 1
        if args.skip_same and len(rewards) > 1 and max(rewards) - min(rewards) < 1e-6 and skips < max_skips:
            RL_STATS["no_signal"] += 1           # every answer scored the same: advantage 0, nothing to learn
            if min(rewards) >= 0.99:
                MASTERED.add(example_idx)
            skips += 1
            continue
        skips = 0
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()
        targets[mask_ids[:, 1:] == 0] = -1
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        advantages = rewards - rewards.mean()
        yield generated_token_sequences, inputs, targets, rewards, advantages


@torch.no_grad()
def ooa_eval(max_examples):
    """Mean reward, valid, solved and ended on the held-out RL rows (one answer each, --eval-temperature)."""
    model.eval()
    n = min(max_examples, len(val_task))
    agg = torch.zeros(5, device=device)
    for idx in range(ddp_rank, n, ddp_world_size):
        conversation = val_task[idx]
        tokens = ooa_prompt_tokens(conversation)
        max_new = min(args.max_new_tokens, WINDOW - len(tokens))
        if max_new < min(args.min_new_tokens, args.max_new_tokens):
            continue
        seqs, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=max_new, temperature=args.eval_temperature,
                                        top_k=args.top_k, seed=idx)
        gen = seqs[0][len(tokens):]
        info = val_task.score(conversation, tokenizer.decode(gen), ended=len(gen) < max_new, tokens=len(gen))
        agg += torch.tensor([1.0, info["total"], float(info.get("valid", False)), float(info.get("solved", False)),
                             float(len(gen) < max_new)], device=device)
    if ddp:
        dist.all_reduce(agg, op=dist.ReduceOp.SUM)
    k = max(1.0, agg[0].item())
    return {"examples": int(agg[0].item()), "reward": agg[1].item() / k, "valid": agg[2].item() / k,
            "solved": agg[3].item() / k, "ended": agg[4].item() / k}


'''
OPT_ANCHOR = '''for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]
'''
OPT_NEW = OPT_ANCHOR + '''# OOA: resume the optimizer state saved with the RL checkpoint (also its learning rates)
_opt_path = os.path.join(_rl_dir, f"optim_{start_step - 1:06d}_rank{ddp_rank:d}.pt")
if _resume_meta is not None and os.path.exists(_opt_path):
    optimizer.load_state_dict(torch.load(_opt_path, map_location=device))
    print0(f"optimizer state resumed from {_opt_path}")
elif _resume_meta is not None:
    print0("no optimizer state saved with the RL checkpoint: the optimizer starts fresh")
'''
ITER_OLD = "batch_iterator = get_batch()\nfor step in range(num_steps):\n"
ITER_NEW = '''batch_iterator = get_batch(start_step * args.examples_per_step)
RL_LOG = os.path.join(_rl_dir, "rl_log.jsonl")
_t_step = _time.time()
_last_eval = None
print0(f"RL steps {start_step}..{num_steps - 1} | {len(train_task)} train / {len(val_task)} val requests | "
       f"{args.examples_per_step} requests x {args.num_samples} answers per step | max new tokens {args.max_new_tokens}")
for step in range(start_step, num_steps):
'''
EVAL_START = "    # Evaluate the model once in a while and log to wandb\n    if step % args.eval_every == 0:\n"
EVAL_END = "    # Forward/Backward on rollouts over multiple examples in the dataset\n"
EVAL_NEW = '''    # OOA: evaluate on the held-out RL rows (mean reward, valid, solved, ended) once in a while
    if args.eval_every > 0 and (step % args.eval_every == 0 or step == num_steps - 1):
        _last_eval = ooa_eval(args.eval_examples)
        print0(f"Step {step} | eval on {_last_eval['examples']} held-out requests | reward {_last_eval['reward']:.3f} | "
               f"valid {_last_eval['valid']:.2f} | solved {_last_eval['solved']:.2f} | ended {_last_eval['ended']:.2f}")
        if master_process:
            os.makedirs(_rl_dir, exist_ok=True)
            with open(RL_LOG, "a", encoding="utf-8") as _f:
                _f.write(_json.dumps({"step": step, "eval": _last_eval, "time": _time.strftime("%Y-%m-%d %H:%M:%S")}) + "\\n")
        wandb_run.log({"step": step, **{f"eval_{k}": v for k, v in _last_eval.items()}})

'''
PASS_PRINT = ('            print0(f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: '
              '{loss.item():.6f} | Average reward: {rewards.mean().item()}")\n')
STEP_PRINT = ('    print0(f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: '
              '{mean_sequence_length:.2f}")\n')
STEP_PRINT_NEW = '''    _a = max(1, RL_STATS["answers"])
    print0(f"Step {step}/{num_steps} | reward {mean_reward:.3f} | answers ended {RL_STATS['ended'] / _a:.2f} | "
           f"valid {RL_STATS['valid'] / _a:.2f} | solved {RL_STATS['solved'] / _a:.2f} | answer tokens "
           f"{RL_STATS['tokens'] / _a:.0f} | sequence {mean_sequence_length:.0f} | {_time.time() - _t_step:.0f} s"
           f" | requests tried {RL_STATS['groups']} (all answers equal: {RL_STATS['no_signal']}) | mastered {len(MASTERED)}"
           + (f" | skipped (prompt too long) {RL_STATS['skipped_long']}" if RL_STATS["skipped_long"] else ""))
    _t_step = _time.time()
    for _k in RL_STATS:
        RL_STATS[_k] = 0
'''
SAVE_OLD = '''        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None, # note: we don't bother to save the optimizer state
            {
                "model_config": model_config_kwargs,
            }
        )
'''
SAVE_NEW = '''        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            optimizer.state_dict(), # OOA: saved so an interrupted run continues (only the newest is kept)
            {
                "step": step,
                "model_config": model_config_kwargs,
                "trained_window": WINDOW, # OOA: the window RL trained with (the serving window check reads it)
                "sft": SFT_ID, # OOA: the SFT model this run started from (resume only continues the same one)
                "tokenizer_sha1": _sft_meta.get("tokenizer_sha1"), # OOA: the start model's tokenizer (the pack must carry it)
                "user_config": user_config,
                "rl_reward": mean_reward,
                "eval": _last_eval,
            }
        )
        # keep the newest --keep-checkpoints model files and only the newest optimizer state
        for _pat, _keep_n in (("model_*.pt", args.keep_checkpoints), ("optim_*.pt", 1)):
            _old = sorted(_glob.glob(os.path.join(checkpoint_dir, _pat)))[:-max(1, _keep_n)]
            for _p in _old:
                os.remove(_p)
        with open(os.path.join(checkpoint_dir, "rl_log.jsonl"), "a", encoding="utf-8") as _f:
            _f.write(_json.dumps({"step": step, "saved": True, "reward": mean_reward}) + "\\n")
'''

BACKWARD_OLD = "            loss.backward()\n"
BACKWARD_NEW = ("            if scaler is not None: # OOA: fp16 (the T4) needs loss scaling, as chat_sft does\n"
                "                scaler.scale(loss).backward()\n"
                "            else:\n"
                "                loss.backward()\n")
STEP_OLD = "    optimizer.step()\n    model.zero_grad(set_to_none=True)\n"
STEP_NEW = '''    if scaler is not None:
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
'''
SCALER = '''# OOA: GradScaler for fp16 training (the T4; bf16 / fp32 need none), as in chat_sft.py
from nanochat.common import COMPUTE_DTYPE as _COMPUTE_DTYPE
scaler = torch.amp.GradScaler() if _COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")
'''

ANCHORS = {"import": IMPORT_OLD, "args": ARGS_ANCHOR, "load": LOAD_OLD, "tasks": TASKS_OLD, "batch start": BATCH_START,
           "batch end": BATCH_END, "optimizer": OPT_ANCHOR, "loop": ITER_OLD, "eval start": EVAL_START,
           "eval end": EVAL_END, "pass print": PASS_PRINT, "step print": STEP_PRINT, "save": SAVE_OLD, "backward": BACKWARD_OLD,
           "optimizer step": STEP_OLD}


def missing_anchors(s):
    """Names of the anchors the chat_rl.py text does not have (empty: the patch applies)."""
    return [k for k, a in ANCHORS.items() if s.count(a) != 1]


def patch_chat_rl(s):
    """chat_rl.py text (pinned nanochat 92d63d4) -> the OOA RL script. Raises AssertionError when an anchor is missing
    or not unique (nanochat changed upstream): the notebook must stop rather than run a half-patched RL."""
    s = s.replace("\r\n", "\n")
    miss = missing_anchors(s)
    assert not miss, "chat_rl.py changed upstream, cannot patch: " + ", ".join(miss)
    s = s.replace(IMPORT_OLD, IMPORT_NEW, 1)
    s = s.replace(ARGS_ANCHOR, ARGS_NEW, 1)
    s = s.replace(LOAD_OLD, LOAD_NEW, 1)
    s = s.replace(TASKS_OLD, TASKS_NEW, 1)
    a = s.index(BATCH_START)
    b = s.index(BATCH_END)
    s = s[:a] + BATCH_NEW + s[b:]
    s = s.replace(OPT_ANCHOR, OPT_NEW + SCALER, 1)
    s = s.replace(BACKWARD_OLD, BACKWARD_NEW, 1)
    s = s.replace(STEP_OLD, STEP_NEW, 1)
    s = s.replace(ITER_OLD, ITER_NEW, 1)
    a = s.index(EVAL_START)
    b = s.index(EVAL_END)
    s = s[:a] + EVAL_NEW + s[b:]
    s = s.replace(PASS_PRINT, "", 1)
    s = s.replace(STEP_PRINT, STEP_PRINT_NEW, 1)
    s = s.replace(SAVE_OLD, SAVE_NEW, 1)
    return s


def rl_args(train_path, val_path, steps, samples=8, examples_per_step=8, device_batch=2, sample_batch=0,
            max_new=1536, every=25, eval_examples=40, temperature=1.0, init_lr_frac=0.02, device_type=""):
    """The patched chat_rl's arguments, as a list (paths may hold spaces). --model-tag=d8: the SFT model is always kept
    in chatsft_checkpoints/d8 and the RL result goes to chatrl_checkpoints/d8, whatever the depth. init_lr_frac: RL moves a trained model, so
    it starts lower than nanochat's 0.05 (a 75M model drifts fast: its JSON breaks before the reward rises)."""
    out = [f"--train-path={train_path}", f"--val-path={val_path}", f"--num-steps={steps}", f"--num-samples={samples}",
           f"--examples-per-step={examples_per_step}", f"--device-batch-size={device_batch}",
           f"--sample-batch-size={sample_batch}", f"--max-new-tokens={max_new}", f"--temperature={temperature}",
           f"--eval-every={every}", f"--eval-examples={eval_examples}", f"--save-every={every}",
           f"--init-lr-frac={init_lr_frac}", "--model-tag=d8", "--run=dummy"]
    return out + ([f"--device-type={device_type}"] if device_type else [])


def rl_command(train_path, val_path, steps, extra="", **kw):
    """The chat_rl command line for a shell (run from the nanochat folder); paths without spaces (Colab)."""
    return " ".join(["python -m scripts.chat_rl"] + rl_args(train_path, val_path, steps, **kw)) + (
        " " + extra if extra else "")


def t4_minutes_per_100_steps(examples_per_step=8, samples=8, answer_tokens=800, decode_ms=9.0, prompt_tokens=1000,
                             train_tokens_per_s=27000, eval_every=25, eval_examples=40, no_signal=0.4):
    """Rough T4 time for 100 RL steps (minutes). Sampling dominates: each request is decoded token by token until the
    longest of its answers ends (answer_tokens: that longest answer; the gold answers of the pool have a median of 250
    and a 90th percentile of 750 tokens, joined requests run longer), decode_ms per token for the whole batch (d8,
    batch 8, KV cache, Python-bound on a T4: an estimate, the CPU here measured about 22 ms at batch 3); no_signal: share of
    requests whose answers all score the same and are skipped (measured 0.67 on the v4 model with 3 answers on the hard
    share of the pool, joined / beginner / question requests; fewer with 8 answers:
    each costs its sampling but gives no step); training reads samples x (prompt + answer) tokens per request at the SFT
    speed (12000 steps x 32,768 tokens in about 4 h on a T4: about 27k tokens/s); the evaluation decodes eval_examples
    answers one by one every eval_every steps."""
    tried = examples_per_step / max(0.05, 1.0 - no_signal)
    sample_s = tried * answer_tokens * decode_ms / 1000
    train_s = examples_per_step * samples * (prompt_tokens + answer_tokens) / train_tokens_per_s
    eval_s = eval_examples * answer_tokens * 0.6 * decode_ms / 1000 / max(1, eval_every)
    return round(100 * (sample_s + train_s + eval_s) / 60)
