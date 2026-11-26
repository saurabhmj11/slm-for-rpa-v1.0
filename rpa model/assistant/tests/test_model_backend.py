"""Tests for model_backend.py.  Run: python tests/test_model_backend.py

Always runs: RepetitionGuard unit tests and FakeModel tests (no torch needed). With OOA_CATALOG, OOA_SCHEMA and
OOA_PLATFORM set, the fake flows are also checked by the verifier and run through serving/ooa_server.Router.
Real model (needs torch + nanochat + a checkpoint): set OOA_MODEL_DIR and OOA_NANOCHAT_DIR. Then a quick check runs
2 looping prompts + 1 good prompt. Set OOA_LONG_CHECK=1 to run the full check: the 20 plain requests (which loop today)
and 5 eval_synth prompts (OOA_SYNTH=path to eval_synth.jsonl) with and without the guard; outputs of the good prompts
must be identical and the time saved per looping request is printed. The real model runs one prompt at a time.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # rpa model/assistant
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "training"))
sys.path.insert(0, str(ROOT.parent / "serving"))

from model_backend import FakeModel, RepetitionGuard, AnswerWatch, Guards  # noqa: E402


class Skip(Exception):
    """Raised by a test that cannot run here; the runner prints SKIP, not PASS."""


def feed(guard, ids):
    for i, t in enumerate(ids):
        if guard.push(t):
            return i + 1
    return None


def test_guard_stops_short_period_loop():
    ids = list(range(100, 160)) + [7, 8, 9] * 200
    n = feed(RepetitionGuard(), ids)
    assert n is not None and n <= 60 + 48 + 8, n


def test_guard_stops_long_period_loop():
    cycle = list(range(1000, 1036))           # a 36-token cycle, like a whole step repeated verbatim
    g = RepetitionGuard()
    n = feed(g, list(range(10, 50)) + cycle * 30)
    assert n is not None and n <= 40 + 3 * 72 + 8, n         # caught as 3 copies of 2 cycles (72 tokens)
    assert "72-token" in g.reason or "36-token" in g.reason, g.reason


def test_guard_spares_varied_repetition():
    # flow-like structure repeated with a different name each time must not trip the guard
    ids = []
    for k in range(120):
        ids += [1, 2, 3, 500 + k, 4, 5, 6, 7, 8, 9, 10, 11]   # {"name":"<k>","type":"String","direction":"local"},
    assert feed(RepetitionGuard(), ids) is None


def test_guard_spares_repeated_clause_in_description():
    # the SFT data has descriptions like "read and parse a JSON file, then " x 9 (11 tokens) and a 25-token clause x 9
    assert feed(RepetitionGuard(), list(range(10, 60)) + list(range(200, 211)) * 10 + [1, 2, 3]) is None
    assert feed(RepetitionGuard(), list(range(10, 60)) + list(range(300, 325)) * 9 + [1, 2, 3]) is None


def test_guard_stops_whole_step_loop():
    # a whole step (59 tokens) written again and again: the n <= 40 rules never see it, the long-period rule does
    step = list(range(2000, 2059))
    g = RepetitionGuard()
    n = feed(g, list(range(10, 400)) + step * 10)
    assert n is not None and n <= 390 + 4 * 59 + 8 and "59-token" in g.reason, (n, g.reason)
    # 2 copies of a long stretch (a description written twice, a step pair) is normal: never stopped
    assert feed(RepetitionGuard(), list(range(10, 100)) + step * 2 + list(range(3000, 3100))) is None
    # the SFT data's worst case for long periods: a 46-token sentence 4 times (185 tokens)
    assert feed(RepetitionGuard(), list(range(10, 60)) + list(range(4000, 4046)) * 4 + [1, 2, 3]) is None


def _watch_feed(watch, text, pieces=3):
    """Feed text to an AnswerWatch a few characters at a time (each piece stands for one token)."""
    for i in range(0, len(text), pieces):
        if watch.push(text[i:i + pieces]):
            return i + pieces
    return None


PLAN3 = "Plan:\n1. List the files (ooa.file/list)\n2. For each file (core/loop)\n  3. Move it (ooa.file/move)\nFlow:\n"


def _step(i, sid=None):
    return ('{"id":"%s","action":"ooa.file/move","description":"Move it","args":{"source":"$f$"}},'
            % (sid or "s%d" % i))


def test_answer_watch_flow_outgrows_plan():
    w = AnswerWatch(lambda piece: piece)
    good = PLAN3 + '{"kind":"TaskBot","steps":[' + "".join(_step(i) for i in range(1, 4)) + "]}"
    assert _watch_feed(w, good) is None and w.plan_lines == 3
    w = AnswerWatch(lambda piece: piece)
    bad = PLAN3 + '{"kind":"TaskBot","steps":[' + "".join(_step(i) for i in range(1, 30))
    n = _watch_feed(w, bad)
    assert n is not None and w.kind == "loop" and "steps for a plan of 3 lines" in w.reason, w.reason
    assert n < len(bad) // 2


def test_answer_watch_repeated_step_id_and_plan_line():
    w = AnswerWatch(lambda piece: piece)
    plan = "Plan:\n" + "".join(f"{i}. Step {i} (ooa.file/move)\n" for i in range(1, 13)) + "Flow:\n"
    text = plan + '{"steps":[' + _step(1) + _step(2) + "".join(_step(0, "summary") for _ in range(4))
    assert _watch_feed(w, text) is not None and "'summary' is written 3 times" in w.reason, w.reason
    # the loop seen on the real model: {"id":...} fragments that are not whole steps (no "action" after the id)
    w = AnswerWatch(lambda piece: piece)
    frag = '{"id":"summary_folder$","folder":"$invoice_folder$","name":"POwner"},"out":{"folder":"noticefile"}},'
    assert _watch_feed(w, plan + '{"id":"f","steps":[' + _step(1) + frag * 4) is not None and "written 3" in w.reason
    w = AnswerWatch(lambda piece: piece)
    looping_plan = "Plan:\n" + "".join(f"{i}. Send the alert to finance (ooa.email/send)\n" for i in range(1, 30))
    assert _watch_feed(w, looping_plan) is not None and "repeats the line" in w.reason
    # up to 10 identical lines happen in real plans ("Set text" for 10 fields): not a loop
    w = AnswerWatch(lambda piece: piece)
    assert _watch_feed(w, "Plan:\n" + "".join(f"{i}. Set text (ooa.browser/type_text)\n" for i in range(1, 11))
                       + "Flow:\n{") is None
    # a JSON-only answer or a question is not watched for plan rules
    assert _watch_feed(AnswerWatch(lambda p: p), "Question: which folder? " * 30) is None


def test_answer_watch_reads_a_restated_request():
    """An answer that opens with "Clear request: ..." (a beginner's request restated) is watched like a plan answer;
    on_plan gets the plan lines only."""
    seen = []
    clear = "Clear request: List the files, read each text file and log it.\n"
    w = AnswerWatch(lambda piece: piece, on_plan=lambda plan: seen.append(plan) or len(plan) > 2)
    n = _watch_feed(w, clear + PLAN3 + '{"kind":"TaskBot","steps":[' + _step(1), pieces=1)
    assert n == len(clear + PLAN3) - 1 and w.kind == "plan" and seen[0][0].startswith("1. List"), (n, seen)
    w = AnswerWatch(lambda piece: piece)
    bad = clear + PLAN3 + '{"kind":"TaskBot","steps":[' + "".join(_step(i) for i in range(1, 30))
    assert _watch_feed(w, bad) is not None and "steps for a plan of 3 lines" in w.reason, w.reason

THINK3 = ("Understand: Move every file of the folder.\nFacts: folder = C:\\In\nMissing: none\nSteps:\n"
          "1. List the files -> ooa.file/list\n2. For each file -> core/loop\n  3. Move it -> ooa.file/move\n"
          "Check: 3 steps, 1 loop, all facts used\nFlow:\n")


def test_answer_watch_reads_a_thinking_answer():
    """A thinking answer (Understand / Facts / Missing / Steps / Check): on_plan gets the Steps lines only, the flow is
    checked against them, the block's header lines never count as a looping plan."""
    seen = []
    w = AnswerWatch(lambda piece: piece, on_plan=lambda plan: seen.append(plan) or len(plan) > 2)
    n = _watch_feed(w, THINK3 + '{"steps":[' + _step(1), pieces=1)
    assert n == len(THINK3) - 1 and w.kind == "plan" and len(seen[0]) == 3, (n, seen)
    assert seen[0][0] == "1. List the files -> ooa.file/list"
    w = AnswerWatch(lambda piece: piece)
    assert _watch_feed(w, THINK3 + '{"steps":[' + "".join(_step(i) for i in range(1, 4)) + "]}") is None
    w = AnswerWatch(lambda piece: piece)
    bad = THINK3 + '{"steps":[' + "".join(_step(i) for i in range(1, 30))
    assert _watch_feed(w, bad) is not None and "steps for a plan of 3 lines" in w.reason, w.reason
    # 40 plan lines plus the five header lines is not "past 40 lines"
    long = THINK3.split("Steps:")[0] + "Steps:\n" + "".join(f"{i}. Step {i} -> ooa.file/move\n" for i in range(1, 41))
    assert _watch_feed(AnswerWatch(lambda p: p), long + "Check: 40 steps\nFlow:\n{") is None


def test_answer_watch_on_plan_stops_after_the_plan():
    seen = []
    w = AnswerWatch(lambda piece: piece, on_plan=lambda plan: seen.append(plan) or len(plan) > 2)
    n = _watch_feed(w, PLAN3 + '{"kind":"TaskBot","steps":[' + _step(1), pieces=1)
    assert n == len(PLAN3) - 1 and w.kind == "plan" and len(seen) == 1 and seen[0][0].startswith("1. List")
    w = AnswerWatch(lambda piece: piece, on_plan=lambda plan: len(plan) > 5)
    assert _watch_feed(w, PLAN3 + '{"kind":"TaskBot","steps":[' + _step(1) + "]}") is None
    g = Guards(RepetitionGuard(), AnswerWatch(lambda piece: piece, on_plan=lambda plan: True))
    assert _watch_feed(g, PLAN3 + "{", pieces=1) and g.kind == "plan"


def test_answer_watch_with_given_plan():
    """A part's answer starts with its plan (prefix, given): the watch knows the plan before the first token."""
    w = AnswerWatch(lambda piece: piece, prefix=PLAN3)
    bad = '{"kind":"TaskBot","steps":[' + "".join(_step(i) for i in range(1, 30))
    assert _watch_feed(w, bad) is not None and "for a plan of 3 lines" in w.reason
    w = AnswerWatch(lambda piece: piece, prefix=PLAN3)
    assert _watch_feed(w, '{"kind":"TaskBot","steps":[' + _step(1) + _step(2) + _step(3) + "]}") is None
    m = FakeModel(style="plan", catalog={})
    req = "Read a CSV file into a table, then write the table to an Excel sheet"
    out = m.generate(req, prefix=PLAN3)
    assert out.startswith(PLAN3) and out[len(PLAN3):].startswith("{")


def test_fake_model_on_plan():
    F = _fmt()
    m = FakeModel(style="plan", catalog={})
    req = "Read a CSV file into a table, then write the table to an Excel sheet"
    full = m.generate(req)
    assert F.parse_answer(full)["flow_text"] and m.last_stats["stop"] == "end"
    cut = m.generate(req, on_plan=lambda plan: len(plan) > 2)
    a = F.parse_answer(cut)
    assert a["flow_text"] in (None, "") and len(a["plan"]) == 4 and m.last_stats["stop"].startswith("plan")
    assert m.generate(req, on_plan=lambda plan: False) == full


def _fmt():
    import ooa_format
    return ooa_format


def test_guard_window_rule_optional():
    ids = list(range(100, 130)) + list(range(5000, 5100)) + list(range(100, 130)) + list(range(6000, 6100)) + list(range(100, 130))
    assert feed(RepetitionGuard(), ids) is None                 # off by default
    g = RepetitionGuard(window=30, max_seen=3)
    assert feed(g, ids) == len(ids) and "30-token" in g.reason


def test_guard_reason_and_reset():
    g = RepetitionGuard()
    assert feed(g, [5] * 100) is not None and "repeated" in g.reason
    g.reset()
    assert g.reason is None and feed(g, list(range(300))) is None


def test_fake_model():
    m = FakeModel()
    flow = json.loads(m.generate("Read a CSV file into a table, then write the table to an Excel sheet", 0))
    assert [s["action"] for s in flow["steps"]][0] == "ooa.csv/read"
    assert not m.generate("make me a sandwich").startswith("{\"")
    assert "ooa.csv/read" in m.ask("Which action reads a CSV?")
    assert set(m.info) == {"device", "params_M", "step", "source"}



def test_fake_model_formats():
    req = "Read a CSV file into a table, then write the table to an Excel sheet"
    m = FakeModel(style="plan")
    text = m.generate(req + "\n### Context\nActions:\n- ooa.csv/read: Read to table | in: path*")   # request + context
    assert text.startswith("Plan:\n1. ") and "(ooa.csv/read)" in text and "\nFlow:\n{" in text
    assert m.generate("automate my work").startswith("Question: ")
    assert FakeModel().generate("automate my work").startswith("garbage")              # the JSON style never asks
    s = FakeModel(script=["a", "b"])
    assert [s.generate("x"), s.generate("y"), s.generate("z")] == ["a", "b", "b"]

def test_fake_flows_pass_verifier():
    cat = os.environ.get("OOA_CATALOG")
    schema = os.environ.get("OOA_SCHEMA")
    if not (cat and schema and Path(cat).exists() and Path(schema).exists()):
        raise Skip("set OOA_CATALOG and OOA_SCHEMA")
    import ooa_reward
    ooa_reward.configure(cat, schema)
    m = FakeModel()
    for prompt in FakeModel.FLOWS:
        assert ooa_reward.problems(m.generate(prompt)) == [], prompt


def test_fake_model_with_router():
    cat, platform = os.environ.get("OOA_CATALOG"), os.environ.get("OOA_PLATFORM")
    if not (cat and platform and Path(cat).exists() and Path(platform).exists()):
        raise Skip("set OOA_CATALOG and OOA_PLATFORM")
    from ooa_server import Router, Policy
    m = FakeModel()
    router = Router(cat, platform, Policy(allow_escalation=False, local_retries=1), local=m)
    ok = router.run("List the files in a folder")
    assert ok["verified"] and ok["source"] == "local", ok
    bad = router.run("make me a sandwich")
    assert not bad["verified"] and bad["problems"], bad
    # a text answer (not a flow) is not retried: the request was read as chat, a retry only repeats it
    assert m.calls[-1] == ("make me a sandwich", 0) and len(m.calls) == 2, m.calls


def test_fake_model_batch():
    """FakeModel.generate_batch answers each item as generate would and records the batch sizes."""
    m = FakeModel(style="plan")
    out = m.generate_batch([{"prompt": "list the files in a folder"}, {"prompt": "nonsense"},
                            {"prompt": "list the files in a folder", "prefix": "Plan:\n1. x\nFlow:\n"}])
    assert len(out) == 3 and out[0].startswith("Plan:") and out[1].startswith("garbage")
    assert out[2].startswith("Plan:\n1. x\nFlow:\n{") and m.batches == [3] and len(m.last_batch_stats) == 3


def test_batched_decode_matches_one_prompt():
    """constrained.generate_batch (several prompts, per-row positions, one forward per step) gives exactly what the
    one-prompt decoder gives, greedy and sampled with the same seed, on a tiny random nanochat model with a sliding
    window shorter than the prompts (needs torch and OOA_NANOCHAT_DIR; no checkpoint)."""
    nd = os.environ.get("OOA_NANOCHAT_DIR")
    try:
        import torch
    except ImportError:
        raise Skip("needs torch")
    if not nd:
        raise Skip("set OOA_NANOCHAT_DIR")
    os.environ.setdefault("NANOCHAT_DTYPE", "float32")
    sys.path.insert(0, str(Path(nd).resolve()))
    from nanochat.gpt import GPT, GPTConfig
    import constrained as C
    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=256, vocab_size=96, n_layer=4, n_head=2, n_kv_head=2, n_embd=64, window_pattern="SL")
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    for prm in model.parameters():                    # init_weights zeroes some projections: make every path count
        prm.data.add_(0.02 * torch.randn_like(prm))
    model.eval()
    assert model.window_sizes[0][0] < 140             # the short window is shorter than the prompts
    prompts = [[1 + (i * 7) % 90 for i in range(150)], [2 + (i * 5) % 90 for i in range(170)], [3, 4, 5, 6] * 40]
    for temp, seed in ((0.0, 1), (0.8, 5)):
        one = [C.generate(model, p, None, 24, temp, 50, seed + k, set(), lambda t: "x")[0]
               for k, p in enumerate(prompts)]
        rows = [C.BatchRow(p, 24 - 6 * k, temp, seed + k, top_k=50) for k, p in enumerate(prompts)]
        C.generate_batch(model, rows, set(), lambda t: "x")
        for k, r in enumerate(rows):
            assert r.out == one[k][:24 - 6 * k], (temp, k, r.out, one[k])
            assert r.reason == "max_tokens"


# ---------------------------------------------------------------- real model
def _real():
    md, nd = os.environ.get("OOA_MODEL_DIR"), os.environ.get("OOA_NANOCHAT_DIR")
    if not (md and nd):
        return None
    if not hasattr(_real, "m"):
        from model_backend import NanochatLocal
        _real.m = NanochatLocal(md, nd)
    return _real.m


def _plain():
    p = ROOT.parent / "eval" / "plain_requests.jsonl"
    return [json.loads(l)["prompt"] for l in open(p, encoding="utf-8")]


def _synth(k=5):
    p = os.environ.get("OOA_SYNTH")
    if not p or not Path(p).exists():
        return []
    return [json.loads(l)["prompt"] for l in open(p, encoding="utf-8")][:k]


def _pair(m, prompt):
    """Run once with the guard off and once on; return (text_off, stats_off, text_on, stats_on)."""
    off = m._run(prompt, 2500, 0.2, 1000, guard=False)
    s_off = dict(m.last_stats)
    on = m.generate(prompt)
    s_on = dict(m.last_stats)
    return off, s_off, on, s_on


def test_real_model_info_and_ask():
    m = _real()
    if m is None:
        raise Skip("set OOA_MODEL_DIR and OOA_NANOCHAT_DIR")
    assert m.info["params_M"] > 1 and m.info["device"] and m.info["source"] == "sft", m.info
    t0 = time.time()
    a = m.ask("Which OOA action would I use to read a CSV file into a table?")
    assert isinstance(a, str) and a and m.last_stats["tokens"] <= 300
    print(f"  info={m.info} ask -> {a[:100]!r} ({time.time() - t0:.1f}s)")


def test_real_model_guard():
    m = _real()
    if m is None:
        raise Skip("set OOA_MODEL_DIR and OOA_NANOCHAT_DIR")
    long_check = os.environ.get("OOA_LONG_CHECK") == "1"
    plain = _plain()
    loops = plain if long_check else [plain[0], plain[13]]     # two that loop early today (fast to check)
    good = _synth(5 if long_check else 1)
    saved = []
    for p in loops:
        off, s_off, on, s_on = _pair(m, p)
        assert off.startswith(on), "guarded output must be a prefix of the unguarded one"
        if s_off["stop"] == "end":                    # ended on its own: the guard must not have touched it
            assert off == on and s_on["stop"] == "end", (s_on["stop"], p[:60])
        else:
            assert s_on["stop"].startswith("loop"), s_on["stop"]
            saved.append(s_off["seconds"] - s_on["seconds"])
        print(f"  {p[:50]!r}: off {s_off['tokens']} tok {s_off['seconds']}s ({s_off['stop']}) | "
              f"on {s_on['tokens']} tok {s_on['seconds']}s ({s_on['stop']})")
    for p in good:
        off, s_off, on, s_on = _pair(m, p)
        assert off == on and s_on["stop"] == "end", (s_on["stop"], p[:60])
        print(f"  good {p[:50]!r}: identical, {s_on['tokens']} tok, stop={s_on['stop']}")
    if saved:
        print(f"  time saved per looping request: mean {sum(saved) / len(saved):.1f}s over {len(saved)} "
              f"(each: {', '.join(f'{x:.1f}' for x in saved)})")


def test_window_check_trained_vs_served():
    from model_backend import window_check, base_window
    import tempfile
    mc = {"model_config": {"sequence_len": 4096}}
    assert window_check({**mc, "trained_window": 4096}) == (4096, None)               # trained with the fix
    tw, w = window_check({**mc, "trained_window": 1024})
    assert tw == 1024 and "mismatch" in w
    tw, w = window_check(mc, base_seq=1024)                                           # before the fix, base known
    assert tw == 1024 and "mismatch" in w and "1024" in w
    assert window_check(mc, base_seq=4096) == (4096, None)
    tw, w = window_check(mc)                                                          # before the fix, base unknown
    assert tw is None and "unknown" in w
    assert window_check({"model_config": {"sequence_len": 2048}}) == (None, None)
    with tempfile.TemporaryDirectory() as d:
        assert base_window(d) is None
        for step, n in ((100, 512), (300, 1024)):
            p = Path(d) / "base_checkpoints" / "d8"
            p.mkdir(parents=True, exist_ok=True)
            (p / f"meta_{step:06d}.json").write_text(json.dumps({"model_config": {"sequence_len": n}}))
        assert base_window(d) == 1024


if __name__ == "__main__":
    fails = skipped = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Skip as ex:
                skipped += 1
                print("SKIP", name, "-", ex)
            except Exception as ex:  # noqa: BLE001
                fails += 1
                print("FAIL", name, type(ex).__name__, ex)
    print((f"ALL PASS ({skipped} skipped)" if skipped else "ALL PASS") if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
