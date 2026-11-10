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

from model_backend import FakeModel, RepetitionGuard  # noqa: E402


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
    assert n is not None and n <= 40 + 320 + 8, n
    assert "36-token" in g.reason


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
    assert m.calls[-2:] == [("make me a sandwich", 0), ("make me a sandwich", 1)], m.calls


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
