"""Tests for rl_setup.py: the RL reward on good / bad / question answers, the prompt pool (leak check, shares, joined
requests), the chat_rl.py patch and the Colab / PC notebook wiring.

    python test_rl_setup.py PLATFORM_REPO [NANOCHAT_DIR]
NANOCHAT_DIR (or env NANOCHAT_DIR): the pinned nanochat; the patch is checked against its scripts/chat_rl.py.
Env OOA_RL_MODEL=<folder with nanochat_base/{chatsft_checkpoints,tokenizer} and sft_render/>: also a tiny RL run on
the CPU (2 steps, 2 answers, 48 new tokens; about a minute) that must write a checkpoint, then resume for 1 more step.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = next((a for a in sys.argv[1:] if os.path.isfile(os.path.join(a, "contracts/flow-schema/flow.v1.json"))),
            os.environ.get("OOA_PLATFORM"))
NANOCHAT = next((a for a in sys.argv[1:] if os.path.isfile(os.path.join(a, "scripts/chat_rl.py"))), None) \
    or os.environ.get("NANOCHAT_DIR")
CATALOG = os.path.join(HERE, "../data/catalog/catalog.jsonl")
SCHEMA = os.path.join(REPO, "contracts/flow-schema/flow.v1.json")
import rl_setup as T          # noqa: E402
import ooa_format as F        # noqa: E402
import ooa_reward as R        # noqa: E402
import gen_flows as G         # noqa: E402

CAT = T.configure(CATALOG, SCHEMA)
FLOWS = G.generate(list(CAT.values()), REPO, 120, 5)
_two = [x for x in FLOWS if len([s for s in x[1]["steps"] if not s["action"].startswith("core/")]) >= 2]
REQ, FLOW, _ = _two[0]
REQ2, FLOW2, _ = next(x for x in _two[1:] if R.actions_of(x[1]) != R.actions_of(FLOW))


def gold_for(req, flow):
    g, ans = T.gold_of({"kind": "flow", "request": req, "flow": flow}, CAT)
    return g, ans


def test_good_answer_scores_one():
    g, ans = gold_for(REQ, FLOW)
    r = T.reward(g, ans, ended=True, tokens=g["tokens"])
    assert r["total"] >= 0.99 and r["solved"] and r["valid"], r
    assert r["kind"] == "flow"
    # the answer is compact (no variables list); the score is on the flow expanded like serving does it
    assert '"variables"' not in ans.split("Flow:", 1)[1]
    # a restatement first ("Clear request: ...", beginner rows) changes nothing
    r2 = T.reward(g, "Clear request: " + REQ + "\n" + ans, ended=True, tokens=g["tokens"])
    assert r2["total"] >= 0.99, r2
    # the end marker is ignored
    assert T.reward(g, ans + "<|assistant_end|>", True, g["tokens"])["total"] >= 0.99


def test_bad_answers_score_lower():
    g, ans = gold_for(REQ, FLOW)
    good = T.reward(g, ans, True, g["tokens"])["total"]
    cut = T.reward(g, ans[: len(ans) // 2], ended=False, tokens=900)
    assert cut["total"] < 0.35 and not cut["valid"], cut                  # cut at the cap: JSON broken
    assert T.reward(g, "Sure, I can help with that.", True, 10)["total"] == 0.0     # a chat reply
    assert T.reward(g, "Question: Which folder do you mean?", True, 10)["total"] == 0.0   # asks on a clear request
    # another job's flow: valid, but the gold actions are not there
    _, other = gold_for(REQ2, FLOW2)
    wrong = T.reward(g, other, True, g["tokens"])
    assert wrong["valid"] and not wrong["solved"] and wrong["total"] < good - 0.25, wrong
    # half of the job (the first step only) loses recall, plan and exactness
    half_flow = dict(FLOW, steps=FLOW["steps"][:1])
    half = T.reward(g, F.render_flow_answer(half_flow, CAT, F.ANSWER_COMPACT), True, g["tokens"])
    assert half["total"] < good and not half["solved"], half
    # an old-format answer (flow JSON only, no plan) misses the plan parts
    plain = T.reward(g, json.dumps(FLOW), True, g["tokens"])
    assert plain["solved"] and abs(plain["total"] - (good - 0.20)) < 0.02, plain
    # not ended on its own: loses the ended part
    noend = T.reward(g, ans, ended=False, tokens=g["tokens"])
    assert abs(noend["total"] - (good - 0.10)) < 1e-6, noend


def test_repetition_and_length_cost():
    g, ans = gold_for(REQ, FLOW)
    good = T.reward(g, ans, True, g["tokens"])["total"]
    plan, flow = ans.split("\nFlow:\n")
    obj = json.loads(flow)
    obj["steps"] = obj["steps"] * 6                                   # the same steps again and again
    lines = plan.split("\n")
    rep = "\n".join([lines[0]] + lines[1:] * 6) + "\nFlow:\n" + json.dumps(obj)
    r = T.reward(g, rep, True, g["tokens"] * 6)
    assert r["repetition"] > 0.5 and r["penalty"] > 0.15 and r["total"] < good - 0.3, r
    long_ = T.reward(g, ans, True, 4 * g["tokens"] + 600)              # rambling past 2 x gold + 128 tokens
    assert long_["total"] < good and long_["penalty"] > 0, long_
    assert T.repetition(["a", "b", "c"]) == 0.0 and T.repetition(["a"] * 4) == 0.75
    assert T.lcs([1, 2, 3, 4], [2, 4, 5]) == 2 and T.lcs([], [1]) == 0


def test_question_reward():
    rec = {"kind": "clarify", "request": "do the monthly report thing", "question": "Which report is it?"}
    g, ans = T.gold_of(rec, CAT)
    assert g["kind"] == "question" and ans == "Question: Which report is it?"
    assert T.reward(g, "Question: Which report do you mean, and where does its data come from?", True, 20)["total"] == 1.0
    assert abs(T.reward(g, "Question: Which report?", ended=False, tokens=900)["total"] - 0.8) < 1e-9
    assert abs(T.reward(g, "Question: " + "which one " * 40, True, 120)["total"] - 0.7) < 1e-9   # over 300 chars
    _, flow_ans = gold_for(REQ, FLOW)
    assert T.reward(g, flow_ans, True, 300)["total"] == 0.0           # a flow built on a guess earns nothing
    assert T.reward(g, "Sure!", True, 3)["total"] == 0.0


def test_leak_check_and_pool():
    tests = ["Copy every PDF from the Invoices folder to the Archive folder and email me when done"]
    leak = T.LeakCheck({T.norm("read the sales csv and total the amount column")}, tests)
    assert not leak.ok("Read the sales CSV, and total the Amount column!")             # in the training data
    assert not leak.ok("copy every pdf from the invoices folder to the archive folder and email me when it is done")
    assert leak.ok("Rename the logs in C:\\Temp to .bak")
    assert leak.dropped == {"in_training": 1, "close_to_test": 1}
    # the real test prompts are found (eval/*.jsonl)
    assert len(T.eval_prompts()) >= 100
    recs = [{"kind": "flow", "request": q, "flow": f, "source": "targeted:x"} for q, f, _ in FLOWS[:30]]
    recs += [{"kind": "clarify", "request": f"automate thing number {i}", "question": "Which thing?",
              "source": "clarify:x"} for i in range(10)]
    seen = []
    train, val, rep = T.build_pool(recs, lambda r: (seen.append(r), r + "\n### Context\nActions:")[1], CAT,
                                   T.LeakCheck(set(), []), 20, n_val=5)
    assert len(train) == 20 and len(val) == 5 and rep["questions"] >= 1, rep
    assert all(x["messages"][0]["content"].endswith("Actions:") for x in train)      # the prompt function's turn
    assert all(x["messages"][1]["role"] == "assistant" and x["gold"]["kind"] in ("flow", "question") for x in train)
    assert {x["gold"]["request"] for x in train}.isdisjoint({x["gold"]["request"] for x in val})
    # a prompt that leaves no room in the window is dropped
    tr2, va2, rep2 = T.build_pool(recs[:5], lambda r: r + " word" * 5000, CAT, T.LeakCheck(set(), []), 5, n_val=0)
    assert not tr2 and rep2["too_long"] == 5


def test_joined_requests_pass_the_checker():
    recs = [{"kind": "flow", "request": q, "flow": f, "source": "targeted:x"} for q, f, _ in FLOWS[:40]]
    out = T.joined_records(recs, 10, seed=3)
    assert len(out) >= 5
    for r in out:
        assert not R.problems(json.dumps(r["flow"])), R.problems(json.dumps(r["flow"]))
        assert r["source"].startswith("joined:") and any(w.strip() in r["request"] for w in T.JOIN_WORDS)
        g, ans = T.gold_of(r, CAT)
        assert T.reward(g, ans, True, g["tokens"])["total"] >= 0.99


def test_shares():
    rows = [{"gold": {"kind": "flow", "source": s}} for s in ["targeted:a"] * 50 + ["joined:a"] * 50 + ["long:a"] * 50]
    rows += [{"gold": {"kind": "question", "source": "clarify:a"}} for _ in range(50)]
    keep = T.pick_shares(rows, 100)
    by = {}
    for x in keep:
        by[T.source_group(x["gold"])] = by.get(T.source_group(x["gold"]), 0) + 1
    assert len(keep) == 100 and by["joined"] > by["targeted"] and by["question"] >= 10, by


def test_patch_and_command():
    args = T.rl_args("/a b/rl_train.jsonl", "/a b/rl_val.jsonl", 5, samples=4, device_batch=2, device_type="cpu")
    assert "--train-path=/a b/rl_train.jsonl" in args and "--num-steps=5" in args and "--device-type=cpu" in args
    assert T.rl_command("x", "y", 3).startswith("python -m scripts.chat_rl --train-path=x")
    assert T.t4_minutes_per_100_steps(8, 8) > T.t4_minutes_per_100_steps(4, 8) > 0
    if not NANOCHAT:
        print("  (patch against chat_rl.py skipped: no NANOCHAT_DIR)")
        return
    s = open(os.path.join(NANOCHAT, "scripts/chat_rl.py"), encoding="utf-8").read()
    assert T.missing_anchors(s) == []
    p = T.patch_chat_rl(s)
    compile(p, "chat_rl.py", "exec")
    for need in ("from tasks.ooa_rl import OOARL", "WINDOW = int(meta.get(\"trained_window\")", "def ooa_prompt_tokens",
                 "train_task.score(conversation", "optimizer.state_dict(), # OOA", "\"trained_window\": WINDOW",
                 "for step in range(start_step, num_steps):", "scaler.scale(loss).backward()", "masks[i] + [1]",
                 "max_new = min(args.max_new_tokens, WINDOW - prefix_length)"):
        assert need in p, need
    assert "GSM8K(" not in p and "render_for_completion(conversation)" not in \
        p.split("def ooa_eval")[0].split("def get_batch")[1]
    try:
        T.patch_chat_rl(s.replace("from tasks.gsm8k import GSM8K", "from tasks.gsm8k import Gsm8k"))
        stopped = None
    except AssertionError as e:
        stopped = str(e)
    assert stopped and "import" in stopped, "a changed chat_rl.py must stop the patch"


def test_notebooks():
    nb = json.load(open(os.path.join(HERE, "ooa_nanochat_colab.ipynb"), encoding="utf-8"))
    src = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    cell1 = next(c for c in src if c.startswith("#@title 1. Config"))
    assert "RL_V5 = False" in cell1 and "RL_MAX_NEW = 1536" in cell1
    cell9 = next(c for c in src if c.startswith("#@title 9. RL"))
    assert 'if not globals().get("RL_V5", False):' in cell9 and "_RL.patch_chat_rl" in cell9 and "_RL.prepare" in cell9
    assert "git show {NANOCHAT_COMMIT}:scripts/chat_rl.py" in cell9
    allsrc = "\n".join(src)
    assert "OOAFlowRL" not in allsrc and "max_completion_tokens=1024" not in allsrc     # the old plain RL cell is gone
    files = next(c for c in src if c.startswith("#@title 6. Build"))
    for f in ("rl_setup.py", "ooa_rl_task.py", "gen_beginner.py", "eval/beginner_requests.jsonl"):
        assert f in files, f
    cell8 = next(c for c in src if c.startswith("#@title 8."))
    assert "_load_model(EVAL_SOURCE" in cell8 and "source=EVAL_SOURCE" in cell8
    gen = open(os.path.join(HERE, "make_local_notebook.py"), encoding="utf-8").read()
    assert '"rl_pc.py"' in gen and '"install" if MODE_NOW == "rl"' in gen


def cpu_smoke(model_dir):
    """A tiny RL run on the CPU from a real model: 2 steps, then resume for 1 more."""
    work = tempfile.mkdtemp(prefix="rl_smoke_")
    try:
        base = os.path.join(work, "base")
        os.makedirs(base)
        for d in ("chatsft_checkpoints", "tokenizer"):
            os.symlink(os.path.join(model_dir, "nanochat_base", d), os.path.join(base, d))
        nano = os.path.join(work, "nano")
        shutil.copytree(NANOCHAT, nano, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        p = os.path.join(nano, "scripts/chat_rl.py")
        s = subprocess.run(["git", "-C", NANOCHAT, "show", "HEAD:scripts/chat_rl.py"], capture_output=True,
                           text=True).stdout or open(p, encoding="utf-8").read()
        open(p, "w", encoding="utf-8").write(T.patch_chat_rl(s))
        shutil.copy2(os.path.join(HERE, "ooa_rl_task.py"), os.path.join(nano, "tasks/ooa_rl.py"))
        rd = os.path.join(model_dir, "sft_render")
        recs = [{"kind": "flow", "request": q, "flow": f, "source": "targeted:x"} for q, f, _ in FLOWS[:6]]
        recs.append({"kind": "clarify", "request": "automate my invoices", "question": "Where do they come from?",
                     "source": "clarify:x"})
        prompt = T.router_prompter(os.path.join(rd, "catalog.jsonl"), REPO, os.path.join(rd, "examples.jsonl"))
        T.configure(os.path.join(rd, "catalog.jsonl"), SCHEMA)
        train, val, _ = T.build_pool(recs, prompt, T._CAT, T.LeakCheck(set(), []), 5, n_val=2)
        T.write_rows(os.path.join(work, "rl_train.jsonl"), train)
        T.write_rows(os.path.join(work, "rl_val.jsonl"), val)
        env = dict(os.environ, NANOCHAT_BASE_DIR=base, NANOCHAT_DTYPE="float32", OOA_DIR=HERE,
                   OOA_CATALOG=os.path.join(rd, "catalog.jsonl"), OOA_SCHEMA=SCHEMA)

        def run(steps):
            args = T.rl_args(os.path.join(work, "rl_train.jsonl"), os.path.join(work, "rl_val.jsonl"), steps,
                             samples=2, examples_per_step=1, device_batch=2, max_new=48, every=2, eval_examples=2,
                             device_type="cpu")
            r = subprocess.run([sys.executable, "-m", "scripts.chat_rl"] + args + ["--min-new-tokens=16"], cwd=nano,
                               env=env, capture_output=True, text=True, timeout=1800)
            out = r.stdout + r.stderr
            assert r.returncode == 0, out[-3000:]
            return out
        out = run(2)
        print("\n".join(l for l in out.splitlines() if l.startswith(("RL ", "Step "))))
        d8 = os.path.join(base, "chatrl_checkpoints", "d8")
        assert os.path.exists(os.path.join(d8, "model_000001.pt")) and os.path.exists(os.path.join(d8, "optim_000001_rank0.pt"))
        meta = json.load(open(os.path.join(d8, "meta_000001.json")))
        assert meta["trained_window"] == 4096 and meta["sft"]["step"] and meta["step"] == 1, meta
        assert "RL attention window: 4096" in out and "| reward " in out
        out2 = run(3)
        assert "RL resumes after step 1" in out2 and "optimizer state resumed" in out2, out2[-2000:]
        assert os.path.exists(os.path.join(d8, "model_000002.pt"))
        print("  cpu smoke: 2 RL steps, checkpoint written, resumed for step 3")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print("ok  ", n)
        except Exception as e:                     # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print("FAIL", n, e)
    if os.environ.get("OOA_RL_MODEL") and NANOCHAT:
        try:
            cpu_smoke(os.environ["OOA_RL_MODEL"])
            print("ok   cpu_smoke")
        except Exception as e:                     # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print("FAIL cpu_smoke", e)
    print(f"{len(tests) - failed}/{len(tests)} passed" if not failed else f"{failed} FAILED")
    sys.exit(1 if failed else 0)
