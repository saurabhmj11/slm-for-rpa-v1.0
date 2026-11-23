"""Notebook cells for the reinforcement layer (imported by make_colab_notebook.py).
Cell 9 is RL in the CURRENT format (plan-first answers, compact flows, retrieval context: rl_setup.py), switched on by
RL_V5 in cell 1. Cell 10 (self-improvement) still uses plain prompts and stays behind OLD_FORMAT_CELLS (gated). The old
plain-format RL cell is gone: it sent plain prompts and read plain JSON capped at 1024 tokens."""


def build(md, code, gated):
    md("""
## Reinforcement layer: the model learns from the platform's own rules

**RL (cell 9, `RL_V5` in cell 1):** nanochat's `chat_rl.py` (pinned commit, patched by `rl_setup.py`) writes
`RL_SAMPLES` answers per request in the format the model is served with: the same prompt the server builds (request +
retrieval context, `Router.prompt`), a plan and a compact flow that is expanded like the server does it. The platform
checker (`ooa_reward.py`) scores every expanded flow; answers above their group's average are reinforced. No human
labels, no LLM judge. Requests are fresh (new generator seeds, beginner rewordings), never in the SFT data and never
close to a test prompt. Reward per answer (0-1): rules pass 0.25, gold actions in order 0.25, no extra steps 0.10,
plan names the gold actions 0.10, plan covered by the flow 0.10, ended on its own 0.10, exact 0.10; minus up to 0.2
for repetition and 0.1 for rambling; questions earn only on requests that need one (`rl_setup.REWARD_DOC`).

When: after the SFT model (cell 7) has finished and cell 8 has scored it. RL continues that model; cell 9b then points
cell 8 at the RL model (`EVAL_SOURCE = "rl"`) so both scores can be compared. Keep the RL model only if cell 8 scores
it better. Checkpoints: `chatrl_checkpoints` (saved every `RL_EVERY` steps; after a disconnect run cells 1, 2 and 9
again and it continues; a new SFT model starts a new RL run and keeps the old one aside).

**Self-improvement rounds (cell 10)** still use plain prompts: off with `OLD_FORMAT_CELLS = False`.
""")

    code('''
#@title 9. RL with the platform checker as the reward (current format; switch on with RL_V5 in cell 1)
import glob, json, os, subprocess, sys, importlib
if not globals().get("RL_V5", False):
    print("Skipped (RL_V5 = False in cell 1). Switch it on once the SFT model of cell 7 has finished and cell 8 has "
          "scored it: RL continues that model (chatrl_checkpoints) and cell 9b points cell 8 at the RL model.")
else:
    sys.path[:0] = ["/content/ooa", "/content/ooa/serving"]
    import rl_setup as _RL
    importlib.reload(_RL)
    _RD = f"{DRIVE_DIR}/sft_render"           # the data the SFT model trained on: catalog, example index, context budget
    if not os.path.exists(f"{_RD}/examples.jsonl"):
        raise SystemExit("cell 9 needs the plan-first data (sft_render/examples.jsonl, cell 6d) and its SFT model (cell 7)")
    _SCHEMA = "/content/platform/contracts/flow-schema/flow.v1.json"
    os.environ.update(OOA_CATALOG=f"{_RD}/catalog.jsonl", OOA_SCHEMA=_SCHEMA, OOA_DIR="/content/ooa")
    _POOL = f"{DRIVE_DIR}/rl"
    _sig = {"examples_bytes": os.path.getsize(f"{_RD}/examples.jsonl"), "prompts": RL_PROMPTS, "seed": RL_SEED}
    _old = json.load(open(f"{_POOL}/rl_pool_report.json")) if os.path.exists(f"{_POOL}/rl_pool_report.json") else {}
    if REBUILD_RL_DATA or _old.get("sig") != _sig:
        # fresh requests with a gold answer, never in the SFT data (sft_render rows + example index), never close to a
        # test prompt (eval/*.jsonl and the generated eval sets of cell 6); prompts built by the server's Router.prompt
        _train_files = [f for f in glob.glob(f"{_RD}/*.jsonl") if not f.endswith(("/catalog.jsonl", "/examples.jsonl"))]
        _eval_files = [f"{DRIVE_DIR}/sft/{n}" for n in ("eval.jsonl", "eval_synth.jsonl", "eval_pattern.jsonl")]
        _rep = _RL.prepare(_POOL, f"{_RD}/catalog.jsonl", _SCHEMA, "/content/platform", f"{_RD}/examples.jsonl",
                           RL_PROMPTS, train_files=_train_files, eval_files=_eval_files, seed=RL_SEED,
                           eval_dir="/content/ooa/eval")
        _rep["sig"] = _sig
        json.dump(_rep, open(f"{_POOL}/rl_pool_report.json", "w"), indent=1)
    else:
        _rep = _old
    print("RL requests:", _rep["train"], "train /", _rep["val"], "held out | by source:", _rep["sources"],
          "| dropped as in training / close to a test prompt:", _rep.get("leak_detail"))
    # chat_rl.py straight from the pinned nanochat, then the patch (it stops when nanochat changed upstream)
    !cd /content/nanochat && git cat-file -e {NANOCHAT_COMMIT} && git show {NANOCHAT_COMMIT}:scripts/chat_rl.py > scripts/chat_rl.py
    _p = "/content/nanochat/scripts/chat_rl.py"
    _s = open(_p).read()
    _miss = _RL.missing_anchors(_s.replace("\\r\\n", "\\n"))
    if _miss:
        raise SystemExit(f"chat_rl.py does not match the pinned nanochat (missing: {_miss}): send this line")
    open(_p, "w").write(_RL.patch_chat_rl(_s))
    !cp /content/ooa/ooa_rl_task.py /content/nanochat/tasks/ooa_rl.py
    _cmd = _RL.rl_command(f"{_POOL}/rl_train.jsonl", f"{_POOL}/rl_val.jsonl", RL_STEPS, samples=RL_SAMPLES,
                          examples_per_step=RL_EXAMPLES_PER_STEP, device_batch=RL_DEVICE_BATCH, max_new=RL_MAX_NEW,
                          every=RL_EVERY, eval_examples=_rep["val"], init_lr_frac=RL_LR_FRAC)
    print(f"RL: {RL_STEPS} steps x {RL_EXAMPLES_PER_STEP} requests x {RL_SAMPLES} answers, window 4096, at most "
          f"{RL_MAX_NEW} new tokens | estimated {_RL.t4_minutes_per_100_steps(RL_EXAMPLES_PER_STEP, RL_SAMPLES)} min per "
          f"100 steps on a T4 | saved every {RL_EVERY} steps (resumes after a disconnect)")
    print(_cmd)
    !cd /content/nanochat && {_cmd}
    _log = sorted(glob.glob(os.path.join(os.environ["NANOCHAT_BASE_DIR"], "chatrl_checkpoints", "*", "rl_log.jsonl")))
    if _log:
        print("\\nHeld-out RL requests per evaluation (reward 0-1 | passes every rule | exact | ended on its own):")
        for _l in open(_log[-1]):
            _x = json.loads(_l)
            if "eval" in _x:
                _e = _x["eval"]
                print(f"  step {_x['step']:>4}: reward {_e['reward']:.3f} | valid {_e['valid']:.2f} | solved {_e['solved']:.2f} | ended {_e['ended']:.2f}")
''')

    code('''
#@title 9b. Point the test at the RL model, then re-run cell 8
if globals().get("RL_V5", False) and glob.glob(os.path.join(os.environ["NANOCHAT_BASE_DIR"], "chatrl_checkpoints", "*", "model_*.pt")):
    EVAL_SOURCE = "rl"     # cell 8 prints "evaluating model source: rl". Set back to "sft" to test the SFT model.
else:
    EVAL_SOURCE = "sft"
print("cell 8 will test:", EVAL_SOURCE)
''')

    gated('''
#@title 10. Self-improvement rounds: attempt fresh requests, keep verified wins, learn from failures, retrain
import json, os, shutil
os.environ["OOA_CATALOG"] = f"{DRIVE_DIR}/sft/catalog.jsonl"
os.environ["OOA_SCHEMA"] = "/content/platform/contracts/flow-schema/flow.v1.json"
!cp /content/ooa/ooa_self_improve.py /content/nanochat/scripts/ooa_self_improve.py
# guards: cell 10 reuses the data and the patched chat_sft.py from cells 6-7. Stop if they do not match the config.
import re as _re
_n = sum(1 for _ in open(f"{DRIVE_DIR}/sft/synth_train.jsonl"))
_patched = open("/content/nanochat/scripts/chat_sft.py").read()
_ep = _re.search(r"\\) \\* (\\d+)\\n", _patched)
print(f"synth_train.jsonl rows: {_n} (config SYNTH_TRAIN={SYNTH_TRAIN}) | patched SFT epochs: {_ep.group(1) if _ep else '?'} (config {SFT_EPOCHS})")
if _n < 0.9 * SYNTH_TRAIN or (_ep and int(_ep.group(1)) != SFT_EPOCHS):
    raise RuntimeError("Data or SFT patch is out of date for the current config. Run cell 1, cell 6 (data), cell 7 (SFT), then run this cell again.")
sft_cmd = (f"python -m scripts.chat_sft --max-seq-len={SFT_SEQ} --device-batch-size={SFT_DEVICE_BATCH} "
           f"--total-batch-size={SFT_SEQ*SFT_DEVICE_BATCH*4} --num-iterations={SFT_ITERS} "
           f"--eval-every=100 --eval-tokens={SFT_SEQ*SFT_DEVICE_BATCH*8} --chatcore-every=-1 --run=dummy")
for r in range(SELF_IMPROVE_ROUNDS):
    print(f"\\n===== self-improvement round {r + 1}/{SELF_IMPROVE_ROUNDS} =====")
    !cd /content/nanochat && python -m scripts.ooa_self_improve --source sft --out {DRIVE_DIR}/sft --seed {100 + r} --num-prompts {SI_PROMPTS} --samples {SI_SAMPLES}
    _ck = os.path.join(os.environ["NANOCHAT_BASE_DIR"], "chatsft_checkpoints")
    if os.path.exists(_ck):                      # keep the previous model (never deleted), like cell 7
        import time as _t
        shutil.move(_ck, _ck + "_before_" + _t.strftime("%Y%m%d_%H%M%S"))
    !cd /content/nanochat && {sft_cmd}
print("\\nPass rate on FRESH requests, per round (this is the progress measure):")
for l in open(f"{DRIVE_DIR}/sft/selfimprove_log.jsonl"):
    x = json.loads(l)
    print(x["time"], "| pass@k", x["fresh_pass_at_k"], "| mean reward", x["mean_reward"], "| new wins", x["added_self_improved"], "| new fixes", x["added_hard_fixed"], "| still hard", x["hard_queue"])
    print("   where answers lose points (1.0 = always fine):", x.get("parts"))
''')
