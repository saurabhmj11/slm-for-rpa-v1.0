"""Tests for sft_setup.py (the SFT data mixture and the chat_sft.py patch) and ooa_task.OOADialogues views.
Standard library only. Run: python test_sft_setup.py [NANOCHAT_DIR]   (the patch is checked against the pinned
nanochat scripts/chat_sft.py when NANOCHAT_DIR or env NANOCHAT_DIR is given; skipped otherwise)."""
import json, os, sys, tempfile, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sft_setup as T

NANOCHAT = next((a for a in sys.argv[1:] if os.path.isfile(os.path.join(a, "scripts/chat_sft.py"))), None) \
    or os.environ.get("NANOCHAT_DIR")

# rows and tokens per row like the real Colab data (render_report of the last run, compact answers about 25% shorter)
STATS = {"synth_train.jsonl": (28381, 1100), "synth_pattern.jsonl": (7533, 1700), "natural_train.jsonl": (5484, 1100),
         "long_train.jsonl": (3000, 1700), "business2_train.jsonl": (1440, 1750), "fix_train.jsonl": (6000, 1500),
         "targeted_train.jsonl": (4000, 900), "combos_train.jsonl": (2048, 1100), "tools_train.jsonl": (1497, 1000),
         "chains_train.jsonl": (832, 1050), "core_train.jsonl": (1015, 900), "scenarios_train.jsonl": (474, 1450),
         "clarify_train.jsonl": (900, 700), "notask_train.jsonl": (359, 250), "train_catalog.jsonl": (2550, 60),
         "train_flows.jsonl": (93, 1700), "self_improved.jsonl": (628, 1100), "hard_fixed.jsonl": (86, 1150)}


def _tokens(views):
    return sum(views[n] * STATS[n][0] * STATS[n][1] for n in views)


def test_plan_fills_the_budget_with_caps():
    b = T.budget(8000, 4096 * 2 * 4)
    v = T.plan_mixture(STATS, b)
    assert abs(_tokens(v) - b) / b < 0.01, (_tokens(v), b)
    for n in T.LONG_FILES:
        cap = T.PROCESS_CAP if n in T.PROCESS_FILES else T.LONG_CAP
        assert v[n] <= cap + 1e-9, (n, v[n])
    for n in T.PROCESS_FILES:                                 # the failing business processes get more practice
        assert v[n] > T.LONG_CAP, (n, v[n])
    assert v["train_catalog.jsonl"] >= T.CATALOG_MIN
    assert min(v.values()) >= 1.0, v                         # nothing is left mostly unseen any more
    assert v["synth_train.jsonl"] < v["targeted_train.jsonl"]
    small = T.plan_mixture(STATS, b // 10)                   # a small budget: the floor and the scale still hold
    assert small["train_catalog.jsonl"] >= T.CATALOG_MIN and small["synth_train.jsonl"] < 1
    big = T.plan_mixture(STATS, b * 4)                      # more budget than the caps allow: never runs out
    assert abs(_tokens(big) - b * 4) / (b * 4) < 0.01 and big["long_train.jsonl"] <= T.PROCESS_CAP
    lines = T.mixture_table(STATS, v, b)
    assert "views" in lines[0] and "fits" in lines[-1]


def test_file_stats_and_plan_for_notebook():
    with tempfile.TemporaryDirectory() as d:
        for name, n in (("long_train.jsonl", 10), ("train_catalog.jsonl", 20), ("val.jsonl", 5), ("catalog.jsonl", 9),
                        ("train.jsonl", 30), ("selfimprove_log.jsonl", 2), ("rl_train.jsonl", 4)):
            with open(os.path.join(d, name), "w") as f:
                for i in range(n):
                    f.write(json.dumps({"messages": [{"role": "user", "content": "x" * 280},
                                                     {"role": "assistant", "content": "y"}]}) + "\n")
        st = T.file_stats(d)
        assert set(st) == {"long_train.jsonl", "train_catalog.jsonl"} and st["long_train.jsonl"][0] == 10
        views, stats, b, lines = T.plan_for_notebook(d, 10, 4096)
        assert set(views) == set(stats) and b > 0


def test_ooa_dialogues_views():
    # OOADialogues imports nanochat's tasks.common.Task: stand in a minimal one
    common = types.ModuleType("tasks.common")

    class Task:
        def __init__(self, start=0, stop=None, step=1):
            self.start, self.stop, self.step = start, stop, step

        def __len__(self):
            return self.num_examples()

        def __getitem__(self, i):
            return self.get_example(i)
    common.Task = Task
    sys.modules["tasks"] = types.ModuleType("tasks")
    sys.modules["tasks.common"] = common
    import importlib
    import ooa_task
    importlib.reload(ooa_task)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x_train.jsonl")
        with open(p, "w") as f:
            for i in range(10):
                f.write(json.dumps({"messages": [{"role": "user", "content": f"q{i}"},
                                                 {"role": "assistant", "content": f"a{i}"}]}) + "\n")
        plain = ooa_task.OOADialogues(p)
        assert len(plain) == 10 and plain[3]["messages"][0]["content"] == "q3"
        t = ooa_task.OOADialogues(p, views=2.4, seed=1)
        assert len(t) == 24 and t.num_rows == 10
        seen = [t[i]["messages"][0]["content"] for i in range(len(t))]
        counts = {q: seen.count(q) for q in set(seen)}
        assert len(counts) == 10 and sorted(counts.values()) == [2] * 6 + [3] * 4
        again = ooa_task.OOADialogues(p, views=2.4, seed=1)
        assert [again[i]["messages"][0]["content"] for i in range(24)] == seen          # seeded
        assert len(ooa_task.OOADialogues(p, views=0.3)) == 3


def test_patch_chat_sft_pinned_nanochat():
    if not NANOCHAT:
        print("  (skipped: give the nanochat checkout as an argument or env NANOCHAT_DIR)")
        return
    src = open(os.path.join(NANOCHAT, "scripts/chat_sft.py"), encoding="utf-8").read()
    out = T.patch_chat_sft(src, "/data/sft_render", {"long_train.jsonl": 2.5, "train_catalog.jsonl": 3.0},
                           smoltalk_rows=3000)
    compile(out, "chat_sft.py", "exec")
    assert out.index("model.config.sequence_len = args.max_seq_len") < out.index("model = torch.compile(model")
    assert out.index("for name, fallback, source in [") < out.index("model.config.sequence_len = args.max_seq_len")
    assert '"trained_window": orig_model.window_sizes[-1][0]' in out
    assert "OOADialogues('/data/sft_render/long_train.jsonl', views=2.5, seed=1)" in out
    assert 'SmolTalk(split="train", stop=3000)' in out and "MMLU(subset" not in out.split("train_dataset =")[0][-2000:]
    assert "_seen[train_dataset.index_map[cursor][0]] += 1" in out and "Training data actually read" in out
    assert "skipped_long[0] += 1" in out
    plain = T.patch_chat_sft(src, "/d", {"a.jsonl": 1.0}, window_fix=False)
    assert "model.config.sequence_len = args.max_seq_len" not in plain
    try:
        T.patch_chat_sft(src.replace("orig_model = model\n", "orig = model\n"), "/d", {"a.jsonl": 1.0})
        raise RuntimeError("a missing anchor must stop the patch")
    except AssertionError as ex:
        assert "changed upstream" in str(ex)


def test_nan_guard_stops_before_saving():
    """The guard patch_chat_sft puts after the loss line / the validation line: NAN_STEPS steps in a row whose loss or
    gradients are not a number (the step guard skipped them), or SKIP_STOP skipped steps in a row for any reason, or a
    nan validation end chat_sft with NAN_EXIT before its last step saves anything."""
    import math
    printed = []
    g = {"_ooa_math": math, "_ooa_bad_loss": [0], "_ooa_skip_run": [0], "print0": printed.append,
         "_ooa_weight_note": lambda m: "the weights are all finite", "orig_model": None}
    exec("def step_(_ooa_bad_step, step):\n" + T.LOSS_GUARD, g)
    exec("def val_(val_bpb, step):\n    if True:\n" + T.VAL_GUARD, g)
    for i, bad in enumerate([False, True, True, True, True, False, True, True, True, True], 1):   # never 5 in a row
        g["step_"](bad, i)
    assert not printed
    try:
        g["step_"](True, 11)
        raise RuntimeError("the 5th bad step in a row must stop")
    except SystemExit as ex:
        assert ex.code == T.NAN_EXIT == 86
    assert printed[0].startswith(T.NAN_MARK) and "(step 11)" in printed[0] and "nothing was saved" in printed[0]
    assert "the weights are all finite" in printed[0]
    g["_ooa_bad_loss"][0], g["_ooa_skip_run"][0] = 0, T.SKIP_STOP                # float16 overflow without end
    try:
        g["step_"](False, 70)
        raise RuntimeError(f"{T.SKIP_STOP} skipped steps in a row must stop")
    except SystemExit as ex:
        assert ex.code == T.NAN_EXIT
    assert f"{T.SKIP_STOP} steps in a row were skipped" in printed[-1]
    g["val_"](1.23, 0)
    try:
        g["val_"](float("nan"), 0)
        raise RuntimeError("a nan validation must stop")
    except SystemExit as ex:
        assert ex.code == T.NAN_EXIT
    assert "the start model is broken before any training step" in printed[-1]
    if not NANOCHAT:
        print("  (pinned-nanochat part skipped: give the nanochat checkout as an argument or env NANOCHAT_DIR)")
        return
    src = open(os.path.join(NANOCHAT, "scripts/chat_sft.py"), encoding="utf-8").read()
    out = T.patch_chat_sft(src, "/d", {"a.jsonl": 1.0})
    compile(out, "chat_sft.py", "exec")
    loss_line = out.index(T.LOSS_LOG_ANCHOR)
    assert loss_line < out.index("_ooa_bad_loss[0] = _ooa_bad_loss[0] + 1 if") < out.index("if step % 10 == 0:", loss_line)
    assert out.index(T.VAL_ANCHOR) < out.index("if not _ooa_math.isfinite(val_bpb)") < out.index("save_checkpoint(\n")
    assert out.index("_ooa_bad_loss = [0]") < out.index("while True:") and "import math as _ooa_math" in out
    assert '"compute_dtype": str(COMPUTE_DTYPE)' in out
    # the step guard sits between the backward passes and the optimizer step, the loss line shows it
    assert out.index("_ooa_losses.append(") < out.index("_ooa_gn = (") < out.index("scaler.step(optimizer)")
    assert out.count("scaler.step(optimizer)") == 1 and out.count("    muon_momentum = get_muon_momentum(step)\n") == 1
    assert "| gnorm: {_ooa_gnorm:.3g}" in out and "def _ooa_weight_note(m):" in out
    for anchor in (T.LOSS_LOG_ANCHOR, T.VAL_ANCHOR, T.OPT_ANCHOR, T.MICRO_ANCHOR, T.EMA_OLD):
        try:
            T.patch_chat_sft(src.replace(anchor, "    pass\n"), "/d", {"a.jsonl": 1.0})
            raise RuntimeError("a missing anchor must stop the patch")
        except AssertionError as ex:
            assert "changed upstream" in str(ex)


def _step_harness():
    """One training step of the patched chat_sft (sft_setup.MICRO_NEW .. OPT_NEW) as code to exec in a namespace."""
    import textwrap
    body = T.MICRO_NEW + ("        loss = loss / grad_accum_steps\n"
                          "        if scaler is not None:\n            scaler.scale(loss).backward()\n"
                          "        else:\n            loss.backward()\n        x, y = next(train_loader)\n") + T.OPT_NEW
    return compile(textwrap.dedent(body), "step_guard", "exec")


def test_step_guard_skips_bad_steps_and_keeps_the_weights_finite():
    """The step guard on a tiny model (CPU, torch): a micro-batch with a nan loss -> the step is skipped (weights and
    optimizer state unchanged, a SKIP line), the next step trains; with a GradScaler whose scale collapsed below 2^-128
    (1/scale = inf: chat_sft's own code then writes nan into every weight without the scaler noticing) the step is
    skipped and the scale is put back to the floor."""
    try:
        import torch
    except ImportError:
        print("  (skipped: no torch)")
        return
    import math
    torch.manual_seed(0)
    code = _step_harness()

    def setup(scaler):
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        for gr in opt.param_groups:
            gr["kind"] = "adamw"
        data = [(torch.randn(16, 4), torch.randn(16, 1)) for _ in range(200)]
        it = iter(data)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = model

            def forward(self, x, y):
                return torch.nn.functional.mse_loss(self.m(x), y)
        m = M()
        printed = []
        ns = {"torch": torch, "model": m, "orig_model": m, "optimizer": opt, "scaler": scaler, "grad_accum_steps": 2,
              "train_loader": it, "device": "cpu", "print0": printed.append, "is_ddp_initialized": lambda: False,
              "dist": None, "_ooa_math": math, "_ooa_skipped": [0], "_ooa_skip_run": [0], "_ooa_step_loss": [0.0],
              "_ooa_inject": set(), "step": 0}
        ns["x"], ns["y"] = next(it)
        return ns, printed

    def snap(ns):
        return [p.detach().clone() for p in ns["orig_model"].parameters()]

    ns, printed = setup(None)                                     # bfloat16 / float32: no scaler
    ns["_ooa_inject"] = {(2, 1)}                                  # step 2, micro-batch 1: loss nan
    for step in range(4):
        ns["step"] = step
        before = snap(ns)
        exec(code, ns)
        same = all(torch.equal(a, b) for a, b in zip(before, snap(ns)))
        assert same == (step == 1), (step, same)
        assert ns["_ooa_bad_step"] == (step == 1) and ns["_ooa_loss_bad"] == (step == 1)
        assert all(bool(torch.isfinite(p).all()) for p in ns["orig_model"].parameters())
        assert all(p.grad is None for p in ns["orig_model"].parameters())        # gradients dropped either way
    assert ns["_ooa_skipped"] == [1] and ns["_ooa_skip_run"] == [0] and len(printed) == 1, printed
    assert printed[0].startswith(T.SKIP_MARK + " step 00002: loss not a number in micro-batch [1] of 2"), printed
    st = ns["optimizer"].state[next(iter(ns["orig_model"].parameters()))]
    assert int(st["step"]) == 3                                   # 4 steps, 1 skipped: the optimizer stepped 3 times

    try:
        torch.amp.GradScaler("cpu")
    except Exception as ex:                                       # noqa: BLE001
        print("  (GradScaler part skipped:", type(ex).__name__, ex, ")")
        return
    # chat_sft's own step (OPT_ANCHOR) with a collapsed scale: 1/scale = inf, nothing flagged, the weights go nan
    import textwrap
    raw = compile(textwrap.dedent(T.MICRO_NEW + "        loss = loss / grad_accum_steps\n"
                                  "        scaler.scale(loss).backward()\n        x, y = next(train_loader)\n"
                                  + T.OPT_ANCHOR), "raw", "exec")
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0 ** -130)
    ns, printed = setup(scaler)
    exec(raw, ns)
    assert not all(bool(torch.isfinite(p).all()) for p in ns["orig_model"].parameters()), \
        "expected chat_sft's own step to break the weights at scale 2^-130 (the incident's mechanism)"
    # the guarded step: skipped, weights finite, the scale back at the floor; then training goes on
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0 ** -130)
    ns, printed = setup(scaler)
    before = snap(ns)
    exec(code, ns)
    assert all(torch.equal(a, b) for a, b in zip(before, snap(ns))) and ns["_ooa_skipped"] == [1]
    assert ns["_ooa_bad_step"] and scaler.get_scale() == T.SCALE_FLOOR, scaler.get_scale()
    assert printed and printed[0].startswith(T.SKIP_MARK + " step 00001: gradient norm ") and "-> 1:" in printed[0], printed
    for step in range(1, 4):
        ns["step"] = step
        exec(code, ns)
        assert all(bool(torch.isfinite(p).all()) for p in ns["orig_model"].parameters())
    assert ns["_ooa_skip_run"] == [0] and scaler.get_scale() >= T.SCALE_FLOOR
    # an overflow the scaler catches (scaled gradients inf, losses finite): skipped, the scale halves, not "bad"
    scaler.update(new_scale=2.0 ** 127)                         # loss x 2^127 overflows float32 in the backward
    ns["step"] = 4
    ns["x"], ns["y"] = torch.randn(16, 4) * 1e6, torch.randn(16, 1) * 1e6
    exec(code, ns)
    assert ns["_ooa_found"] > 0 and not ns["_ooa_bad_step"] and ns["_ooa_skip_run"] == [1]
    assert scaler.get_scale() == 2.0 ** 126 and "overflow caught by GradScaler" in printed[-1], printed[-1]


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, type(e).__name__, e)
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
