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
        assert v[n] <= T.LONG_CAP + 1e-9, (n, v[n])
    assert v["train_catalog.jsonl"] >= T.CATALOG_MIN
    assert min(v.values()) >= 1.0, v                         # nothing is left mostly unseen any more
    assert v["synth_train.jsonl"] < v["targeted_train.jsonl"]
    small = T.plan_mixture(STATS, b // 10)                   # a small budget: the floor and the scale still hold
    assert small["train_catalog.jsonl"] >= T.CATALOG_MIN and small["synth_train.jsonl"] < 1
    big = T.plan_mixture(STATS, b * 4)                      # more budget than the caps allow: never runs out
    assert abs(_tokens(big) - b * 4) / (b * 4) < 0.01 and big["long_train.jsonl"] <= T.LONG_CAP
    lines = T.mixture_table(STATS, v, b)
    assert "views" in lines[0] and "fits" in lines[-1]


def test_file_stats_and_plan_for_notebook():
    with tempfile.TemporaryDirectory() as d:
        for name, n in (("long_train.jsonl", 10), ("train_catalog.jsonl", 20), ("val.jsonl", 5), ("catalog.jsonl", 9)):
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
