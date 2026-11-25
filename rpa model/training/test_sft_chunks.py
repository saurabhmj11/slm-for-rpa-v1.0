"""Tests for sft_chunks.py: SFT in saved chunks, resumable across Colab sessions (cell 7), and the skip logic of cells
6 / 6d. The planner, the state handling and the chunk loop run with a fake chat_sft (no torch); the tiny CPU run (a
real nanochat chat_sft, a session break between chunk 1 and chunks 2-3) needs torch and NANOCHAT_DIR (the pinned
nanochat). Run: python test_sft_chunks.py   or   python -m pytest -q test_sft_chunks.py"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sft_chunks as SC      # noqa: E402
import sft_setup as T        # noqa: E402

NANO = os.environ.get("NANOCHAT_DIR")


class MP:
    """A minimal monkeypatch for the plain runner (python test_sft_chunks.py)."""
    def __init__(self):
        self.undo = []

    def setattr(self, obj, name, value):
        self.undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def restore(self):
        for obj, n, v in reversed(self.undo):
            setattr(obj, n, v)


def fake_nonfinite(path):
    """sft_setup.nonfinite_tensors without torch: a file whose bytes start with b"nan" is a broken state dict."""
    return ["wte.weight (nan)"] if Path(path).read_bytes().startswith(b"nan") else []


def put_model(d8, step, data=b"m", optim=b"o", meta=None):
    d8 = Path(d8)
    d8.mkdir(parents=True, exist_ok=True)
    (d8 / f"model_{step:06d}.pt").write_bytes(data)
    (d8 / f"meta_{step:06d}.json").write_text(json.dumps(meta or {"step": step}), encoding="utf-8")
    if optim is not None:
        (d8 / f"optim_{step:06d}_rank0.pt").write_bytes(optim)


def fake_drive(root, report="r1"):
    """A Drive base folder (base model d8 + tokenizer) and a rendered data folder."""
    root = Path(root)
    base = root / "drive" / "nanochat_base"
    put_model(base / "base_checkpoints" / "d8", 5000, meta={"step": 5000, "user_config": {"unembedding_lr": 0.008}})
    (base / "tokenizer").mkdir(parents=True)
    (base / "tokenizer" / "tokenizer.pkl").write_bytes(b"tok")
    (base / "tokenizer" / "token_bytes.pt").write_bytes(b"tb")
    data = root / "drive" / "sft_render"
    data.mkdir(parents=True)
    for n in ("val.jsonl", "beginner_train.jsonl", "long_train.jsonl"):
        (data / n).write_text("".join(json.dumps({"messages": [{"role": "user", "content": "x" * 300},
                                                               {"role": "assistant", "content": "y"}]}) + "\n"
                                      for _ in range(20)), encoding="utf-8")
    (data / "render_report.json").write_text(json.dumps({"settings": {"max_tokens": 4068}, "r": report}),
                                             encoding="utf-8")
    return base, data


class FakeSFT:
    """Stands in for run_chunk: 'trains' by writing a model to the chunk's chatsft_checkpoints, records every call."""
    def __init__(self, nan_when=None, broken_out=False):
        self.calls = []
        self.nan_when = nan_when or (lambda call: False)
        self.broken_out = broken_out

    def __call__(self, cmd, cwd, env, log, total, offset, prio, watch=None, every=10):
        flags = dict(re.match(r"--([\w-]+)=(.*)", c).groups() for c in cmd if c.startswith("--"))
        call = {"flags": flags, "offset": int(env["OOA_STEP_OFFSET"]), "base_dir": env["NANOCHAT_BASE_DIR"]}
        self.calls.append(call)
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text("log\n", encoding="utf-8")
        if self.nan_when(call):
            watch["nan"] = "the training loss was not a number (nan/inf) for 5 steps in a row (step 3)"
            return -15, 1.0
        tag = flags["model-tag"]
        steps = int(flags["num-iterations"]) * int(flags["max-seq-len"]) * int(flags["device-batch-size"]) \
            // int(flags["total-batch-size"])
        put_model(Path(env["NANOCHAT_BASE_DIR"]) / "chatsft_checkpoints" / tag, steps,
                  data=b"nan" if self.broken_out else f"model after {offset_plus(call, steps)}".encode(),
                  meta={"step": steps, "user_config": {"init_lr_frac": float(flags["init-lr-frac"])},
                        "trained_window": int(flags["max-seq-len"])})
        return 0, 1.0


def offset_plus(call, steps):
    return call["offset"] + steps


def setup_fakes(mp, sft):
    mp.setattr(T, "nonfinite_tensors", fake_nonfinite)
    mp.setattr(SC, "run_chunk", sft)
    mp.setattr(SC, "write_script", lambda *a, **k: None)


def run(base, data, work, **kw):
    args = dict(steps=12, chunks=3, seq=4096, device_batch=2, tokens_per_step=32768, smoltalk=0,
                nano=str(work / "nano"), training_dir=str(HERE), work=str(work / "work"))
    args.update(kw)
    return SC.colab_sft(base, data, **args)


# ------------------------------------------------------------------------------------------- the planner
def test_chunks_follow_one_run_schedule():
    """The chunks' learning-rate flags put together give one run's schedule (constant, then down to 0)."""
    def single(g):
        return 0.8 * (1.0 if g <= 0.5 else (1 - g) / 0.5)
    for steps, chunks in ((12000, 12), (12000, 7), (100, 2), (6, 3), (1001, 4)):
        b = SC.chunk_bounds(steps, chunks)
        assert b[0] == 0 and b[-1] == steps and all(b[i] < b[i + 1] for i in range(chunks))
        for k in range(chunks):
            lr0, wd, final = SC.lr_for_span(b[k] / steps, b[k + 1] / steps, 0.8)
            for p in (0.0, 0.3, 0.5, 0.9, 1.0):
                lr = lr0 if p <= 1 - wd else lr0 * ((1 - (p - (1 - wd)) / wd) + (p - (1 - wd)) / wd * final)
                g = (b[k] + p * (b[k + 1] - b[k])) / steps
                assert abs(lr - single(g)) < 1e-9, (steps, chunks, k, p)
    assert SC.lr_for_span(0, 1, 0.8) == (0.8, 0.5, 0.0)                  # one chunk = nanochat's defaults


def test_plan_run_cases():
    s = {"steps": 12000, "chunks": 12, "seq": 4096, "device_batch": 2, "tokens_per_step": 32768, "lr": 0.8,
         "smoltalk": 3000, "seed": 1, "data": "/d", "data_sig": "abc", "base": {"tag": "d8", "step": 5000}}
    assert SC.plan_run({}, s) == ("new", [])
    st = dict(s, done=4, finished=False)
    assert SC.plan_run(st, s) == ("continue", [])
    a, notes = SC.plan_run(st, dict(s, data_sig="xyz"))
    assert a == "refuse" and "training data" in notes[0] and "abc -> xyz" in notes[0]
    a, notes = SC.plan_run(st, dict(s, chunks=6, seq=2048))
    assert a == "refuse" and len(notes) == 2 and "SFT_CHUNKS: 12 -> 6" in notes[0]
    a, notes = SC.plan_run(st, dict(s, data_sig="xyz"), restart=True)
    assert a == "new" and "4 of 12 chunks done; they are thrown away" in notes[0]
    assert SC.plan_run(st, s, chunks_ok=False)[0] == "new"
    fin = dict(s, done=12, finished=True)
    assert SC.plan_run(fin, s) == ("finished", [])
    a, notes = SC.plan_run(fin, dict(s, steps=8000))
    assert a == "new" and "finished" in notes[0] and "12000 -> 8000" in notes[0]
    assert SC.plan_run(fin, s, restart=True)[0] == "new"
    # RESTART_SFT left on: the run it started is continued (also in the next session), never thrown away again
    rst = dict(st, started_by_restart=True)
    a, notes = SC.plan_run(rst, s, restart=True)
    assert a == "continue" and "RESTART_SFT = True is ignored" in notes[0]
    assert SC.plan_run(dict(fin, started_by_restart=True), s, restart=True)[0] == "finished"
    assert SC.plan_run(rst, dict(s, steps=8000), restart=True)[0] == "new"        # new settings: a new run
    msg = SC.refuse_message(notes, st)
    assert "4 of 12 chunks" in msg and "RESTART_SFT = True" in msg and "FORCE_DATA = False" in msg


# ------------------------------------------------------------------------------------------- the chunk loop
def test_resume_across_sessions_with_a_fake_chat_sft(monkeypatch=None):
    mp = monkeypatch or MP()
    try:
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            base, data = fake_drive(t)
            put_model(base / "chatsft_checkpoints" / "d8", 9000, data=b"old model")      # the previous finished model
            sft = FakeSFT()
            setup_fakes(mp, sft)
            # session 1: the first chunk, then the session ends
            r = run(base, data, t, stop_after=1)
            assert r["status"] == "paused" and r["trained_now"] == 1
            st = json.loads((base / "sft_run/state.json").read_text(encoding="utf-8"))
            assert st["done"] == 1 and st["chunks"] == 3 and st["data_sig"] == SC.data_sig(data)
            assert st["base"]["tag"] == "d8" and st["base"]["step"] == 5000
            assert st["base"]["tokenizer_sha1"] == hashlib.sha1(b"tok").hexdigest()
            assert SC.newest(base / "sft_run/chunks/chunk_000") == 4
            assert not (base / "chatsft_checkpoints").exists()                   # moved aside once, at the new run
            kept = sorted(base.glob("chatsft_checkpoints_before_*"))
            assert len(kept) == 1 and (kept[0] / "d8/model_009000.pt").read_bytes() == b"old model"
            # a disconnect in the middle of chunk 2: a half-copied chunk and the old work folder are left behind
            (base / "sft_run/chunks/chunk_001.part").mkdir()
            (base / "sft_run/chunks/chunk_001.part/model_000004.pt").write_bytes(b"half")
            # a new session with other settings: refused, nothing changes
            try:
                run(base, data, t, steps=24)
                raise AssertionError("other settings must be refused")
            except SystemExit as ex:
                assert "optimizer steps (SFT_ITERS / 4): 12 -> 24" in str(ex) and "RESTART_SFT = True" in str(ex)
            assert json.loads((base / "sft_run/state.json").read_text(encoding="utf-8"))["done"] == 1
            # session 2: continues with chunks 2 and 3, then installs the model
            r = run(base, data, t)
            assert r["status"] == "finished" and r["trained_now"] == 2
            assert len(sft.calls) == 3
            for k, c in enumerate(sft.calls):                                       # one run's schedule
                lr0, wd, final = SC.lr_for_span(k / 3, (k + 1) / 3, 0.8)
                f = c["flags"]
                assert (float(f["init-lr-frac"]), float(f["warmdown-ratio"]), float(f["final-lr-frac"])) == \
                    (round(lr0, 6), round(wd, 6), round(final, 6)), (k, f)
                assert c["offset"] == 4 * k and f["num-iterations"] == str(4 * 4)  # 4 steps x 4 micro-batches
                assert f["unembedding-lr"] == "0.008" and f["embedding-lr"] == "0.3" and f["matrix-lr"] == "0.02"
                assert f["load-optimizer"] == "1" and f["model-tag"] == "d8"
            assert sft.calls[0]["flags"]["model-step"] == "5000" and sft.calls[1]["flags"]["model-step"] == "4"
            st = json.loads((base / "sft_run/state.json").read_text(encoding="utf-8"))
            assert st["finished"] and st["done"] == 3 and [c["seed"] for c in st["chunk_log"]] == [1, 2, 3]
            assert sorted(p.name for p in (base / "sft_run/chunks").iterdir()) == ["chunk_001", "chunk_002"]
            d8 = base / "chatsft_checkpoints/d8"
            assert SC.newest(d8) == 12 and (d8 / "optim_000012_rank0.pt").exists()     # renamed to the run's step
            assert (d8 / "model_000012.pt").read_bytes() == b"model after 12"
            meta = json.loads((d8 / "meta_000012.json").read_text(encoding="utf-8"))
            assert meta["step"] == 12 and meta["ooa_colab"]["chunks"] == 3 and meta["trained_window"] == 4096
            # session 3: nothing to train; the finished model is not moved aside again
            r = run(base, data, t)
            assert r["status"] == "finished" and r["trained_now"] == 0 and len(sft.calls) == 3
            assert len(list(base.glob("chatsft_checkpoints_before_*"))) == 1
            # RESTART_SFT: a new run (the finished model moved aside); left on, it does not restart that run again
            r = run(base, data, t, restart=True, stop_after=1)
            assert r["status"] == "paused" and len(list(base.glob("chatsft_checkpoints_before_*"))) == 2
            assert not (base / "chatsft_checkpoints").exists()
            r = run(base, data, t, restart=True, stop_after=1)
            st = json.loads((base / "sft_run/state.json").read_text(encoding="utf-8"))
            assert st["done"] == 2 and st["started_by_restart"]
            # new data in the middle of a run (cells 6 / 6d made it again): refused
            (data / "render_report.json").write_text(json.dumps({"r": "r2"}), encoding="utf-8")
            try:
                run(base, data, t)
                raise AssertionError("changed data must be refused")
            except SystemExit as ex:
                assert "training data" in str(ex)
    finally:
        if not monkeypatch:
            mp.restore()


def test_a_nan_chunk_is_never_saved_and_is_retried_with_a_fresh_optimizer(monkeypatch=None):
    mp = monkeypatch or MP()
    try:
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            base, data = fake_drive(t)
            # chunk 2 goes nan with the loaded optimizer state, then trains with a fresh one
            sft = FakeSFT(nan_when=lambda c: c["offset"] == 4 and c["flags"]["load-optimizer"] == "1")
            setup_fakes(mp, sft)
            r = run(base, data, t)
            assert r["status"] == "finished"
            assert [c["flags"]["load-optimizer"] for c in sft.calls] == ["1", "1", "0", "1"]
            st = json.loads((base / "sft_run/state.json").read_text(encoding="utf-8"))
            assert st["nan_stops"][0]["chunk"] == 1 and st["nan_stops"][0]["optimizer"] == "loaded"
            assert [c["optimizer"] for c in st["chunk_log"]] == ["loaded", "fresh", "loaded"]
            # nan with a fresh optimizer too: stopped, the chunk is not saved, the run does not advance
            shutil.rmtree(base / "sft_run")
            shutil.rmtree(base / "chatsft_checkpoints")
            sft = FakeSFT(nan_when=lambda c: c["offset"] == 4)
            mp.setattr(SC, "run_chunk", sft)
            try:
                run(base, data, t)
                raise AssertionError("must stop")
            except SystemExit as ex:
                assert "running it again will not help" in str(ex)
            st = json.loads((base / "sft_run/state.json").read_text(encoding="utf-8"))
            assert st["done"] == 1 and len(st["nan_stops"]) == 2 and not (base / "sft_run/chunks/chunk_001").exists()
            assert not (base / "chatsft_checkpoints").exists()
            # the watchdog missed it but the saved weights are nan: treated the same, never saved
            shutil.rmtree(base / "sft_run")
            mp.setattr(SC, "run_chunk", FakeSFT(broken_out=True))
            try:
                run(base, data, t)
                raise AssertionError("must stop")
            except SystemExit as ex:
                assert "the model it saved has nan/inf" in str(ex)
            assert not (base / "sft_run/chunks/chunk_000").exists()
    finally:
        if not monkeypatch:
            mp.restore()


def test_v6_base_uses_its_own_tokenizer_and_state(monkeypatch=None):
    """BASE "v6": the run lives in nanochat_base_v6/sft_run and records that base's tokenizer."""
    mp = monkeypatch or MP()
    try:
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            base, data = fake_drive(t)
            v6 = base.with_name("nanochat_base_v6")
            put_model(v6 / "base_checkpoints/d8", 5760)
            (v6 / "tokenizer").mkdir(parents=True)
            (v6 / "tokenizer/tokenizer.pkl").write_bytes(b"v6 tok")
            setup_fakes(mp, FakeSFT())
            run(base, data, t, stop_after=1)
            run(v6, data, t, stop_after=1)
            s1 = json.loads((base / "sft_run/state.json").read_text(encoding="utf-8"))
            s6 = json.loads((v6 / "sft_run/state.json").read_text(encoding="utf-8"))
            assert s6["base"]["tokenizer_sha1"] == hashlib.sha1(b"v6 tok").hexdigest() != s1["base"]["tokenizer_sha1"]
            assert s6["base"]["step"] == 5760 and s1["done"] == s6["done"] == 1
            assert (t / "work/nanochat_base/tokenizer/tokenizer.pkl").read_bytes() == b"v6 tok"
    finally:
        if not monkeypatch:
            mp.restore()


# ------------------------------------------------------------------------------------------- cells 6 / 6d
def test_data_skip():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        base = t / "nanochat_base"
        out = ["sft/a.jsonl", "sft_render/render_report.json"]
        want = {"code": "c1", "synth_train": 30000, "platform": "p1"}
        skip, why = SC.data_skip(t, SC.STAMP6, want, out)
        assert not skip and "not on Drive yet" in why
        for o in out:
            (t / o).parent.mkdir(parents=True, exist_ok=True)
            (t / o).write_text("x", encoding="utf-8")
        skip, why = SC.data_skip(t, SC.STAMP6, want, out)
        assert not skip and "older notebook" in why                          # outputs without a record: made again
        SC.write_stamp(t, SC.STAMP6, want)
        skip, why = SC.data_skip(t, SC.STAMP6, dict(want, platform="p2"), out)   # the platform moved on: kept
        assert skip and "same inputs" in why
        assert not SC.data_skip(t, SC.STAMP6, want, out, force=True)[0]
        skip, why = SC.data_skip(t, SC.STAMP6, dict(want, synth_train=1500), out, base_dir=base)
        assert not skip and "synth_train" in why
        # an SFT run in progress on the data: kept even when the inputs differ now
        (base / "sft_run").mkdir(parents=True)
        SC.write_state(base / "sft_run/state.json", {"done": 3, "chunks": 12, "finished": False})
        skip, why = SC.data_skip(t, SC.STAMP6, dict(want, synth_train=1500), out, base_dir=base)
        assert skip and "3 of 12 chunks" in why and "synth_train" in why and "FORCE_DATA = True" in why
        assert not SC.data_skip(t, SC.STAMP6, dict(want, synth_train=1500), out, force=True, base_dir=base)[0]
        SC.write_state(base / "sft_run/state.json", {"done": 12, "chunks": 12, "finished": True})
        assert not SC.data_skip(t, SC.STAMP6, dict(want, synth_train=1500), out, base_dir=base)[0]
        (t / "sft/a.jsonl").write_text("", encoding="utf-8")                  # an empty output is not finished
        assert not SC.data_skip(t, SC.STAMP6, want, out)[0]
        SC.clear_stamp(t, SC.STAMP6)
        assert not (t / SC.STAMP6).exists()
        assert SC.code_sig(HERE, ("sft_chunks.py",)) != SC.code_sig(HERE, ("sft_setup.py",))


def nb_code():
    nb = json.loads((HERE / "ooa_nanochat_colab.ipynb").read_text(encoding="utf-8"))
    return nb, ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def strip_magics(src):
    """The cell as plain Python: ! / % lines (and their \\ continuation lines) become pass."""
    out, cont = [], False
    for line in src.splitlines():
        st = line.lstrip()
        if cont:
            cont = line.endswith("\\")
            continue
        if st.startswith(("!", "%")):
            out.append(line[:len(line) - len(st)] + "pass")
            cont = line.endswith("\\")
        else:
            out.append(line)
    return "\n".join(out)


def test_notebook_cells():
    nb, code = nb_code()
    for i, c in enumerate(code):
        compile(strip_magics(c), f"cell {i}", "exec")
    c1 = code[0]
    assert 'SFT_CHUNKS = 2 if PROFILE == "smoke" else 12' in c1 and "RESTART_SFT = False" in c1
    assert "FORCE_DATA = False" in c1
    c6 = next(c for c in code if c.startswith("#@title 6. Build"))
    assert '"sft_chunks.py"' in c6 and "_SC.data_skip(" in c6 and "_SC.write_stamp(" in c6
    assert c6.index("FILES.items()") < c6.index("if SKIP6:")                 # code files written even when skipped
    assert c6.index("cp /content/ooa/ooa_task.py") < c6.index("if SKIP6:")
    c6d = next(c for c in code if c.startswith("#@title 6d."))
    assert "SKIP6D" in c6d and c6d.index("_SC.data_skip(") < c6d.index("render_dataset.py")
    c7 = next(c for c in code if c.startswith("#@title 7."))
    assert "_SC.colab_sft(" in c7 and "restart=RESTART_SFT" in c7 and "SFT_CHUNKS" in c7
    assert "!python -m scripts.chat_sft " not in c7                          # no single unsaved run any more
    md = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "run 1 -> 2 -> 6 -> 6d -> 7" in md
    gen = (HERE / "make_colab_notebook.py").read_text(encoding="utf-8")
    assert '"sft_chunks.py": (H / "sft_chunks.py")' in gen


# ------------------------------------------------------------------------------------------- a real tiny run
def tiny_setup(t):
    """A copy of the pinned nanochat, a tiny base model (d2-sized, window 128, 300-token tokenizer) on a fake Drive and
    a few short training rows."""
    sys.path.insert(0, str(HERE / "pretrain"))
    import base_setup as B
    import random
    import pyarrow as pa
    import pyarrow.parquet as pq
    nano = t / "nanochat"
    shutil.copytree(NANO, nano, ignore=shutil.ignore_patterns("*_ooa.py", "ooa*.py", "__pycache__"))
    base = t / "drive" / "nanochat_base"
    rnd = random.Random(7)
    words = ["flow", "step", "read", "the", "file", "and", "send", "mail", "to", "team", "sheet", "copy", "rows"]
    d = base / "base_data_climbmix"
    d.mkdir(parents=True)
    for name, n in (("shard_00000.parquet", 400), ("shard_99999.parquet", 80)):
        pq.write_table(pa.table({"text": [" ".join(rnd.choice(words) for _ in range(rnd.randint(20, 80)))
                                          for _ in range(n)]}), d / name, row_group_size=100)
    env = dict(os.environ, NANOCHAT_BASE_DIR=str(base), NANOCHAT_DTYPE="float32", PYTHONUTF8="1", OMP_NUM_THREADS="2")
    r = subprocess.run([sys.executable, "-m", "scripts.tok_train", "--max-chars=100000", "--vocab-size=300"],
                       cwd=nano, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    pinned = subprocess.run(["git", "-C", str(nano), "show", f"{SC.NANOCHAT_COMMIT}:scripts/base_train.py"],
                            capture_output=True, text=True, check=True).stdout
    (nano / "scripts" / "base_train_t.py").write_text(B.patch_base_train(pinned, compile_ok=False), encoding="utf-8")
    p = dict(B.PROFILES["tiny"], iterations=3, save_every=3, eval_every=-1)
    r = subprocess.run([sys.executable, "-m", "scripts.base_train_t"] + B.base_train_args(p, device_type="cpu"),
                       cwd=nano, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    shutil.rmtree(d)
    data = t / "drive" / "sft_render"
    data.mkdir(parents=True)
    rows = [{"messages": [{"role": "user", "content": f"read the file {i}"},
                          {"role": "assistant", "content": "send mail to team"}]} for i in range(40)]
    for n, rr in (("beginner_train.jsonl", rows), ("val.jsonl", rows[:30])):
        (data / n).write_text("".join(json.dumps(x) + "\n" for x in rr), encoding="utf-8")
    (data / "render_report.json").write_text(json.dumps({"settings": {"max_tokens": 100}}), encoding="utf-8")
    return nano, base, data, env


SESSION = """
import sys, json
sys.path.insert(0, {here!r})
import sft_chunks as SC
r = SC.colab_sft({base!r}, {data!r}, 6, 3, 128, 1, 256, smoltalk=0, nano={nano!r}, training_dir={here!r},
                 work={work!r}, eval_every=-1, device_type="cpu", compile_ok=False, stop_after={stop}, every=1)
print("RESULT", json.dumps({{"status": r["status"], "trained_now": r["trained_now"]}}))
"""


def test_tiny_cpu_run_with_a_session_break():
    """Chunk 1 of 3 in one process, then a new process (a new Colab session, after a disconnect in the middle of
    chunk 2) continues chunks 2-3; the finished model is installed to chatsft_checkpoints/<tag>."""
    if not NANO or not Path(NANO, "scripts/chat_sft.py").exists():
        print("  (skipped: set NANOCHAT_DIR to the pinned nanochat)")
        return
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  (skipped: no torch)")
        return
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        nano, base, data, env = tiny_setup(t)
        tag = SC.base_tag(base)
        bstep = SC.newest(base / "base_checkpoints" / tag)

        def session(stop, work):
            code = SESSION.format(here=str(HERE), base=str(base), data=str(data), nano=str(nano), work=str(work),
                                  stop=stop)
            r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=1200)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
            return r.stdout, json.loads(r.stdout.rsplit("RESULT ", 1)[1])

        out, res = session(1, t / "work1")
        assert res == {"status": "paused", "trained_now": 1}, out
        assert "NEW SFT RUN" in out and "SFT attention window" in out and "chunk 1/3 saved to Drive" in out
        assert "continues at chunk 2" in out
        st = json.loads((base / "sft_run/state.json").read_text(encoding="utf-8"))
        assert st["done"] == 1 and not (base / "chatsft_checkpoints").exists()
        # the session ended in the middle of chunk 2: a half-written chunk on Drive, a new machine (other work folder)
        (base / "sft_run/chunks/chunk_001.part").mkdir()
        (base / "sft_run/chunks/chunk_001.part/model_000002.pt").write_bytes(b"half")
        out, res = session("None", t / "work2")
        assert res == {"status": "finished", "trained_now": 2}, out
        assert "CONTINUING the SFT run" in out and "Next: chunk 2" in out and "SFT FINISHED" in out
        st = json.loads((base / "sft_run/state.json").read_text(encoding="utf-8"))
        assert st["finished"] and [c["steps"] for c in st["chunk_log"]] == [[0, 2], [2, 4], [4, 6]]
        for k, c in enumerate(st["chunk_log"]):
            assert c["lr"] == [round(x, 6) for x in SC.lr_for_span(k / 3, (k + 1) / 3, 0.8)]
            assert c["optimizer"] == ("loaded" if k else c["optimizer"])     # chunks 2-3 carry the optimizer state
        d8 = base / "chatsft_checkpoints" / tag
        assert SC.newest(d8) == 6 and (d8 / "optim_000006_rank0.pt").exists()
        meta = json.loads((d8 / "meta_000006.json").read_text(encoding="utf-8"))
        uc = meta["user_config"]
        lr0, wd, final = SC.lr_for_span(2 / 3, 1, 0.8)                     # the last chunk: the end of the warmdown
        assert abs(uc["init_lr_frac"] - lr0) < 1e-6 and abs(uc["warmdown_ratio"] - wd) < 1e-6
        assert abs(uc["final_lr_frac"] - final) < 1e-6 and uc["unembedding_lr"] == 0.008
        assert meta["step"] == 6 and meta["trained_window"] == 128 and meta["ooa_colab"]["chunks"] == 3
        assert meta["tokenizer_sha1"] == SC.file_sha1(base / "tokenizer/tokenizer.pkl")
        assert meta["ooa_colab"]["base"]["step"] == bstep
        assert T.check_model_dir(d8) == (6, [])                              # finite weights
        assert sorted(p.name for p in (base / "sft_run/chunks").iterdir()) == ["chunk_001", "chunk_002"]
        out, res = session("None", t / "work2")                              # a later session: nothing to do
        assert res == {"status": "finished", "trained_now": 0} and "already finished" in out


if __name__ == "__main__":
    import inspect
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(*([None] if "monkeypatch" in inspect.signature(fn).parameters else []))
                print("ok  ", name)
            except Exception as ex:                                   # noqa: BLE001
                failed += 1
                print("FAIL", name, type(ex).__name__, str(ex)[:800])
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
