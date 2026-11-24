"""Tests for the local PC pipeline (data_pc.py, train_pc.py, eval_pc.py, make_local_notebook.py) that need no torch
and no model: the chunked learning-rate schedule, the top-up mixture, the Drive sync (with a local folder as Drive),
the start-model choice, the test sample, the install and the generated notebook.
Run: python test_train_pc.py"""
import ast
import json
import os
import re
import subprocess
import sys
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import data_pc as D      # noqa: E402
import eval_pc as E      # noqa: E402
import setup_pc as SP    # noqa: E402
import sft_setup as T    # noqa: E402
import train_pc as P     # noqa: E402

COLAB = (HERE.parent / "make_colab_notebook.py").read_text(encoding="utf-8")


def single_run_lr(g, lr):
    return lr * (1.0 if g <= 0.5 else (1 - g) / 0.5)


def chunk_lr(k, chunks, lr, p):
    """nanochat chat_sft get_lr_multiplier with the chunk's flags, at chunk-local progress p."""
    lr0, wd, final = P.lr_for_chunk(k, chunks, lr)
    if p <= 1.0 - wd:
        return lr0
    decay = (p - (1.0 - wd)) / wd
    return lr0 * ((1 - decay) + decay * final)


def test_chunks_follow_one_run_schedule():
    assert P.lr_for_chunk(0, 1, 0.8) == (0.8, 0.5, 0.0)           # one chunk = nanochat's defaults = Colab
    for chunks in (2, 3, 6, 7, 10):
        for k in range(chunks):
            for p in (0.0, 0.25, 0.5, 0.75, 1.0):
                g = (k + p) / chunks
                assert abs(chunk_lr(k, chunks, 0.3, p) - single_run_lr(g, 0.3)) < 1e-9, (chunks, k, p)


def test_render_settings_match_colab():
    m = re.search(r"NEW_FILES = (\{.*?\})", COLAB, re.S)
    names = set(re.findall(r'"(\w+_train\.jsonl)"', m.group(1)))
    assert names == set(T.RENDER_EXTRA), (names, T.RENDER_EXTRA)
    assert f'ANSWER_FORMAT = "{D.ANSWER_FORMAT}"' in COLAB and f"GOLD_DROPOUT = {D.GOLD_DROPOUT}" in COLAB
    assert "ROW_CHARS = 6500 * SFT_SEQ // 2048" in COLAB and D.row_chars(4096) == 13000
    assert "ROW_TOKENS = SFT_SEQ - 28" in COLAB and D.row_tokens(4096) == 4068
    assert f"--cap {D.FILTER_CAP}" in COLAB
    assert f'NANOCHAT_COMMIT = "{P.NANOCHAT_COMMIT}"' in COLAB
    assert "SFT_ITERS = 400 if PROFILE == \"smoke\" else 48000" in COLAB and P.FULL_STEPS == 48000 // 4
    assert "SMOLTALK_ROWS = 3000" in COLAB and P.SMOLTALK_ROWS == 3000
    assert P.TOKENS_PER_STEP == 4096 * 2 * 4                        # SFT_SEQ x SFT_DEVICE_BATCH x 4


STATS = {"beginner_train.jsonl": (10884, 1547), "long_train.jsonl": (3000, 2329), "business2_train.jsonl": (1440, 2208),
         "fix_train.jsonl": (6000, 1883), "synth_pattern.jsonl": (6290, 2305), "targeted_train.jsonl": (4000, 1088),
         "combos_train.jsonl": (2048, 1154), "tools_train.jsonl": (1498, 1063), "chains_train.jsonl": (832, 1212),
         "scenarios_train.jsonl": (474, 1809), "clarify_train.jsonl": (900, 736), "train_flows.jsonl": (81, 2232),
         "notask_train.jsonl": (359, 332), "train_catalog.jsonl": (2550, 60)}


def test_topup_mixture_is_beginner_and_business_first():
    b = T.budget(300, P.TOKENS_PER_STEP)
    v = T.plan_topup(STATS, b)
    tok = {n: v[n] * STATS[n][0] * STATS[n][1] for n in v}
    total = sum(tok.values())
    assert abs(total - b) / b < 0.02, (total, b)
    focus = sum(tok[n] for n in ("beginner_train.jsonl", "long_train.jsonl", "business2_train.jsonl"))
    assert focus / total > 0.7, focus / total
    assert tok["beginner_train.jsonl"] / total > 0.3
    assert all(v[n] > 0 for n in v)                                  # replay keeps every file in the mix
    full = T.plan_mixture(STATS, b)                                   # the full mix at the same budget
    ftok = {n: full[n] * STATS[n][0] * STATS[n][1] for n in full}
    ffocus = sum(ftok[n] for n in ("beginner_train.jsonl", "long_train.jsonl", "business2_train.jsonl"))
    assert focus / total > ffocus / sum(ftok.values()) + 0.1
    for n in ("beginner_train.jsonl", "business2_train.jsonl", "clarify_train.jsonl"):
        assert v[n] > full[n], n
    assert v["synth_pattern.jsonl"] < full["synth_pattern.jsonl"]
    # a missing focus file's share goes to the others; feedback gets its share when present
    s2 = dict(STATS, **{"feedback_train.jsonl": (50, 900)})
    v2 = T.plan_topup(s2, b)
    assert v2["feedback_train.jsonl"] * 50 * 900 / b > 0.015


def test_plan_for_local_modes():
    with tempfile.TemporaryDirectory() as d:
        for n, (rows, _) in list(STATS.items())[:5]:
            Path(d, n).write_text("".join(json.dumps({"messages": [{"role": "user", "content": "x" * 300}]}) + "\n"
                                          for _ in range(min(rows, 40))), encoding="utf-8")
        Path(d, "val.jsonl").write_text("{}\n", encoding="utf-8")
        vf, _, bf, _ = T.plan_for_local(d, "full", 8000, P.TOKENS_PER_STEP, smoltalk_rows=3000)
        vn, _, bn, _ = T.plan_for_notebook(d, 8000, P.TOKENS_PER_STEP, smoltalk_rows=3000)
        assert (vf, bf) == (vn, bn)                                   # full = exactly Colab cell 7's plan
        vt, _, _, _ = T.plan_for_local(d, "topup", 300, P.TOKENS_PER_STEP)
        assert set(vt) == set(vf) and "val.jsonl" not in vt


V4 = {"answer_format": "compact", "gold_dropout": 0.5, "max_tokens": 4068}


def make_drive(root, report="r1"):
    root = Path(root)
    (root / "nanochat_base/tokenizer").mkdir(parents=True)
    (root / "nanochat_base/tokenizer/tokenizer.pkl").write_bytes(b"tok")
    (root / "nanochat_base/tokenizer/token_bytes.pt").write_bytes(b"tb")
    for folder, step in (("chatsft_checkpoints_before_20251101_000000", 2500), ("chatsft_checkpoints", 8000)):
        d8 = root / "nanochat_base" / folder / "d8"
        d8.mkdir(parents=True)
        (d8 / f"model_{step:06d}.pt").write_bytes(b"m" * step)
        (d8 / f"meta_{step:06d}.json").write_text(json.dumps({"step": step}), encoding="utf-8")
    r = root / "sft_render"
    r.mkdir()
    for n in ("catalog.jsonl", "examples.jsonl", "val.jsonl", *T.RENDER_EXTRA):
        (r / n).write_text('{"a": 1}\n', encoding="utf-8")
    (r / "long_train (1).jsonl").write_text('{"copy": 1}\n', encoding="utf-8")
    (r / "render_report.json").write_text(json.dumps({"seconds": report, "settings": V4}), encoding="utf-8")
    return root


def test_drive_sync_with_a_local_folder():
    with tempfile.TemporaryDirectory() as t:
        drive, rpa = make_drive(Path(t) / "drive"), Path(t) / "rpa"
        logs = []
        log = lambda *x: logs.append(" ".join(str(y) for y in x))
        out = D.sync(D.FolderDrive(drive), rpa, "topup", log=log)
        data = Path(out["data"])
        assert out["render"] == "downloaded" and (data / "examples.jsonl").exists()
        assert not (data / "long_train (1).jsonl").exists()
        assert any("(n) in the name" in l and "long_train (1).jsonl" in l for l in logs), logs
        assert out["drive_model"] == ("nanochat_base/chatsft_checkpoints/d8", 8000)
        dm = rpa / "train_data/drive_model"
        assert (dm / "chatsft_checkpoints/d8/model_008000.pt").exists() and (dm / "tokenizer/tokenizer.pkl").exists()
        assert P.newest(dm / "chatsft_checkpoints/d8") == 8000
        # unchanged on Drive: nothing downloaded again; a new render on Drive: downloaded again
        assert D.sync(D.FolderDrive(drive), rpa, "topup", log=log)["render"] == "up to date"
        (drive / "sft_render/render_report.json").write_text(json.dumps({"seconds": "r2", "settings": V4}), encoding="utf-8")
        (drive / "sft_render/long_train.jsonl").write_text('{"b": 2}\n', encoding="utf-8")
        assert D.sync(D.FolderDrive(drive), rpa, "topup", log=log)["render"] == "downloaded"
        assert (data / "long_train.jsonl").read_text(encoding="utf-8") == '{"b": 2}\n'
        # full mode: the pretrained base model with its optimizer
        bd8 = drive / "nanochat_base/base_checkpoints/d8"
        bd8.mkdir(parents=True)
        for n in ("model_000500.pt", "meta_000500.json", "optim_000500_rank0.pt", "model_004577.pt",
                  "meta_004577.json", "optim_004577_rank0.pt"):
            (bd8 / n).write_text("{}", encoding="utf-8")
        assert D.Sync(D.FolderDrive(drive), rpa, log).base_model() == 4577
        got_b = sorted(p.name for p in (rpa / "train_data/base/base_checkpoints/d8").iterdir())
        assert got_b == ["meta_004577.json", "model_004577.pt", "optim_004577_rank0.pt"], got_b
        assert P.newest(rpa / "train_data/base/base_checkpoints/d8") == 4577
        # Colab run in progress (chatsft_checkpoints empty): the newest kept model
        import shutil
        shutil.rmtree(drive / "nanochat_base/chatsft_checkpoints")
        got = D.Sync(D.FolderDrive(drive), rpa, log).sft_model()
        assert got == ("nanochat_base/chatsft_checkpoints_before_20251101_000000/d8", 2500)


def test_drive_render_must_fit_the_run():
    with tempfile.TemporaryDirectory() as t:
        drive, rpa = make_drive(Path(t) / "drive"), Path(t) / "rpa"
        logs = []
        log = lambda *x: logs.append(" ".join(str(y) for y in x))
        assert D.Sync(D.FolderDrive(drive), rpa, log).render_from_drive(seq=4096) == "downloaded"
        # a 2048 PC does not train on the 4096 render (most long/business rows would be skipped)
        assert D.Sync(D.FolderDrive(drive), rpa, log).render_from_drive(seq=2048) is None
        assert any("max_tokens 4068 (needed 2020)" in l for l in logs), logs
        # an older render: no beginner rows, other settings
        (drive / "sft_render/beginner_train.jsonl").unlink()
        assert D.Sync(D.FolderDrive(drive), rpa, log).render_from_drive(seq=4096) is None
        assert any("no beginner_train.jsonl" in l for l in logs), logs
        old = dict(V4, gold_dropout=0.0, answer_format="full")
        assert len(D.render_problems({"settings": old}, 4096, set(T.RENDER_EXTRA))) == 2
        assert D.render_problems({"settings": V4}, 4096, set(T.RENDER_EXTRA)) == []
        assert D.render_problems({}, 4096) != []


def test_drive_model_keeps_its_identity_when_moved():
    """Colab cell 7 moves chatsft_checkpoints to chatsft_checkpoints_before_<date>; Drive keeps the file id."""
    class Moved:                                                  # a Drive listing: path -> [file id]
        def __init__(self, files):
            self.files = files

        def download(self, path, dest, i=0):
            Path(dest).write_bytes(self.files[path][i].encode())
    with tempfile.TemporaryDirectory() as t:
        rpa = Path(t) / "rpa"
        tokf = {"nanochat_base/tokenizer/tokenizer.pkl": ["t1"], "nanochat_base/tokenizer/token_bytes.pt": ["t2"]}
        a = dict(tokf, **{"nanochat_base/chatsft_checkpoints/d8/model_008000.pt": ["FILEID"],
                          "nanochat_base/chatsft_checkpoints/d8/meta_008000.json": ["META"],
                          "nanochat_base/chatsft_checkpoints/d8/optim_008000_rank0.pt": ["OPT"]})
        logs = []
        log = lambda *x: logs.append(" ".join(str(y) for y in x))
        s1 = D.Sync(Moved(a), rpa, log)
        assert s1.sft_model() == ("nanochat_base/chatsft_checkpoints/d8", 8000)
        dm = rpa / "train_data/drive_model"
        assert (dm / "chatsft_checkpoints/d8/optim_008000_rank0.pt").exists()      # top-up keeps the momentum buffers
        id1 = (dm / ".drive_id").read_text(encoding="utf-8")
        # the PC's model grew from it
        here = rpa / "nanochat_base/chatsft_checkpoints/d8"
        here.mkdir(parents=True)
        (here / "model_008300.pt").write_bytes(b"x")
        (here / "meta_008300.json").write_text(json.dumps({"ooa_local": {"drive_origin": id1}}), encoding="utf-8")
        assert P.topup_start(rpa) == rpa / "nanochat_base"
        # Colab starts a new run: the same file now sits in chatsft_checkpoints_before_*
        b = dict(tokf, **{k.replace("chatsft_checkpoints/", "chatsft_checkpoints_before_20261118_010203/"): v
                          for k, v in a.items() if "chatsft" in k})
        n0 = D.Sync(Moved(b), rpa, log)
        before = n0.downloaded
        assert n0.sft_model() == ("nanochat_base/chatsft_checkpoints_before_20261118_010203/d8", 8000)
        assert n0.downloaded == before                                          # not downloaded again
        assert P.same_drive_model((dm / ".drive_id").read_text(encoding="utf-8"), id1)
        assert P.topup_start(rpa) == rpa / "nanochat_base"                      # the PC's grown model stays the start
        assert not P.same_drive_model("x/model_1.pt:A", "x/model_1.pt:B") and not P.same_drive_model(None, None)


def test_topup_starts_from_the_newest_model():
    with tempfile.TemporaryDirectory() as t:
        rpa = Path(t)
        here, drive = rpa / "nanochat_base/chatsft_checkpoints/d8", rpa / "train_data/drive_model/chatsft_checkpoints/d8"
        here.mkdir(parents=True)
        (here / "model_004000.pt").write_bytes(b"x")
        (here / "meta_004000.json").write_text("{}", encoding="utf-8")
        assert P.topup_start(rpa) == rpa / "nanochat_base"                 # no Drive model
        drive.mkdir(parents=True)
        (drive / "model_008000.pt").write_bytes(b"x")
        (drive / "meta_008000.json").write_text("{}", encoding="utf-8")
        (rpa / "train_data/drive_model/.drive_id").write_text("idA", encoding="utf-8")
        assert P.topup_start(rpa) == rpa / "train_data/drive_model"         # Drive's is newer
        (here / "meta_004000.json").write_text(json.dumps({"ooa_local": {"drive_origin": "idA"}}), encoding="utf-8")
        assert P.topup_start(rpa) == rpa / "nanochat_base"                  # the PC's grew from it: newer
        (rpa / "train_data/drive_model/.drive_id").write_text("idB", encoding="utf-8")
        assert P.topup_start(rpa) == rpa / "train_data/drive_model"         # a new Colab model came


def test_eval_sample_is_spread_and_interleaved():
    rows = E.sample_rows(HERE.parent.parent / "eval", 3)
    labels = [r[1] for r in rows]
    assert labels[:4] == ["set 1", "set 2", "set 3", "beginner"] and len(rows) == 12
    assert rows == E.sample_rows(HERE.parent.parent / "eval", 3)          # the same requests for both models
    assert len(E.sample_rows(HERE.parent.parent / "eval", 0)) == 20 + 15 + 24 + 80
    assert {r[0] for r in rows if r[1] == "beginner"} == {"beginner:0", "beginner:26", "beginner:53"}


def test_install_moves_the_old_model_and_writes_the_pack():
    with tempfile.TemporaryDirectory() as t:
        rpa = Path(t) / "rpa"
        d8 = rpa / "nanochat_base/chatsft_checkpoints/d8"
        d8.mkdir(parents=True)
        (d8 / "model_004000.pt").write_bytes(b"old")
        (d8 / "meta_004000.json").write_text("{}", encoding="utf-8")
        (rpa / "sft").mkdir()
        (rpa / "sft/catalog.jsonl").write_text("oldcat\n", encoding="utf-8")
        new, data, tok = Path(t) / "new", Path(t) / "data", Path(t) / "tok"
        for d in (new, data, tok):
            d.mkdir()
        (new / "model_000003.pt").write_bytes(b"new")
        (new / "meta_000003.json").write_text(json.dumps({"step": 3, "trained_window": 4096}), encoding="utf-8")
        (data / "catalog.jsonl").write_text("cat\n", encoding="utf-8")
        (data / "examples.jsonl").write_text("ex\n", encoding="utf-8")
        (tok / "tokenizer.pkl").write_bytes(b"t")
        (tok / "token_bytes.pt").write_bytes(b"b")
        E.install(rpa, new, 3, data, {"install_step": 4003, "drive_origin": "idA", "tokenizer": str(tok)}, "20990101_000000",
                  log=lambda *x: None)
        assert sorted(p.name for p in d8.iterdir()) == ["meta_004003.json", "model_004003.pt"]
        meta = json.loads((d8 / "meta_004003.json").read_text(encoding="utf-8"))
        assert meta["step"] == 4003 and meta["ooa_local"]["drive_origin"] == "idA" and meta["trained_window"] == 4096
        keep = rpa / "nanochat_base/old_model_20990101_000000"
        assert (keep / "model_004000.pt").read_bytes() == b"old" and (keep / "catalog.jsonl").exists()
        assert (rpa / "sft/catalog.jsonl").read_text(encoding="utf-8") == "cat\n"
        assert (rpa / "sft/examples.jsonl").exists()                          # the app then uses context prompts
        assert E.current_pack(rpa)[3] == "context"
        assert (rpa / "nanochat_base/tokenizer/tokenizer.pkl").read_bytes() == b"t"


def test_seq_default_follows_the_ram_rule():
    import types
    assert P.default_seq(True) == 4096
    real = sys.modules.get("psutil")
    try:
        for gb, want in ((8, 2048), (16, 4096)):
            sys.modules["psutil"] = types.SimpleNamespace(
                virtual_memory=lambda gb=gb: types.SimpleNamespace(total=gb * 2**30))
            assert P.default_seq(False) == want, gb
    finally:
        if real is None:
            sys.modules.pop("psutil", None)
        else:
            sys.modules["psutil"] = real


def test_notebook_generates_without_the_drive_id():
    with tempfile.TemporaryDirectory() as t:
        gen = HERE.parent / "make_local_notebook.py"
        for fid, out in (("", Path(t) / "a.ipynb"), ("FAKE_ID_123", Path(t) / "b.ipynb")):
            subprocess.run([sys.executable, str(gen), fid, str(out)], check=True, capture_output=True)
            nb = json.loads(out.read_text(encoding="utf-8"))
            src = ["".join(c["source"]) for c in nb["cells"]]
            assert f'DRIVE_FOLDER_ID = "{fid}"' in src[1]
            for s, c in zip(src, nb["cells"]):
                if c["cell_type"] == "code":
                    ast.parse(s)
            text = "\n".join(src)
            for old in ("topup_local.py", "train_long.py", "big_local.py", "curriculum_local.py", "refine_local.py"):
                assert old not in text, old
            assert "train_pc.py" in text and "eval_pc.py" in text and "data_pc" in text
            assert '"taskkill", "/T", "/F"' in text and "1.15" not in text
            assert 'get("finished")' in src[4]                     # TOPUP_HOURS sizes every new round
        shipped = json.loads((HERE.parent.parent / "ooa_local_pc.ipynb").read_text(encoding="utf-8"))
        assert 'DRIVE_FOLDER_ID = ""' in "".join(shipped["cells"][1]["source"])


def test_patch_against_pinned_nanochat():
    """train_pc.write_script on the pinned chat_sft.py (when NANOCHAT_DIR is set): window fix, mix, no-compile."""
    nano = os.environ.get("NANOCHAT_DIR")
    if not nano or not Path(nano, "scripts/chat_sft.py").exists():
        print("  (skipped: set NANOCHAT_DIR to the pinned nanochat)")
        return
    import shutil
    with tempfile.TemporaryDirectory() as t:
        n2 = Path(t) / "nanochat"
        shutil.copytree(nano, n2)
        out = P.write_script(n2, Path(t) / "data", {"long_train.jsonl": 0.5}, 0, 3, compile_ok=False)
        s = out.read_text(encoding="utf-8")
        assert "SFT attention window" in s and '"trained_window"' in s and "seed=3" in s
        assert "torch.compile switched off" in s and s.startswith("import torch as _torch_nc")
        assert P.MOMENTUM_NEW in s and P.MOMENTUM_OLD not in s                 # momentum ramp over the whole run
        assert (n2 / "tasks/ooa.py").exists()
        assert "OOADialogues" not in (n2 / "scripts/chat_sft.py").read_text(encoding="utf-8")
        compile(s, str(out), "exec")


# ---------------------------------------------------------------------------------------- Windows
RPA = HERE.parent.parent
# the code the PC notebook runs (cells 1-9 and everything they start)
PC_FILES = ["training/local/data_pc.py", "training/local/train_pc.py", "training/local/eval_pc.py",
            "training/local/setup_pc.py", "training/make_local_notebook.py", "training/sft_setup.py",
            "training/render_dataset.py", "training/data_filter.py", "training/ooa_format.py",
            "training/ingest_feedback.py", "training/ooa_task.py", "training/ooa_reward.py", "training/repair.py",
            "training/retrieval.py", "training/safety.py", "assistant/model_backend.py", "assistant/app.py",
            "assistant/constrained.py", "assistant/knowledge.py", "assistant/runner_bridge.py",
            "serving/ooa_server.py", "serving/backends.py", "eval/beginner_score.py", "eval/flow_check.py"]
KNOWN = {("serving/ooa_server.py", "Policy.load")}     # fixed in its own change (that file is being edited elsewhere)


def no_encoding_calls(src):
    """Text-mode open() / read_text() / write_text() / subprocess(text=True) without encoding=: on Windows they use
    the ANSI code page (cp1252) and crash on the UTF-8 JSONL. -> [(line, what, enclosing def)]"""
    out, tree = [], ast.parse(src)
    parents = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parents[c] = n
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        kw = {k.arg for k in n.keywords}
        f = n.func
        name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
        if "encoding" in kw:
            continue
        what = None
        if name == "open" and isinstance(f, ast.Name):
            mode = n.args[1] if len(n.args) > 1 else next((k.value for k in n.keywords if k.arg == "mode"), None)
            if not (isinstance(mode, ast.Constant) and "b" in str(mode.value)):
                what = "open"
        elif name in ("read_text", "write_text"):
            what = name
        elif name in ("run", "Popen", "check_output") and {"text", "universal_newlines"} & kw:
            what = "subprocess " + name
        if what:
            d, p = None, parents.get(n)
            while p is not None and d is None:
                d = p.name if isinstance(p, (ast.FunctionDef, ast.ClassDef)) else None
                p = parents.get(p)
            out.append((n.lineno, what, d))
    return out


def notebook_cells(fid=""):
    with tempfile.TemporaryDirectory() as t:
        out = Path(t) / "nb.ipynb"
        subprocess.run([sys.executable, str(HERE.parent / "make_local_notebook.py"), fid, str(out)], check=True,
                       capture_output=True)
        nb = json.loads(out.read_text(encoding="utf-8"))
    return [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]


def test_every_text_file_access_names_utf8():
    bad = []
    for f in PC_FILES:
        for ln, what, d in no_encoding_calls((RPA / f).read_text(encoding="utf-8")):
            if (f, f"{d}" if d != "load" else "Policy.load") not in KNOWN:
                bad.append(f"{f}:{ln} {what} in {d}")
    for i, (src, kind) in enumerate(notebook_cells()):
        if kind == "code":
            bad += [f"notebook cell {i}: line {ln} {what}" for ln, what, d in no_encoding_calls(src)]
    assert not bad, bad


def test_notebook_cells_are_windows_safe():
    cells = notebook_cells()
    code = [s for s, k in cells if k == "code"]
    for i, s in enumerate(code):
        for line in s.splitlines():
            assert not line.lstrip().startswith(("!", "%")), (i, line)   # shell lines: cmd.exe on Windows
        assert "python3" not in s and "os.system" not in s and "shell=True" not in s, i
        for n in ast.walk(ast.parse(s)):          # every command is a list of arguments
            if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) in ("run", "Popen", "stream"):
                if n.args and isinstance(n.args[0], (ast.Constant, ast.JoinedStr)):
                    raise AssertionError(f"cell {i}: command given as a string")
    # cell 1 runs on a bare Python: plain settings + the standard library only
    imported = {a.name.split(".")[0] for n in ast.walk(ast.parse(code[0])) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in (n.names if isinstance(n, ast.Import) else [type("M", (), {"name": n.module})])}
    assert imported <= {"os", "sys", "json", "shutil", "subprocess", "time", "pathlib"}, imported
    assert "__vsc_ipynb_file__" in code[0] and "sys.executable" in code[0]
    c2 = code[1]
    assert "setup_pc.ensure_nanochat(ROOT)" in c2 and "setup_pc.adopt_old(ROOT)" in c2
    assert '[sys.executable, "-m", "pip", "install"' in c2 and "gdown>=6" in c2
    assert "stream(TRAIN + [\"--check\"])" in code[3]                  # the check's output shows in the notebook
    assert "stdout=_app_log" in code[6]


def test_python_and_path_checks():
    assert SP.python_problems((3, 12, 1), True) == []
    assert len(SP.python_problems((3, 9, 0), True)) == 1 and len(SP.python_problems((3, 12, 0), False)) == 1
    assert SP.path_notes("C:/OOA/rpa model", True) == []
    assert any("OneDrive" in n for n in SP.path_notes("C:/Users/x/OneDrive/Desktop/rpa model", True))
    assert any("260" in n for n in SP.path_notes("C:/" + "x" * 130 + "/rpa model", True))
    assert SP.path_notes("/home/x/" + "y" * 200, False) == []
    assert SP.NANOCHAT_COMMIT == P.NANOCHAT_COMMIT


def fake_nanochat(d, patched=False):
    d = Path(d)
    for f in ("nanochat/engine.py", "nanochat/gpt.py", "scripts/chat_sft.py", "tasks/common.py"):
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_text("# OOADialogues\n" if patched and f.endswith("chat_sft.py") else "x = 1\n", encoding="utf-8")
    return d


def test_nanochat_without_git_comes_from_the_zip():
    """No Git (most Windows PCs): GitHub's zip of the pinned commit, unpacked to rpa/nanochat with its commit noted."""
    import zipfile
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        src = fake_nanochat(t / "src")
        zf = t / "nc.zip"
        with zipfile.ZipFile(zf, "w") as z:
            for f in src.rglob("*"):
                if f.is_file():
                    z.write(f, f"nanochat-{SP.NANOCHAT_COMMIT}/" + f.relative_to(src).as_posix())
        rpa = t / "new" / "rpa model"
        rpa.mkdir(parents=True)
        (rpa / "nanochat").mkdir()                                     # a half-finished earlier try
        logs = []
        how = SP.ensure_nanochat(rpa, log=logs.append, olds=[], zip_url=str(zf), use_git=False)
        assert how == "downloaded the zip", how
        assert SP.nanochat_ok(rpa / "nanochat") and SP.nanochat_commit(rpa / "nanochat") == SP.NANOCHAT_COMMIT
        assert not list(rpa.glob("*.zip")) and not (rpa / "nanochat.unzip").exists()
        assert SP.ensure_nanochat(rpa, log=logs.append, olds=[], zip_url="unused", use_git=False) == "already here"
        # git missing: train_pc reads the script from the folder (no crash on the missing git program)
        (rpa / "nanochat" / ".git").mkdir()
        real = subprocess.run
        try:
            def no_git(*a, **k):
                raise FileNotFoundError("git")
            subprocess.run = no_git
            assert P.pinned_chat_sft(rpa / "nanochat") == "x = 1\n"
        finally:
            subprocess.run = real


def test_update_zip_takes_what_it_lacks_from_the_old_folder():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        old, new = t / "rpa model OLD", t / "rpa model"
        fake_nanochat(old / "nanochat")
        for f in ("platform/contracts/flow-schema/flow.v1.json", "nanochat_base/chatsft_checkpoints/d8/model_000001.pt",
                  "nanochat_base/old_model_1/model_000000.pt", "sft/catalog.jsonl",
                  "assistant/assistant_data/feedback.jsonl", "assistant/app.py", "train_data/sft_render/x.jsonl"):
            (old / f).parent.mkdir(parents=True, exist_ok=True)
            (old / f).write_text("é\n", encoding="utf-8")
        (new / "assistant").mkdir(parents=True)
        (new / "assistant" / "app.py").write_text("new", encoding="utf-8")
        (new / "sft").mkdir()
        (new / "sft" / "catalog.jsonl").write_text("mine", encoding="utf-8")
        assert SP.old_folders(new) == [old]
        done = SP.adopt_old(new, log=lambda *a: None)
        assert set(done) == {"platform", "nanochat_base", "train_data", "assistant/assistant_data"}, done
        assert SP.platform_ok(new) and (new / "sft" / "catalog.jsonl").read_text(encoding="utf-8") == "mine"
        assert (new / "nanochat_base/chatsft_checkpoints/d8/model_000001.pt").exists()
        assert not (new / "nanochat_base" / "old_model_1").exists()            # backups stay behind
        assert (old / "nanochat_base/chatsft_checkpoints/d8/model_000001.pt").exists()   # copied, not moved
        assert SP.ensure_nanochat(new, log=lambda *a: None, use_git=False).startswith("copied from")
        assert SP.nanochat_ok(new / "nanochat")
        # an old nanochat whose chat_sft.py an older tool patched in place is not copied
        shutil.rmtree(new / "nanochat")
        fake_nanochat(old / "nanochat", patched=True)
        try:
            SP.ensure_nanochat(new, log=lambda *a: None, use_git=False, zip_url=str(t / "missing.zip"))
            raise AssertionError("expected the download to be tried")
        except SystemExit as ex:
            assert "could not download nanochat" in str(ex)


def test_windows_runs_list_commands_and_shows_child_output():
    """data_pc passes a child's output on line by line (Windows notebooks do not show it otherwise) as UTF-8."""
    lines = []
    rc = D.run_step([sys.executable, "-c", "print('caf\\u00e9 \\u2192 ok'); import sys; sys.exit(3)"], lines.append)
    assert rc == 3 and lines == ["café → ok"], lines
    r = D.run_quiet([sys.executable, "-c", "print('\\u00fc')"])
    assert r.stdout.strip() == "ü"


def test_calculator_works_without_sigalrm():
    """Windows has no signal.SIGALRM: nanochat's calculator (engine.eval_with_timeout) must not crash an answer."""
    import signal
    import types
    sys.path.insert(0, str(RPA / "assistant"))
    import model_backend as MB
    fake = types.ModuleType("nanochat.engine")
    fake.eval_with_timeout = lambda formula, max_time=3: signal.SIGALRM     # what nanochat does first
    pkg = types.ModuleType("nanochat")
    pkg.engine = fake
    saved = {k: sys.modules.get(k) for k in ("nanochat", "nanochat.engine")}
    had = hasattr(signal, "SIGALRM")
    alarm = getattr(signal, "SIGALRM", None)
    try:
        sys.modules.update({"nanochat": pkg, "nanochat.engine": fake})
        if had:
            del signal.SIGALRM
        MB._calculator_without_alarm()
        assert fake.eval_with_timeout("2+3") == 5 and fake.eval_with_timeout("open('x')") is None
    finally:
        if had:
            signal.SIGALRM = alarm
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as ex:
                fails += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, ex)
    print("all passed" if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)


def test_copy_tree_when_windows_refuses_the_rename(tmp_path, monkeypatch):
    """Windows can refuse renaming a just-copied folder (antivirus / editor watching it): copy straight into place."""
    import setup_pc
    src = tmp_path / "old" / "platform"
    (src / "docs").mkdir(parents=True)
    (src / "docs" / "a.md").write_text("x", encoding="utf-8")
    (src / "README.md").write_text("y", encoding="utf-8")
    dst = tmp_path / "new" / "platform"
    dst.parent.mkdir()

    def refuse(a, b):
        raise PermissionError(5, "Access is denied", str(a))
    monkeypatch.setattr(setup_pc.os, "replace", refuse)
    monkeypatch.setattr(setup_pc.time, "sleep", lambda s: None)
    setup_pc.copy_tree(src, dst, direct=False)
    assert (dst / "docs" / "a.md").read_text(encoding="utf-8") == "x"
    assert (dst / "README.md").exists()
    assert not (tmp_path / "new" / "platform.copying").exists()


def test_windows_copy_is_direct_and_resumes(tmp_path):
    """Windows: copied straight into place; a copy stopped half-way (marker still there) is finished on the next run,
    and a leftover .copying folder that cannot be removed does not stop it."""
    import setup_pc
    old = tmp_path / "rpa model OLD"
    (old / "platform" / "docs").mkdir(parents=True)
    (old / "platform" / "docs" / "a.md").write_text("x", encoding="utf-8")
    (old / "assistant").mkdir()
    (old / "assistant" / "app.py").write_text("", encoding="utf-8")
    new = tmp_path / "rpa model"
    (new / "platform").mkdir(parents=True)                         # stopped half-way: marker, one file missing
    (new / "platform" / setup_pc.COPYING_MARK).write_text("", encoding="utf-8")
    (new / "platform.copying").mkdir()                              # leftover from the old version
    orig = setup_pc.copy_tree
    setup_pc.copy_tree = lambda s, d: orig(s, d, direct=True)
    try:
        done = setup_pc.adopt_old(new, log=lambda *a: None, olds=[old])
    finally:
        setup_pc.copy_tree = orig
    assert "platform" in done
    assert (new / "platform" / "docs" / "a.md").exists()
    assert not (new / "platform" / setup_pc.COPYING_MARK).exists()
    assert setup_pc.adopt_old(new, log=lambda *a: None, olds=[old]) == [] or "platform" not in setup_pc.adopt_old(new, log=lambda *a: None, olds=[old])


# ---------------------------------------------------------------------------------------- refine (GPU, after Colab)
def test_auto_mode_choice():
    with tempfile.TemporaryDirectory() as t:
        rpa = Path(t)
        drive = {"nanochat_base/chatsft_checkpoints/d8/model_012000.pt": ["a"],
                 "nanochat_base/chatsft_checkpoints/d8/meta_012000.json": ["b"]}
        assert D.choose_mode("auto", False, drive, rpa)[0] == "topup"                 # no GPU: a CPU top-up
        mode, why = D.choose_mode("auto", True, drive, rpa)
        assert mode == "refine" and "12000" in why and "full" in why                 # GPU + Colab model: refine it
        kept = {k.replace("chatsft_checkpoints/", "chatsft_checkpoints_before_20261120_010203/"): v
                for k, v in drive.items()}
        assert D.choose_mode("auto", True, kept, rpa)[0] == "refine"                  # a kept one counts too
        no_meta = {"nanochat_base/chatsft_checkpoints/d8/model_012000.pt": ["a"]}      # an upload in progress
        assert D.choose_mode("auto", True, no_meta, rpa)[0] == "full"
        assert D.choose_mode("auto", True, {}, rpa)[0] == "full"                      # nothing trained anywhere
        here = rpa / "nanochat_base/chatsft_checkpoints/d8"                           # ... but this PC has one
        here.mkdir(parents=True)
        (here / "model_008300.pt").write_bytes(b"x")
        (here / "meta_008300.json").write_text("{}", encoding="utf-8")
        assert D.choose_mode("auto", True, {}, rpa)[0] == "refine"
        for m in ("full", "topup", "refine", "install", "rl"):                        # an explicit MODE is kept
            assert D.choose_mode(m, True, drive, rpa)[0] == m


REFINE_STATS = dict(STATS, **{"v5_train.jsonl": (1555, 1450), "session_train.jsonl": (733, 1500),
                              "synth_train.jsonl": (28380, 1290), "natural_train.jsonl": (5484, 1050)})


def test_refine_plan_shares():
    assert abs(sum(T.REFINE_SHARES.values()) + T.REFINE_REPLAY - 1.0) < 1e-9
    assert 0.15 <= T.REFINE_REPLAY <= 0.20
    for n in ("v5_train.jsonl", "session_train.jsonl", "beginner_train.jsonl", "long_train.jsonl",
              "business2_train.jsonl", "clarify_train.jsonl", "targeted_train.jsonl", "fix_train.jsonl",
              "feedback_train.jsonl"):
        assert n in T.REFINE_SHARES, n
    b = T.budget(P.REFINE_STEPS, P.TOKENS_PER_STEP)
    v = T.plan_topup(REFINE_STATS, b, T.REFINE_SHARES, T.REFINE_REPLAY)
    tok = {n: v[n] * REFINE_STATS[n][0] * REFINE_STATS[n][1] for n in v}
    total = sum(tok.values())
    assert abs(total - b) / b < 0.02, (total, b)
    replay = sum(t_ for n, t_ in tok.items() if n not in T.REFINE_SHARES) / total
    assert 0.15 <= replay <= 0.21, replay                                  # ~18.6% (no feedback file here)
    assert all(v[n] > 0 for n in v)                                        # every file stays in the mix
    assert v["v5_train.jsonl"] >= 2 and v["session_train.jsonl"] >= 2     # the newest rows are seen several times
    assert tok["v5_train.jsonl"] / total > 0.12 and tok["session_train.jsonl"] / total > 0.06
    full = T.plan_mixture(REFINE_STATS, b)
    assert v["synth_train.jsonl"] < full["synth_pattern.jsonl"] + full["synth_train.jsonl"]
    with tempfile.TemporaryDirectory() as d:                               # plan_for_local takes the refine table
        for n in ("v5_train.jsonl", "session_train.jsonl", "synth_train.jsonl", "beginner_train.jsonl"):
            Path(d, n).write_text((json.dumps({"messages": [{"role": "user", "content": "x" * 900}]}) + "\n") * 400,
                                  encoding="utf-8")
        Path(d, "val.jsonl").write_text("{}\n", encoding="utf-8")
        vr, _, _, _ = T.plan_for_local(d, "refine", 10, P.TOKENS_PER_STEP)
        vt, _, _, _ = T.plan_for_local(d, "topup", 10, P.TOKENS_PER_STEP)
        assert vr["session_train.jsonl"] > vt["session_train.jsonl"] and vr != vt


def fake_train_setup(rpa):
    """The files train_pc.py --check looks for (a tiny rendered data folder, nanochat, a Drive model + tokenizer)."""
    rpa = Path(rpa)
    data = rpa / "train_data/sft_render"
    data.mkdir(parents=True)
    for n in ("val.jsonl", "catalog.jsonl", "examples.jsonl", "v5_train.jsonl", "session_train.jsonl",
              "synth_train.jsonl"):
        (data / n).write_text(json.dumps({"messages": [{"role": "user", "content": "x" * 900}]}) + "\n",
                              encoding="utf-8")
    (data / "render_report.json").write_text(json.dumps({"settings": V4}), encoding="utf-8")
    for f in ("scripts/chat_sft.py", "nanochat/gpt.py"):
        (rpa / "nanochat" / f).parent.mkdir(parents=True, exist_ok=True)
        (rpa / "nanochat" / f).write_text("#\n", encoding="utf-8")
    dm = rpa / "train_data/drive_model"
    (dm / "chatsft_checkpoints/d8").mkdir(parents=True)
    (dm / "chatsft_checkpoints/d8/model_012000.pt").write_bytes(b"m")
    (dm / "chatsft_checkpoints/d8/meta_012000.json").write_text("{}", encoding="utf-8")
    (dm / "tokenizer").mkdir()
    for f in ("tokenizer.pkl", "token_bytes.pt"):
        (dm / "tokenizer" / f).write_bytes(b"t")
    (dm / ".drive_id").write_text("nanochat_base/chatsft_checkpoints/d8/model_012000.pt:V5ID", encoding="utf-8")
    return data


def test_refine_check_and_state_dir_separation(monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as t:
        rpa = Path(t)
        data = fake_train_setup(rpa)
        # a CPU top-up round in progress must not be taken for the refine run (and the other way round)
        top = rpa / "local_run/topup"
        top.mkdir(parents=True)
        (top / "state.json").write_text(json.dumps({"mode": "topup", "steps": 300, "chunks": 6, "seq": 4096, "done": 3,
                                                    "start": str(rpa / "nanochat_base"), "start_step": 8300, "lr": 0.3,
                                                    "device_batch": 1}), encoding="utf-8")
        monkeypatch.setattr(P, "device_info", lambda: ("cuda", "Fake T4", 15.0, (7, 5)))
        assert P.main(["--rpa", str(rpa), "--mode", "refine", "--data", str(data), "--check"]) == 0
        out = capsys.readouterr().out
        assert "mode             : refine" in out and "drive_model" in out and "step 12000" in out
        assert "Drive's newest trained model" in out
        assert f"{P.REFINE_STEPS} steps" in out and f"in {P.REFINE_STEPS // P.REFINE_CHUNK} chunk(s)" in out
        assert "0 done" in out and f"lr {P.REFINE_LR}" in out and "continuing the run in progress" not in out
        assert "session_train.jsonl" in out and "SmolTalk rows 0" in out
        assert json.loads((top / "state.json").read_text(encoding="utf-8"))["done"] == 3       # untouched
        assert not (rpa / "local_run/refine").exists()                                          # --check writes nothing
        # without a GPU refine is refused (it is sized for one); --allow-cpu is for tests only
        monkeypatch.setattr(P, "device_info", lambda: ("cpu", "4 CPU threads", 0.0, None))
        try:
            P.main(["--rpa", str(rpa), "--mode", "refine", "--data", str(data), "--check"])
            raise AssertionError("refine on a CPU must be refused")
        except SystemExit:
            pass
        assert "use --mode topup" in capsys.readouterr().out
        assert P.main(["--rpa", str(rpa), "--mode", "refine", "--data", str(data), "--check", "--allow-cpu"]) == 0
    assert P.lr_for_chunk(0, 6, P.REFINE_LR)[0] == P.REFINE_LR and 0.3 <= P.REFINE_LR <= 0.4


def test_chunks_pass_colabs_base_learning_rates(monkeypatch):
    """A saved SFT checkpoint stores the lrs as None, so chat_sft would fall back to unembedding 0.004 (half of the
    0.008 Colab's run inherited from the base model): every chunk passes the base rates explicitly."""
    with tempfile.TemporaryDirectory() as t:
        rpa = Path(t)
        data = fake_train_setup(rpa)
        assert P.base_lrs(rpa) == {"embedding_lr": 0.3, "unembedding_lr": 0.008, "matrix_lr": 0.02}
        cmds = []

        def fake_chunk(cmd, *a, **k):
            cmds.append(cmd)
            raise KeyboardInterrupt()
        monkeypatch.setattr(P, "device_info", lambda: ("cuda", "Fake T4", 15.0, (7, 5)))
        monkeypatch.setattr(P, "run_chunk", fake_chunk)
        monkeypatch.setattr(P, "write_script", lambda *a, **k: None)
        monkeypatch.setattr(P, "kill_leftover", lambda *a, **k: None)
        try:
            P.main(["--rpa", str(rpa), "--mode", "refine", "--data", str(data)])
        except KeyboardInterrupt:
            pass
        assert cmds and "--unembedding-lr=0.008" in cmds[0] and "--embedding-lr=0.3" in cmds[0] \
            and "--matrix-lr=0.02" in cmds[0], cmds
        st = json.loads((rpa / "local_run/refine/state.json").read_text(encoding="utf-8"))
        assert st["base_lrs"]["unembedding_lr"] == 0.008
        # the base model's own record wins when it is on the PC
        b = rpa / "train_data/base/base_checkpoints/d8"
        b.mkdir(parents=True)
        (b / "model_005000.pt").write_bytes(b"m")
        (b / "meta_005000.json").write_text(json.dumps({"user_config": {"unembedding_lr": 0.006}}), encoding="utf-8")
        assert P.base_lrs(rpa)["unembedding_lr"] == 0.006 and P.base_lrs(rpa)["matrix_lr"] == 0.02


def session_rows(n):
    rows = (RPA / "data/extra_sft/session_train.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 733
    return "\n".join(rows[:n]) + "\n"


def test_data_pc_renders_a_missing_extra_file():
    """Drive's Colab render lacks session_train.jsonl (made after the run): refine renders only that file here, from
    the repo (data/extra_sft) when Drive sft/ does not have it, with the Colab render's catalog and settings; the full
    run does not take it."""
    with tempfile.TemporaryDirectory() as t:
        drive, rpa = make_drive(Path(t) / "drive"), Path(t) / "rpa"
        shutil.copyfile(RPA / "data/catalog/catalog.jsonl", drive / "sft_render/catalog.jsonl")
        (drive / "sft_render/examples.jsonl").write_text(json.dumps(
            {"request": "copy a file to a folder", "outline": "1. copy", "actions": ["ooa.file/copy"], "sig": "s"})
            + "\n", encoding="utf-8")
        (drive / "sft_render/session_train.jsonl").unlink()
        (rpa / "data/extra_sft").mkdir(parents=True)
        (rpa / "data/extra_sft/session_train.jsonl").write_text(session_rows(12), encoding="utf-8")
        logs = []
        log = lambda *x: logs.append(" ".join(str(y) for y in x))
        out = D.sync(D.FolderDrive(drive), rpa, "refine", log=log)
        data = Path(out["data"])
        assert out["render"] == "downloaded" and out["rendered_here"] == ["session_train.jsonl"], (out, logs)
        rows = [json.loads(l) for l in (data / "session_train.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) >= 10 and all(r["messages"][-1]["role"] == "assistant" for r in rows)
        assert any("Plan:" in r["messages"][-1]["content"] for r in rows)          # the plan-first format
        rep = json.loads((data / "render_report.json").read_text(encoding="utf-8"))
        assert rep["files"]["session_train.jsonl"]["rendered_here"] and rep["local_extra"]
        assert rep["settings"] == V4                                               # Colab's settings stay the record
        ex = (data / "examples.jsonl").read_text(encoding="utf-8").splitlines()
        assert json.loads(ex[0])["request"] == "copy a file to a folder" and len(ex) > 1   # Colab's first, then new
        assert (data / "catalog.jsonl").read_bytes() == (drive / "sft_render/catalog.jsonl").read_bytes()
        assert any("from the repo" in l for l in logs), logs
        assert T.file_stats(str(data)).get("session_train.jsonl")                  # in the training mixture
        # again: nothing to do; a changed source file: rendered again from Colab's example index
        out2 = D.sync(D.FolderDrive(drive), rpa, "refine", log=log)
        assert out2["render"] == "up to date" and any("rendered here earlier" in l for l in logs)
        (rpa / "data/extra_sft/session_train.jsonl").write_text(session_rows(5), encoding="utf-8")
        D.sync(D.FolderDrive(drive), rpa, "refine", log=log)
        ex2 = (data / "examples.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(ex2) < len(ex) and json.loads(ex2[0])["request"] == "copy a file to a folder"
        # Drive sft/ has it (uploaded): Drive's copy is used
        (drive / "sft").mkdir()
        (drive / "sft/session_train.jsonl").write_text(session_rows(8), encoding="utf-8")
        logs.clear()
        D.sync(D.FolderDrive(drive), rpa, "refine", log=log)
        assert any("from Drive sft/" in l for l in logs), logs
        # full: exactly the Colab data, so Drive's incomplete render is not used and nothing is added from the repo
        assert D.Sync(D.FolderDrive(drive), rpa, log).render_from_drive(seq=4096) is None
        assert D.extra_sources(rpa, repo_extra=False) == {"session_train.jsonl": rpa / "train_data/sft/session_train.jsonl"}
        (rpa / "train_data/sft/session_train.jsonl").unlink()
        assert D.extra_sources(rpa, repo_extra=False) == {}
        assert list(D.extra_sources(rpa, repo_extra=True)) == ["session_train.jsonl"]


def test_eval_compares_with_the_start_model_not_an_older_installed_one():
    with tempfile.TemporaryDirectory() as t:
        rpa = Path(t)
        data = fake_train_setup(rpa)
        tok = rpa / "train_data/drive_model/tokenizer"
        st = {"mode": "refine", "start": str(rpa / "train_data/drive_model"), "start_step": 12000}
        sp = E.start_pack(rpa, st, data, tok)
        assert sp is not None and sp[4] == 12000 and sp[3] == "context" and sp[1] == data / "catalog.jsonl"
        # a run that started from the installed model: the installed one is the reference
        assert E.start_pack(rpa, dict(st, start=str(rpa / "nanochat_base")), data, tok) is None
        assert E.start_pack(rpa, dict(st, mode="full"), data, tok) is None
        assert E.start_pack(rpa, {}, data, tok) is None


def test_rl_starts_from_the_newest_model():
    src = (HERE / "rl_pc.py").read_text(encoding="utf-8")
    assert 'src_root = Path(a.start).resolve() if a.start else topup_start(rpa)' in src   # never an old run's start
    assert "local_origin(rpa) if src_root == rpa / \"nanochat_base\"" in src                # keeps the Drive origin


def test_notebook_refine_mode():
    cells = notebook_cells()
    text = "\n".join(s for s, _ in cells)
    code = [s for s, k in cells if k == "code"]
    assert 'REFINE_STEPS = 1500' in code[0] and "REFINE_THEN_RL = False" in code[0] and 'MODE = "auto"' in code[0]
    assert 'data_pc.choose_mode("auto", CUDA, DRIVE.files, ROOT)' in code[2]
    assert '"--steps", str(REFINE_STEPS), "--chunks", str(REFINE_CHUNKS)' in code[3]
    assert '"rl_pc.py"' in code[5] and "REFINE_THEN_RL" in code[5]
    assert "After a Colab run" in text and '`"refine"`' in text


def test_notebook_rerun_of_cell_6_continues_rl_without_testing_the_refine_again():
    """REFINE_THEN_RL: after RL stops, cell 6 runs again; it must not score the finished refine again (up to 2 x
    EVAL_MINUTES), and cell 5 must not start a new refine while that RL run is going."""
    code = [s for s, k in notebook_cells() if k == "code"]
    fn = next(n for n in ast.parse(code[3]).body if isinstance(n, ast.FunctionDef) and n.name == "_refine_states")
    assert "_refine_states()[1] == \"going\"" in code[4]
    assert "_ref_tested and not RESTART_TRAINING" in code[5] and code[5].index("_ref_tested and") < \
        code[5].index("rc = stream(EVAL + _EV_OPTS)")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        g = {"ROOT": root, "json": json}
        exec(compile(ast.Module([fn], []), "cell4", "exec"), g)
        rs, ls = root / "local_run/refine/state.json", root / "local_run/rl/state.json"
        assert g["_refine_states"]() == (False, None)
        ls.parent.mkdir(parents=True)
        ls.write_text(json.dumps({"done": 0, "finished": "old"}), encoding="utf-8")   # an RL run from before
        rs.parent.mkdir(parents=True)
        rs.write_text(json.dumps({"result": "x"}), encoding="utf-8")
        assert g["_refine_states"]() == (False, None)                                   # refine not tested yet
        rs.write_text(json.dumps({"result": "x", "finished": "now"}), encoding="utf-8")
        os.utime(ls, (1, 1))
        assert g["_refine_states"]() == (True, None)                                    # old RL does not count
        ls.write_text(json.dumps({"done": 0}), encoding="utf-8")
        assert g["_refine_states"]() == (True, "going")
        ls.write_text(json.dumps({"done": 1, "finished": "later"}), encoding="utf-8")
        assert g["_refine_states"]() == (True, "done")


def test_notebook_generator_refuses_options():
    r = subprocess.run([sys.executable, str(HERE.parent / "make_local_notebook.py"), "--help"], capture_output=True,
                       text=True)
    assert r.returncode != 0
    shipped = json.loads((HERE.parent.parent / "ooa_local_pc.ipynb").read_text(encoding="utf-8"))
    assert 'DRIVE_FOLDER_ID = ""' in "".join(shipped["cells"][1]["source"])


def test_eval_gate_after_a_refine(monkeypatch, capsys):
    """Refine of Drive's v5 while the PC still has an older model: installed only if it beats v5 (and the old one);
    when it does not, v5 itself is installed if it beats the old one."""
    def setup(t):
        rpa = Path(t)
        data = fake_train_setup(rpa)
        (rpa / "eval").mkdir()
        (rpa / "eval/plain_requests.jsonl").write_text("".join(json.dumps({"prompt": f"p{i}", "must": []}) + "\n"
                                                               for i in range(4)), encoding="utf-8")
        old = rpa / "nanochat_base/chatsft_checkpoints/d8"                       # the PC's older model (v4)
        old.mkdir(parents=True)
        (old / "model_008300.pt").write_bytes(b"v4")
        (old / "meta_008300.json").write_text("{}", encoding="utf-8")
        new = rpa / "local_run/refine/chunks/chunk_005"
        new.mkdir(parents=True)
        (new / "model_001500.pt").write_bytes(b"refined")
        (new / "meta_001500.json").write_text("{}", encoding="utf-8")
        st = rpa / "local_run/refine/state.json"
        st.write_text(json.dumps({"mode": "refine", "start": str(rpa / "train_data/drive_model"), "start_step": 12000,
                                  "steps": 1500, "install_step": 13500, "result": str(new), "data": str(data),
                                  "tokenizer": str(rpa / "train_data/drive_model/tokenizer"),
                                  "drive_origin": "nanochat_base/chatsft_checkpoints/d8/model_012000.pt:V5ID"}),
                      encoding="utf-8")
        return rpa, st

    def fake_scores(right):
        def run(rpa, name, *args, **kw):
            n = right[name]
            return {f"set 1:{i}": {"set": "set 1", "right": i < n, "verified": i < n} for i in range(4)}
        return run
    for scores, want in (({"installed model": 1, "start model": 2, "new model": 3}, "new"),
                         ({"installed model": 1, "start model": 3, "new model": 2}, "v5"),
                         ({"installed model": 3, "start model": 2, "new model": 2}, None)):
        with tempfile.TemporaryDirectory() as t:
            rpa, st = setup(t)
            monkeypatch.setattr(E, "run_scoring", fake_scores(scores))
            E.main(["--rpa", str(rpa), "--run-state", str(st), "--install", "--per-set", "0"])
            out = capsys.readouterr().out
            assert "compared with: the start model" in out, out
            d8 = rpa / "nanochat_base/chatsft_checkpoints/d8"
            got = sorted(p.name for p in d8.iterdir())
            if want == "new":
                assert got == ["meta_013500.json", "model_013500.pt"], got
                assert (d8 / "model_013500.pt").read_bytes() == b"refined"
                meta = json.loads((d8 / "meta_013500.json").read_text(encoding="utf-8"))
                assert meta["ooa_local"]["drive_origin"].endswith(":V5ID")       # the next refine / RL starts here
                assert P.topup_start(rpa) == rpa / "nanochat_base"
            elif want == "v5":
                assert got == ["meta_012000.json", "model_012000.pt"], got
                assert "installing it instead" in out
                assert P.topup_start(rpa) == rpa / "nanochat_base"
            else:
                assert got == ["meta_008300.json", "model_008300.pt"], got
            assert json.loads(st.read_text(encoding="utf-8"))["finished"]
