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
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import data_pc as D      # noqa: E402
import eval_pc as E      # noqa: E402
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
    m = re.search(r"NEW_FILES = (\{.*?\})   # beginner", COLAB, re.S)
    names = set(re.findall(r'"(\w+_train\.jsonl)"', m.group(1)))
    assert names == set(T.RENDER_EXTRA), (names, T.RENDER_EXTRA)
    assert f'ANSWER_FORMAT = "{D.ANSWER_FORMAT}"' in COLAB and f"GOLD_DROPOUT = {D.GOLD_DROPOUT}" in COLAB
    assert "ROW_CHARS = 6500 * SFT_SEQ // 2048" in COLAB and D.row_chars(4096) == 13000
    assert "ROW_TOKENS = SFT_SEQ - 28" in COLAB and D.row_tokens(4096) == 4068
    assert f"--cap {D.FILTER_CAP}" in COLAB
    assert f'NANOCHAT_COMMIT = "{P.NANOCHAT_COMMIT}"' in COLAB
    assert "SFT_ITERS = 400 if PROFILE == \"smoke\" else 32000" in COLAB and P.FULL_STEPS == 32000 // 4
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
