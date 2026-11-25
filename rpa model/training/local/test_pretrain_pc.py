"""Tests for the clean base model (v6): base_setup.py (profiles, estimates, the base_train patch, the tokenizer card,
the pack), pretrain_pc.py, the tokenizer pairing of a model and its pack (model_backend.tokenizer_problem, train_pc,
eval_pc install), data_pc's v6 downloads and the notebooks' v6 settings. The tests that run nanochat (the NaN guard,
the patch against the pinned scripts) need NANOCHAT_DIR (the pinned nanochat) and torch; the others need neither.
Run: python test_pretrain_pc.py   (or pytest)"""
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RPA = HERE.parent.parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "pretrain"), str(RPA / "assistant")]
import base_setup as B          # noqa: E402
import data_pc as D             # noqa: E402
import eval_pc as E             # noqa: E402
import model_backend as MB      # noqa: E402
import pretrain_data as PD      # noqa: E402
import pretrain_pc as PP        # noqa: E402
import sft_setup as T           # noqa: E402
import train_pc as TP           # noqa: E402

NANO = os.environ.get("NANOCHAT_DIR")


# ------------------------------------------------------------------------------------------- profiles / estimates
def test_profiles_match_nanochat_rules():
    pc = B.param_counts(8)
    assert round(pc["total"] / 1e6, 1) == 75.5 and round(pc["scaling"] / 1e6, 1) == 33.6
    assert round(B.param_counts(10)["total"] / 1e6, 1) == 122.6
    target, it, tokens = B.horizon(B.PROFILES["pc"])
    assert it == 5760 and abs(tokens - 1.51e9) < 0.01e9
    assert B.horizon(B.PROFILES["t4"])[1] == it                       # Colab trains the same model
    for name in ("pc", "t4"):                                         # nanochat's own batch for that horizon
        p = B.PROFILES[name]
        assert B.auto_batch(target, p["ratio"]) == p["total_batch"]
        assert p["total_batch"] % (p["dbs"] * p["seq"]) == 0
    est = B.estimate(B.PROFILES["pc"])
    assert 9 < est["hours"] < 13 and est["tokens_per_param"] == 20.0 and 4 < est["disk_gb"] < 7
    assert B.train_chars_for(B.PROFILES["pc"]) >= tokens * B.CHARS_PER_TOKEN_MIX
    args = B.base_train_args(B.PROFILES["pc"], resume_step=400)
    assert "--target-param-data-ratio=45" in args and "--resume-from-step=400" in args and "--window-pattern=L" in args
    assert "--model-tag=d8" in args and "--head-dim=64" in args
    assert B.device_batch(8, 2048) == 2 and B.device_batch(15, 2048) == 4 and B.device_batch(8, 4096) == 1


def test_resolve_profile_keeps_tokens_per_step():
    class A:
        profile, depth, seq, ratio, iterations, save_every, device_batch = "pc", 0, 0, 0, 0, 0, 0
    p = PP.resolve_profile(A, 6, True)
    assert p["dbs"] == 1 and p["total_batch"] == 262144
    A.seq = 4096
    p = PP.resolve_profile(A, 8, True)
    assert p["dbs"] == 1 and p["total_batch"] % (p["dbs"] * 4096) == 0


# ------------------------------------------------------------------------------------------- the base_train patch
def pinned_base_train():
    return PP.pinned_script(NANO, "scripts/base_train.py")


def test_patch_base_train_against_pinned_nanochat():
    if not NANO:
        print("  (skipped: set NANOCHAT_DIR to the pinned nanochat)")
        return
    s = pinned_base_train()
    assert B.missing_anchors(s) == []
    out = B.patch_base_train(s)
    compile(out, "base_train_ooa.py", "exec")
    for k in ("_ooa_stop(", "OOA_KEEP_CHECKPOINTS", '"tokenizer_sha1": _ooa_tok_sha1', "_ooa_nf = [n for n",
              "_ooa_nan_steps = 5 if scaler is not None else 1"):
        assert k in out, k
    # the weights are checked before the checkpoint is written, the pruning runs after it
    assert out.index("_ooa_nf = [n for n") < out.index("        save_checkpoint(\n") < out.index("_ooa_keep = int(")
    nc = B.patch_base_train(s, compile_ok=False)
    compile(nc, "base_train_ooa.py", "exec")
    assert "model = torch.compile(model" not in nc and nc.startswith("import torch as _torch_nc")
    try:
        B.patch_base_train(s.replace(B.LOSS_ANCHOR, ""))
        raise AssertionError("another version must be refused")
    except SystemExit as ex:
        assert "loss" in str(ex)


def test_sft_patch_records_the_tokenizer():
    if not NANO:
        print("  (skipped: set NANOCHAT_DIR to the pinned nanochat)")
        return
    s = TP.pinned_chat_sft(NANO)
    out = T.patch_chat_sft(s, "/data", {"a_train.jsonl": 1.0})
    compile(out, "chat_sft_ooa.py", "exec")
    assert '"tokenizer_sha1": _ooa_tok_sha1()' in out and "def _ooa_tok_sha1" in out


def tiny_base(root):
    """A nanochat base folder with 2 tiny shards of made-up text and a 300-token tokenizer (tok_train)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    rnd = random.Random(7)
    words = ["flow", "step", "read", "the", "file", "and", "send", "mail", "to", "team", "sheet", "copy", "rows",
             "if", "then", "loop", "each", "invoice", "report", "folder", '{"action":', '"ooa.excel/read"}', "data"]
    d = Path(root) / "base_data_climbmix"
    d.mkdir(parents=True)
    for name, n in (("shard_00000.parquet", 600), ("shard_99999.parquet", 120)):
        docs = [" ".join(rnd.choice(words) for _ in range(rnd.randint(20, 120))) for _ in range(n)]
        pq.write_table(pa.table({"text": docs}), d / name, row_group_size=100)
    env = dict(os.environ, NANOCHAT_BASE_DIR=str(root), NANOCHAT_DTYPE="float32", PYTHONUTF8="1")
    r = subprocess.run([sys.executable, "-m", "scripts.tok_train", "--max-chars=200000", "--vocab-size=300"],
                       cwd=NANO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return env


def run_patched(env, text, steps=6, save_every=2, name="base_train_ooa_test"):
    (Path(NANO) / "scripts" / f"{name}.py").write_text(text, encoding="utf-8")
    p = dict(B.PROFILES["tiny"], iterations=steps, save_every=save_every, eval_every=-1)
    try:
        return subprocess.run([sys.executable, "-m", f"scripts.{name}"] + B.base_train_args(p, device_type="cpu"),
                              cwd=NANO, env=dict(env, OOA_KEEP_CHECKPOINTS="2"), capture_output=True, text=True)
    finally:
        (Path(NANO) / "scripts" / f"{name}.py").unlink(missing_ok=True)


def test_nan_guard_stops_before_anything_broken_is_saved():
    if not NANO:
        print("  (skipped: set NANOCHAT_DIR to the pinned nanochat)")
        return
    with tempfile.TemporaryDirectory() as t:
        env = tiny_base(t)
        ck = Path(t) / "base_checkpoints" / "d8"
        patched = B.patch_base_train(pinned_base_train(), compile_ok=False)
        # 1. a clean run: checkpoints at 2, 4, 6, only the newest 2 kept, each records the tokenizer
        r = run_patched(env, patched)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        assert B.checkpoint_steps(ck) == [4, 6] and B.complete_checkpoint(ck, 6)
        meta = json.loads((ck / "meta_000006.json").read_text())
        assert meta["tokenizer_sha1"] == B.tokenizer_sha1(Path(t) / "tokenizer") and meta["compute_dtype"] == "float32"
        shutil.rmtree(ck)
        # 2. the loss goes nan at step 3 (float32: the weights took that step): stopped, nothing saved after step 2
        bad = patched.replace(B.LOSS_ANCHOR, "    train_loss_f = float('nan') if step >= 3 else train_loss.item()\n")
        r = run_patched(env, bad)
        out = r.stdout + r.stderr
        assert r.returncode == B.NAN_EXIT and B.NAN_MARK in out and "step 3" in out, out[-3000:]
        assert B.checkpoint_steps(ck) == [2]
        shutil.rmtree(ck)
        # 3. broken weights with a finite loss (the guard before every save)
        bad = patched.replace(B.LOSS_ANCHOR, B.LOSS_ANCHOR + "    if step == 2:\n        "
                              "next(orig_model.parameters()).data.fill_(float('nan'))\n")
        r = run_patched(env, bad, save_every=3)
        out = r.stdout + r.stderr
        assert r.returncode == B.NAN_EXIT and "weights with nan/inf at step 3" in out, out[-3000:]
        assert B.checkpoint_steps(ck) == []


# ------------------------------------------------------------------------------------------- checkpoints / resume
def put_ckpt(d, step, model=b"m", optim=True):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"model_{step:06d}.pt").write_bytes(model)
    (d / f"meta_{step:06d}.json").write_text(json.dumps({"step": step}))
    if optim:
        (d / f"optim_{step:06d}_rank0.pt").write_bytes(b"o")


def test_resume_takes_the_newest_complete_healthy_checkpoint(monkeypatch):
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        put_ckpt(d, 200)
        put_ckpt(d, 400, model=b"nan")
        put_ckpt(d, 600, optim=False)                          # cut while it was saved
        put_ckpt(d, 800)
        (d / "optim_000800_rank0.pt").write_bytes(b"cut")      # cut while the optimizer state was written

        def nonfinite(f):
            if Path(f).read_bytes() == b"cut":
                raise EOFError("truncated")
            return []
        monkeypatch.setattr(PP, "model_health", lambda dd, s: ("broken", ["wte (nan)"]) if
                            (Path(dd) / f"model_{s:06d}.pt").read_bytes() == b"nan" else ("ok", []))
        monkeypatch.setattr(PP.T, "nonfinite_tensors", nonfinite)
        assert PP.healthy_resume_step(d, log=lambda *_: None) == 200
        assert B.checkpoint_steps(d) == [200] and not list(d.glob("*000600*")) and not list(d.glob("*000400*"))
        assert not list(d.glob("*000800*"))


def test_data_ready_reads_the_manifest():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        assert PP.data_ready(d, 10) == (False, 0)
        (d / "shard_00000.parquet").write_bytes(b"x")
        (d / "shard_99999.parquet").write_bytes(b"x")
        (d / "manifest.json").write_text(json.dumps({"shards": [{"file": "shard_00000.parquet", "web_chars": 80,
                                                                 "domain_chars": 20}],
                                                     "val": {"file": "shard_99999.parquet"}}))
        assert PP.data_ready(d, 100) == (True, 100) and PP.data_ready(d, 200)[0] is False
        (d / "shard_99999.parquet").unlink()
        assert PP.data_ready(d, 100)[0] is False


def test_repeat_scale_doubles_the_domain_stream_and_old_manifests_still_resume():
    src = {"render": {"docs": [f"r{i}" * 50 for i in range(20)]}, "code": {"docs": [f"c{i}" * 50 for i in range(5)]}}
    one, two = PD.domain_stream(src, 1), PD.domain_stream(src, 1, repeat_scale=2)
    assert len(two) == 2 * len(one)
    assert PD.SETTING_DEFAULTS["repeat_scale"] == 1 and "repeat_scale" in PD.SETTING_KEYS


def test_write_pack_never_replaces_silently():
    with tempfile.TemporaryDirectory() as t:
        work, pack = Path(t) / "work", Path(t) / "train_data" / "base_v6"
        put_ckpt(work / "base_checkpoints" / "d8", 8)
        (work / "tokenizer").mkdir(parents=True)
        (work / "tokenizer" / "tokenizer.pkl").write_bytes(b"tok")
        B.write_pack(work, 8, pack, info={"x": 1}, log=lambda *_: None)
        meta = json.loads((pack / "base_checkpoints" / "d8" / "meta_000008.json").read_text())
        assert meta["tokenizer_sha1"] == B.tokenizer_sha1(work / "tokenizer") and meta["ooa_base"]["x"] == 1
        assert (pack / "VERSION").read_text().startswith("ooa-base-v6-")
        B.write_pack(work, 8, pack, log=lambda *_: None)
        assert len(list(pack.parent.glob("base_v6_before_*"))) == 1
        (work / "tokenizer" / "tokenizer.pkl").write_bytes(b"other")      # trained with another tokenizer
        meta_f = work / "base_checkpoints" / "d8" / "meta_000008.json"
        meta_f.write_text(json.dumps({"tokenizer_sha1": "0" * 40}))
        try:
            B.write_pack(work, 8, pack, log=lambda *_: None)
            raise AssertionError("a pack with another tokenizer must be refused")
        except SystemExit:
            pass


# ------------------------------------------------------------------------------------------- tokenizer identity
def make_tok(d, data=b"tok", card=False):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    (d / "tokenizer.pkl").write_bytes(data)
    (d / "token_bytes.pt").write_bytes(b"tb")
    if card:
        (d / "tokenizer_card.json").write_text(json.dumps({"name": "v6", "sha1": B.tokenizer_sha1(d)}))
    return d


def test_a_model_is_only_paired_with_its_own_tokenizer():
    with tempfile.TemporaryDirectory() as t:
        old = make_tok(Path(t) / "old", b"web")
        new = make_tok(Path(t) / "new", b"mix", card=True)
        stale = make_tok(Path(t) / "stale", b"web")
        (stale / "tokenizer_card.json").write_text(json.dumps({"sha1": B.tokenizer_sha1(new)}))   # card of another
        v6 = {"tokenizer_sha1": B.tokenizer_sha1(new)}
        assert MB.tokenizer_problem(v6, new) is None
        assert "trained with tokenizer" in MB.tokenizer_problem(v6, old)
        assert MB.tokenizer_problem({}, old) is None and MB.tokenizer_problem({}, stale) is None
        assert "before tokenizers were recorded" in MB.tokenizer_problem({}, new)
        assert MB.tokenizer_problem({}, Path(t) / "none").startswith("no tokenizer.pkl")
        # train_pc / rl_pc: the start pack's own tokenizer; the installed one only for an older model
        rpa = Path(t) / "rpa"
        make_tok(rpa / "nanochat_base" / "tokenizer", b"web")
        src = Path(t) / "pack"
        assert TP.pick_tokenizer(rpa, src, {})[0] == rpa / "nanochat_base" / "tokenizer"
        tok, why = TP.pick_tokenizer(rpa, src, v6)
        assert tok is None and "no tokenizer" in why
        shutil.copytree(new, src / "tokenizer")
        assert TP.pick_tokenizer(rpa, src, v6)[0] == src / "tokenizer"
        assert TP.pick_tokenizer(rpa, src, {})[0] == rpa / "nanochat_base" / "tokenizer"   # an old model: not v6's
        assert B.read_card(new)["name"] == "v6" and B.read_card(stale) is None
        assert B.plan_chars_per_token(new, 2.8) == 2.8
        (new / "tokenizer_card.json").write_text(json.dumps({"sha1": B.tokenizer_sha1(new), "jsonl_chars_per_token": 4.1}))
        assert B.plan_chars_per_token(new, 2.8) == 4.1


def test_install_switches_the_tokenizer_and_keeps_the_old_one():
    with tempfile.TemporaryDirectory() as t:
        rpa = Path(t)
        data = rpa / "data"
        data.mkdir()
        for f in ("catalog.jsonl", "examples.jsonl"):
            (data / f).write_text("{}\n")
        base = rpa / "nanochat_base"
        put_ckpt(base / "chatsft_checkpoints" / "d8", 5)
        make_tok(base / "tokenizer", b"web")
        new = make_tok(rpa / "pack" / "tokenizer", b"mix", card=True)
        put_ckpt(rpa / "run", 2)
        E.install(rpa, rpa / "run", 2, data, {"tokenizer": str(new)}, "s1", log=lambda *_: None)
        assert (base / "tokenizer" / "tokenizer.pkl").read_bytes() == b"mix"
        assert (base / "tokenizer" / "tokenizer_card.json").exists()
        assert (base / "old_model_s1" / "tokenizer.pkl").read_bytes() == b"web"
        old = make_tok(rpa / "oldpack" / "tokenizer", b"web")              # back to an older model: no card stays
        put_ckpt(rpa / "run2", 3)
        E.install(rpa, rpa / "run2", 3, data, {"tokenizer": str(old)}, "s2", log=lambda *_: None)
        assert (base / "tokenizer" / "tokenizer.pkl").read_bytes() == b"web"
        assert not (base / "tokenizer" / "tokenizer_card.json").exists()
        assert (base / "old_model_s2" / "tokenizer_card.json").exists()


def test_full_run_on_v6_has_its_own_state_and_base():
    assert TP.run_name("full", "v6") == "full_v6" and TP.run_name("full") == "full" and TP.run_name("topup", "v6") == "topup"
    assert TP.base_root("/r", "v6") == Path("/r/train_data/base_v6") and TP.base_root("/r") == Path("/r/train_data/base")
    with tempfile.TemporaryDirectory() as t:
        d8 = Path(t) / "base_checkpoints" / "d8"
        d8.mkdir(parents=True)
        (d8 / "model_000010.pt").write_bytes(b"m")
        (d8 / "meta_000010.json").write_text(json.dumps({"user_config": {"matrix_lr": 0.03}}))
        assert TP.base_lrs("/nowhere", t)["matrix_lr"] == 0.03


# ------------------------------------------------------------------------------------------- Drive (data_pc)
def test_data_pc_gets_the_v6_base_and_tokenizer_from_drive():
    with tempfile.TemporaryDirectory() as t:
        drive, rpa = Path(t) / "drive", Path(t) / "rpa"
        put_ckpt(drive / "nanochat_base_v6" / "base_checkpoints" / "d8", 5760)
        make_tok(drive / "nanochat_base_v6" / "tokenizer", b"mix", card=True)
        s = D.Sync(D.FolderDrive(drive), rpa, log=lambda *_: None)
        assert s.base_model("v6") == 5760
        root = rpa / "train_data" / "base_v6"
        assert (root / "base_checkpoints" / "d8" / "model_005760.pt").exists() and (root / ".drive_id").exists()
        assert (root / "tokenizer" / "tokenizer_card.json").exists()
        # this PC's own pretrained pack (no .drive_id) is used as it is, never replaced by a download
        shutil.rmtree(root)
        put_ckpt(root / "base_checkpoints" / "d8", 99)
        assert s.base_model("v6") == 99 and not (root / "base_checkpoints" / "d8" / "model_005760.pt").exists()
        got = s.pretrain_inputs()
        assert got["tokenizer"] and (Path(got["tokenizer"]) / "tokenizer.pkl").read_bytes() == b"mix"
        try:
            s.pretrain_inputs(shards=True)
            raise AssertionError("no shards on Drive must be said")
        except SystemExit as ex:
            assert "PRETRAIN_DATA" in str(ex)
        # no v6 base anywhere: the message names both ways to make one
        s2 = D.Sync(D.FolderDrive(Path(t) / "drive"), Path(t) / "rpa2", log=lambda *_: None)
        shutil.rmtree(drive / "nanochat_base_v6" / "base_checkpoints")
        s2 = D.Sync(D.FolderDrive(drive), Path(t) / "rpa2", log=lambda *_: None)
        try:
            s2.base_model("v6")
            raise AssertionError("must refuse")
        except SystemExit as ex:
            assert 'MODE "pretrain"' in str(ex)


# ------------------------------------------------------------------------------------------- notebooks
def nb_code(path):
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_pc_notebook_has_pretrain_and_base_and_keeps_the_defaults():
    with tempfile.TemporaryDirectory() as t:
        out = Path(t) / "nb.ipynb"
        subprocess.run([sys.executable, str(HERE.parent / "make_local_notebook.py"), "", str(out)], check=True,
                       capture_output=True)
        code = nb_code(out)
    c1 = code[0]
    assert 'MODE = "auto"' in c1 and 'BASE = "current"' in c1 and 'PRETRAIN_PROFILE = "pc"' in c1
    assert 'PRETRAIN_MEASURE = 0' in c1 and 'PRETRAIN_DATA = "build"' in c1
    assert '"pretrain"' in code[1] and '"pyyaml"' in code[1]
    assert "base=BASE, pretrain_data=PRETRAIN_DATA" in code[2]
    assert '"pretrain_pc.py"' in code[3] and '"--base", BASE' in code[3] and 'RUN = "full_v6"' in code[3]
    assert 'local_run" / RUN / "state.json"' in code[5]


def test_colab_v6_profile_and_base():
    code = nb_code(HERE.parent / "ooa_nanochat_colab.ipynb")
    c1 = code[0]
    assert 'PROFILE = "t4"' in c1 and 'BASE = "auto"' in c1 and '"v6":' in c1 and "SMOLTALK_ROWS = 0" in c1
    assert '"/nanochat_base_v6" if BASE_NOW == "v6"' in code[1]
    c3 = next(c for c in code if c.startswith("#@title 3."))
    c4 = next(c for c in code if c.startswith("#@title 4."))
    assert 'if PROFILE != "v6":' in c3 and "pretrain_data.py --out" in c3 and "tokenizer_card(" in c3
    assert "data_manifest.json" in c3 and "platform_commit.txt" in c3 and "domain_render" in c3
    assert "patch_base_train(" in c4 and "OOA_KEEP_CHECKPOINTS=2" in c4 and "nonfinite_tensors" in c4
    assert "!python -m scripts.base_train --depth" in c4                        # the old path is still there
    c7 = next(c for c in code if c.startswith("#@title 7."))
    assert "plan_chars_per_token(" in c7 and "chars_per_token=_cpt" in c7


if __name__ == "__main__":
    import inspect

    class MP:
        def __init__(self):
            self.undo = []

        def setattr(self, obj, name, value):
            self.undo.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            mp = MP()
            try:
                fn(*([mp] if "monkeypatch" in inspect.signature(fn).parameters else []))
                print("ok  ", name)
            except Exception as ex:                                  # noqa: BLE001
                failed += 1
                print("FAIL", name, type(ex).__name__, str(ex)[:500])
            finally:
                for obj, n, v in reversed(mp.undo):
                    setattr(obj, n, v)
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
