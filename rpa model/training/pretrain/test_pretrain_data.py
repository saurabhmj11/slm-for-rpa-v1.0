"""Tests for pretrain_data.py on tiny fake sources (a local "web" parquet folder, a fake platform checkout, rendered
rows, an eval set). Needs pyarrow. Run: python test_pretrain_data.py"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pretrain_data as P                                    # noqa: E402
import pyarrow as pa                                         # noqa: E402
import pyarrow.parquet as pq                                 # noqa: E402

EVAL_PROMPT = "Read the invoices spreadsheet and email the overdue customers a polite payment reminder"
FAKE_KEY = "AKIA" + "ABCDEFGHIJKLMNOP"                       # an access-key-shaped string, split so no scanner trips
SOFT = "hunter2-very-secret"


def _msgs(req, ans):
    return {"messages": [{"role": "user", "content": req + "\n### Context\nActions:\n- ooa.csv/read: Read"},
                         {"role": "assistant", "content": ans}]}


def make_sources(root):
    root = Path(root)
    web = root / "web"
    web.mkdir(parents=True)
    for f in range(2):                                      # 2 files x 6 row groups x 20 docs of ~500 chars
        docs = [f"web file {f} doc {i} " + ("lorem ipsum dolor sit amet " * 18) for i in range(120)]
        pq.write_table(pa.table({"text": docs, "id": list(range(120))}), web / f"{f:03d}_00000.parquet",
                       row_group_size=20)
    plat = root / "plat"
    (plat / "docs").mkdir(parents=True)
    (plat / "contracts").mkdir()
    (plat / "platform" / "src").mkdir(parents=True)
    (plat / "docs" / "guide.md").write_text("# Guide\n" + "How flows run on the agent.\n" * 40)
    (plat / "docs" / "copy.md").write_text("# Guide\n" + "How flows run on the agent.\n" * 40)    # exact duplicate
    (plat / "docs" / "quote.md").write_text("Example request: " + EVAL_PROMPT + "\n")
    (plat / "contracts" / "flow.json").write_text(json.dumps({"steps": [{"action": "ooa.csv/read"}]}, indent=1))
    (plat / "platform" / "src" / "Keys.java").write_text(f'class Keys {{ String k = "{FAKE_KEY}"; }}\n')
    (plat / "platform" / "src" / "Db.java").write_text(f'class Db {{ String password = "{SOFT}"; }}\n')
    (plat / "platform" / "src" / ".env").write_text("TOKEN=abc\n")
    (plat / "platform" / "src" / "server.pem.json").write_text("{}\n")
    render = root / "render"
    render.mkdir()
    rows = [_msgs(f"Copy report number {i} to the archive folder and log it", f"Flow:\n{{\"steps\":[{i}]}}")
            for i in range(60)]
    rows.append(_msgs(EVAL_PROMPT, "Flow: eval-leak"))
    rows.append(_msgs("Rename every file in the drop folder by date", "Flow: val-dup-in-train"))
    with open(render / "core_train.jsonl", "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    with open(render / "val.jsonl", "w") as f:
        f.write(json.dumps(_msgs("Rename every file in the drop folder by date", "Plan:\n1. Rename\nFlow:\n{\"steps\":[\"held-out-answer\"]}")) + "\n")
    with open(render / "rl_train.jsonl", "w") as f:
        f.write(json.dumps(_msgs("rl only prompt", "Flow: rl-row")) + "\n")
    cat = root / "catalog.jsonl"
    cat.write_text("\n".join(json.dumps({"id": f"ooa.x/a{i}", "name": f"Action {i}", "toolkit": "x"})
                             for i in range(30)) + "\n")
    ev = root / "eval"
    ev.mkdir()
    (ev / "set.jsonl").write_text(json.dumps({"prompt": EVAL_PROMPT, "must": ["ooa.csv/read"]}) + "\n")
    return {"platform": str(plat), "render_dirs": [str(render)], "catalog": str(cat), "web_dir": str(web),
            "eval_dir": str(ev)}


def run(out, src, train_chars=60_000, shard_chars=20_000, **kw):
    return P.build(out, train_chars, shard_chars=shard_chars, domain_share=0.15, val_groups=2, seed=1,
                   log=lambda *a: None, **src, **kw)


def texts(path):
    return pq.read_table(path).column("text").to_pylist()


def all_train(out):
    return [t for f in sorted(Path(out).glob("shard_0*.parquet")) for t in texts(f)]


class Tmp:
    def __enter__(self):
        self.d = tempfile.mkdtemp()
        self.src = make_sources(self.d)
        return self

    def __exit__(self, *a):
        shutil.rmtree(self.d, ignore_errors=True)


def test_layout_and_nanochat_order():
    with Tmp() as t:
        out = Path(t.d) / "out"
        card = run(out, t.src)
        names = sorted(p.name for p in out.glob("*.parquet"))
        assert names == ["shard_00000.parquet", "shard_00001.parquet", "shard_00002.parquet", P.VAL_NAME], names
        assert names[-1] == P.VAL_NAME                           # nanochat's val split is the last file
        f = pq.ParquetFile(out / "shard_00000.parquet")
        assert f.schema_arrow.names == ["text"]
        assert card["totals"]["train_shards"] == 3
        assert (out / "data_card.json").exists() and (out / "DATA_CARD.md").exists()
        assert 0.10 < card["totals"]["domain_share"] < 0.20, card["totals"]
        assert "ClimbMix (CC BY-NC 4.0)" in card["not_included"]


def test_deterministic():
    with Tmp() as t:
        a, b = Path(t.d) / "a", Path(t.d) / "b"
        run(a, t.src)
        run(b, t.src)
        ma, mb = json.loads((a / "manifest.json").read_text()), json.loads((b / "manifest.json").read_text())
        assert [s["sha256"] for s in ma["shards"]] == [s["sha256"] for s in mb["shards"]]
        assert ma["val"]["sha256"] == mb["val"]["sha256"]


def test_resume_extends_and_matches_a_fresh_build():
    with Tmp() as t:
        a, b = Path(t.d) / "a", Path(t.d) / "b"
        run(a, t.src, train_chars=40_000)
        m0 = (a / "shard_00000.parquet").stat().st_mtime_ns
        run(a, t.src, train_chars=80_000)                        # more data later: the first shards stay
        assert (a / "shard_00000.parquet").stat().st_mtime_ns == m0
        run(b, t.src, train_chars=80_000)
        sa = [s["sha256"] for s in json.loads((a / "manifest.json").read_text())["shards"]]
        sb = [s["sha256"] for s in json.loads((b / "manifest.json").read_text())["shards"]]
        assert len(sa) == 4 and sa == sb


def test_resume_after_a_broken_shard():
    with Tmp() as t:
        a = Path(t.d) / "a"
        run(a, t.src)
        good = json.loads((a / "manifest.json").read_text())["shards"]
        (a / "shard_00001.parquet").write_bytes(b"cut off")      # e.g. the machine stopped mid-write
        run(a, t.src)
        again = json.loads((a / "manifest.json").read_text())["shards"]
        assert [s["sha256"] for s in again] == [s["sha256"] for s in good]


def test_other_settings_are_refused():
    with Tmp() as t:
        a = Path(t.d) / "a"
        run(a, t.src)
        try:
            P.build(a, 60_000, shard_chars=20_000, domain_share=0.3, log=lambda *x: None, **t.src)
        except SystemExit as e:
            assert "other settings" in str(e)
        else:
            raise AssertionError("a different domain share was mixed into the same folder")
        (Path(t.src["platform"]) / "docs" / "new.md").write_text("A new page about schedules.\n" * 30)
        try:
            run(a, t.src)
        except SystemExit as e:
            assert "domain sources" in str(e)
        else:
            raise AssertionError("changed domain sources were mixed into the same folder")


def test_repeat_scale_is_a_setting_and_old_manifests_resume():
    with Tmp() as t:
        out = Path(t.d) / "out"
        run(out, t.src)
        man_f = out / "manifest.json"
        man = json.loads(man_f.read_text())
        assert man["settings"]["repeat_scale"] == 1
        del man["settings"]["repeat_scale"]                    # a manifest written before the setting existed
        man_f.write_text(json.dumps(man))
        run(out, t.src, train_chars=80_000)                    # resumes (it used the default)
        try:
            run(out, t.src, train_chars=80_000, repeat_scale=2)
            raise AssertionError("other passes over the domain text must be refused")
        except SystemExit:
            pass
        src = {"render": {"docs": ["r" * 100] * 10}, "code": {"docs": ["c" * 100] * 4}}
        assert len(P.domain_stream(src, 1, repeat_scale=2)) == 2 * len(P.domain_stream(src, 1))


def test_val_split_is_held_out():
    with Tmp() as t:
        out = Path(t.d) / "out"
        run(out, t.src)
        man = json.loads((out / "manifest.json").read_text())
        train_units = {tuple(u) for s in man["shards"] for u in s["web_units"]}
        val_units = {tuple(u) for u in man["val"]["web_units"]}
        assert val_units == {(0, 4), (0, 5)} and not (train_units & val_units)
        train = all_train(out)
        val = texts(out / P.VAL_NAME)
        assert not set(train) & set(val)
        assert any("held-out-answer" in v for v in val)
        assert not any("held-out-answer" in x for x in train)
        assert not any("val-dup-in-train" in x for x in train)  # a train row asking a val request is dropped
        assert not any("rl-row" in x for x in train + val)


def test_secrets_are_scrubbed():
    with Tmp() as t:
        out = Path(t.d) / "out"
        card = run(out, t.src)
        blob = "\n".join(all_train(out) + texts(out / P.VAL_NAME))
        assert FAKE_KEY not in blob and SOFT not in blob
        assert 'password = "<redacted>"' in blob
        assert "TOKEN=abc" not in blob
        assert card["domain"]["code"]["dropped_secret"] == 1
        assert card["domain"]["code"]["redacted_literals"] == 1
    st = {}
    assert P.scrub('{"password": "$password$"}', st) == '{"password": "$password$"}'   # flow variables stay
    assert P.scrub("-----BEGIN RSA PRIVATE KEY-----\nxx", st) is None


def test_eval_prompts_are_excluded():
    with Tmp() as t:
        out = Path(t.d) / "out"
        card = run(out, t.src)
        blob = "\n".join(all_train(out))
        assert "eval-leak" not in blob
        assert P.norm(EVAL_PROMPT) not in P.norm(blob)
        assert card["domain"]["render"]["excluded_eval_close"] == 1
        assert card["domain"]["docs"]["excluded_eval_quote"] == 1
        assert card["domain"]["docs"]["dropped_duplicates"] == 1
    g = P.EvalGuard([EVAL_PROMPT])
    assert g.close("Please read the invoices spreadsheet and email overdue customers a polite payment reminder")
    assert not g.close("Copy report number 7 to the archive folder and log it")


def test_domain_stream_keeps_weights_and_repeat_caps():
    src = {"render": {"docs": ["r" * 100] * 50}, "code": {"docs": ["c" * 100] * 50}}
    s = P.domain_stream(src, seed=3, weights={"render": 0.75, "code": 0.25}, max_repeat={"render": 2, "code": 2})
    head = s[:40]
    assert abs(sum(1 for k, _ in head if k == "render") / 40 - 0.75) < 0.05
    assert sum(1 for k, _ in s if k == "code") == 100 and sum(1 for k, _ in s if k == "render") == 100
    assert s == P.domain_stream(src, seed=3, weights={"render": 0.75, "code": 0.25}, max_repeat={"render": 2, "code": 2})


def test_pieces_fit_the_tokenizer_cap():
    text = "\n".join("x" * 300 for _ in range(100)) + "\n" + "y" * 20000
    ps = P._pieces(text)
    assert all(len(p) <= P.DOC_CHARS for p in ps) and "".join(ps) == text


def test_tok_compare_checks_the_chat_special_tokens():
    """tok_compare on a tiny tokenizer trained by nanochat itself (skipped without NANOCHAT_DIR)."""
    nc = os.environ.get("NANOCHAT_DIR")
    if not nc or not os.path.isfile(os.path.join(nc, "nanochat", "tokenizer.py")):
        print("  (skipped: set NANOCHAT_DIR)")
        return
    sys.path.insert(0, nc)
    from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS
    import tok_compare as T
    assert T.SPECIAL == SPECIAL_TOKENS                       # the pinned nanochat's chat tokens, same order
    with Tmp() as t:
        tok = RustBPETokenizer.train_from_iterator(iter(['{"action":"ooa.csv/read"} hello world'] * 50), 300)
        d = Path(t.d) / "tok"
        tok.save(str(d))
        enc = T.load(d)
        ok, ids = T.specials_ok(enc)
        assert ok and ids["<|bos|>"] == enc.n_vocab - len(SPECIAL_TOKENS), ids
        rows = T.main(["--tok", f"a={d}", "--tok", f"b={d}", "--render", t.src["render_dirs"][0],
                       "--eval-dir", t.src["eval_dir"], "--catalog", t.src["catalog"],
                       "--web-dir", t.src["web_dir"], "--web-groups", "2"])
        assert rows[1]["flow_tokens_vs_first"] == 0 and rows[0]["flow"] > 0 and rows[0]["specials_ok"]


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
