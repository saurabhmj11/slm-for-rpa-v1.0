"""Tests for knowledge.py. Run: python tests/test_knowledge.py

Uses the real catalog and the platform repo's example flows when found:
  OOA_CATALOG   path to catalog.jsonl (default: ../data/catalog/catalog.jsonl)
  OOA_PLATFORM  path to the platform repo (its examples/**/*.flow.json are explained)
A tiny built-in catalog is used when no real one is found, so the tests always run.
"""
import json, os, random, sys, tempfile, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from knowledge import Knowledge  # noqa: E402


class Skip(Exception):
    """Raised by a test that cannot run here; the runner prints SKIP, not PASS."""

TINY = [
    {"id": "ooa.csv/read", "name": "Read to table", "toolkit": "CSV/TXT", "toolkit_id": "ooa.csv", "kind": "action",
     "description": "Read a CSV file into a table.", "notes": "Time limit 60 s.",
     "inputs": [{"name": "path", "type": "Text", "required": True, "description": "CSV file path."},
                {"name": "delimiter", "type": "Text", "required": False, "description": "Column separator."}],
     "outputs": [{"name": "table", "type": "Table", "description": "The table read."}]},
    {"id": "ooa.csv/write", "name": "Write table", "toolkit": "CSV/TXT", "toolkit_id": "ooa.csv", "kind": "action",
     "description": "Write a table to a CSV file.", "notes": "",
     "inputs": [{"name": "path", "type": "Text", "required": True, "description": "CSV file path."},
                {"name": "table", "type": "Table", "required": True, "description": "Table to write."}],
     "outputs": []},
    {"id": "ooa.file/delete", "name": "Delete file", "toolkit": "File", "toolkit_id": "ooa.file", "kind": "action",
     "description": "Delete a file.", "notes": "",
     "inputs": [{"name": "path", "type": "Text", "required": True, "description": "File to delete."}], "outputs": []},
    {"id": "ooa.file/copy", "name": "Copy file", "toolkit": "File", "toolkit_id": "ooa.file", "kind": "action",
     "description": "Copy a file.", "notes": "",
     "inputs": [{"name": "source", "type": "Text", "required": True, "description": "File to copy."},
                {"name": "target", "type": "Text", "required": True, "description": "Where to copy it."}], "outputs": []},
    {"id": "ooa.file/exists", "name": "File exists", "toolkit": "File", "toolkit_id": "ooa.file", "kind": "condition",
     "description": "True when the file exists.", "notes": "",
     "inputs": [{"name": "path", "type": "Text", "required": True, "description": "File."}], "outputs": []},
    {"id": "core/if", "name": "If", "toolkit": "Core", "toolkit_id": "core", "kind": "action",
     "description": "Run steps when a condition holds.", "notes": "",
     "inputs": [{"name": "condition", "type": "Condition", "required": True, "description": "Condition."}], "outputs": []},
    {"id": "core/loop", "name": "Loop", "toolkit": "Core", "toolkit_id": "core", "kind": "action",
     "description": "Repeat steps.", "notes": "",
     "inputs": [{"name": "over", "type": "List", "required": False, "description": "Items."}], "outputs": []},
    {"id": "core/try", "name": "Try", "toolkit": "Core", "toolkit_id": "core", "kind": "action",
     "description": "Handle errors.", "notes": "", "inputs": [], "outputs": []},
    {"id": "core/log", "name": "Log message", "toolkit": "Core", "toolkit_id": "core", "kind": "action",
     "description": "Write a log line.", "notes": "",
     "inputs": [{"name": "message", "type": "Text", "required": True, "description": "Message."}], "outputs": []},
    {"id": "ooa.mail/send", "name": "Send email", "toolkit": "Email", "toolkit_id": "ooa.mail", "kind": "action",
     "description": "Send an email.", "notes": "",
     "inputs": [{"name": "to", "type": "Text", "required": True, "description": "Recipient."},
                {"name": "subject", "type": "Text", "required": True, "description": "Subject."}], "outputs": []},
]


def find_catalog():
    env = os.environ.get("OOA_CATALOG")
    if env and Path(env).is_file():
        return env
    for p in (HERE.parent.parent / "data" / "catalog" / "catalog.jsonl",):
        if p.is_file():
            return str(p)
    return None


def find_platform():
    env = os.environ.get("OOA_PLATFORM")
    return env if env and Path(env, "examples").is_dir() else None


CATALOG = find_catalog()
if CATALOG is None:
    _tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    _tmp.write("\n".join(json.dumps(e) for e in TINY))
    _tmp.close()
    CATALOG = _tmp.name
K = Knowledge(CATALOG)
REAL = len(K.entries) > 100

NESTED = {
    "kind": "TaskBot", "id": "demo", "name": "Demo", "description": "Read a CSV and log rows.",
    "variables": [{"name": "path", "type": "String", "direction": "input", "description": "CSV to read"},
                  {"name": "table", "type": "Table", "direction": "local"},
                  {"name": "count", "type": "Number", "direction": "output", "description": "Rows"}],
    "steps": [
        {"id": "s1", "action": "core/try", "children": [
            {"id": "s2", "action": "ooa.csv/read", "args": {"path": "$path"}, "out": {"table": "table"}},
            {"id": "s3", "action": "core/loop", "args": {"iterator": "list", "over": "$table", "as": "row"}, "children": [
                {"id": "s4", "action": "core/if", "args": {"condition": {"op": "empty", "left": "$row"}},
                 "children": [{"id": "s5", "action": "core/log", "args": {"message": "empty"}}],
                 "elseIf": [{"condition": {"op": "eq", "left": "$row.a", "right": 1},
                             "children": [{"id": "s6", "action": "core/log", "args": {"message": "one"}}]}],
                 "else": [{"id": "s7", "action": "ooa.made/up", "args": {"x": 1}}]}]}],
         "catch": [{"codes": ["FileNotFound"], "children": [{"id": "s8", "action": "core/log", "args": {"message": "x"}}]}],
         "finally": [{"id": "s9", "action": "core/log", "args": {"message": "done"}}]}]}


def test_catalog_basics():
    tks = K.toolkits()
    assert tks and all(set(t) >= {"id", "name", "count"} for t in tks)
    assert [t["name"].lower() for t in tks] == sorted(t["name"].lower() for t in tks)
    assert sum(t["count"] for t in tks) == len(K.entries)
    if REAL:
        print(f"  {len(K.entries)} entries, {len(tks)} toolkits")
        if "ooa.datetime/is_working_day" in K.by_id:      # id exists as condition and action: action wins
            assert K.by_id["ooa.datetime/is_working_day"]["kind"] == "action"


def test_action_card():
    c = K.action_card("ooa.csv/read")
    assert c and c["id"] == "ooa.csv/read" and c["toolkit"]
    assert {"id", "name", "toolkit", "description", "notes", "required", "optional", "outputs"} <= set(c)
    assert any(i["name"] == "path" for i in c["required"]), c["required"]
    assert all(set(i) == {"name", "type", "description"} for i in c["required"] + c["optional"] + c["outputs"])
    assert K.action_card("ooa.nope/nothing") is None


def test_search():
    r = K.search("read a csv file into a table", 5)
    assert 0 < len(r) <= 5 and set(r[0]) >= {"id", "name", "toolkit", "description"}
    assert "ooa.csv/read" in [x["id"] for x in r], [x["id"] for x in r]
    assert K.search("", 5) == [] and K.search("   ") == []
    assert K.search("ooa.file/delete", 3)[0]["id"] == "ooa.file/delete"


def test_lesson():
    tid = "ooa.csv"
    L = K.lesson(tid)
    assert L and L["toolkit"] and L["intro"] and L["actions"]
    assert all(a["id"] in K.by_id for a in L["actions"])
    assert L["intro"].count(".") <= 4 and len(L["intro"]) < 400, L["intro"]
    assert K.lesson("ooa.does_not_exist") is None
    for t in K.toolkits():          # every toolkit gives a lesson
        assert K.lesson(t["id"])["actions"]


def test_explain_nested():
    x = K.explain_flow(NESTED)
    ids = [s["id"] for s in x["steps"]]
    assert ids == ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"], ids
    by = {s["id"]: s for s in x["steps"]}
    assert by["s1"]["block"] is None and by["s1"]["depth"] == 0
    assert by["s2"]["block"] == "try" and by["s2"]["depth"] == 1
    assert by["s4"]["block"] == "loop" and by["s4"]["depth"] == 2
    assert (by["s5"]["block"], by["s6"]["block"], by["s7"]["block"]) == ("if", "elseif", "else")
    assert by["s5"]["depth"] == 3
    assert by["s8"]["block"] == "catch" and by["s9"]["block"] == "finally"
    assert by["s7"]["known"] is False and "not in the catalog" in by["s7"]["what"]
    assert by["s2"]["known"] and "Reads from: path=$path" in by["s2"]["what"], by["s2"]["what"]
    assert "in: table" in by["s2"]["what"] and by["s2"]["outs"] == [{"name": "table", "variable": "table"}]
    assert "inside the loop" in by["s4"]["what"] and "is empty" in by["s4"]["what"]
    assert by["s2"]["args"][0]["name"] == "path" and by["s2"]["args"][0]["meaning"]
    assert [i["name"] for i in x["inputs"]] == ["path"] and [o["name"] for o in x["outputs"]] == ["count"]
    assert "ooa.made/up" in x["summary"]
    keys = {"id", "depth", "action", "title", "what", "args", "outs", "known", "block"}
    assert all(set(s) == keys for s in x["steps"])


def test_explain_garbage():
    for bad in (None, [], "x", {}, {"steps": "nope"}, {"steps": [1, None, {"action": 5}, {"id": "a"}]},
                {"steps": [{"action": "core/if", "args": {"condition": [1, {"all": [{"op": "zz"}]}]},
                            "elseIf": [None, {"children": None}], "catch": "x"}]}):
        x = K.explain_flow(bad)
        assert isinstance(x["summary"], str) and isinstance(x["steps"], list)


def test_explain_platform_examples():
    root = find_platform()
    if not root:
        raise Skip("set OOA_PLATFORM to the platform repo to explain its examples")
    files = sorted(Path(root, "examples").rglob("*.flow.json"))
    assert files
    n = 0
    for f in files:
        x = K.explain_flow(json.loads(f.read_text(encoding="utf-8")))
        assert x["summary"] and all(s["what"] for s in x["steps"]), f
        n += len(x["steps"])
    print(f"  explained {len(files)} example flows, {n} steps")


def test_quiz_roundtrip():
    rng = random.Random(7)
    for kind in ("which_action", "required_inputs", "which_toolkit"):
        for _ in range(30 if REAL else 5):
            q = K.quiz(rng, kind)
            assert q["kind"] == kind and len(q["choices"]) == 4 and len(set(q["choices"])) == 4, q
            assert set(q) == {"id", "kind", "question", "choices"}
            results = [K.check_quiz(q["id"], i) for i in range(4)]
            assert sum(r["correct"] for r in results) == 1
            ans = results[0]["answer_index"]
            assert results[ans]["correct"] and results[ans]["explanation"]
    q = K.quiz()                     # random kind, default rng
    assert q["kind"] in ("which_action", "required_inputs", "which_toolkit")


def test_quiz_cannot_be_forged_or_read():
    q = K.quiz(random.Random(1), "which_action")
    import base64
    raw = base64.urlsafe_b64decode(q["id"] + "=" * (-len(q["id"]) % 4))
    assert b'"x"' not in raw and q["choices"][0].encode() not in raw    # encrypted, answer not readable
    tampered = q["id"][:-2] + ("AA" if q["id"][-2:] != "AA" else "BB")
    for bad in (tampered, "", "!!!", "abc", base64.urlsafe_b64encode(b"{}" * 20).decode()):
        try:
            K.check_quiz(bad, 0)
            raise AssertionError("forged id accepted: " + bad)
        except ValueError:
            pass
    other = Knowledge(CATALOG)       # another process (different key) rejects it too
    try:
        other.check_quiz(q["id"], 0)
        raise AssertionError("id from another key accepted")
    except ValueError:
        pass
    for bad_choice in (4, -1, "x", None):
        try:
            K.check_quiz(q["id"], bad_choice)
            raise AssertionError("bad choice accepted")
        except ValueError:
            pass
    try:
        K.quiz(kind="nope")
        raise AssertionError("bad kind accepted")
    except ValueError:
        pass


def test_quiz_distractors_same_toolkit():
    if not REAL:
        raise Skip("set OOA_CATALOG to the full catalog")
    rng, same = random.Random(3), 0
    for _ in range(40):
        q = K.quiz(rng, "which_action")
        tk = {c.split(" -> ")[0] for c in q["choices"]}
        same += len(tk) == 1
    assert same >= 25, same          # most toolkits have 4+ actions


def _ambiguous(k, q):
    """True when a wrong choice of a which_toolkit quiz also has the asked action (same name and description)."""
    st = k._open(q["id"])
    e = k.by_id[st["a"]]
    for i, name in enumerate(q["choices"]):
        if i == st["x"]:
            continue
        for t in k._tk.values():
            if t["name"] == name and any(x["kind"] == "action" and x["name"] == e["name"]
                                         and x["description"] == e["description"] for x in t["entries"]):
                return True
    return False


def test_which_toolkit_has_one_right_answer():
    """Fix: 'Mark read or unread' exists in Email and EWS alike, and EWS was offered as a wrong choice."""
    dup = dict(TINY[0], id="ooa.tsv/read", toolkit="TSV", toolkit_id="ooa.tsv")      # same name + description
    extra = [dict(TINY[0], id=f"ooa.t{i}/x", name=f"Other {i}", description=f"Other action {i}.",
                  toolkit=f"T{i}", toolkit_id=f"ooa.t{i}") for i in range(3)]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for e in TINY + [dup] + extra:
            f.write(json.dumps(e) + "\n")
    k = Knowledge(f.name)
    rng = random.Random(0)
    seen = 0
    for _ in range(300):
        q = k.quiz(rng, "which_toolkit")
        assert not _ambiguous(k, q), q
        seen += k._open(q["id"])["a"] in ("ooa.csv/read", "ooa.tsv/read")
    assert seen, "the duplicated action was never asked"
    os.unlink(f.name)
    if REAL:   # the reviewer's seed on the full catalog asked about ooa.email mark with EWS as a "wrong" choice
        assert not _ambiguous(K, K.quiz(random.Random(8392), "which_toolkit"))
        rng = random.Random(1)
        for _ in range(500):
            assert not _ambiguous(K, K.quiz(rng, "which_toolkit"))


def test_explain_malformed_flow():
    """Fix: non-list variables / elseIf / catch / codes raised TypeError."""
    for f in ({"variables": 5, "steps": []},
              {"steps": [{"id": "a", "action": "core/if", "elseIf": 5, "args": {"condition": "true"}}]},
              {"steps": [{"id": "a", "action": "core/try", "catch": 5}]},
              {"steps": [{"id": "a", "action": "core/try", "catch": [{"codes": 3, "children": 7}]}]},
              {"steps": 5}):
        x = K.explain_flow(f)
        assert isinstance(x["summary"], str) and isinstance(x["steps"], list), f


if __name__ == "__main__":
    print(f"catalog: {CATALOG} ({len(K.entries)} entries){'' if REAL else ' [tiny built-in]'}")
    fails = skipped = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Skip as ex:
                skipped += 1
                print("SKIP", name, "-", ex)
            except Exception:
                fails += 1
                print("FAIL", name)
                traceback.print_exc()
    print((f"all passed ({skipped} skipped)" if skipped else "all passed") if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)
