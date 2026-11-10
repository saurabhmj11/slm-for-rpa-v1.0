"""Tests for constrained.py (catalog-locked decoding). Pure Python, no model needed.
Run: python tests/test_constrained.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import constrained as C

CAT = [
    {"id": "ooa.file/copy", "inputs": [{"name": "source"}, {"name": "target"}, {"name": "overwrite"}], "outputs": [{"name": "files"}]},
    {"id": "ooa.file/list", "inputs": [{"name": "folder"}, {"name": "pattern"}], "outputs": [{"name": "files"}]},
    {"id": "ooa.folder/list_subfolders", "inputs": [{"name": "folder"}], "outputs": [{"name": "folders"}]},
    {"id": "core/if", "inputs": [{"name": "condition"}], "outputs": []},
]


def slot_after(text):
    t = C.JsonTracker()
    t.feed(text)
    return t.name_slot()


def test_tracker_slots():
    assert slot_after('{"steps":[{"id":"s1","action":"ooa.fi') == ("action", None, (), "ooa.fi")
    s = slot_after('{"steps":[{"id":"s1","action":"ooa.file/copy","description":"a {b} \\"c\\"","args":{"source":"$x","ta')
    assert s[0] == "args" and s[1] == "ooa.file/copy" and s[3] == "ta" and "source" in s[2], s
    s = slot_after('{"steps":[{"action":"ooa.file/copy","args":{"source":{"expr":"$a"},"tar')
    assert s[0] == "args" and s[3] == "tar", s                      # a nested value object closes correctly
    s = slot_after('{"steps":[{"action":"ooa.file/copy","args":{},"out":{"fi')
    assert s[0] == "out" and s[3] == "fi", s
    assert slot_after('{"steps":[{"action":"ooa.file/copy","args":{"source":"$x') is None   # inside a value: free
    assert slot_after('{"description":"the action is') is None
    assert slot_after('{"variables":[{"name":"ac') is None            # variable names are free


def test_allowed_tokens():
    vocab = ["", "ooa", ".", "file", "folder", "/", "copy", "list", "_sub", "folders", '"', '",', "ooa.file/list", "x", "target", "tar", "get", '":"', "foo"]
    con = C.NameConstraint(CAT, vocab, lock_keys=True)
    ids = range(len(vocab))
    allowed = lambda slot: {vocab[i] for i in con.allowed(slot, ids)}
    a = allowed(("action", None, (), ""))
    assert {"ooa", "ooa.file/list"} <= a and "x" not in a and '"' not in a, a
    a = allowed(("action", None, (), "ooa.file/"))
    assert {"copy", "list"} <= a and "folder" not in a, a
    a = allowed(("action", None, (), "ooa.file/list"))
    assert '"' in a and '",' in a and "_sub" not in a, a                      # the name may end here
    a = allowed(("action", None, (), "ooa.folder/list"))
    assert "_sub" in a and '"' not in a, a                                   # "ooa.folder/list" is not a real id
    a = allowed(("args", "ooa.file/copy", {"source"}, ""))
    assert {"tar", "target"} <= a and "x" not in a and "foo" not in a, a
    a = allowed(("args", "ooa.file/copy", {"source"}, "target"))
    assert '":"' in a and '"' in a, a
    con2 = C.NameConstraint(CAT, ['":older', '":"', '"x', '",'], lock_keys=True)
    a2 = {con2.tokens[i] for i in con2.allowed(("args", "ooa.file/copy", set(), "target"), range(4))}
    assert a2 == {'":"', '",'}, a2                                          # no letters after the closing quote
    assert con.allowed(("args", "ooa.unknown/x", set(), ""), ids) is None      # unknown action: not constrained
    assert con.allowed(("args", "core/if", set(), ""), ids) is None            # control steps: not constrained
    assert con.allowed(("args", "ooa.file/list", {"folder", "pattern"}, ""), ids) is None   # all used: free
    default = C.NameConstraint(CAT, vocab)
    assert default.allowed(("args", "ooa.file/copy", set(), ""), ids) is None   # keys are not locked by default
    assert default.allowed(("action", None, (), "ooa.file/"), ids)              # action ids always are


def test_not_a_flow():
    vocab = ["", " ", "{", '{"', "The", " answer", "\n"]
    dec = lambda t: vocab[t]
    assert not C.not_a_flow([], dec)
    assert not C.not_a_flow([1, 6], dec)                 # only spaces so far: undecided
    assert not C.not_a_flow([1, 3], dec)                 # opens a JSON object
    assert C.not_a_flow([4], dec) and C.not_a_flow([1, 4, 5], dec)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                fails += 1
                print("FAIL", name, e)
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
