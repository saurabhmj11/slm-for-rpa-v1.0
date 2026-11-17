"""Tests for the beginner test set and its scorer. Run: python test_beginner_score.py (works under pytest too)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import beginner_score as B

CATALOG = os.path.join(HERE, "../data/catalog/catalog.jsonl")
IDS = {json.loads(l)["id"] for l in open(CATALOG, encoding="utf-8") if l.strip()}
ROWS = B.load(os.path.join(HERE, "beginner_requests.jsonl"))


def test_the_set_is_valid():
    assert B.validate(ROWS, IDS) == []
    kinds = [r["kind"] for r in ROWS]
    assert len(ROWS) == 80 and kinds.count("flow") == 61 and kinds.count("question") == 19
    styles = {s for r in ROWS for s in r["style"]}
    assert {"vague", "business", "casual", "typo", "hinglish", "indian_english", "voice", "story", "missing_detail"} <= styles
    assert all("missing_detail" in r["style"] for r in ROWS if r["kind"] == "question")


def test_validate_finds_problems():
    bad = [{"prompt": "x", "kind": "flow", "must": ["ooa.nope/never"], "style": ["casual"]},
           {"prompt": "y", "kind": "question", "ask_about": [], "style": ["weird"]}, {"prompt": "x", "kind": "flow", "must": []}]
    probs = B.validate(bad, IDS)
    assert any("unknown action" in p for p in probs) and any("ask_about" in p for p in probs)
    assert any("unknown style" in p for p in probs) and any("duplicate" in p for p in probs) and any("needs must" in p for p in probs)


def test_word_overlap():
    assert B.word_overlap("Zip the folder", "zip THE folder!") == 1.0
    assert B.word_overlap("a b c d", "a b x y") == 2 / 6
    reqs = ["zip the reports folder", "send an email", "read a csv file"]
    assert B.closest("pls zip my reports folder", reqs, B.overlap_index(reqs))[1] == "zip the reports folder"


def test_must_alternatives():
    need = ["ooa.excel/open", "ooa.excel/set_cell", ["ooa.excel/save", "ooa.excel/close"]]
    assert B.in_order(need, ["ooa.excel/open", "ooa.excel/set_cell", "ooa.excel/close"])
    assert B.in_order(need, ["ooa.excel/open", "ooa.excel/set_cell", "ooa.excel/save", "ooa.excel/close"])
    assert not B.in_order(need, ["ooa.excel/open", "ooa.excel/close", "ooa.excel/set_cell"])
    assert B.validate([{"prompt": "p", "kind": "flow", "must": [["ooa.excel/save", "ooa.nope/x"]], "style": []}], IDS) \
        == ["row 0: unknown action ooa.nope/x"]
    row = next(r for r in ROWS if r["prompt"].startswith("Please put today's date in cell A1"))
    assert B.score(row, "flow", True, ["ooa.excel/open", "ooa.excel/set_cell", "ooa.excel/close"])["right"]


def test_containment_catches_a_close_paraphrase():
    test = "Kindly save the attachments of all unread mails in inbox to D:\\Mails and mark them as read also"
    train = "As discussed, kindly save the attachments of each new mail in Inbox to Exports and mark them read pls"
    assert B.word_overlap(test, train) <= B.MAX_OVERLAP and B.too_close(test, train)[0]
    assert not B.too_close(test, "Save the attachments of new mail to a folder, then post a summary to the team "
                                 "channel, archive the mail and log how many there were")[0]
    assert B.containment("rename them properly", "rename them properly now") == 0.0     # too few content words
    reqs = [train, "send an email"]
    assert B.closest_contained(test, reqs, B.overlap_index(reqs))[1] == train


def test_scoring_rules():
    flow_row = {"kind": "flow", "must": ["ooa.csv/read", "ooa.csv/write"], "ask_about": [], "style": ["casual"]}
    assert B.score(flow_row, "flow", True, ["ooa.csv/read", "ooa.datatable/filter", "ooa.csv/write"])["right"]
    assert B.score(flow_row, "flow", True, ["ooa.csv/write", "ooa.csv/read"])["outcome"] == "wrong_actions"
    assert B.score(flow_row, "flow", False, ["ooa.csv/read", "ooa.csv/write"])["outcome"] == "invalid_flow"
    assert B.score(flow_row, "question", question="Which file?")["outcome"] == "asked"
    q_row = {"kind": "question", "must": [], "ask_about": ["folder", "which files"], "style": ["vague", "missing_detail"]}
    assert B.score(q_row, "question", question="Which FOLDER are they in?") == {"right": True, "outcome": "asked_right"}
    assert B.score(q_row, "question", question="What should the bot do?")["outcome"] == "asked_off_topic"
    assert B.score(q_row, "flow", True, ["ooa.file/move"]) == {"right": False, "outcome": "assumed"}
    assert B.score(q_row, "reply")["outcome"] == "no_answer"
    res = {"verified": True, "flow": json.dumps({"steps": [{"action": "ooa.csv/read"}, {"action": "ooa.csv/write"}]})}
    acts = lambda t: [s["action"] for s in json.loads(t)["steps"]]
    assert B.score_router(flow_row, res, acts)["right"]
    assert B.score_router(q_row, {"verified": False, "question": "Which folder?"}, acts)["right"]


def test_summary_per_kind_and_style():
    rows = [{"kind": "flow", "style": ["casual", "typo"]}, {"kind": "question", "style": ["missing_detail"]},
            {"kind": "flow", "style": ["casual"]}]
    res = [{"right": True, "outcome": "right"}, {"right": False, "outcome": "assumed"}, {"right": False, "outcome": "no_flow"}]
    s = B.summarize(rows, res)
    assert s["all"] == [1, 3] and s["kind"] == {"flow": [1, 2], "question": [0, 1]}
    assert s["style"]["casual"] == [1, 2] and s["style"]["typo"] == [1, 1] and s["outcomes"]["assumed"] == 1
    assert B.summary_text(s).startswith("right 1/3 | flow 1/2 question 0/1")


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
