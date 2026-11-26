"""Tests for the thinking answer format (ooa_format "thinking answers": Understand / Facts / Missing / Steps / Check
before the flow or the question) and render_dataset.py --think. Standard library only.
Run: python test_think_format.py   (works under pytest too)."""
import json, os, random, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ooa_format as F, retrieval, render_dataset as RD

CATALOG = os.path.join(HERE, "../data/catalog/catalog.jsonl")
ENTRIES = [json.loads(l) for l in open(CATALOG, encoding="utf-8")]
CAT = {e["id"]: e for e in ENTRIES}
INDEX = retrieval.ActionIndex(ENTRIES)

REQUEST = ("Every Monday at 9 read C:\\Invoices\\march.csv, and for each row with amount above 1000 email "
           "ap@acme.com, then log done")
FLOW = {"kind": "TaskBot", "id": "inv", "name": "Invoices", "description": "Email AP about the large March invoices.",
        "variables": [{"name": "rows", "type": "Table", "direction": "local"}],
        "steps": [
            {"id": "s1", "action": "ooa.csv/read", "description": "Read the invoices",
             "args": {"path": "C:\\Invoices\\march.csv"}, "out": {"table": "rows"}},
            {"id": "s2", "action": "core/loop", "args": {"iterator": "list", "over": "$rows$", "as": "row"},
             "children": [
                 {"id": "s3", "action": "core/if",
                  "args": {"condition": {"op": ">", "left": "$row.amount$", "right": 1000}},
                  "children": [{"id": "s4", "action": "ooa.email/send", "description": "Email AP",
                                "args": {"to": "ap@acme.com", "subject": "Invoice ${row.id}"}}]}]},
            {"id": "s5", "action": "core/log", "args": {"message": "done"}}]}


def test_think_answer_layout():
    ans = F.render_think_answer(FLOW, CAT, F.ANSWER_COMPACT, {"request": REQUEST})
    lines = ans.split("\n")
    assert lines[0] == "Understand: Email AP about the large March invoices."            # the flow's description
    assert lines[1].startswith("Facts: schedule = Every Monday at 9; path = C:\\Invoices\\march.csv")
    assert "filter = row.amount > 1000" in lines[1] and "to = ap@acme.com" in lines[1] and "message = done" in lines[1]
    assert lines[2] == "Missing: none" and lines[3] == "Steps:"
    assert lines[4] == "1. Read the invoices -> ooa.csv/read"
    assert lines[5] == "2. For each row in rows -> core/loop"
    assert lines[6] == "  3. If row.amount > 1000 -> core/if"
    assert lines[7] == "    4. Email AP -> ooa.email/send"
    assert lines[9] == "Check: 5 steps, 1 loop, 1 if, all facts used"
    assert lines[10] == "Flow:" and json.loads(lines[11]) == F.compact_flow(FLOW)


def test_think_answer_round_trip():
    ans = F.render_think_answer(FLOW, CAT, F.ANSWER_COMPACT, {"request": REQUEST})
    a = F.parse_answer(ans, expand=True, catalog=CAT, request=REQUEST)
    assert a["kind"] == "flow" and a["think"] is True
    assert a["clear"] == a["understand"] == "Email AP about the large March invoices."
    assert a["plan"] == F.plan_from_flow(FLOW, CAT)                    # Steps come back as the old plan lines
    assert a["steps"][0] == "1. Read the invoices -> ooa.csv/read"
    assert ("to", "ap@acme.com") in a["facts"] and a["missing"] == [] and a["check"].startswith("5 steps")
    flow = json.loads(a["flow_text"])
    assert F.actions_of(flow) == F.actions_of(FLOW) and a.get("expanded")
    assert F.plan_coverage(a["plan"], flow, CAT) == [] and F.plan_coverage(a["steps"], flow, CAT) == []
    assert F.fact_coverage(a["facts"], flow) == []
    # a full (not compact) flow too
    full = F.parse_answer(F.render_think_answer(FLOW, CAT, F.ANSWER_FULL, {"request": REQUEST}))
    assert json.loads(full["flow_text"])["steps"][0]["id"] == "s1"


def test_old_formats_still_parse():
    plan = F.render_flow_answer(FLOW, CAT, F.ANSWER_COMPACT)
    a = F.parse_answer(plan)
    assert a["kind"] == "flow" and a["plan"] == F.plan_from_flow(FLOW, CAT) and not a.get("think")
    c = F.parse_answer(F.render_flow_answer(FLOW, CAT, F.ANSWER_COMPACT, clear="Read it and email AP."))
    assert c["clear"] == "Read it and email AP." and c["plan"] == a["plan"] and c["flow_text"] == a["flow_text"]
    assert F.parse_answer(F.compact(FLOW))["kind"] == "flow"
    q = F.parse_answer("Question: Which folder?")
    assert q["kind"] == "question" and q["question"] == "Which folder?"
    assert F.parse_answer("Hello there")["kind"] == "reply"


def test_steps_lines_work_as_plan_lines():
    steps = F.plan_to_steps(F.plan_from_flow(FLOW, CAT))
    assert F.steps_to_plan(steps) == F.plan_from_flow(FLOW, CAT)
    assert F.plan_actions(steps) == F.actions_of(FLOW)                 # plan_items reads " -> id" too
    assert [i["depth"] for i in F.plan_items(steps)] == [0, 0, 1, 2, 0]
    assert F.plan_coverage(steps + ["6. Zip it -> ooa.file/zip"], FLOW, CAT) == ["ooa.file/zip"]
    assert len(F.plan_parts(steps, max_lines=2)) >= 2                  # the builder can split Steps


def test_question_answer():
    rec = {"kind": "clarify", "request": "move the files to the usual folder",
           "question": "Which folder do you mean? Please tell me its name or path."}
    t = F.render_think_question(rec)
    lines = t.split("\n")
    assert lines[0].startswith("Understand: Move the files to the usual folder. Not said: which folder")
    assert lines[1] == "Facts: none" and lines[2].startswith("Missing: folder")
    assert lines[-1] == "Question: Which folder do you mean? Please tell me its name or path."
    a = F.parse_answer(t)
    assert a["kind"] == "question" and a["question"].startswith("Which folder") and "folder" in a["missing"]
    # the record's own slots win; a question about records
    assert F.missing_slots({"request": "x", "question": "q", "missing": ["recipient"]}) == ["recipient"]
    assert F.missing_slots({"request": "close the old ones", "question": "Which cases do you mean (a file or a list)?"}) \
        == ["records"]
    # Missing without a question: one is made from the slots
    m = F.parse_answer("Understand: Send it.\nFacts: none\nMissing: recipient")
    assert m["kind"] == "question" and m["question"] == F.ask_question(["recipient"])
    m = F.parse_answer("Understand: Close them.\nFacts: none\nMissing: which tickets")
    assert m["kind"] == "question" and "which tickets" in m["question"]


def test_cut_off_answers():
    full = F.render_think_answer(FLOW, CAT, F.ANSWER_COMPACT, {"request": REQUEST})
    upto_flow = full[:full.index("\nFlow:") + len("\nFlow:")]               # stopped after the plan (parts)
    a = F.parse_answer(upto_flow)
    assert a["kind"] == "flow" and a["flow_text"] in (None, "") and len(a["plan"]) == 5
    b = F.parse_answer(full[:full.index("\nSteps:")])                       # stopped inside the block
    assert b["kind"] == "flow" and b["plan"] == [] and b["understand"]
    c = F.parse_answer(full[:full.index("\nCheck:")])                       # no Check line
    assert len(c["plan"]) == 5 and c["check"] is None


def test_hand_written_fields_win():
    own = ["1. Read march.csv -> ooa.csv/read", "2. For each row -> core/loop", "  3. If it is over 1000 -> core/if",
           "    4. Mail AP -> ooa.email/send", "5. Log it -> core/log"]
    rec = {"request": REQUEST, "understand": "Mail AP about each big invoice", "facts": {"file": "march.csv"},
           "steps": own}
    t = F.render_think_answer(FLOW, CAT, F.ANSWER_COMPACT, rec)
    assert t.startswith("Understand: Mail AP about each big invoice.\nFacts: file = march.csv\n")
    assert "\n".join(own) in t
    bad = dict(rec, steps=own[:2])                                          # steps that do not match the flow
    assert F.plan_to_steps(F.plan_from_flow(FLOW, CAT))[0] in F.render_think_answer(FLOW, CAT, "compact", bad)
    rw = F.render_think_answer(FLOW, CAT, "compact", {"request": REQUEST, "clear": "Read it, mail AP."})
    assert rw.startswith("Understand: Read it, mail AP.\n")                 # a rewrite record's restatement


def test_facts():
    facts = F.flow_facts(FLOW, REQUEST)
    assert facts[0] == ("schedule", "Every Monday at 9")
    assert F.flow_facts(FLOW, "do it") == []                               # nothing the request says
    templ = {"steps": [{"action": "ooa.csv/write", "args": {"path": "${folder}\\report_q1.csv"}}]}
    assert F.flow_facts(templ, "save report_q1.csv") == [("path", "report_q1.csv")]
    assert F.flow_facts({"steps": [{"action": "x/y", "args": {"sheet": "Sheet1"}}]}, "sheet 1") == []
    assert F.parse_facts(F.facts_text(facts)) == facts
    assert F.parse_facts("none") == [] and F.facts_text([]) == "none"
    assert F.fact_coverage([("to", "boss@acme.com"), ("schedule", "daily")], FLOW) == [("to", "boss@acme.com")]
    assert F.schedule_of("roz subah 9 baje") == "roz subah" and F.schedule_of("daily at 7 am send it") == "daily at 7 am"


def test_think_prefix_parses_like_a_plan():
    lines = F.plan_from_flow(FLOW, CAT)[:2]
    p = F.think_prefix("Read the invoices, then for each row: check it.", lines)
    assert p.startswith("Understand: Read the invoices") and p.endswith("\nFlow:\n")
    a = F.parse_answer(p + F.compact({"steps": [{"action": "ooa.csv/read"}]}))
    assert a["plan"] == lines and a["kind"] == "flow" and a["flow_text"].startswith("{")


def test_render_chat_think_and_override():
    ex = F.ExampleIndex.from_flows([("Read the invoices csv and email AP", FLOW)])
    rec = {"kind": "flow", "request": REQUEST, "flow": FLOW}
    row = F.render_chat(rec, CAT, INDEX, ex, random.Random(1), think=True)
    ans = row["messages"][1]["content"]
    assert ans.startswith("Understand: ") and "\nSteps:\n" in ans and F.CONTEXT_HEADER in row["messages"][0]["content"]
    plain = F.render_chat(dict(rec, think=False), CAT, INDEX, ex, random.Random(1), think=True)
    assert plain["messages"][1]["content"].startswith("Plan:")             # the record's flag wins
    q = F.render_chat({"kind": "clarify", "request": "automate my work", "question": "Which app?"}, CAT, INDEX, ex,
                      think=True)
    assert q["messages"][1]["content"].endswith("\nQuestion: Which app?")
    assert q["messages"][1]["content"].startswith("Understand: ")
    fix = F.render_chat({"kind": "fix", "request": REQUEST, "previous": {"steps": []}, "problems": ["No steps."],
                         "flow": FLOW}, CAT, INDEX, ex, random.Random(0), think=True)
    assert fix["messages"][1]["content"].startswith("Understand: ")


def _records(d):
    rows = [{"kind": "flow", "request": f"Read orders{i}.csv and sort it",
             "flow": {"kind": "TaskBot", "id": f"f{i}", "name": "f", "variables": [],
                      "steps": [{"id": "s1", "action": "ooa.csv/read", "args": {"path": f"orders{i}.csv"},
                                 "out": {"table": "t"}},
                                {"id": "s2", "action": "ooa.datatable/sort", "args": {"table": "$t$", "column": "id"}}]}}
            for i in range(4)]
    rows.append({"kind": "clarify", "request": "move the files to the usual folder", "question": "Which folder?"})
    a, b = os.path.join(d, "a_train.jsonl"), os.path.join(d, "b_train.jsonl")
    for p in (a, b):
        with open(p, "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    return a, b


def test_render_dataset_think_per_file():
    src, out = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        a, b = _records(src)
        os.makedirs(os.path.join(src, "empty"))
        rep = RD.render(os.path.join(src, "empty"), out, CATALOG, k=8, extra=[a, b], log=lambda *x: None,
                        think=["a_train.jsonl"])
        ra = [json.loads(l)["messages"][1]["content"] for l in open(os.path.join(out, "a_train.jsonl"))]
        rb = [json.loads(l)["messages"][1]["content"] for l in open(os.path.join(out, "b_train.jsonl"))]
        assert all(t.startswith("Understand: ") for t in ra) and len(ra) == 5
        assert all(t.startswith(("Plan:", "Question:")) for t in rb) and len(rb) == 5
        assert "\nFacts: path = orders0.csv\n" in ra[0], ra[0]
        assert rep["settings"]["think"] == ["a_train.jsonl"] and rep["files"]["a_train.jsonl"]["think"] == 5
        assert rep["files"]["b_train.jsonl"]["think"] == 0 and rep["totals"]["think"] == 5
        for t in ra:
            p = F.parse_answer(t, expand=True, catalog=CAT)
            assert p["think"] and p["kind"] in ("flow", "question")
        # the CLI flag
        out2 = tempfile.mkdtemp()
        p = subprocess.run([sys.executable, os.path.join(HERE, "render_dataset.py"), os.path.join(src, "empty"), out2,
                            "--catalog", CATALOG, "--k", "8", "--think", "all", "--extra", a, b],
                           capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, p.stderr
        r2 = json.load(open(os.path.join(out2, "render_report.json")))
        assert r2["settings"]["think"] == ["all"] and r2["totals"]["think"] == 10
        shutil.rmtree(out2)
    finally:
        shutil.rmtree(src, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


def test_long_flow_parts_think():
    steps = [{"id": f"s{i}", "action": a, "description": f"Step {i} " + "x" * 70, "args": {}}
             for i, a in enumerate(["ooa.csv/read", "ooa.datatable/filter", "ooa.datatable/sort", "ooa.excel/open",
                                    "ooa.excel/write_table", "ooa.excel/close", "ooa.email/send", "ooa.file/move"])]
    rec = {"kind": "flow", "request": "a long one", "flow": {"kind": "TaskBot", "id": "l", "name": "l", "steps": steps,
                                                            "variables": [], "description": "y" * 3000}}
    src, out = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        p = os.path.join(src, "long_train.jsonl")
        open(p, "w").write(json.dumps(rec) + "\n")
        os.makedirs(os.path.join(src, "empty"))
        rep = RD.render(os.path.join(src, "empty"), out, CATALOG, k=8, extra=[p], log=lambda *x: None,
                        think=["all"], max_chars=4500, answer_format=F.ANSWER_FULL)
        rows = [json.loads(l)["messages"] for l in open(os.path.join(out, "long_train.jsonl"))]
        assert rep["files"]["long_train.jsonl"]["part"] >= 2, rep["files"]
        for m in rows:
            req, ans = F.request_of(m[0]["content"]), m[1]["content"]
            assert ans.startswith("Understand: " + req[:20]) and "\nFacts: none\n" in ans   # as think_prefix writes it
    finally:
        shutil.rmtree(src, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


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
