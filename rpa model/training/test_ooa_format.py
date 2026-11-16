"""Tests for ooa_format.py and render_dataset.py. Standard library only, no platform
repo needed. Run: python test_ooa_format.py   (works under pytest too)."""
import json, os, random, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ooa_format as F, retrieval, render_dataset as RD

CATALOG = os.path.join(HERE, "../data/catalog/catalog.jsonl")
ENTRIES = [json.loads(l) for l in open(CATALOG, encoding="utf-8")]
CAT = {e["id"]: e for e in ENTRIES}
INDEX = retrieval.ActionIndex(ENTRIES)

LOOP_FLOW = {
    "kind": "TaskBot", "id": "zip_pngs", "name": "Zip PNGs", "description": "Zip the png files",
    "variables": [{"name": "folder", "type": "String", "direction": "input", "required": True},
                  {"name": "archive", "type": "String", "direction": "input", "required": True},
                  {"name": "files", "type": "List", "direction": "local"},
                  {"name": "count", "type": "Number", "direction": "local"}],
    "steps": [
        {"id": "s1", "action": "ooa.file/list", "description": "List files in the folder",
         "args": {"folder": "$folder"}, "out": {"files": "files", "count": "count"}},
        {"id": "s2", "action": "core/loop", "args": {"iterator": "list", "over": "$files", "as": "file"},
         "children": [
             {"id": "s3", "action": "core/if", "description": "Only for .png files",
              "args": {"condition": {"condition": "ooa.file/has_extension", "args": {"path": "$file", "extensions": "png"}}},
              "children": [{"id": "s4", "action": "ooa.file/zip", "args": {"source": "$file", "target": "$archive"}}],
              "else": [{"id": "s5", "action": "ooa.file/delete", "args": {"path": "$file"}}]}]}]}


def linear_flow(actions, fid="f"):
    return {"kind": "TaskBot", "id": fid, "name": fid, "variables": [{"name": "p", "type": "String", "direction": "input"}],
            "steps": [{"id": f"s{i + 1}", "action": a, "description": f"Step {i + 1} in plain words"}
                      for i, a in enumerate(actions)]}


# ---------------------------------------------------------------------------------------------- format
def test_outline_nesting():
    assert F.outline(LOOP_FLOW) == ("ooa.file/list > [loop: [if: ooa.file/zip | else: ooa.file/delete]]")
    t = F.truncate_outline("a > [loop: b > c > d > e > f > g > h]", 22)
    assert t.endswith("]") and "..." in t and t.count("[") == t.count("]"), t


def test_plan_from_flow_and_round_trip():
    plan = F.plan_from_flow(LOOP_FLOW, CAT)
    assert plan == ["1. List files in the folder (ooa.file/list)",
                    "2. For each file in files (core/loop)",
                    "  3. Only for .png files (core/if)",
                    "    4. Zip (ooa.file/zip)" if CAT["ooa.file/zip"]["name"] == "Zip" else plan[3],
                    "    5. Otherwise: " + CAT["ooa.file/delete"]["name"] + " (ooa.file/delete)"], plan
    text = F.render_flow_answer(LOOP_FLOW, CAT)
    a = F.parse_answer(text + "<|assistant_end|>")
    assert a["kind"] == "flow" and a["plan"] == plan and json.loads(a["flow_text"]) == LOOP_FLOW
    items = F.plan_items(a["plan"])
    assert [i["depth"] for i in items] == [0, 0, 1, 2, 2]
    assert items[4]["text"] == CAT["ooa.file/delete"]["name"] and items[4]["action"] == "ooa.file/delete"
    assert F.plan_coverage(a["plan"], LOOP_FLOW) == []
    no_zip = json.loads(json.dumps(LOOP_FLOW))
    no_zip["steps"][1]["children"][0]["children"] = []
    assert F.plan_coverage(a["plan"], no_zip) == ["ooa.file/zip"]


def test_parse_answer_kinds():
    old = json.dumps(LOOP_FLOW)
    a = F.parse_answer(old)
    assert a["kind"] == "flow" and a["plan"] == [] and a["flow_text"] == old          # models before plans
    assert F.parse_answer("```json\n" + old + "\n```")["kind"] == "flow"
    q = F.parse_answer("Question: Which folder should it watch?\n\nmore")
    assert q["kind"] == "question" and q["question"] == "Which folder should it watch?"
    r = F.parse_answer("Canberra. Tell me a task to automate.")
    assert r["kind"] == "reply" and r["flow_text"] is None
    cut = F.parse_answer("Plan:\n1. Read it (ooa.csv/read)\n2. Wri")              # stopped before Flow:
    assert cut["kind"] == "flow" and cut["flow_text"] is None and len(cut["plan"]) == 2
    assert F.parse_answer("Flow:\n{\"a\":1}")["flow_text"] == '{"a":1}'


def test_build_user_layout_and_trim():
    acts = [CAT[a] for a in ("ooa.csv/read", "ooa.excel/write_table", "core/loop")]
    u = F.build_user("Read sales.csv into Excel", acts, [("Read a csv", "ooa.csv/read")], [("folder", "String")])
    lines = u.split("\n")
    assert lines[0] == "Read sales.csv into Excel" and lines[1] == F.CONTEXT_HEADER
    assert lines[2] == "Variables: folder (String)" and lines[3] == "Actions:"
    assert lines[4].startswith("- ooa.csv/read: ") and "path*" in lines[4] and "| out: table" in lines[4]
    assert lines[-1] == "Similar: Read a csv => ooa.csv/read"
    assert F.request_of(u) == "Read sales.csv into Excel" and F.split_user(u)[0] == "Read sales.csv into Excel"
    # trimming: deterministic, keeps the required ids, stays under the budget
    many = INDEX.search("read excel workbook write table email send file", 60)
    req = ["ooa.email/send"]
    big = F.build_user("x", many + [CAT["ooa.email/send"]], [("r" * 300, "a > b")] * 2, required=req)
    assert big == F.build_user("x", many + [CAT["ooa.email/send"]], [("r" * 300, "a > b")] * 2, required=req)
    ctx = big[big.index(F.CONTEXT_HEADER):]
    assert len(ctx) <= F.MAX_CONTEXT_CHARS and "- ooa.email/send:" in ctx
    # self-correction sections; a long previous flow becomes its outline
    fix = F.build_user("x", acts, previous=LOOP_FLOW, problems=["Unknown action 'a'."] * 9)
    assert F.PREVIOUS_HEADER + "\n{" in fix and fix.count("\n- Unknown") == F.MAX_PROBLEMS
    huge = linear_flow(["ooa.csv/read"] * 200)
    assert "\nOutline: ooa.csv/read > " in F.build_user("x", acts, previous=huge, problems=["p"])


def test_split_user_formats():
    assert F.split_user('Existing variables: [{"name":"a","type":"String"}]\nRequest: do it') == ("do it", [("a", "String")])
    assert F.split_user("Actions:\nooa.csv/read | in: path\n\nRequest: read it")[0] == "read it"
    assert F.split_user("just this") == ("just this", [])


def test_chunk_and_merge():
    plan = F.plan_from_flow(linear_flow(["ooa.csv/read", "ooa.datatable/filter", "ooa.datatable/sort",
                                         "ooa.excel/open", "ooa.excel/write_table", "ooa.excel/close",
                                         "ooa.email/send"]), CAT)
    chunks = F.chunk_plan(plan)
    assert [len(c) for c in chunks] == [4, 3], [len(c) for c in chunks]
    assert all(2 <= len(c) <= 4 for c in chunks)
    assert F.chunk_request(chunks[0]) == "Step 1 in plain words, then Step 2 in plain words, then Step 3 in plain words, then Step 4 in plain words."
    assert F.chunk_actions(chunks[1]) == ["ooa.excel/write_table", "ooa.excel/close", "ooa.email/send"]
    nested = F.chunk_plan(F.plan_from_flow(LOOP_FLOW, CAT))
    assert len(nested) == 1                                      # one loop item: nothing to split
    assert "For each file in files: Only for .png files: " in F.chunk_request(nested[0])
    a = {"kind": "TaskBot", "id": "a", "name": "a", "variables": [{"name": "t", "type": "Table"}],
         "steps": [{"id": "s1", "action": "ooa.csv/read"}, {"id": "s2", "action": "core/loop", "children": [{"id": "s3", "action": "core/log"}]}]}
    b = {"kind": "TaskBot", "id": "b", "name": "b", "variables": [{"name": "t", "type": "Table"}, {"name": "u", "type": "String"}],
         "steps": [{"id": "s1", "action": "ooa.excel/open"}, {"id": "s3", "action": "ooa.excel/close"}]}
    m = F.merge_flows([a, b], "whole request")
    ids = [s["id"] for s, _ in F.walk(m["steps"])]
    assert len(ids) == len(set(ids)) == 5 and ids[:3] == ["s1", "s2", "s3"], ids
    assert [v["name"] for v in m["variables"]] == ["t", "u"] and m["description"] == "whole request" and m["id"] == "a"
    assert F.actions_of(m) == ["ooa.csv/read", "core/loop", "core/log", "ooa.excel/open", "ooa.excel/close"]


def test_plan_parts_cut_a_long_loop():
    plan = ["1. List the files (ooa.file/list)", "2. For each file (core/loop)", "  3. Read it (ooa.file/read_text)",
            "  4. Compute the HMAC (ooa.crypto/hmac)", "  5. Read the signature (ooa.file/read_text)",
            "  6. If they match (core/if)", "    7. Move to verified (ooa.file/move)",
            "    8. Otherwise: Move to quarantine (ooa.file/move)", "9. Email the summary (ooa.email/send)"]
    assert len(F.chunk_plan(plan)) == 1                 # the loop is one group: chunk_plan cannot cut it
    parts = F.plan_parts(plan)
    assert [p["inside"] for p in parts] == [False, True, False], parts
    assert [[i["line"].split(".")[0].strip() for g in p["chunk"] for i in g] for p in parts] == \
        [["1", "2", "3", "4", "5"], ["6", "7", "8"], ["9"]]
    assert F.chunk_request(parts[1]["chunk"]) == "If they match: Move to verified, otherwise: Move to quarantine."
    assert F.chunk_actions(parts[0]["chunk"]) == ["ooa.file/list", "core/loop", "ooa.file/read_text", "ooa.crypto/hmac"]
    # short plans and long non-loop groups are not cut
    assert [p["inside"] for p in F.plan_parts(plan[:4])] == [False]
    iff = ["1. If it is late (core/if)"] + [f"  {i}. Step {i} (core/log)" for i in range(2, 9)]
    assert len(F.plan_parts(iff)) == 1
    # merging: an inside part goes into the loop that ends the parts before it (also when it wrapped itself in the
    # same loop again); ids stay unique
    p1 = linear_flow(["ooa.file/list"], "a")
    p1["steps"].append({"id": "s2", "action": "core/loop", "args": {"iterator": "list", "over": "$files$", "as": "file"},
                        "children": [{"id": "s3", "action": "ooa.file/read_text"}]})
    p2 = linear_flow(["ooa.file/move", "ooa.file/move"], "b")
    p3 = linear_flow(["ooa.email/send"], "c")
    m = F.merge_flows([p1, p2, p3], "whole", [False, True, False])
    assert F.outline(m) == "ooa.file/list > [loop: ooa.file/read_text > ooa.file/move > ooa.file/move] > ooa.email/send"
    ids = [s["id"] for s, _ in F.walk(m["steps"])]
    assert len(ids) == len(set(ids)) == 6
    wrapped = {"kind": "TaskBot", "id": "w", "name": "w", "steps": [{"id": "s1", "action": "core/loop",
               "args": {"iterator": "list", "over": "$files$", "as": "file"}, "children": p2["steps"]}]}
    m = F.merge_flows([p1, wrapped], None, [False, True])
    assert F.outline(m) == "ooa.file/list > [loop: ooa.file/read_text > ooa.file/move > ooa.file/move]"
    assert F.outline(F.merge_flows([p3, p2], None, [False, True])) == "ooa.email/send > ooa.file/move > ooa.file/move"


def test_split_flow_parts_are_complete():
    flow = linear_flow(["ooa.csv/read", "ooa.datatable/filter", "ooa.datatable/sort", "ooa.excel/open",
                        "ooa.excel/write_table", "ooa.excel/close", "ooa.email/send"])
    flow["variables"] += [{"name": "t", "type": "Table", "direction": "local"}]
    flow["steps"][0]["out"] = {"table": "t"}
    flow["steps"][4]["args"] = {"table": "$t"}
    parts = F.split_flow(flow, CAT)
    assert len(parts) == 2
    (r1, p1, c1), (r2, p2, c2) = parts
    assert r1.startswith("Step 1") and [s["id"] for s in p2["steps"]] == ["s5", "s6", "s7"]
    assert {v["name"] for v in p1["variables"]} == {"t"} and ("t", "Table") in c2      # carried to the second part


def test_context_actions_gold_and_core():
    rng = random.Random(3)
    acts, req = F.context_actions("read invoices from csv files", ["ooa.csv/read", "ooa.pdf/merge"], CAT, INDEX, 12, rng)
    ids = [e["id"] for e in acts]
    assert {"ooa.csv/read", "ooa.pdf/merge", "core/loop", "core/if"} <= set(ids) and len(ids) == 14
    assert set(req) == {"ooa.csv/read", "ooa.pdf/merge", "core/loop", "core/if"}
    serve, _ = F.context_actions("read invoices from csv", (), CAT, INDEX, 12)
    assert [e["id"] for e in serve][-2:] == ["core/loop", "core/if"] and len(serve) <= 14
    shown = [{"request": "r", "outline": "o", "actions": ["ooa.email/send", "ooa.csv/read", "nope/x"]}]
    serve, _ = F.context_actions("read invoices from csv", (), CAT, INDEX, 12, examples=shown)
    assert serve[0]["id"] == "ooa.email/send" and len({e["id"] for e in serve}) == len(serve)     # examples' actions
    train, _ = F.context_actions("read invoices", ["ooa.pdf/merge"], CAT, INDEX, 12, random.Random(1), shown)
    assert {"ooa.email/send", "ooa.csv/read", "ooa.pdf/merge"} <= {e["id"] for e in train} and len(train) == 16


LONG_REQUEST = ("For each new invoice PDF in the inbox folder, read the total; if it's over 10,000 forward it for "
                "approval by email, otherwise send it to the department manager, then move the file to the archive folder.")


def test_request_clauses():
    cl = F.request_clauses(LONG_REQUEST)
    assert cl[0] == "new invoice PDF in the inbox folder" and "read the total" in cl, cl
    assert any("10,000" in c for c in cl) and cl[-1] == "move the file to the archive folder", cl   # 10,000 stays whole
    assert F.request_clauses("Read a CSV file into a table") == ["Read a CSV file into a table"]
    assert F.request_clauses("then and if") == [] and F.request_clauses("") == []


def test_context_actions_per_clause():
    """A long request gets per-clause candidates after the whole-request ones (votes from similar training requests
    and BM25 per clause); a one-clause request gets exactly what it got before; training rows use the same function."""
    flows = [("List the files in a folder and move each one to the archive folder",
              linear_flow(["ooa.file/list", "core/loop", "ooa.file/move"], "mv")),
             ("Send an email to the manager with the report", linear_flow(["ooa.email/send"], "em")),
             ("Read the text of a PDF invoice", linear_flow(["ooa.pdf/extract_text"], "pdf"))]
    ex = F.ExampleIndex.from_flows(flows)
    assert ex.vote("move each file to the archive folder", 2) == ["ooa.file/list", "ooa.file/move"]
    assert ex.vote("move each file to the archive folder", 2, exclude_sig=F.flow_signature(flows[0][1])) != \
        ["ooa.file/list", "ooa.file/move"]
    whole = [e["id"] for e in INDEX.search(LONG_REQUEST, 12)]
    acts, req = F.context_actions(LONG_REQUEST, (), CAT, INDEX, 12, example_index=ex)
    ids = [e["id"] for e in acts]
    assert ids[:len(whole)] == list(dict.fromkeys(whole))[:len(whole)] or ids[:3] == whole[:3], ids
    assert {"ooa.file/move", "ooa.email/send", "ooa.pdf/extract_text"} <= set(ids), ids
    assert ids[-2:] == ["core/loop", "core/if"] and len(ids) <= F.MAX_CONTEXT_ACTIONS and len(set(ids)) == len(ids)
    few, _ = F.context_actions(LONG_REQUEST, (), CAT, INDEX, 12, example_index=ex, max_actions=16)
    assert len(few) == 16 and [e["id"] for e in few][-2:] == ["core/loop", "core/if"]
    one, _ = F.context_actions("Read a CSV file into a table", (), CAT, INDEX, 12, example_index=ex)
    old, _ = F.context_actions("Read a CSV file into a table", (), CAT, INDEX, 12)
    assert [e["id"] for e in one] == [e["id"] for e in old]
    # training: gold always in, the same per-clause candidates fill it up, the row itself never votes
    train, required = F.context_actions(LONG_REQUEST, ["ooa.file/list", "ooa.file/move"], CAT, INDEX, 12,
                                        random.Random(2), example_index=ex, exclude_request=LONG_REQUEST)
    tids = {e["id"] for e in train}
    assert {"ooa.file/list", "ooa.file/move", "core/loop", "core/if"} <= tids and len(train) > 16
    assert set(required) == {"ooa.file/list", "ooa.file/move", "core/loop", "core/if"}
    # the context budget: a 4096-token model gets the larger one and keeps the per-clause candidates
    assert F.context_for(4096) == F.MAX_CONTEXT_CHARS_LONG and F.context_for(2048) == F.context_for(None) == F.MAX_CONTEXT_CHARS
    have = {e["id"] for e in acts}
    acts = acts + [e for e in INDEX.search("excel workbook sheet chart pivot table", 12) if e["id"] not in have]
    small = F.build_user(LONG_REQUEST, acts, (), (), required=req)
    big = F.build_user(LONG_REQUEST, acts, (), (), required=req, context_limit=F.context_for(4096))
    assert small.count("\n- ") < big.count("\n- ") == len(acts)
    assert len(F.request_of(big) + big.split(F.CONTEXT_HEADER, 1)[1]) <= len(LONG_REQUEST) + F.MAX_CONTEXT_CHARS_LONG + 2


def test_render_chat_rows():
    flows = [("Read the orders csv and filter it", linear_flow(["ooa.csv/read", "ooa.datatable/filter"], "a")),
             ("Read the orders csv and sort it", linear_flow(["ooa.csv/read", "ooa.datatable/sort"], "b")),
             ("Read the orders csv and filter it again", linear_flow(["ooa.csv/read", "ooa.datatable/filter"], "a"))]
    ex = F.ExampleIndex.from_flows(flows)
    for seed in range(20):
        row = F.render_chat({"kind": "flow", "request": flows[0][0], "flow": flows[0][1]}, CAT, INDEX, ex, random.Random(seed))
        user, ans = row["messages"][0]["content"], row["messages"][1]["content"]
        assert "- ooa.csv/read:" in user and "- ooa.datatable/filter:" in user          # gold always there
        assert "Similar: Read the orders csv and filter it =>" not in user              # never the row itself
        assert "filter it again" not in user                                            # nor the same flow
        assert ans.startswith("Plan:\n1. Step 1 in plain words (ooa.csv/read)") and "\nFlow:\n{" in ans
    fix = F.render_chat({"kind": "fix", "request": "r", "previous": {"steps": []}, "problems": ["The flow has no steps."],
                         "flow": flows[0][1]}, CAT, INDEX, ex, random.Random(0))
    assert F.PROBLEMS_HEADER in fix["messages"][0]["content"] and F.PREVIOUS_HEADER in fix["messages"][0]["content"]
    q = F.render_chat({"kind": "clarify", "request": "automate my work", "question": "Which app?"}, CAT, INDEX, ex)
    assert q["messages"][1]["content"] == "Question: Which app?" and F.CONTEXT_HEADER in q["messages"][0]["content"]
    assert F.render_chat({"kind": "fix", "request": "r", "flow": flows[0][1]}, CAT, INDEX) is None   # no problems



def test_approx_tokens():
    assert F.approx_tokens("") == 0 and F.approx_tokens("read the file") == 3
    assert F.approx_tokens('{"a":1}') == 7 and F.approx_tokens("°") == 2
    assert F.approx_tokens("internationalization") > 3                         # long words count as several
    row = {"messages": [{"role": "user", "content": "x " * 3000}, {"role": "assistant", "content": "y"}]}
    assert not F.row_fits(row) and F.row_fits({"messages": [{"role": "user", "content": "hi"}]})

# ---------------------------------------------------------------------------------------------- render_dataset
def _fixture(d):
    def chat(u, a):
        return json.dumps({"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]})
    rows = [chat(f"Read the orders csv number {i} and sort it", json.dumps(linear_flow(["ooa.csv/read", "ooa.datatable/sort"], f"f{i}")))
            for i in range(6)]
    rows.append(chat("Read the orders csv and email it", json.dumps(linear_flow(["ooa.csv/read", "ooa.email/send"], "e"))))
    rows.append(chat("What is the capital of France?", "Paris. Tell me a task to automate."))
    long_flow = linear_flow(["ooa.csv/read", "ooa.datatable/filter", "ooa.datatable/sort", "ooa.excel/open",
                             "ooa.excel/write_table", "ooa.excel/close", "ooa.email/send", "ooa.file/move"], "long")
    for s in long_flow["steps"]:
        s["description"] = "x" * 80
    long_flow["description"] = "y" * 3000
    rows.append(chat("A long one", json.dumps(long_flow)))
    open(os.path.join(d, "train.jsonl"), "w").write("\n".join(rows) + "\n")
    open(os.path.join(d, "val.jsonl"), "w").write(chat("Read the orders csv and write it to excel",
                                                       json.dumps(linear_flow(["ooa.csv/read", "ooa.excel/write_table"], "v"))) + "\n")
    extra = os.path.join(d, "records.jsonl")
    with open(extra, "w") as f:
        f.write(json.dumps({"kind": "clarify", "request": "automate my work", "question": "Which app and which steps?", "source": "t"}) + "\n")
        f.write(json.dumps({"kind": "fix", "request": "Read the orders csv", "previous": {"kind": "TaskBot", "steps": []},
                            "problems": ["The flow has no steps."], "flow": linear_flow(["ooa.csv/read"], "fx"), "source": "t"}) + "\n")
        f.write(json.dumps({"kind": "flow", "request": "Move the report file", "flow": linear_flow(["ooa.file/move"], "mv"), "source": "t"}) + "\n")
    return extra


def test_render_dataset_fixture():
    src, out = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        extra = _fixture(src)
        rep = RD.render(src, out, CATALOG, k=8, seed=7, extra=[extra], max_chars=4500, log=lambda *a: None)
        names = sorted(os.listdir(out))
        assert names == ["catalog.jsonl", "examples.jsonl", "records.jsonl", "render_report.json", "train.jsonl", "val.jsonl"], names
        train = [json.loads(l) for l in open(os.path.join(out, "train.jsonl"))]
        flows = [r for r in train if r["messages"][1]["content"].startswith("Plan:")]
        assert len(flows) >= 7
        for r in flows:
            user, ans = r["messages"][0]["content"], r["messages"][1]["content"]
            a = F.parse_answer(ans)
            for act in set(F.actions_of(json.loads(a["flow_text"]))):
                assert f"- {act}:" in user, (act, user)                                   # gold always present
            req = F.request_of(user)
            assert f"Similar: {req} =>" not in user                                       # never itself
            assert F.row_chars(r) <= 4500 and F.row_fits(r, 4500)                         # length cap
        assert any("Paris" in r["messages"][1]["content"] for r in train)                 # chat row kept
        f = rep["files"]["train.jsonl"]
        assert f["skipped_too_long"] + f["split_long"] >= 1 and f["rows"] == 9, f
        ex = [json.loads(l) for l in open(os.path.join(out, "examples.jsonl"))]
        reqs = {e["request"] for e in ex}
        assert "Read the orders csv and write it to excel" not in reqs                    # val is never an example
        assert "Move the report file" in reqs and all(e["outline"] for e in ex)
        val = [json.loads(l) for l in open(os.path.join(out, "val.jsonl"))]
        assert val[0]["messages"][1]["content"].startswith("Plan:")                       # val rendered the same way
        recs = [json.loads(l)["messages"] for l in open(os.path.join(out, "records.jsonl"))]
        assert [m[1]["content"].split("\n")[0].split(":")[0] for m in recs] == ["Question", "Plan", "Plan"]
        assert F.PROBLEMS_HEADER in recs[1][0]["content"]
        # same seed -> same output
        out2 = tempfile.mkdtemp()
        RD.render(src, out2, CATALOG, k=8, seed=7, extra=[extra], max_chars=4500, log=lambda *a: None)
        assert open(os.path.join(out, "train.jsonl")).read() == open(os.path.join(out2, "train.jsonl")).read()
        shutil.rmtree(out2)
    finally:
        shutil.rmtree(src, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


def test_render_dataset_cli():
    src, out = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        _fixture(src)
        p = subprocess.run([sys.executable, os.path.join(HERE, "render_dataset.py"), src, out, "--catalog", CATALOG,
                            "--k", "8", "--seed", "3"], capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, p.stderr
        assert "rows written" in p.stdout and os.path.exists(os.path.join(out, "render_report.json"))
    finally:
        shutil.rmtree(src, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


def test_dollar_name_dollar_platform_saving_rule():
    """Studio's upgradeRefs cases (platform/ui/src/studio/refs.test.ts) and conformance case 029."""
    flow = {"kind": "TaskBot", "id": "a", "name": "A", "variables": [{"name": "count", "type": "Number"}], "steps": [
        {"id": "s1", "action": "core/set", "args": {"variable": "count", "value": "$count"}, "note": "uses $count"},
        {"id": "s2", "action": "core/log", "args": {"message": "Total ${count}", "other": "${count}",
                                                     "calc": {"expr": "$count + ${count} * 2"}, "keep": {"literal": "$count"}}},
        {"id": "s3", "action": "core/if", "args": {"condition": {"op": "gt", "left": "$count", "right": 1}},
         "children": [{"id": "s4", "action": "core/log", "args": {"message": "$count$"}}]}],
        "subflows": [{"name": "Tidy", "steps": [{"id": "t1", "action": "core/log", "args": {"message": "$sys.date"}}]}]}
    g = F.to_dollar_name_dollar(flow)
    assert g["steps"][0] == {"id": "s1", "action": "core/set", "args": {"variable": "count", "value": "$count$"}, "note": "uses $count"}
    assert g["steps"][1]["args"] == {"message": "Total $count$", "other": "$count$", "calc": {"expr": "$count$ + $count$ * 2"},
                                     "keep": {"literal": "$count"}}
    assert g["steps"][2]["args"] == {"condition": {"op": "gt", "left": "$count$", "right": 1}}
    assert g["steps"][2]["children"][0]["args"] == {"message": "$count$"}
    assert g["subflows"][0]["steps"][0]["args"] == {"message": "$sys.date$"}
    assert F.to_dollar_name_dollar(g) == g                     # saving again changes nothing
    assert flow["steps"][0]["args"]["value"] == "$count"       # the input is not changed
    # conformance 029: spreadsheet references, US$ amounts, a variable index and an unknown name stay as typed
    c29 = {"variables": [{"name": n} for n in ("customer", "count", "rates", "reason")], "steps": [
        {"id": "s2", "action": "core/set", "args": {"variable": "msg", "value": "Total $count for ${customer} (US$ 5, $other)"}},
        {"id": "s3", "action": "core/set", "args": {"variable": "f", "value": "=B2*$F$1 for $customer$"}},
        {"id": "s4", "action": "core/set", "args": {"variable": "rate", "value": "$rates[$reason]"}},
        {"id": "s5", "action": "core/loop", "args": {"over": "$rows", "as": "row"}, "children": [
            {"id": "s6", "action": "core/log", "args": {"message": "Row $row.Name, done."}},
            {"id": "s7", "action": "ooa.powershell/run", "args": {"script": "Write-Host $count"}},
            {"id": "s8", "action": "core/comment", "args": {"text": "uses $count"}}]}]}
    a = [s["args"] for s, _ in F.walk(F.to_dollar_name_dollar(c29)["steps"])]
    assert a[0]["value"] == "Total $count$ for $customer$ (US$ 5, $other)"
    assert a[1]["value"] == "=B2*$F$1 for $customer$"
    assert a[2]["value"] == "$rates[$reason]"
    assert a[3]["over"] == "$rows$"
    assert a[4]["message"] == "Row $row.Name$, done."
    assert a[5]["script"] == "Write-Host $count" and a[6]["text"] == "uses $count"
    # names given by the caller (Studio's open agent) count as known
    steps = [{"id": "s1", "action": "core/log", "args": {"message": "Hello $who"}}]
    assert F.to_dollar_name_dollar(steps, ["who"])[0]["args"]["message"] == "Hello $who$"
    assert F.to_dollar_name_dollar(steps)[0]["args"]["message"] == "Hello $who"


def test_dollar_name_dollar_reads_back():
    """$name$ flows: the plan words, the checker's references and render_chat targets."""
    g = F.to_dollar_name_dollar(LOOP_FLOW)
    assert g["steps"][1]["args"]["over"] == "$files$"
    assert F.plan_from_flow(g, CAT) == F.plan_from_flow(LOOP_FLOW, CAT)
    assert "For each file in files" in "\n".join(F.plan_from_flow(g, CAT))
    import ooa_reward as R
    assert R.refs(g["steps"][1]["args"]) == {"files"}
    row = F.render_chat({"kind": "flow", "request": "zip the png files", "flow": LOOP_FLOW}, CAT, INDEX, None,
                        random.Random(1))
    ans = F.parse_answer(row["messages"][1]["content"])
    assert '"$folder$"' in ans["flow_text"] and '"$folder"' not in ans["flow_text"]
    bad = json.loads(F.compact(LOOP_FLOW))
    del bad["steps"][1]
    fix = F.render_chat({"kind": "fix", "request": "zip the png files", "flow": LOOP_FLOW, "previous": bad,
                         "problems": ["The plan names core/loop but no step of the flow uses it."]}, CAT, INDEX, None,
                        random.Random(1))
    prev = fix["messages"][0]["content"].split(F.PREVIOUS_HEADER, 1)[1]
    assert '"$folder$"' in prev


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
