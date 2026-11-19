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
    assert F.chunk_plan_lines(parts[1]["chunk"]) == ["1. If they match (core/if)", "  2. Move to verified (ooa.file/move)",
                                                     "  3. Otherwise: Move to quarantine (ooa.file/move)"]
    pre = F.answer_prefix(F.chunk_plan_lines(parts[0]["chunk"]))
    assert pre.startswith("Plan:\n1. List the files (ooa.file/list)\n2. For each file (core/loop)\n  3. Read it")
    assert pre.endswith("\nFlow:\n") and F.parse_answer(pre + "{}")["plan"] == F.chunk_plan_lines(parts[0]["chunk"])
    # short plans are not cut; nor is a long group that is not a loop, try or if
    assert [p["inside"] for p in F.plan_parts(plan[:4])] == [False]
    sw = ["1. Run them all at once (core/parallel)"] + [f"  {i}. Step {i} (core/log)" for i in range(2, 9)]
    assert len(F.plan_parts(sw)) == 1
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


def _part_lines(parts):
    return [[i["line"].split(".")[0].strip() for g in p["chunk"] for i in g] for p in parts]


def test_plan_parts_cut_a_long_try_and_nested_containers():
    # a try around the whole process (seen on the real model: a part of 12 lines): the try line, its first children
    # and its "On error" lines in one part (the part that writes the try and its catch), the rest of its body inside
    plan = ["1. Try to build the summary (core/try)", "  2. Open the mailbox (ooa.email/connect)",
            "  3. List the invoices (ooa.email/list)", "  4. Read the totals (ooa.excel/read_range)",
            "  5. Write the summary (ooa.file/write_text)", "  6. Move the file (ooa.file/move)",
            "  7. Post it in Slack (ooa.slack/post_message)", "  8. On error: Email the error (ooa.email/send)",
            "  9. Log the error (core/log)", "10. Log that it is done (core/log)"]
    parts = F.plan_parts(plan)
    assert [p["inside"] for p in parts] == [False, True, False], parts
    assert _part_lines(parts) == [["1", "2", "3", "8", "9"], ["4", "5", "6", "7"], ["10"]]
    assert F.chunk_plan_lines(parts[0]["chunk"])[-2:] == ["  4. On error: Email the error (ooa.email/send)",
                                                          "  5. Log the error (core/log)"]
    # a loop whose child is a long try: the try is cut inside the loop (inside=2 goes into the try in the loop)
    plan = ["1. List the files (ooa.file/list)", "2. For each file (core/loop)", "  3. Read it (ooa.file/read_text)",
            "  4. Try to check it (core/try)", "    5. Compute the HMAC (ooa.crypto/hmac)",
            "    6. Read the signature (ooa.file/read_text)", "    7. Compare them (core/if)",
            "    8. Move it to verified (ooa.file/move)", "    9. On error: Log the failure (core/log)",
            "  10. Log the file (core/log)", "11. Email the summary (ooa.email/send)"]
    parts = F.plan_parts(plan)
    assert [p["inside"] for p in parts] == [False, True, 2, True, False], parts
    assert _part_lines(parts) == [["1", "2", "3"], ["4", "5", "9"], ["6", "7", "8"], ["10"], ["11"]]
    # merging follows the levels: part 3 into the try inside the loop, part 4 into the loop after the try
    p = [linear_flow(["ooa.file/list"], "a"), linear_flow(["ooa.crypto/hmac"], "b"), linear_flow(["core/log"], "c"),
         linear_flow(["ooa.file/move"], "d"), linear_flow(["ooa.email/send"], "e")]
    p[0]["steps"].append({"id": "s2", "action": "core/loop", "args": {"iterator": "list", "over": "$files$",
                          "as": "file"}, "children": [{"id": "s3", "action": "ooa.file/read_text"}]})
    p[1]["steps"] = [{"id": "s1", "action": "core/try", "children": p[1]["steps"],
                      "catch": [{"children": [{"id": "s9", "action": "core/log"}]}]}]
    m = F.merge_flows(p, "whole", [False, True, 2, True, False])
    assert F.actions_of(m) == ["ooa.file/list", "core/loop", "ooa.file/read_text", "core/try", "ooa.crypto/hmac",
                               "core/log", "core/log", "ooa.file/move", "ooa.email/send"], F.actions_of(m)
    loop = m["steps"][1]
    assert [s["action"] for s in loop["children"]] == ["ooa.file/read_text", "core/try", "ooa.file/move"]
    assert [s["action"] for s in loop["children"][1]["children"]] == ["ooa.crypto/hmac", "core/log"]
    assert F.container_path(m["steps"], 2) is None and F.container_path(m["steps"][:2], 1) == [loop]
    # a loop whose only child is a long try (seen on the real model): the cut is inside the try
    plan = (["1. List the files (ooa.file/list)", "2. For each file (core/loop)", "  3. Try to check it (core/try)"]
            + [f"    {i}. Step {i} (core/log)" for i in range(4, 10)] + ["    10. On error: Log it (core/log)"])
    parts = F.plan_parts(plan)
    assert [p["inside"] for p in parts] == [False, False, 2], parts
    assert _part_lines(parts) == [["1"], ["2", "3", "4", "5", "10"], ["6", "7", "8", "9"]]
    # a long if is cut too, its "Otherwise" branch stays with the if line
    iff = (["1. If it is late (core/if)"] + [f"  {i}. Step {i} (core/log)" for i in range(2, 8)]
           + ["  8. Otherwise: Wait (core/log)"])
    parts = F.plan_parts(iff)
    assert [p["inside"] for p in parts] == [False, True] and _part_lines(parts) == [["1", "2", "3", "8"], ["4", "5", "6", "7"]]


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
    serve, _ = F.context_actions("read invoices from csv", (), CAT, INDEX, 12, examples=shown, app_aware=False)
    assert serve[0]["id"] == "ooa.email/send" and len({e["id"] for e in serve}) == len(serve)     # examples' actions
    serve, _ = F.context_actions("read invoices from csv", (), CAT, INDEX, 12, examples=shown)
    assert [e["id"] for e in serve][:2] == ["ooa.csv/read", "ooa.email/send"]       # a verb hint, then the examples'
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
    acts, req = F.context_actions(LONG_REQUEST, (), CAT, INDEX, 12, example_index=ex, app_aware=False)
    ids = [e["id"] for e in acts]
    assert ids[:len(whole)] == list(dict.fromkeys(whole))[:len(whole)] or ids[:3] == whole[:3], ids
    acts, req = F.context_actions(LONG_REQUEST, (), CAT, INDEX, 12, example_index=ex)
    ids = [e["id"] for e in acts]
    hints = [a for a in F.verb_hints(LONG_REQUEST) if a in CAT]
    assert hints and ids[:len(hints)] == hints, (hints, ids)     # app-aware: the verb hints first
    assert {"ooa.file/move", "ooa.email/send", "ooa.pdf/extract_text"} <= set(ids), ids
    assert ids[-2:] == ["core/loop", "core/if"] and len(ids) <= F.MAX_CONTEXT_ACTIONS and len(set(ids)) == len(ids)
    few, _ = F.context_actions(LONG_REQUEST, (), CAT, INDEX, 12, example_index=ex, max_actions=16)
    assert len(few) == 16 and [e["id"] for e in few][-2:] == ["core/loop", "core/if"]
    one, _ = F.context_actions("Read a CSV file into a table", (), CAT, INDEX, 12, example_index=ex)
    old, _ = F.context_actions("Read a CSV file into a table", (), CAT, INDEX, 12)
    assert [e["id"] for e in one] == [e["id"] for e in old]
    # training: gold always in, the same per-clause candidates fill it up, the row itself never votes
    train, required = F.context_actions(LONG_REQUEST, ["ooa.file/list", "ooa.file/move"], CAT, INDEX, 12,
                                        random.Random(2), example_index=ex, exclude_request=LONG_REQUEST,
                                        app_aware=False)
    tids = {e["id"] for e in train}
    assert {"ooa.file/list", "ooa.file/move", "core/loop", "core/if"} <= tids and len(train) > 16
    assert set(required) == {"ooa.file/list", "ooa.file/move", "core/loop", "core/if"}
    # app-aware (the default): fewer, because the other apps' mail actions (Exchange, Outlook, Microsoft 365) are out
    train, required = F.context_actions(LONG_REQUEST, ["ooa.file/list", "ooa.file/move"], CAT, INDEX, 12,
                                        random.Random(2), example_index=ex, exclude_request=LONG_REQUEST)
    tids = {e["id"] for e in train}
    assert {"ooa.file/list", "ooa.file/move", "core/loop", "core/if"} <= tids and len(train) >= 14
    assert not any(a.split("/")[0] in F.APP_NAMES for a in tids), tids
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
        rep = RD.render(src, out, CATALOG, k=8, seed=7, extra=[extra], max_chars=4500, log=lambda *a: None,
                        gold_dropout=0.0, answer_format=F.ANSWER_FULL)
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
        RD.render(src, out2, CATALOG, k=8, seed=7, extra=[extra], max_chars=4500, log=lambda *a: None, gold_dropout=0.0,
                  answer_format=F.ANSWER_FULL)
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


# ---------------------------------------------------------------------------------------------- compact answers
def test_compact_flow_drops_what_code_rebuilds():
    c = F.compact_flow(LOOP_FLOW)
    assert set(c) == {"steps"}, c                                      # no kind / id / name / description / variables
    for s, _ in F.walk(c["steps"]):
        assert "id" not in s and "description" not in s and "note" not in s
    assert F.actions_of(c) == F.actions_of(LOOP_FLOW)
    assert F.is_compact(c) and not F.is_compact(LOOP_FLOW)
    out = dict(LOOP_FLOW, variables=LOOP_FLOW["variables"] + [{"name": "n", "type": "Number", "direction": "output"}])
    assert F.compact_flow(out)["outputs"] == ["n"]
    assert len(F.compact(c)) < 0.6 * len(F.compact(LOOP_FLOW))


def test_expand_flow_rebuilds_a_full_flow():
    c = F.compact_flow(LOOP_FLOW)
    plan = F.plan_from_flow(LOOP_FLOW, CAT)
    ex = F.expand_flow(c, CAT, "Zip the png files in a folder, delete the others", plan)
    assert ex["kind"] == "TaskBot" and ex["id"] and ex["name"] and ex["description"]
    ids = [s["id"] for s, _ in F.walk(ex["steps"])]
    assert len(ids) == len(set(ids)) == 5
    v = {x["name"]: x for x in ex["variables"]}
    assert v["folder"]["direction"] == "input" and v["archive"]["direction"] == "input"      # only read
    assert v["files"]["direction"] == "local" and v["file"]["direction"] == "local"          # made by a step / the loop
    assert v["files"]["type"] == "List"                                                       # catalog output type
    assert F.actions_of(ex) == F.actions_of(LOOP_FLOW)
    assert ex["steps"][0]["description"]                                                      # from the plan
    assert not F.is_compact(ex)


def test_expand_flow_credentials_out_targets_and_known():
    c = {"steps": [{"action": "ooa.email/send", "args": {"to": "$boss$", "subject": "Hi", "password": "${smtp_password}",
                                                          "body": "$api_token"}, "out": {"message_id": "$mid$"}},
                   {"id": "s1", "action": "core/log", "args": {"message": "$mid$ $known_var$"}},
                   {"id": "s1", "action": "core/log", "args": {"message": {"literal": "$not_a_var$"}}}],
         "outputs": ["mid"]}
    ex = F.expand_flow(c, CAT, "Send it", known=[{"name": "known_var", "type": "String"}])
    v = {x["name"]: x for x in ex["variables"]}
    assert v["smtp_password"]["type"] == "Credential" and v["api_token"]["type"] == "Credential"
    assert v["boss"]["type"] == "String" and v["boss"]["direction"] == "input"
    assert v["mid"]["direction"] == "output"
    assert "known_var" not in v and "not_a_var" not in v
    assert ex["steps"][0]["out"] == {"message_id": "mid"}                                     # "$mid$" -> "mid"
    ids = [s["id"] for s, _ in F.walk(ex["steps"])]
    assert len(set(ids)) == 3 and "s1" in ids                                                 # duplicate id renamed
    assert c["steps"][0]["out"] == {"message_id": "$mid$"}                                    # input unchanged


def test_parse_answer_expands_only_compact_flows():
    full = F.render_flow_answer(LOOP_FLOW, CAT)
    a = F.parse_answer(full, expand=True, catalog=CAT, request="Zip the png files")
    assert not a.get("expanded") and json.loads(a["flow_text"]) == LOOP_FLOW               # old model: as written
    comp = F.render_flow_answer(LOOP_FLOW, CAT, F.ANSWER_COMPACT)
    assert comp.split(F.FLOW_HEAD)[0] == full.split(F.FLOW_HEAD)[0]                         # same plan
    assert len(comp) < len(full)
    b = F.parse_answer(comp)
    assert F.is_compact(json.loads(b["flow_text"]))                                          # no expand asked
    b = F.parse_answer(comp, expand=True, catalog=CAT, request="Zip the png files")
    assert b["expanded"] and F.is_compact(json.loads(b["compact_text"]))
    ex = json.loads(b["flow_text"])
    assert ex["variables"] and F.actions_of(ex) == F.actions_of(LOOP_FLOW)


def test_drop_gold():
    rng = random.Random(3)
    gold = ["ooa.csv/read", "core/loop", "ooa.datatable/sort", "core/if", "ooa.email/send", "ooa.file/move"]
    assert F.drop_gold(gold, rng, 0.0) == []
    hits = [F.drop_gold(gold, rng, 0.5) for _ in range(400)]
    share = sum(1 for h in hits if h) / len(hits)
    assert 0.4 < share < 0.6, share
    for h in hits:
        assert not set(h) & {"core/loop", "core/if"}
        assert len(h) <= 3                                    # 4 droppable: at most 45% -> 2, and one always kept
    assert all(len(F.drop_gold(["ooa.csv/read"], rng, 1.0)) == 1 for _ in range(5))
    assert F.drop_gold(["core/loop", "core/if"], rng, 1.0) == []


def test_render_gold_dropout_keeps_the_answer():
    src, out0, out1 = tempfile.mkdtemp(), tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        extra = _fixture(src)
        RD.render(src, out0, CATALOG, k=8, seed=7, extra=[extra], max_chars=4500, log=lambda *a: None, gold_dropout=0.0)
        rep = RD.render(src, out1, CATALOG, k=8, seed=7, extra=[extra], max_chars=4500, log=lambda *a: None,
                        gold_dropout=1.0)
        assert rep["settings"]["gold_dropout"] == 1.0 and rep["settings"]["answer_format"] == F.ANSWER_COMPACT
        assert rep["settings"]["format"] == F.FORMAT_COMPACT
        a0 = [json.loads(l)["messages"][1]["content"] for l in open(os.path.join(out0, "train.jsonl"))]
        r1 = [json.loads(l)["messages"] for l in open(os.path.join(out1, "train.jsonl"))]
        assert a0 == [m[1]["content"] for m in r1]                                    # plan and flow unchanged
        missing = 0
        for user, ans in ((m[0]["content"], m[1]["content"]) for m in r1):
            if not ans.startswith("Plan:"):
                continue
            flow = json.loads(F.parse_answer(ans)["flow_text"])
            assert F.is_compact(flow)
            acts = set(F.actions_of(flow))
            gone = [a for a in acts if f"- {a}:" not in user]
            assert gone and not set(gone) & {"core/loop", "core/if"}, (acts, user)
            missing += len(gone)
        assert missing >= 7
        f = rep["files"]["train.jsonl"]
        assert f["split_long"] == 0 and f["skipped_too_long"] == 0, f      # compact: the long flow fits whole
    finally:
        for d in (src, out0, out1):
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------------------------- rewrite ("Clear request:")
def test_clear_request_names_the_steps():
    c = F.clear_request(LOOP_FLOW, CAT)
    assert c.startswith("List files in the folder") and c.endswith(".") and "otherwise" in c, c
    assert F.clear_request(linear_flow(["ooa.csv/read", "ooa.excel/close", "ooa.csv/write"]), CAT) == \
        "Step 1 in plain words, then step 3 in plain words."               # closing steps are left out
    long = linear_flow(["ooa.csv/read"] * 40)
    c = F.clear_request(long, CAT, limit=120)
    assert len(c) <= 121 and c.endswith("and more."), c
    # a body that does not fit is cut to its first step, never to nothing
    c = F.clear_request(LOOP_FLOW, CAT, limit=90)
    assert c.endswith("then for each file in files: only for .png files: zip."), c
    loop = {"steps": [{"action": "core/loop", "description": "Repeat for each item", "args": {"over": "$rows", "as": "row"},
                       "children": [{"action": "ooa.salesforce/create_record", "description": "Create a Salesforce task"}]
                       + [{"action": "core/log", "description": "Log step %d of the row" % i} for i in range(30)]}]}
    c = F.clear_request(loop, CAT, limit=100)
    assert c.startswith("Repeat for each item: create a Salesforce task"), c
    # a description copied from the catalog gives way to the catalog name; one naming an app the request does not
    # name too
    e = CAT["ooa.csv/read"]
    flow = {"steps": [{"action": "ooa.csv/read", "description": e["description"]}]}
    assert not F.clear_step_ok(flow["steps"][0], CAT)
    assert F.clear_request(flow, CAT) == e["name"][:1].upper() + e["name"][1:].rstrip(".") + "."
    up = {"action": "ooa.microsoft365/upload_file", "description": "Upload the report to SharePoint"}
    assert F.clear_step_ok(up, CAT) and F.clear_step_ok(up, CAT, "put it on SharePoint")
    assert not F.clear_step_ok(up, CAT, "Put the CSV on OneDrive under Reports")
    c = F.clear_request({"steps": [up]}, CAT, request="Put the CSV on OneDrive under Reports")
    assert "Upload the report to SharePoint" not in c, c


def test_rewrite_answer_parses_back():
    clear = "Read the CSV, then write it to Excel."
    ans = F.render_flow_answer(LOOP_FLOW, CAT, F.ANSWER_COMPACT, clear)
    assert ans.startswith(F.CLEAR_HEAD + " " + clear + "\n" + F.PLAN_HEAD)
    a = F.parse_answer(ans + "<|assistant_end|>")
    assert a["kind"] == "flow" and a["clear"] == clear and a["plan"] == F.plan_from_flow(LOOP_FLOW, CAT)
    assert F.flow_object(a["flow_text"]) == F.compact_flow(LOOP_FLOW)
    assert F.parse_answer(F.render_flow_answer(LOOP_FLOW, CAT))["clear"] is None        # no restatement
    q = F.parse_answer("Clear request: Move the files.\nQuestion: Which folder?")
    assert q["kind"] == "question" and q["question"] == "Which folder?" and q["clear"] == "Move the files."
    cut = F.parse_answer("Clear request: Move the files")                               # stopped before the plan
    assert cut["kind"] == "flow" and cut["flow_text"] is None and cut["clear"] == "Move the files"


def test_render_rewrite_record():
    import random as _r
    rec = {"kind": "rewrite", "request": "pls tidy my pngs into a zip", "flow": LOOP_FLOW}
    row = F.render_chat(rec, CAT, INDEX, None, _r.Random(1))
    ans = row["messages"][1]["content"]
    assert ans.startswith(F.CLEAR_HEAD + " " + F.clear_request(LOOP_FLOW, CAT))      # clear made from the flow
    row = F.render_chat({**rec, "clear": "Zip the png files."}, CAT, INDEX, None, _r.Random(1))
    assert row["messages"][1]["content"].startswith("Clear request: Zip the png files.\nPlan:")
    assert row["messages"][0]["content"].startswith("pls tidy my pngs into a zip\n" + F.CONTEXT_HEADER)
    flow_row = F.render_chat({**rec, "kind": "flow"}, CAT, INDEX, None, _r.Random(1))
    assert flow_row["messages"][1]["content"].startswith(F.PLAN_HEAD)

def test_named_apps():
    assert F.named_apps("nudge the team in the #sales slack channel") == ["ooa.slack"]
    assert F.named_apps("Download the report from SFTP and load it into the database") == ["ooa.file_transfer",
                                                                                           "ooa.database"]
    assert set(F.named_apps("book it in Outlook")) == {"ooa.microsoft365", "ooa.outlook_desktop"}
    assert F.named_apps("message the sales head on Teams") == ["ooa.microsoft365"]
    assert F.named_apps("put prices.xlsx up on our google drive") == ["ooa.google"]
    assert F.named_apps("stop the EC2 instances tagged dev") == ["ooa.aws_ec2"]
    # local things are no app: a folder on the E drive, a text box, the inbox, an email, the team
    for r in ["keep a spare copy of my Projects folder on the E drive", "type it into the search box",
              "save the attachments of the unread mails in the inbox", "Make a new folder called Reports",
              "Delete all the files in the temp folder", "the sales team", "print invoice.pdf"]:
        assert F.named_apps(r) == [], (r, F.named_apps(r))


def test_app_aware_context():
    """A named app's best actions are offered and kept by trimming; apps the request does not name are left out
    (or only fill a short list); without app_aware the context is as before."""
    req = "nudge the team in the #sales slack channel that timesheets are due on friday"
    acts, keep = F.context_actions(req, (), CAT, INDEX, 12)
    ids = [e["id"] for e in acts]
    slack = [a for a in ids if a.startswith("ooa.slack/")]
    assert "ooa.slack/post_message" in ids and set(slack[:F.APP_TOP]) <= set(keep), (ids, keep)
    assert ids[:len(slack)] == slack                                       # the named app first
    for req in ["Make a new folder called Reports", "Delete all the files in the temp folder",
                "List the files in a folder", "keep a spare copy of my whole Projects folder on the E drive"]:
        ids = [e["id"] for e in F.context_actions(req, (), CAT, INDEX, 12)[0]]
        other = [a for a in ids if a.split("/")[0] in F.APP_NAMES]
        assert len(other) <= F.DEMOTED_FILL and all(ids.index(a) >= len(ids) - len(other) - 2 for a in other), ids
    ids = [e["id"] for e in F.context_actions("Make a new folder called Reports", (), CAT, INDEX, 12)[0]]
    assert ids[0] == "ooa.folder/create", ids
    old = [e["id"] for e in F.context_actions("Make a new folder called Reports", (), CAT, INDEX, 12,
                                              app_aware=False)[0]]
    assert any(a.split("/")[0] in F.APP_NAMES for a in old[:12])          # what it fixes: cloud folders came first
    # training rows: the gold is in whatever app it uses, the named app's best actions are kept too
    train, req = F.context_actions("post the total in the #finance slack channel", ["ooa.csv/read"], CAT, INDEX, 12,
                                   random.Random(4))
    assert "ooa.csv/read" in req and any(a.startswith("ooa.slack/") for a in req)
    assert {e["id"] for e in train} >= set(req)


def test_verb_hints():
    h = F.verb_hints
    assert set(h("print invoice.pdf for me")) >= {"ooa.printer/print_file", "ooa.file/print_files"}
    assert "ooa.excel_desktop/export_pdf" in h("convrt the reprot.xlsx to pdf and mail it to boss@company.com")
    assert "ooa.email/send" in h("convrt the reprot.xlsx to pdf and mail it to boss@company.com")
    assert h("word doc minutes.docx -> pdf")[:2] == ["ooa.word/open", "ooa.word/save_as"]
    assert "ooa.word/save_as" not in h("pdf to text pls, file is notes.pdf")          # a PDF in, not out
    assert set(h("summary of minutes.pdf pls")) >= {"ooa.genai/summarize", "ooa.pdf/extract_text"}
    assert set(h("letter.txt but in hindi")) >= {"ooa.genai/translate", "ooa.file/read_text"}
    assert "ooa.folder/create" in h("new folder on the desktop called Clients 2026")
    assert "ooa.folder/create" not in h("For each new invoice PDF in the inbox folder, read the total")
    assert "ooa.file/delete" in h("get rid of the junk in C:\\Temp")
    assert "ooa.file/delete" not in h("remove duplicates from customers.csv")
    assert "ooa.datatable/remove_duplicates" in h("remove dupliactes from custmers.csv")
    assert "ooa.file/unzip" in h("open up backup.zip onto the desktop")
    assert "ooa.datatable/row_count" in h("how many rows does data.csv have")
    assert "ooa.file/move" not in h("move the emails to Processed")
    assert "ooa.pdf/extract_text" not in h("save their PDF attachments into the BGV folder, mark each email read")
    assert h("Read a CSV file into a table") == ["ooa.csv/read"] and h("") == []


def test_ask_first_gate():
    """Requests that only point at something the user knows get a question naming it; requests that name what
    they need (even generally: 'a CSV file', 'a folder', 'the website ... as a CSV file') never do."""
    for req, word in [("rename them properly", "files"), ("put todays entries into the tracker", "file"),
                      ("copy the important ones over to the other folder", "folder"),
                      ("send the sales data to the app", "app"), ("check the website for me and tell me", "website"),
                      ("move the invoices into the usual folder", "folder"),
                      ("Kindly change the format of that document", "file"),
                      ("remind people about the deadline", "who")]:
        a = F.ask_first(req)
        assert a and word in a["question"].lower(), (req, a)
    for req in ["Read a CSV file into a table, then write the table to an Excel sheet",
                "Open the website and save its product table as a CSV file", "Merge some PDF files into one",
                "Copy a file to another folder, then delete the original", "print invoice.pdf for me",
                "For each file in a list, move it to the archive folder", "save all the attachments for me",
                "okay so whenever there's stuff in the Incoming folder i want it moved over to Processed",
                "List all repositories of our GitHub organization, save them as a CSV file and upload that file",
                "make it smaller", ""]:
        assert F.ask_first(req) is None or req == "make it smaller", (req, F.ask_first(req))
    assert F.ask_first("rename them properly", has_variables=True) is None          # the agent's variables say
    assert F.ask_first("copy it into the usual folder", has_variables=True)         # ... but never which folder
    two = F.ask_question(["folder", "recipient"])
    assert "Which folder" in two and "who should get it" in two


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


def test_plan_coverage_ignores_ids_that_are_not_real_actions():
    flow = {"steps": [{"id": "s1", "action": "ooa.folder/create", "args": {"path": "C:/x"}}]}
    plan = ["1. Make the folder (ooa.folder/create_symbolic)", "2. Make it (ooa.folder/create)"]
    cat = {"ooa.folder/create": {}, "ooa.folder/create_symbolic_link": {}}
    assert F.plan_coverage(plan, flow, cat) == []                       # a garbled id does not fail the flow
    assert F.plan_coverage(["1. Zip it (ooa.file/zip)"], flow, {"ooa.file/zip": {}}) == ["ooa.file/zip"]
    assert F.plan_coverage(plan, flow) == ["ooa.folder/create_symbolic"]   # without a catalog: as before
