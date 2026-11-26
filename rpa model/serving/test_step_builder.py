"""Tests of the step-by-step builder (serving/step_builder.py): the rule-based splitter (steps, loops, ifs, else
branches, a summary after a loop), the plain step requests, the code-built loop, the assembly of checked pieces into
one flow, the choice between the step-built flow and the single answer, and the Router's use of it in each mode,
with scripted models (no real model).
Run: python test_step_builder.py PLATFORM_REPO   (or OOA_PLATFORM=... python -m pytest test_step_builder.py)"""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_layers as TL                                    # noqa: E402  (shared catalog, platform, scripted models)
from test_layers import S, F, CAT                           # noqa: E402
import step_builder as SB                                   # noqa: E402
from test_modes import BatchAware                           # noqa: E402

# requests written for these tests (none of them is an eval prompt)
LOOP_REQ = ("For each CSV file in the exports folder, read it and drop the empty rows; then email the team how many "
            "files were merged.")
IF_REQ = ("Read the stock list from stock.csv, keep the rows where Qty is below 5, and if there are any email the "
          "list to purchasing, otherwise log that stock is fine.")
APPS_REQ = "Get the open Jira bugs, write them to an Excel file and post the file in the Slack channel #qa."
EACH_REQ = "Read the vendor list from vendors.csv and send each vendor a reminder email; finally log the count."


def kinds(steps):
    return [(s["kind"], s["parent"], s["branch"]) for s in steps]


# ------------------------------------------------------------------------------------------- the splitter
def test_split_loop_body_and_summary_after_it():
    st = SB.split_steps(LOOP_REQ)
    assert [s["text"] for s in st] == ["For each CSV file in the exports folder", "read it", "drop the empty rows",
                                       "email the team how many files were merged"]
    assert kinds(st) == [("loop", None, "children"), ("do", 0, "children"), ("do", 0, "children"),
                         ("do", None, "children")]               # "how many ..." after "then" closes the loop
    assert st[0]["refers"] is False and st[0]["source"] == "List the CSV files in the exports folder"


def test_split_if_with_else_branch():
    st = SB.split_steps(IF_REQ)
    assert [s["kind"] for s in st] == ["do", "do", "if", "do", "do"]
    assert st[2]["text"] == "if there are any"                  # "email" after "any" + "the" is a verb: its own step
    assert (st[3]["text"], st[3]["parent"], st[3]["branch"]) == ("email the list to purchasing", 2, "children")
    assert (st[4]["text"], st[4]["parent"], st[4]["branch"]) == ("log that stock is fine", 2, "else")
    st = SB.split_steps("Read the inbox, and if any email arrives from the bank, save it to bank.eml, then log it")
    assert [(s["kind"], s["parent"]) for s in st] == [("do", None), ("if", None), ("do", 1), ("do", None)]
    assert st[1]["text"] == "if any email arrives from the bank"     # here "email" is a noun


def test_split_each_inside_a_step_is_a_loop():
    st = SB.split_steps(EACH_REQ)
    assert kinds(st) == [("do", None, "children"), ("loop", None, "children"), ("do", 1, "children"),
                         ("do", None, "children")]             # "finally" closes the loop "each vendor" opened
    assert st[1]["refers"] is True and st[1]["item"] == "vendor"


def test_split_short_request_is_not_split():
    assert SB.split_steps("Copy report.pdf to the archive folder") == []
    assert SB.split_steps("Read the 10,000 rows of data.csv") == []
    assert not SB.wants("Copy report.pdf to the archive folder")


def test_wants_long_or_multi_app_requests():
    assert SB.wants(APPS_REQ)                                  # three steps across Jira, Excel and Slack
    assert SB.wants(LOOP_REQ)
    assert not SB.wants("Read a CSV file into a table, then write the table to an Excel sheet")


def test_quotes_and_numbers_are_not_cut():
    st = SB.split_steps("Find the rows with 'Paid, partly' in Status, sum the 1,250 rows' totals and email the result "
                        "to ap@example.com")
    assert st[0]["text"] == "Find the rows with 'Paid, partly' in Status"
    assert "1,250" in st[1]["text"]


# ------------------------------------------------------------------------------------------- step requests
def test_step_text_names_the_loop_item():
    st = SB.split_steps(LOOP_REQ)
    run = SB._StepRun(None, LOOP_REQ, [], [], lambda: False, None, st, (0.2,))
    st[0]["item_var"] = "csv"
    assert run.text(1) == "Read the CSV file"                  # "it" is the loop item in words
    assert run.text(2) == "Drop the empty rows for the CSV file"
    assert run.text(3) == "Email the team how many files were merged"
    st2 = SB.split_steps(IF_REQ)
    run2 = SB._StepRun(None, IF_REQ, [], [], lambda: False, None, st2, (0.2,))
    assert run2.text(1) == "Keep the rows where Qty is below 5"


def test_thing_and_item_names():
    assert SB.thing_of("new invoice PDF") == "invoice PDF"
    assert SB.thing_of("rejected payments") == "rejected payment"
    assert SB.thing_of("one") == ""
    assert SB.item_name("new invoice PDF") == "invoice" and SB.item_name("payments") == "payment"


# ------------------------------------------------------------------------------------------- assembly
def _piece(*steps, variables=()):
    return {"kind": "TaskBot", "id": "p", "name": "p", "variables": list(variables), "steps": list(steps)}


def test_loop_piece_turns_a_table_into_records():
    p = SB.loop_piece({"name": "rows", "type": "Table"}, "row", CAT)
    assert [s["action"] for s in p["steps"]] == ["ooa.datatable/to_records", "core/loop"]
    assert p["steps"][1]["args"] == {"iterator": "list", "over": "$row_records$", "as": "row"}
    p = SB.loop_piece({"name": "files", "type": "List"}, "file", CAT)
    assert [s["action"] for s in p["steps"]] == ["core/loop"] and p["steps"][0]["args"]["over"] == "$files$"


def test_assemble_puts_pieces_into_their_containers():
    steps = SB.split_steps(IF_REQ)
    read = _piece({"id": "s1", "action": "ooa.csv/read", "args": {"path": "stock.csv"}, "out": {"table": "stock"}})
    keep = _piece({"id": "s1", "action": "ooa.datatable/filter", "args": {"table": "$stock", "column": "Qty",
                   "operator": "lt", "value": 5}, "out": {"table": "low"}})
    cond = _piece({"id": "s1", "action": "ooa.datatable/row_count", "args": {"table": "$low"}, "out": {"count": "n"}},
                  {"id": "s2", "action": "core/if", "args": {"condition": "${n} > 0"},    # its body is a guess: dropped
                   "children": [{"id": "s3", "action": "ooa.file/delete", "args": {"path": "x"}}]})
    mail = _piece({"id": "s1", "action": "ooa.email/send", "args": {"to": "purchasing@example.com",
                                                                     "subject": "Low stock", "body": "see list"}})
    other = _piece({"id": "s1", "action": "core/if", "args": {"condition": "$flag"},     # an if nobody asked for
                    "children": [{"id": "s2", "action": "core/log", "args": {"message": "stock is fine"}}]})
    flow = SB.assemble([read, keep, cond, mail, other], steps)
    acts = [s["action"] for s, _ in F.walk(flow["steps"])]
    assert acts == ["ooa.csv/read", "ooa.datatable/filter", "ooa.datatable/row_count", "core/if", "ooa.email/send",
                    "core/log"]
    box = flow["steps"][3]
    assert [s["action"] for s in box["children"]] == ["ooa.email/send"]
    assert [s["action"] for s in box["else"]] == ["core/log"]     # "otherwise" went into the else, unwrapped


def test_assemble_drops_a_repeated_sign_in():
    steps = [{"text": "a", "kind": "do", "parent": None, "branch": "children"},
             {"text": "b", "kind": "do", "parent": None, "branch": "children"}]
    conn = {"id": "s1", "action": "ooa.email/connect", "args": {"server": "imap.example.com"}, "out": {"session": "m"}}
    one = _piece(conn, {"id": "s2", "action": "ooa.email/list", "args": {"session": "$m"}, "out": {"emails": "e"}})
    two = _piece(dict(conn), {"id": "s2", "action": "ooa.email/send", "args": {"to": "a@example.com", "subject": "x",
                                                                                "body": "y"}})
    flow = SB.assemble([one, two], steps)
    assert [s["action"] for s in flow["steps"]] == ["ooa.email/connect", "ooa.email/list", "ooa.email/send"]


def test_fragment_score_prefers_what_the_step_asked_for():
    step = {"text": "Email the team", "kind": "do", "parent": None}
    plain = _piece({"id": "s1", "action": "ooa.email/send", "args": {"to": "t@example.com"}})
    wrapped = _piece({"id": "s1", "action": "core/if", "args": {"condition": "$flag"}, "children": [
        {"id": "s2", "action": "ooa.email/send", "args": {"to": "t@example.com"}}]})
    rel = {"ooa.email/send"}
    assert SB.fragment_score(plain, step, [], rel) > SB.fragment_score(wrapped, step, [], rel)
    ifstep = {"text": "If any are late", "kind": "if", "parent": None}
    assert SB.fragment_score(wrapped, ifstep, [], rel) > SB.fragment_score(plain, ifstep, [], rel)


def test_step_hints_rephrase_and_referents():
    assert "ooa.genai/classify" in SB.step_verb_ids("Ask AI whether the letter is a complaint")
    assert "ooa.genai/prompt_with_images" in SB.step_verb_ids("Have the AI describe what the photos show")
    assert "ooa.rest/request" in SB.step_verb_ids("Pull the price list from the supplier's API")
    assert SB.step_verb_ids("Email the finance team the totals", strong=True) == ["ooa.email/send"]
    assert SB.step_verb_ids("Drop the empty rows", strong=True) == []
    assert SB.rephrase("Email the HR ops team how many came in") == \
        "Send an email to the HR ops team about how many came in"
    assert SB.rephrase("Move the file") == "Move the file"
    st = SB.split_steps("Check the shared mailbox for unread orders from the shop, save their PDF attachments to "
                        "the orders folder, then log how many were saved.")
    run = SB._StepRun(None, "", [], [], lambda: False, None, st, (0.2,))
    assert run.text(1) == "Save the unread orders' PDF attachments to the orders folder"


def test_app_swap_uses_the_named_apps_action():
    r = TL.router(BatchAware(lambda *a: ""))
    other, _ = r.verify('{"steps":[{"action":"ooa.salesforce/revoke_authentication","args":{"instance_url":'
                        '"$instance_url$","access_token":"$access_token$"}}]}', [], None, "Revoke the token")
    assert other is not None
    swapped = SB.app_swaps(r, other, {"ooa.servicenow"}, ["ooa.servicenow/revoke_authentication"], [], "Revoke the token")
    assert [F.actions_of(f) for f in swapped] == [["ooa.servicenow/revoke_authentication"]]
    assert SB.app_swaps(r, other, set(), ["ooa.servicenow/revoke_authentication"], [], "x") == []


# ------------------------------------------------------------------------------------------- with a scripted model
READ_CSV = {"steps": [{"action": "ooa.csv/read", "args": {"path": "${file}"}, "out": {"table": "rows"}}]}
DROP = {"steps": [{"action": "ooa.datatable/remove_empty_rows", "args": {"table": "$rows"}, "out": {"table": "clean"}}]}
LIST = {"steps": [{"action": "ooa.file/list", "args": {"folder": "exports", "pattern": "*.csv"},
                   "out": {"files": "files"}}]}
MAIL = {"steps": [{"action": "ooa.email/send", "args": {"to": "team@example.com", "subject": "Merged",
                                                        "body": "${count} files merged"}}]}


def step_model(answers, single=None):
    """A batch model that answers a step request by the first key it contains (lower case), and the whole request
    with `single` (an answer text, or None: a flow that does not pass the checker)."""
    def answer(req, prompt, attempt):
        low = req.lower()
        for key, flow in answers.items():
            if key in low:
                return "Plan:\n" + "\n".join(F.plan_from_flow(F.expand_flow(flow, CAT, req), CAT)) + "\nFlow:\n" + \
                    F.compact(flow)
        if single is not None and req.strip() in (LOOP_REQ, IF_REQ, APPS_REQ, EACH_REQ):
            return single
        return "Plan:\n1. Do it (ooa.nothing/here)\nFlow:\n{\"steps\":[{\"action\":\"ooa.nothing/here\"}]}"
    return BatchAware(answer)


def test_build_joins_checked_pieces_with_a_code_built_loop():
    m = step_model({"list the csv files": LIST, "read the csv": READ_CSV, "drop the empty": DROP, "email the team": MAIL})
    r = TL.router(m)
    tried = []
    flow, notes, info = SB.build(r, LOOP_REQ, [], tried, lambda: False, None)
    assert flow is not None, notes
    acts = [s["action"] for s, _ in F.walk(flow["steps"])]
    assert acts == ["ooa.file/list", "core/loop", "ooa.csv/read", "ooa.datatable/remove_empty_rows", "ooa.email/send"]
    loop = flow["steps"][1]
    assert loop["args"]["over"] == "$files$" and [s["action"] for s in loop["children"]] == [
        "ooa.csv/read", "ooa.datatable/remove_empty_rows"]
    assert info["built"] == 4 and not info["failed"] and any("step by step" in n for n in notes)
    assert m.batches == [4 * len(SB.STEP_TEMPERATURES)]     # all 4 requests (the loop's source + 3 steps) at once,
    #                                                         best-of-N each, in ONE batch; the loop is built by code
    assert info["batches"] == 1
    assert any(t.get("step") for t in tried)


def test_build_retries_a_bad_step_with_its_problems():
    seen = {"n": 0}

    def answer(req, prompt, attempt):
        low = req.lower()
        if "drop the empty" in low:
            seen["n"] += 1
            if "### Problems" not in prompt:
                return "Plan:\n1. Drop (ooa.datatable/remove_empty_rows)\nFlow:\n{\"steps\":[{\"action\":\"ooa.nope/x\"}]}"
            return "Plan:\n1. Drop the empty rows (ooa.datatable/remove_empty_rows)\nFlow:\n" + F.compact(DROP)
        for key, flow in (("list the csv", LIST), ("read the csv", READ_CSV), ("email the team", MAIL)):
            if key in low:
                return "Plan:\n1. x (%s)\nFlow:\n" % flow["steps"][0]["action"] + F.compact(flow)
        return "no"
    m = BatchAware(answer)
    flow, notes, info = SB.build(TL.router(m), LOOP_REQ, [], [], lambda: False, None)
    assert flow is not None and seen["n"] == 2 * len(SB.STEP_TEMPERATURES)
    assert "ooa.datatable/remove_empty_rows" in F.actions_of(flow)


def test_a_weak_step_is_asked_again_in_other_words():
    asked = []

    def answer(req, prompt, attempt):
        low = req.lower()
        asked.append(req)
        if low.startswith("send an email to the team"):
            return "Plan:\n1. Email the team (ooa.email/send)\nFlow:\n" + F.compact(MAIL)
        if low.startswith("email the team"):                # answered without any email (a weak piece)
            return "Plan:\n1. Log it (core/log)\nFlow:\n" + F.compact(
                {"steps": [{"action": "core/log", "args": {"message": "files merged"}}]})
        for key, flow in (("list the csv", LIST), ("read the csv", READ_CSV), ("drop the empty", DROP)):
            if key in low:
                return "Plan:\n1. x (%s)\nFlow:\n" % flow["steps"][0]["action"] + F.compact(flow)
        return "no"
    m = BatchAware(answer)
    flow, notes, info = SB.build(TL.router(m), LOOP_REQ, [], [], lambda: False, None)
    assert flow is not None and "ooa.email/send" in F.actions_of(flow) and "core/log" not in F.actions_of(flow)
    assert any(a.startswith("Send an email to the team about how many") for a in asked)
    assert m.batches == [4 * len(SB.STEP_TEMPERATURES), len(SB.STEP_TEMPERATURES)]   # only the weak step again


def test_a_runaway_piece_is_never_used():
    many = {"steps": [{"action": "ooa.datatable/remove_empty_rows", "args": {"table": "$rows"},
                       "out": {"table": "t%d" % k}} for k in range(4)]}
    m = step_model({"list the csv files": LIST, "read the csv": READ_CSV, "drop the empty": many,
                    "email the team": MAIL})
    flow, notes, info = SB.build(TL.router(m), LOOP_REQ, [], [], lambda: False, None)
    assert flow is None or F.actions_of(flow).count("ooa.datatable/remove_empty_rows") <= SB.MAX_SAME_ACTION


def test_build_gives_up_when_too_many_steps_fail():
    m = step_model({"list the csv files": LIST})
    flow, notes, info = SB.build(TL.router(m), LOOP_REQ, [], [], lambda: False, None)
    assert flow is None and info["failed"] and "could not be built" in notes[0]


# ------------------------------------------------------------------------------------------- inside the Router
def _single_missing_the_mail():
    """The single answer: verified, but it leaves out the email step."""
    one = {"steps": [{"action": "ooa.file/list", "args": {"folder": "exports", "pattern": "*.csv"},
                      "out": {"files": "files"}}]}
    return "Plan:\n1. List the CSV files (ooa.file/list)\nFlow:\n" + F.compact(one)


def test_smart_mode_keeps_the_step_built_flow_when_it_covers_more():
    m = step_model({"list the csv files": LIST, "read the csv": READ_CSV, "drop the empty": DROP, "email the team": MAIL},
                   single=_single_missing_the_mail())
    r = TL.router(m, allow_escalation=False)
    res = r.run(LOOP_REQ, mode="smart")
    assert res["verified"] and res["step_builder"]["kept"] and res["step_builder"]["built"] == 4
    assert "step builder" in [a["agent"] for a in res["agents"]]
    acts = [s["action"] for s, _ in F.walk(json.loads(res["flow"])["steps"])]
    assert "ooa.email/send" in acts and "core/loop" in acts


def test_single_answer_stays_when_the_step_path_fails():
    m = step_model({}, single=_single_missing_the_mail())
    res = TL.router(m, allow_escalation=False).run(LOOP_REQ, mode="smart")
    assert res["verified"] and res["step_builder"]["kept"] is False and res["step_builder"]["verified"] is False
    assert [s["action"] for s in json.loads(res["flow"])["steps"]] == ["ooa.file/list"]


def test_step_builder_rescues_a_request_the_single_answer_could_not_build():
    m = step_model({"list the csv files": LIST, "read the csv": READ_CSV, "drop the empty": DROP, "email the team": MAIL},
                   single="Plan:\n1. x (ooa.nothing/here)\nFlow:\n{\"steps\":[{\"action\":\"ooa.nothing/here\"}]}")
    res = TL.router(m, allow_escalation=False).run(LOOP_REQ, mode="smart")
    assert res["verified"] and res["step_builder"]["kept"]


LONG_REQ = ("For each CSV file in the exports folder, read it, drop the empty rows and save it as an Excel file; then "
            "email the team how many files were merged.")
SAVE = {"steps": [{"action": "ooa.datatable/write_to_file", "args": {"table": "$clean", "path": "merged.xlsx"},
                   "out": {"path": "saved"}}]}


def test_whole_coverage_of_a_long_request_skips_the_single_answer():
    m = step_model({"list the csv files": LIST, "read the csv": READ_CSV, "drop the empty": DROP,
                    "save the csv file": SAVE, "email the team": MAIL}, single=_single_missing_the_mail())
    res = TL.router(m, allow_escalation=False).run(LONG_REQ, mode="smart")
    sb = res["step_builder"]
    assert res["verified"] and sb["kept"] and sb["single"].startswith("skipped"), sb
    assert agents(res)[:1] == ["step builder"] and "planner" not in agents(res)
    assert not any(LONG_REQ in p for p in m.prompts)        # the whole request was never asked of the model


def test_prefer_goes_by_coverage_and_a_tie_keeps_the_steps():
    st = SB.split_steps(APPS_REQ)
    assert not SB.long_request(APPS_REQ, st) and SB.long_request(LONG_REQ)

    class R:                                                  # coverage() needs only the router's index
        index = TL.router(BatchAware(lambda *a: "")).index
    full = {"steps": [{"action": "ooa.jira/search"}, {"action": "ooa.excel/write_table"},
                      {"action": "ooa.slack/upload_file"}]}
    part = {"steps": [{"action": "ooa.jira/search"}]}
    assert SB.coverage(R, APPS_REQ, part) > SB.coverage(R, APPS_REQ, full)
    assert SB.prefer(R, APPS_REQ, full, full, st) is True         # a tie: the step-built flow
    assert SB.prefer(R, APPS_REQ, part, full, st) is False        # the single answer covers more
    assert SB.prefer(R, APPS_REQ, part, None, st) is True         # the single answer did not verify


def test_the_single_answer_still_asks_when_the_step_path_fails():
    def answer(req, prompt, attempt):
        if req.strip() == LONG_REQ:
            return "Question: Which folder are the CSV files in?"
        return "Plan:\n1. x (ooa.nothing/here)\nFlow:\n{\"steps\":[{\"action\":\"ooa.nothing/here\"}]}"
    res = TL.router(BatchAware(answer), allow_escalation=False, ask_first=False).run(LONG_REQ, mode="smart")
    assert not res["verified"] and res.get("question", "").startswith("Which folder")


def agents(res):
    return [a["agent"] for a in res.get("agents") or []]


def test_fast_mode_and_short_requests_never_use_it():
    m = step_model({"list the csv files": LIST, "read the csv": READ_CSV, "drop the empty": DROP, "email the team": MAIL},
                   single=_single_missing_the_mail())
    res = TL.router(m, allow_escalation=False).run(LOOP_REQ, mode="fast")
    assert "step_builder" not in res and "step builder" not in [a["agent"] for a in res["agents"]]
    assert S.Policy().for_mode("fast").step_builder == "off"
    assert S.Policy().for_mode("smart").step_builder == "long" and S.Policy().for_mode("ultra").step_builder == "long"
    m2 = step_model({}, single=_single_missing_the_mail())
    res = TL.router(m2, allow_escalation=False).run("List the CSV files in the exports folder", mode="smart")
    assert "step_builder" not in res


def test_auto_sends_a_short_multi_app_request_to_a_step_building_mode():
    mode, why = S.choose_mode(APPS_REQ)
    assert mode in ("smart", "ultra")


def test_story_requests_are_plain_steps_and_stay_with_the_single_answer():
    story = ("First I open the vendor portal, then I log in with my user name, then I click Invoices and save the "
             "newest one. Can the bot do this?")
    st = SB.split_steps(story)
    assert [s["text"] for s in st] == ["open the vendor portal", "log in with my user name", "click Invoices",
                                       "save the newest one"]           # no "I", no question about the bot
    assert SB.is_story(story) and not SB.wants(story)
    st = SB.split_steps("We copy the Q3 folder to the backup drive, then we zip the copy. That's our quarter end.")
    assert [s["text"] for s in st] == ["copy the Q3 folder to the backup drive", "zip the copy"]
    assert not SB.is_story(LOOP_REQ) and not SB.is_story(APPS_REQ)


def test_strong_step_hints_for_filters_links_dates():
    assert SB.step_verb_ids("keep the tickets whose priority is High", strong=True) == ["ooa.datatable/filter"]
    assert SB.step_verb_ids("keep the tickets", strong=True) == []
    assert "ooa.browser/extract_list" in SB.step_verb_ids("grab all the download links", strong=True)
    ids = SB.step_verb_ids("If the order was placed more than 30 days ago", strong=True)
    assert ids == ["ooa.datetime/now", "ooa.datetime/parse", "ooa.datetime/difference"]
    assert SB.step_verb_ids("add 5 working days to the start date", strong=True) == ["ooa.datetime/add_working_days"]
    assert "ooa.microsoft365/find_meeting_times" in SB.step_verb_ids(
        "find a free 30-minute slot when the managers are available in Outlook", strong=True)


def test_the_step_path_has_its_own_time_budget():
    """Policy.step_time: the step path's generations are cut at its own limit (the single answer keeps the rest of
    the mode's time), and its summary stays in the reply when nothing verified."""
    import time as _t
    m = step_model({}, single="Plan:\n1. x (ooa.nothing/here)\nFlow:\n{\"steps\":[{\"action\":\"ooa.nothing/here\"}]}")
    t0 = _t.time()
    res = TL.router(m, allow_escalation=False, step_time=20.0).run(LOOP_REQ, mode="smart")
    assert not res["verified"] and res["step_builder"]["verified"] is False, res
    assert m.deadlines[0] is not None and m.deadlines[0] <= t0 + 20.0 + 1     # the step batch: its own cut
    assert max(d for d in m.deadlines[1:] if d is not None) > t0 + 100       # the single answer: the mode's cut
    m2 = step_model({})
    TL.router(m2, allow_escalation=False, step_time=0.0).run(LOOP_REQ, mode="smart")
    assert m2.deadlines[0] > t0 + 100                                       # 0: only the mode's own limits


def test_a_guided_sample_starts_with_the_actions_the_step_words_point_at():
    """'keep the ones whose ...' points at a filter: one more sample of that step starts with the filter as its plan;
    the model's own samples dedupe instead, and the guided piece is the one kept."""
    req = "Read the tickets from tickets.csv, keep the ones whose Status is Open, then email the list to qa@example.com."
    dedupe = {"steps": [{"action": "ooa.datatable/remove_duplicates", "args": {"table": "$rows"},
                         "out": {"table": "kept"}}]}
    filt = {"steps": [{"action": "ooa.datatable/filter", "args": {"table": "$rows", "column": "Status",
                                                                   "operator": "eq", "value": "Open"},
                       "out": {"table": "kept"}}]}
    read = {"steps": [{"action": "ooa.csv/read", "args": {"path": "tickets.csv"}, "out": {"table": "rows"}}]}
    mail = {"steps": [{"action": "ooa.email/send", "args": {"to": "qa@example.com", "subject": "Open tickets",
                                                            "body": "${kept}"}}]}
    holder = {}

    def answer(req_, prompt, attempt):
        low = req_.lower()
        if low.startswith("keep the ones"):
            pre = (holder["m"]._item or {}).get("prefix") or ""
            flow = filt if "ooa.datatable/filter" in pre else dedupe
            return "Plan:\n1. x (%s)\nFlow:\n" % flow["steps"][0]["action"] + F.compact(flow)
        for key, flow in (("read the tickets", read), ("email the list", mail)):
            if key in low:
                return "Plan:\n1. x (%s)\nFlow:\n" % flow["steps"][0]["action"] + F.compact(flow)
        return "no"
    m = holder["m"] = BatchAware(answer)
    flow, notes, info = SB.build(TL.router(m), req, [], [], lambda: False, None)
    assert flow is not None, notes
    assert F.actions_of(flow) == ["ooa.csv/read", "ooa.datatable/filter", "ooa.email/send"], F.actions_of(flow)
    guided = [p for p in m.prefixes if p]
    assert len(guided) == 1 and guided[0].startswith("Plan:\n1. ") and "(ooa.datatable/filter)" in guided[0]
    assert m.batches == [3 * len(SB.STEP_TEMPERATURES) + 1]          # the guided sample is in the same batch


def test_app_pick_guides_a_step_that_names_its_app():
    r = TL.router(BatchAware(lambda *a: ""))
    assert SB.app_pick(r, "Cancel the long running statement in Snowflake") == ["ooa.snowflake/cancel_statement"]
    assert SB.app_pick(r, "Move the files to the archive folder") == []          # no app named in the step
    for text in ("Send the customer a follow-up from Gmail", "Find the running EC2 instances of the dev team"):
        picked = SB.app_pick(r, text)                    # never a delete / reboot / ... the step does not say
        assert not any(set(a.split("/")[-1].split("_")) & SB.SAID_VERBS for a in picked), picked


def test_find_the_one_inside_a_loop_is_an_if_and_or_that_is_an_else():
    st = SB.split_steps("For each order in the backlog sheet, find the ones with a missing address and email the "
                        "customer a reminder; then log how many were reminded.")
    assert kinds(st)[:3] == [("loop", None, "children"), ("if", 0, "children"), ("do", 1, "children")]
    assert st[1]["text"] == "If the order has a missing address"
    st = SB.split_steps("Read the failed builds from builds.csv, open a ticket for each one; then email the release "
                        "manager how many failed, or that all builds passed.")
    tail = [(s["kind"], s["branch"], s["text"]) for s in st[-3:]]
    assert tail == [("if", "children", "if there are any"), ("do", "children", "email the release manager how many failed"),
                    ("do", "else", "email the release manager that all builds passed")], tail


def test_now_is_read_once_before_the_loop():
    flow = {"steps": [{"action": "ooa.csv/read", "out": {"table": "t"}},
                      {"action": "core/loop", "args": {"over": "$t$", "as": "row"}, "children": [
                          {"action": "ooa.datetime/now", "out": {"value": "now"}},
                          {"action": "core/if", "args": {"condition": "x"}, "children": [{"action": "core/log"}]}]}]}
    got = SB.hoist_now(flow)
    assert [s["action"] for s in got["steps"]] == ["ooa.csv/read", "ooa.datetime/now", "core/loop"]
    assert [s["action"] for s in got["steps"][2]["children"]] == ["core/if"]


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as ex:                     # noqa: BLE001
                import traceback
                traceback.print_exc()
                print("FAIL", name, ex)
                fails += 1
    print(f"{fails} failed")
    sys.exit(1 if fails else 0)
