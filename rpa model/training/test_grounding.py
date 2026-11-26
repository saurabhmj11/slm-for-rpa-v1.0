"""Tests for grounding_check.py (the detector) and fix_grounding.py (the fix) + the shipped data/extra_sft files.
Run: python test_grounding.py   (works under pytest too). Standard library only."""
import copy, glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grounding_check as G
import fix_grounding as FX

EXTRA = sorted(glob.glob(os.path.join(HERE, "../data/extra_sft/*.jsonl")))
MAIL = {"smtp_host": "$smtp_host", "user": "$mail_user", "password": "$mail_password"}


def rec(request, steps, variables=None, **kw):
    r = {"kind": "rewrite", "request": request,
         "flow": {"kind": "TaskBot", "id": "t", "name": "T", "description": kw.pop("description", "Test."),
                  "variables": [], "steps": steps}}
    if variables is not None:
        r["kind"], r["variables"] = "flow", variables
    r.update(kw)
    return r


def kinds(r):
    return [i[0] for i in G.check_record(r)]


def test_detector_flags_made_up_details():
    r = rec("Create the Jira tickets for the open rows and email the summary to the AP team",
            [{"id": "s1", "action": "ooa.jira/connect", "args": {"connection": "jira-prod"}},
             {"id": "s2", "action": "ooa.email/send", "description": "Email the AP team",
              "args": {"to": "ap-team@ourfirm.com", "subject": "x", **MAIL}},
             {"id": "s3", "action": "ooa.microsoft365/upload_file", "description": "Upload it to the HR SharePoint site",
              "args": {"local_path": "D:\\Recon\\out.csv", "remote_path": "Shared Documents/out.csv",
                       "drive": "ourfirm.sharepoint.com:/sites/HR"}}])
    found = {(i[2], i[3]) for i in G.check_record(r) if i[0] == "flow_constant"}
    assert ("connection", "jira-prod") in found and ("to", "ap-team@ourfirm.com") in found
    assert ("drive", "ourfirm.sharepoint.com:/sites/HR") in found and ("local_path", "D:\\Recon\\out.csv") in found
    assert ("text_detail", "step.description", "", "ourfirm.sharepoint.com:/sites/HR") in G.check_record(r)


def test_detector_accepts_what_the_request_says():
    r = rec("Read D:\\Sales\\orders.csv, keep Quantity over 500, save D:\\Sales\\big.csv and mail it to a@b.com",
            [{"id": "s1", "action": "ooa.csv/read", "args": {"path": "D:\\Sales\\orders.csv"}, "out": {"table": "t"}},
             {"id": "s2", "action": "ooa.csv/write", "args": {"table": "$t", "path": "D:\\Sales\\big.csv"}},
             {"id": "s3", "action": "ooa.csv/write", "args": {"table": "$t", "path": "D:\\Sales\\big_copy.csv"}},
             {"id": "s4", "action": "ooa.email/send", "args": {"to": "a@b.com", "attachments": ["D:\\Sales\\big.csv"],
                                                             **MAIL}},
             {"id": "s5", "action": "ooa.json/get", "args": {"path": "data.price"}}],
            clear="Read D:\\Sales\\orders.csv, save D:\\Sales\\big.csv and big_copy.csv, mail it to a@b.com.")
    assert G.check_record(r) == []                    # said, a file made in a said folder, a JSON path


def test_detector_step_builder_known_variables():
    r = rec("With the current row: email it to the address in row_email.",
            [{"id": "s1", "action": "ooa.email/send", "args": {"to": "$row_email", **MAIL}}],
            variables=[{"name": "row_email", "type": "String", "direction": "local"}])
    assert G.check_record(r) == []
    r["flow"]["steps"][0]["args"]["to"] = "meera.iyer@ourfirm.com"
    assert kinds(r) == ["flow_constant"]


def test_bare_reference_inside_text():
    r = rec("For each row of D:\\P\\b.csv write D:\\P\\out\\<Name>.txt",
            [{"id": "s1", "action": "ooa.file/write_text",
              "args": {"path": "D:\\P\\out\\$row.Name.txt", "text": "Dear $row.Name, ${row.Detail}"}},
             {"id": "s2", "action": "core/if", "args": {"condition": "$row.A < $row.B"}}])
    assert kinds(r).count("bare_ref_in_text") == 2      # the condition is an expression: fine
    FX.fix_record(r)
    a = r["flow"]["steps"][0]["args"]
    assert a["path"] == "D:\\P\\out\\${row.Name}.txt" and a["text"] == "Dear ${row.Name}, ${row.Detail}"
    assert FX.braced_refs("$row.Name") == "$row.Name" and FX.braced_refs("x ${a.b} $c$") == "x ${a.b} $c$"


def test_fix_makes_input_variables_and_cleans_texts():
    r = rec("Create the Jira tickets for the open rows, upload open_list.csv to SharePoint and email the AP team",
            [{"id": "s1", "action": "ooa.jira/connect", "description": "Connect to Jira (saved connection jira-prod)",
              "args": {"connection": "jira-prod"}},
             {"id": "s2", "action": "ooa.microsoft365/upload_file", "description": "Upload it to the HR SharePoint site",
              "args": {"local_path": "D:\\Recon\\open_list.csv", "remote_path": "Shared Documents/open_list.csv",
                       "drive": "ourfirm.sharepoint.com:/sites/HR"}},
             {"id": "s3", "action": "ooa.email/send", "description": "Email the AP team",
              "args": {"to": "ap-team@ourfirm.com", "subject": "x", "attachments": ["D:\\Recon\\open_list.csv"],
                       **MAIL}}],
            clear="Use the saved connection jira-prod, upload D:\\Recon\\open_list.csv to the HR SharePoint site and "
                  "email ap-team@ourfirm.com.",
            facts=[["to", "ap-team@ourfirm.com"], ["filter", "x > 1"]])
    notes = FX.fix_record(r)
    assert notes
    fl = r["flow"]
    a1, a2, a3 = (s["args"] for s in fl["steps"])
    assert a1["connection"] == "$jira_connection"
    assert a2["drive"] == "$sharepoint_site" and a2["local_path"] == "${local_folder}\\open_list.csv"
    assert a3["to"] == "$ap_team_email" and a3["attachments"] == ["${local_folder}\\open_list.csv"]
    names = {v["name"]: v for v in fl["variables"]}
    for n in ("jira_connection", "sharepoint_site", "local_folder", "ap_team_email"):
        assert names[n]["direction"] == "input" and names[n]["required"] and "default" not in names[n]
    assert names["jira_connection"]["description"] == "Saved Jira connection (under Connections)."
    assert G.check_record(r) == [], G.check_record(r)
    assert "jira-prod" not in r["clear"] and "HR" not in r["clear"] and "ourfirm" not in r["clear"], r["clear"]
    assert "Recon" not in r["clear"] and "open_list.csv" in r["clear"]
    assert r["facts"] == [["filter", "x > 1"]]
    before = copy.deepcopy(r)
    assert FX.fix_record(r) == [] and r == before     # idempotent


def test_clarify_gets_missing():
    r = {"kind": "clarify", "request": "post it on teams", "question": "What should the message say?"}
    assert kinds(r) == ["clarify_without_missing"]
    FX.fix_record(r)
    assert r["missing"] == ["details", "place"] and G.check_record(r) == []


def test_extra_sft_files_are_grounded():
    """The shipped files: no made-up detail, no bare reference in text, every clarify has 'missing', and the fix
    has nothing left to do."""
    assert EXTRA
    for p in EXTRA:
        for i, line in enumerate(open(p, encoding="utf-8"), 1):
            r = json.loads(line)
            assert G.check_record(r) == [], (os.path.basename(p), i, G.check_record(r))
            assert FX.fix_record(copy.deepcopy(r)) == [], (os.path.basename(p), i)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
