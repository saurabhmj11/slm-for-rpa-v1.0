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
             {"id": "s5", "action": "ooa.json/get", "args": {"path": "data.quantity"}}],
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
    assert a2["drive"] == "$sharepoint_site" and a2["local_path"] == "${input_folder}\\open_list.csv"
    assert a3["to"] == "$ap_team_email" and a3["attachments"] == ["${input_folder}\\open_list.csv"]
    names = {v["name"]: v for v in fl["variables"]}
    for n in ("jira_connection", "sharepoint_site", "input_folder", "ap_team_email"):
        assert names[n]["direction"] == "input" and names[n]["required"] and "default" not in names[n]
    assert names["jira_connection"]["description"] == "Saved Jira connection (under Connections)."
    assert G.check_record(r) == [], G.check_record(r)
    assert "jira-prod" not in r["clear"] and "HR" not in r["clear"] and "ourfirm" not in r["clear"], r["clear"]
    assert "Recon" not in r["clear"] and "open_list.csv" in r["clear"]
    assert r["facts"] == [["filter", "x > 1"]]
    before = copy.deepcopy(r)
    assert FX.fix_record(r) == [] and r == before     # idempotent


def fixed(r):
    FX.fix_record(r)
    return r, {s["id"]: s["args"] for s, _ in FX.F.walk(r["flow"]["steps"])}


def test_download_and_unzip_use_the_same_variable():
    """The producer (download path) and the consumer (unzip archive) change together; a made-up unzip folder named
    only inside the link ('rates') is not said."""
    r = rec("download + unzip https://downloads.example.org/rates/fx_rates.zip",
            [{"id": "s1", "action": "ooa.rest/download_file", "description": "Download fx_rates.zip",
              "args": {"url": "https://downloads.example.org/rates/fx_rates.zip", "path": "D:\\Temp\\fx_rates.zip"}},
             {"id": "s2", "action": "ooa.file/unzip", "description": "Unzip it into D:\\Treasury\\Rates",
              "args": {"archive": "D:\\Temp\\fx_rates.zip", "folder": "D:\\Treasury\\Rates"}}],
            clear="Download https://downloads.example.org/rates/fx_rates.zip to D:\\Temp\\fx_rates.zip and unzip it "
                  "into D:\\Treasury\\Rates.")
    assert {(i[2], i[3]) for i in G.check_record(r) if i[0] == "flow_constant"} >= {
        ("path", "D:\\Temp\\fx_rates.zip"), ("archive", "D:\\Temp\\fx_rates.zip"), ("folder", "D:\\Treasury\\Rates")}
    half = copy.deepcopy(r)                           # round 1: only the producer changed
    half["flow"]["steps"][0]["args"]["path"] = "${folder}\\fx_rates.zip"
    half["flow"]["steps"][1]["args"]["folder"] = "${folder_2}\\Rates"
    found = {(i[2], i[3]) for i in G.check_record(half) if i[0] == "flow_constant"}
    assert ("archive", "D:\\Temp\\fx_rates.zip") in found and ("folder", "${folder_2}\\Rates") in found
    r, a = fixed(r)
    assert a["s1"]["path"] == a["s2"]["archive"] == "${download_folder}\\fx_rates.zip"
    assert a["s2"]["folder"] == "${extract_folder}"
    assert r["clear"] == ("Download https://downloads.example.org/rates/fx_rates.zip to fx_rates.zip in the download "
                          "folder and unzip it into the extract folder."), r["clear"]
    assert "given" not in json.dumps(r) and G.check_record(r) == []


def test_upload_and_sharing_link_use_the_same_variable():
    r = rec("upload the new leave policy to onedrive and send all staff a link to it",
            [{"id": "s1", "action": "ooa.microsoft365/upload_file", "description": "Upload policy_2027.pdf",
              "args": {"local_path": "D:\\HR\\policy_2027.pdf", "remote_path": "HR/Policies/policy_2027.pdf"}},
             {"id": "s2", "action": "ooa.microsoft365/create_sharing_link", "args": {"item": "HR/Policies/policy_2027.pdf"}}],
            clear="Upload policy_2027.pdf to HR/Policies and email all staff a link.")
    assert ("item", "HR/Policies/policy_2027.pdf") in {(i[2], i[3]) for i in G.check_record(r)}
    r, a = fixed(r)
    assert a["s1"]["remote_path"] == a["s2"]["item"] == "${remote_folder}/policy_2027.pdf"
    assert a["s1"]["local_path"] == "$policy_file"
    assert r["clear"] == "Upload the policy PDF to the OneDrive folder and email all staff a link.", r["clear"]


def test_a_read_file_is_said_or_written_earlier():
    """'derived' (a new name in a said folder) only where the flow writes it; a read of it must match the write."""
    r = rec("Read D:\\Sales\\orders.csv and email the summary to a@b.com",
            [{"id": "s1", "action": "ooa.csv/read", "args": {"path": "D:\\Sales\\orders_2026.csv"}},
             {"id": "s2", "action": "ooa.email/send", "args": {"to": "a@b.com", **MAIL}}])
    assert kinds(r) == ["flow_constant"]
    assert not G._word_in("ab", "a b c") and not G._word_in("Rates", "x https://d.org/rates/fx.zip".replace(
        "https://d.org/rates/fx.zip", "")) and G._word_in("statement", "the statements")


def test_ad_names_become_variables():
    r = rec("With the current p: move them to Leavers, add them to All Staff.",
            [{"id": "s1", "action": "ooa.active_directory/move_user", "description": "Move them to Leavers",
              "args": {"target_ou_dn": "OU=Leavers,DC=ourfirm,DC=local"}},
             {"id": "s2", "action": "ooa.active_directory/add_to_group",
              "args": {"group_dn": "CN=All Staff,OU=Groups,DC=ourfirm,DC=local"}},
             {"id": "s3", "action": "ooa.active_directory/search",
              "args": {"filter": "(&(objectClass=user)(memberOf=CN=All Staff,OU=Groups,DC=ourfirm,DC=local))"}}],
            variables=[{"name": "p", "type": "Any", "direction": "local"}])
    r["understand"] = r["request"]
    assert {i[0] for i in G.check_record(r)} == {"flow_constant", "ad_name"}
    r, a = fixed(r)
    assert a["s1"]["target_ou_dn"] == "$leavers_ou" and a["s2"]["group_dn"] == "$all_staff_group_dn"
    assert a["s3"]["filter"] == "(&(objectClass=user)(memberOf=${all_staff_group_dn}))"
    assert r["understand"] == r["request"] and "ourfirm" not in json.dumps(r)


def test_type_filter_template_text_db_type_company_and_link():
    r = rec("Need all files attached to the case in Salesforce downloaded; book the interview; email the 2 page(s); "
            "reply with the portal link",
            [{"id": "s1", "action": "ooa.salesforce/get_attachments", "description": "Download the jpg files",
              "args": {"extension": "jpg", "folder": "$download_folder"}},
             {"id": "s2", "action": "ooa.microsoft365/send_mail",
              "args": {"to": "$c.Email", "subject": "Your interview with OurFirm",
                       "body": "Please use https://help.ourfirm.com"}},
             {"id": "s3", "action": "ooa.database/connect", "args": {"database_type": "sqlserver"}},
             {"id": "s4", "action": "ooa.microsoft365/create_event", "args": {"subject": "Interview", "location": "Teams"}}],
            clear="Download the jpg files, print 2 copy/copies and query the SQL Server database.")
    k = kinds(r)
    assert "type_filter" in k and "flow_text" in k and "text_style" in k and "flow_constant" in k
    r, a = fixed(r)
    assert "extension" not in a["s1"] and "location" not in a["s4"]
    assert a["s2"]["subject"] == "Your interview" and a["s2"]["body"] == "Please use ${help_link}"
    assert a["s3"]["database_type"] == "$database_type"
    assert r["request"].endswith("email the 2 pages; reply with the portal link")
    assert r["clear"] == "Download the files, print 2 copies and query the database.", r["clear"]
    assert G.check_record(r) == []


def test_unfixable_records_are_dropped():
    r = rec("With the current lead: save the score on the lead.",
            [{"id": "s1", "action": "ooa.salesforce/update_record",
              "args": {"object": "Lead", "fields": {"AI_Score__c": "$lead_score"}}}],
            variables=[{"name": "lead", "type": "Any", "direction": "local"}])
    r["understand"] = r["request"]
    assert kinds(r) == ["flow_constant"]
    try:
        FX.fix_record(r)
        assert False, "should be dropped"
    except FX.Unfixable:
        pass


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
