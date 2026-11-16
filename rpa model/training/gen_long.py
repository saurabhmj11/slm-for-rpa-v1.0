"""Long office processes (5-10 non-core steps) composed from realistic segments, with natural multi-sentence requests
that mention every step. The measured failure: long requests came back empty or lost their middle and last steps.

A process is: intake (mailbox attachments, a folder, SFTP, a website, an API, a database, a Google Sheet, Excel, CSV,
a ticket system) -> per-file work inside a loop (PDF text or OCR, AI field extraction, a condition, a try/catch,
filing the file) and/or table work (filter, de-duplicate, sort, total, count) -> an output document (Excel report with
a bold header, Excel to PDF, Word report from a template, CSV, merged PDF, zip, HTML table) -> an app update
(ServiceNow, Jira, Zendesk, Salesforce, a database, Google Sheets, OneDrive, Slack) -> a notification (mail server,
Outlook, Gmail, Teams, Slack; sometimes only when there is something to report). Sessions that were opened are
closed at the end. Each segment writes its own sentence; the request joins them in order.

Diversity: one action sequence is kept at most 20 times. Every flow passes repair + ooa_reward.problems == [];
requests close to a test request (difflib ratio > 0.8) are dropped.

Writes NEUTRAL records: {"kind": "flow", "request", "flow", "source": "long:<intake>"}.
Usage: python gen_long.py CATALOG.jsonl SCHEMA.json OUT.jsonl [n=3000] [seed=23]
"""
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ooa_reward as R   # noqa: E402
from gen_targeted import Checker, FlowBuilder, LeakGuard, person, typo, write_jsonl, flow_record, snake  # noqa: E402

THEMES = [
    {"items": "invoices", "one": "invoice", "subj": "Invoice", "col": "Amount", "team": "accounts payable", "ext": "pdf",
     "fields": {"invoice_number": "the invoice number", "vendor": "the vendor name", "total": "the total amount"}, "key": "invoice_number"},
    {"items": "purchase orders", "one": "purchase order", "subj": "PO", "col": "Total", "team": "procurement", "ext": "pdf",
     "fields": {"po_number": "the PO number", "supplier": "the supplier", "total": "the order total"}, "key": "po_number"},
    {"items": "expense claims", "one": "expense claim", "subj": "Expense claim", "col": "Amount", "team": "finance", "ext": "pdf",
     "fields": {"employee": "the employee name", "claim_date": "the claim date", "total": "the amount claimed"}, "key": "employee"},
    {"items": "timesheets", "one": "timesheet", "subj": "Timesheet", "col": "Hours", "team": "payroll", "ext": "xlsx",
     "fields": {"employee": "the employee name", "week": "the week", "total": "the total hours"}, "key": "employee"},
    {"items": "bank statements", "one": "bank statement", "subj": "Statement", "col": "Balance", "team": "treasury", "ext": "pdf",
     "fields": {"account": "the account number", "period": "the statement period", "total": "the closing balance"}, "key": "account"},
    {"items": "delivery notes", "one": "delivery note", "subj": "Delivery note", "col": "Quantity", "team": "warehouse", "ext": "pdf",
     "fields": {"note_number": "the delivery note number", "customer": "the customer", "total": "the number of items"}, "key": "note_number"},
    {"items": "resumes", "one": "resume", "subj": "Application", "col": "Experience", "team": "HR", "ext": "pdf",
     "fields": {"name": "the candidate name", "email": "the candidate email", "total": "years of experience"}, "key": "name"},
    {"items": "sales orders", "one": "sales order", "subj": "Order", "col": "Value", "team": "sales ops", "ext": "csv",
     "fields": {"order_number": "the order number", "customer": "the customer", "total": "the order value"}, "key": "order_number"},
    {"items": "insurance claims", "one": "claim", "subj": "Claim", "col": "Claim Amount", "team": "claims", "ext": "pdf",
     "fields": {"claim_number": "the claim number", "policy": "the policy number", "total": "the amount claimed"}, "key": "claim_number"},
    {"items": "utility bills", "one": "bill", "subj": "Bill", "col": "Amount Due", "team": "facilities", "ext": "pdf",
     "fields": {"account": "the account number", "due_date": "the due date", "total": "the amount due"}, "key": "account"},
    {"items": "quotes", "one": "quote", "subj": "Quotation", "col": "Price", "team": "purchasing", "ext": "pdf",
     "fields": {"quote_number": "the quote number", "supplier": "the supplier", "total": "the quoted price"}, "key": "quote_number"},
    {"items": "payslips", "one": "payslip", "subj": "Payslip", "col": "Net Pay", "team": "payroll", "ext": "pdf",
     "fields": {"employee": "the employee name", "month": "the month", "total": "the net pay"}, "key": "employee"},
]
OPENERS = ["", "", "", "Every morning: ", "Each Monday, ", "At month end, ", "Daily at 9: ", "Every evening, ", "Once a week, ",
           "Here's what I do by hand every day. ", "Can you automate this for the {team} team? ", "For the {team} team: ",
           "I need a bot for this. ", "Our {team} process: "]
THEN = ["Then", "After that,", "Next,", "Once that's done,", "Then", "After that", "Next"]
FINALLY = ["Finally,", "At the end,", "Lastly,", "When everything is done,", "Finally"]


class Ctx:
    def __init__(self, theme, rng):
        self.t, self.rng = theme, rng
        self.files = self.table = self.file = self.count = self.report = None
        self.file_kind = None
        self.closers, self.n = [], Counter()
        self.mail = None            # "imap" when a mailbox session is open

    def name(self, base):
        self.n[base] += 1
        return base if self.n[base] == 1 else f"{base}_{self.n[base]}"


def folder_var(b, label):
    return b.inp(snake(label) + "_folder", f"The {label} folder.")


# ------------------------------------------------------------------------------------------------------------ intake

def i_mail(b, c, rng):
    t = c.t
    box = rng.choice([t["team"], "shared", "AP", "ops", "orders"])
    subj = rng.choice([t["subj"], t["subj"].lower(), t["items"]])
    ext = t["ext"] if t["ext"] != "csv" else rng.choice(["csv", "xlsx"])
    steps = [b.s("ooa.email/connect", f"Sign in to the {box} mailbox", {"imap_host": b.inp("imap_host", "Incoming mail server."),
                                                                         "user": b.inp("mail_user", "Mailbox user name."),
                                                                         "password": b.inp("mail_password", "Mailbox password.", "Credential"),
                                                                         "smtp_host": b.inp("smtp_host", "Outgoing mail server.")})]
    steps.append(b.s("ooa.email/save_all_attachments", f"Save the {ext.upper()} attachments of unread {subj} emails",
                     {"unread_only": True, "mark_read": True, "subject_contains": subj, "name_pattern": f"*.{ext}",
                      "target_folder": folder_var(b, "incoming")}, {"files": "files", "email_count": "email_count"}))
    c.files, c.file_kind, c.mail, c.count = "files", ext, "imap", "email_count"
    c.closers.append(b.s("ooa.email/disconnect", "Sign out of the mailbox", {}))
    return steps, rng.choice([f"sign in to the {box} mailbox and save the {ext.upper()} attachments of the unread emails with '{subj}' in the subject into our incoming folder",
                              f"check the {box} inbox for unread {subj} emails and download their .{ext} attachments to the incoming folder",
                              f"pull the {t['items']} that came in by email ({ext} attachments, unread mails with {subj} in the subject) into a local folder"])


def i_gmail(b, c, rng):
    t = c.t
    ext = t["ext"] if t["ext"] != "csv" else "csv"
    step = b.s("ooa.google/gmail_save_all_attachments", f"Save the {ext.upper()} attachments from Gmail",
               {"access_token": b.inp("google_token", "Google access token.", "Credential"), "target_folder": folder_var(b, "incoming"),
                "subject_contains": t["subj"], "name_pattern": f"*.{ext}"}, {"files": "files"})
    c.files, c.file_kind = "files", ext
    return [step], rng.choice([f"save the {ext} attachments of the Gmail messages about {t['items']} into the incoming folder",
                               f"from our Gmail, download every {ext.upper()} attached to emails with '{t['subj']}' in the subject"])


def i_folder(b, c, rng):
    t = c.t
    where = rng.choice(["the shared drive", "the scanner folder", "the inbound folder", f"the {t['items']} folder", "the network share"])
    step = b.s("ooa.file/list", f"List the {t['ext'].upper()} files in the folder",
               {"folder": b.inp("source_folder", f"The folder with the {t['items']}."), "pattern": f"*.{t['ext']}"},
               {"files": "files", "count": "file_count"})
    c.files, c.file_kind, c.count = "files", t["ext"], "file_count"
    return [step], rng.choice([f"take all the {t['ext']} {t['items']} in {where}", f"pick up every .{t['ext']} file in {where}",
                               f"look in {where} for {t['items']} ({t['ext']} files)"])


def i_sftp(b, c, rng):
    t = c.t
    party = rng.choice(["the bank", "our logistics partner", "the payroll provider", "the distributor", "the card processor"])
    steps = [b.s("ooa.file_transfer/connect", f"Connect to the SFTP server of {party}", {"protocol": "sftp", "host": b.inp("sftp_host", "SFTP server."),
                                                                                         "user": b.inp("sftp_user", "SFTP user name."),
                                                                                         "password": b.inp("sftp_password", "SFTP password.", "Credential")}),
             b.s("ooa.file_transfer/download", "Download today's file", {"remote_path": b.inp("remote_path", "Path of the file on the server."),
                                                                         "local_path": b.inp("local_file", "Where to save it locally.")})]
    steps2 = [b.s("ooa.csv/read", "Read the downloaded file", {"path": "$local_file"}, {"table": "data", "row_count": "row_count"})]
    c.closers.append(b.s("ooa.file_transfer/disconnect", "Disconnect from the SFTP server", {}))
    c.table, c.count = "data", "row_count"
    return steps + steps2, rng.choice([f"download today's {t['items']} file from {party}'s SFTP server and read it",
                                       f"connect to {party}'s SFTP, grab the {t['items']} CSV and load it as a table"])


def i_web(b, c, rng):
    t = c.t
    site = rng.choice(["the supplier portal", "the bank portal", "the vendor portal", "the government portal", "the carrier portal"])
    steps = [b.s("ooa.browser/open", f"Open {site}", {"url": b.inp("portal_url", f"Address of {site}.")}),
             b.s("ooa.browser/type_text", "Type the user name", {"text": b.inp("portal_user", "Portal user name."), "selector": "#username"}),
             b.s("ooa.browser/type_text", "Type the password", {"text": b.inp("portal_password", "Portal password.", "Credential"), "selector": "#password"}),
             b.s("ooa.browser/click", "Click Sign in", {"selector": "button[type=submit]"})]
    if rng.random() < 0.5:
        steps.append(b.s("ooa.browser/extract_table", f"Read the {t['items']} table", {}, {"table": "data"}))
        c.table = "data"
        what = rng.choice([f"read the {t['items']} table on the page", f"copy the {t['items']} list shown after login"])
    else:
        steps.append(b.s("ooa.browser/click", "Click the export link", {"selector": "text=Export"}))
        steps.append(b.s("ooa.browser/download", "Download the export", {"folder": folder_var(b, "downloads")}, {"path": "export_file"}))
        steps.append(b.s("ooa.csv/read", "Read the exported file", {"path": "$export_file"}, {"table": "data", "row_count": "row_count"}))
        c.table, c.count = "data", "row_count"
        what = rng.choice([f"click Export to download the {t['items']} CSV and read it", f"download the {t['items']} export and open it as a table"])
    c.closers.append(b.s("ooa.browser/close", "Close the browser", {}))
    return steps, rng.choice([f"log in to {site} with our user name and password, {what}", f"open {site}, sign in, then {what}"])


def i_api(b, c, rng):
    t = c.t
    steps = [b.s("ooa.rest/request", f"Get the {t['items']} from the API", {"url": b.inp("api_url", f"Address of the {t['items']} API."),
                                                                            "method": "GET", "bearer_token": b.inp("api_token", "API token.", "Credential")},
                 {"json": "response"}),
             b.s("ooa.json/to_table", "Turn the items into a table", {"value": "$response", "path": "items"}, {"table": "data"})]
    c.table = "data"
    return steps, rng.choice([f"call our {t['items']} API and turn the items in the response into a table",
                              f"pull the {t['items']} from the REST API (the items list) as a table"])


def i_db(b, c, rng):
    t = c.t
    steps = [b.s("ooa.database/connect", "Connect to the database", {"database_type": "sqlserver", "server": b.inp("db_server", "Database server."),
                                                                    "database": b.inp("db_name", "Database name."), "user": b.inp("db_user", "Database user."),
                                                                    "password": b.inp("db_password", "Database password.", "Credential")}),
             b.s("ooa.database/query", f"Read the open {t['items']}", {"sql": f"SELECT * FROM {snake(t['items'])} WHERE status = 'Open'"},
                 {"table": "data", "row_count": "row_count"})]
    c.table, c.count, c.db = "data", "row_count", True
    c.closers.append(b.s("ooa.database/disconnect", "Disconnect from the database", {}))
    return steps, rng.choice([f"query the open {t['items']} from our SQL Server database", f"connect to the {t['team']} database and read all {t['items']} with status Open"])


def i_gsheet(b, c, rng):
    t = c.t
    step = b.s("ooa.google/sheets_read", f"Read the {t['items']} Google Sheet", {"access_token": b.inp("google_token", "Google access token.", "Credential"),
                                                                                 "spreadsheet_id": b.inp("spreadsheet_id", f"Id of the {t['items']} Google Sheet.")},
               {"table": "data"})
    c.table, c.gsheet = "data", True
    return [step], rng.choice([f"read the {t['items']} tracker from Google Sheets", f"open our {t['items']} Google Sheet and read all rows"])


def i_excel(b, c, rng):
    t = c.t
    steps = [b.s("ooa.excel/open", f"Open the {t['items']} workbook", {"path": b.inp("tracker_path", f"The {t['items']} Excel workbook.")}),
             b.s("ooa.excel/get_all_cells", "Read the sheet as a table", {"sheet": t["items"].title()}, {"table": "data", "row_count": "row_count"}),
             b.s("ooa.excel/close", "Close the workbook", {})]
    c.table, c.count = "data", "row_count"
    return steps, rng.choice([f"open the {t['items']} Excel tracker and read the {t['items'].title()} sheet",
                              f"read the {t['items'].title()} sheet of our Excel workbook (then close it)"])


def i_csv(b, c, rng):
    t = c.t
    step = b.s("ooa.csv/read", f"Read the {t['items']} CSV", {"path": b.inp("csv_path", f"The {t['items']} CSV export.")}, {"table": "data", "row_count": "row_count"})
    c.table, c.count = "data", "row_count"
    return [step], rng.choice([f"read the {t['items']} CSV export", f"load the daily {t['items']} csv file"])


def i_tickets(b, c, rng):
    kind = rng.choice(["servicenow", "jira", "zendesk"])
    if kind == "servicenow":
        step = b.s("ooa.servicenow/query", "Get the open incidents", {"instance_url": b.inp("instance_url", "ServiceNow address."),
                                                                     "access_token": b.inp("sn_token", "ServiceNow token.", "Credential"),
                                                                     "table": "incident", "query": "active=true^priority=1"}, {"table": "data", "count": "ticket_count"})
        what = rng.choice(["get the active priority 1 incidents from ServiceNow", "pull all open P1 incidents out of ServiceNow"])
    elif kind == "jira":
        step = b.s("ooa.jira/search", "Find the overdue issues", {"site_url": b.inp("jira_url", "Jira site address."),
                                                                  "email": b.inp("jira_email", "Jira user email."),
                                                                  "api_token": b.inp("jira_token", "Jira API token.", "Credential"),
                                                                  "jql": "duedate < now() AND statusCategory != Done"}, {"table": "data", "count": "ticket_count"})
        what = rng.choice(["search Jira for issues past their due date that aren't done", "find the overdue Jira issues"])
    else:
        step = b.s("ooa.zendesk/search_tickets", "Find the open urgent tickets", {"site": b.inp("zendesk_site", "Zendesk site."),
                                                                                  "email": b.inp("zendesk_email", "Zendesk user email."),
                                                                                  "api_token": b.inp("zendesk_token", "Zendesk API token.", "Credential"),
                                                                                  "query": "status:open priority:urgent"}, {"table": "data", "count": "ticket_count"})
        what = rng.choice(["find all open urgent tickets in Zendesk", "search Zendesk for urgent open tickets"])
    c.table, c.count, c.tickets = "data", "ticket_count", True
    return [step], what


INTAKE = {"mail": (i_mail, 16), "gmail": (i_gmail, 5), "folder": (i_folder, 14), "sftp": (i_sftp, 8), "web": (i_web, 8),
          "api": (i_api, 6), "db": (i_db, 8), "gsheet": (i_gsheet, 9), "excel": (i_excel, 9), "csv": (i_csv, 9), "tickets": (i_tickets, 8)}


# -------------------------------------------------------------------------------------------------- per-file work

def p_files(b, c, rng):
    """Loop over the files: read (text / OCR / table), maybe extract fields with AI, decide, file it."""
    t = c.t
    kind = c.file_kind
    body, says = [], []
    summary = None
    if kind == "pdf":
        if rng.random() < 0.4:
            body.append(b.s("ooa.pdf/read_text_ocr", "Read the scanned PDF with OCR", {"path": "$file"}, {"text": "text"}))
            says.append(rng.choice(["read it with OCR (they are scans)", "OCR the scanned pages"]))
        else:
            body.append(b.s("ooa.pdf/extract_text", "Get the text of the PDF", {"path": "$file"}, {"text": "text"}))
            says.append(rng.choice(["get the text out of the PDF", "read the PDF text"]))
        body.append(b.s("ooa.genai/extract", f"Pull out the {', '.join(w.split(' ', 1)[1] for w in list(t['fields'].values())[:2])} and total",
                        {"text": "$text", "fields": dict(t["fields"])}, {"fields": "fields"}))
        says.append(rng.choice([f"have AI pick out {', '.join(t['fields'].values())}", f"extract {', '.join(t['fields'].values())} with AI"]))
        if rng.random() < 0.6:
            summary = b.inp("summary_csv", "CSV file that collects one line per file.")
            body.append(b.s("ooa.file/write_text", "Add a line to the summary file", {"path": summary, "append": True,
                                                                                     "text": "${fields." + t["key"] + "},${fields.total}\n"}))
            says.append(rng.choice(["add a line with those values to a summary CSV", "append them to our summary csv"]))
        lim = rng.choice([1000, 5000, 10000, 25000, 50000])
        if rng.random() < 0.55:
            hi, lo = b.inp("approval_folder", "Folder for files that need approval."), b.inp("processed_folder", "Folder for processed files.")
            body.append(b.if_(f"$fields.total > {lim}", [b.s("ooa.file/move", "Move it to the approval folder", {"source": "$file", "target": hi})],
                              f"If the total is over {lim}", [b.s("ooa.file/move", "Move it to the processed folder", {"source": "$file", "target": lo})]))
            says.append(rng.choice([f"if the total is over {lim:,} move it to the approval folder, otherwise to processed",
                                    f"files with a total above {lim:,} go to approval, the rest to the processed folder"]))
        else:
            body.append(b.s("ooa.file/move", "Move it to the processed folder", {"source": "$file", "target": b.inp("processed_folder", "Folder for processed files.")}))
            says.append(rng.choice(["then move it to the processed folder", "file it in the processed folder"]))
    elif kind in ("csv", "xlsx"):
        if kind == "csv":
            body.append(b.s("ooa.csv/read", "Read the file", {"path": "$file"}, {"table": "part", "row_count": "part_rows"}))
        else:
            body += [b.s("ooa.excel/open", "Open the workbook", {"path": "$file", "read_only": True}),
                     b.s("ooa.excel/get_all_cells", "Read the sheet as a table", {}, {"table": "part", "row_count": "part_rows"}),
                     b.s("ooa.excel/close", "Close the workbook", {})]
        says.append(rng.choice(["read it", f"open it and read the rows"]))
        summary = b.inp("combined_csv", "The combined CSV file.")
        body.append(b.s("ooa.csv/write", "Append the rows to the combined CSV", {"table": "$part", "path": summary, "append": True}))
        says.append(rng.choice(["append its rows to one combined CSV", "add the rows to a master csv"]))
        body.append(b.log("${file}: ${part_rows} rows", "Log the row count"))
        says.append("log how many rows it had")
    else:
        return None
    if rng.random() < 0.35:
        err = b.inp("error_folder", "Folder for files that could not be processed.")
        body = [b.try_(body, [b.s("ooa.file/move", "Move the file to the error folder", {"source": "$file", "target": err}),
                              b.log("Could not process ${file}: ${error_message}", "Log the error", "error")], "Try to process the file")]
        says.append(rng.choice(["if anything fails for a file, move it to the error folder and log the error",
                                "anything that can't be read goes to an error folder (log it)"]))
    c.moved = any(s.get("action") == "ooa.file/move" for s in R.walk(body))
    loop = b.loop("$files", "file", body, f"For each {t['one']} file", "String")
    steps = [loop]
    phrase = rng.choice([f"for each file, ", f"then for every {t['one']}: ", f"go through the files one by one: "]) + ", ".join(says)
    if summary:
        steps.append(b.s("ooa.csv/read", "Read the summary file", {"path": summary}, {"table": "summary", "row_count": "summary_rows"}))
        c.table, c.count = "summary", "summary_rows"
        phrase += rng.choice(["; afterwards read the summary CSV back as a table", ". Then load the combined file as a table"])
    c.files_done = True
    return steps, phrase


# --------------------------------------------------------------------------------------------------- table work

def p_table(b, c, rng):
    t = c.t
    ops = rng.sample(["filter", "dedupe", "sort", "sum", "count"], rng.choice([1, 2, 2, 3]))
    order = ["dedupe", "filter", "sort", "sum", "count"]
    steps, says = [], []
    for op in sorted(ops, key=order.index):
        src = "$" + c.table
        if op == "filter":
            col, val = rng.choice([("Status", "Unpaid"), ("Status", "Pending"), ("Status", "Open"), ("Region", "North"), ("Approved", "No"),
                                   ("Status", "Rejected"), ("Priority", "High")])
            nv = c.name("filtered")
            steps.append(b.s("ooa.datatable/filter", f"Keep the rows where {col} is {val}", {"table": src, "column": col, "operator": "eq", "value": val},
                             {"table": nv, "count": nv + "_count"}))
            c.table, c.count = nv, nv + "_count"
            says.append(rng.choice([f"keep only the rows where {col} is {val}", f"filter to {col} = {val}"]))
        elif op == "dedupe":
            nv = c.name("unique")
            steps.append(b.s("ooa.datatable/remove_duplicates", "Remove duplicate rows", {"table": src}, {"table": nv}))
            c.table = nv
            says.append(rng.choice(["remove the duplicate rows", "drop duplicates"]))
        elif op == "sort":
            col = rng.choice([t["col"], "Date", "Due Date"])
            nv = c.name("sorted")
            steps.append(b.s("ooa.datatable/sort", f"Sort by {col}", {"table": src, "column": col, "descending": True}, {"table": nv}))
            c.table = nv
            says.append(rng.choice([f"sort by {col.lower()}, largest first", f"sort it on {col} descending"]))
        elif op == "sum":
            steps.append(b.s("ooa.datatable/sum_column", f"Add up the {t['col']} column", {"table": src, "column": t["col"]}, {"total": "grand_total"}))
            c.total = "grand_total"
            says.append(rng.choice([f"add up the {t['col'].lower()} column", f"work out the total {t['col'].lower()}"]))
        else:
            steps.append(b.s("ooa.datatable/row_count", "Count the rows", {"table": src}, {"count": "row_total"}))
            c.count = "row_total"
            says.append(rng.choice(["count the rows", "count how many there are"]))
    return steps, ", ".join(says)


# ------------------------------------------------------------------------------------------------------- outputs

def o_excel(b, c, rng):
    t = c.t
    rp = b.inp("report_path", f"The {t['items']} Excel report to create.")
    steps = [b.s("ooa.excel/create_workbook", "Create the report workbook", {"path": rp}),
             b.s("ooa.excel/write_table", "Write the table to the sheet", {"table": "$" + c.table, "include_header": True})]
    says = ["write it into a new Excel workbook"]
    if rng.random() < 0.6:
        steps.append(b.s("ooa.excel/format_cells", "Make the header row bold", {"range": "A1:Z1", "bold": True}))
        steps.append(b.s("ooa.excel/autofit_columns", "Fit the column widths", {}))
        says.append(rng.choice(["make the header bold and fit the columns", "bold header row, autofit columns"]))
    steps += [b.s("ooa.excel/save", "Save the workbook", {}), b.s("ooa.excel/close", "Close the workbook", {})]
    c.report, c.report_label = rp, "the Excel report"
    if rng.random() < 0.3:
        pdf = b.inp("report_pdf", "The PDF copy of the report.")
        steps.append(b.s("ooa.excel_desktop/export_pdf", "Export the report to PDF", {"path": rp, "target": pdf}))
        says.append(rng.choice(["export it to PDF as well", "and save a PDF version"]))
        c.report, c.report_label = pdf, "the PDF report"
    return steps, ", ".join(says)


def o_word(b, c, rng):
    t = c.t
    steps = [b.s("ooa.word/open", "Open the report template", {"path": b.inp("template_path", "The Word report template.")}),
             b.s("ooa.word/replace_text", "Put today's date in the template", {"find": "{{date}}", "replace_with": "${sys.date}"}),
             b.s("ooa.word/add_table", "Insert the table", {"table": "$" + c.table, "after_text": "Details"}),
             b.s("ooa.word/save_as", "Save the report as PDF", {"path": b.inp("report_pdf", "The PDF report to create."), "format": "pdf"}),
             b.s("ooa.word/close", "Close the document", {"save": False})]
    c.report, c.report_label = "$report_pdf", "the PDF report"
    return steps, rng.choice(["fill our Word report template (today's date in place of {{date}}, the table after 'Details') and save it as a PDF",
                              "open the Word template, replace {{date}} with today's date, insert the table under Details and export it to PDF"])


def o_csv(b, c, rng):
    t = c.t
    p = b.inp("output_csv", f"The {t['items']} CSV to create.")
    c.report, c.report_label = p, "the CSV"
    return [b.s("ooa.csv/write", "Save the table as CSV", {"table": "$" + c.table, "path": p})], \
        rng.choice(["save the result as a CSV file", "write it out to a new csv"])


def o_merge(b, c, rng):
    if not c.files or getattr(c, "moved", False):     # the listed paths are gone once the loop moved the files
        return None
    p = b.inp("merged_pdf", "The merged PDF to create.")
    c.report, c.report_label = p, "the merged PDF"
    return [b.s("ooa.pdf/merge", "Merge the PDFs into one file", {"files": "$files", "target": p}, {"page_count": "page_count"})], \
        rng.choice(["merge all those PDFs into one file", "combine the PDFs into a single document"])


def o_zip(b, c, rng):
    if not c.files:
        return None
    src = next(("$" + v for v in ("processed_folder", "incoming_folder", "source_folder") if v in b.vars), None)
    if src is None:
        return None
    p = b.inp("zip_path", "The zip file to create.")
    c.report, c.report_label = p, "the zip"
    return [b.s("ooa.file/zip", "Zip the files", {"source": src, "target": p}, {"path": "zip_file"})], \
        rng.choice(["zip the folder", "put the files into one zip"])


def o_html(b, c, rng):
    c.html = "html"
    return [b.s("ooa.datatable/to_html", "Turn the table into an HTML table", {"table": "$" + c.table}, {"html": "html"})], \
        rng.choice(["turn the table into an HTML table for the email body", "make an HTML table of it for the mail"])


OUTPUTS = {"excel": (o_excel, 10), "word": (o_word, 5), "csv": (o_csv, 6), "merge": (o_merge, 4), "zip": (o_zip, 3), "html": (o_html, 5)}


# -------------------------------------------------------------------------------------------------- app updates

def u_rows(b, c, rng):
    """For each row: create a record in a business app."""
    t = c.t
    rec = b.s("ooa.datatable/to_records", "Turn the rows into records", {"table": "$" + c.table}, {"records": "records"})
    kind = rng.choice(["servicenow", "jira", "zendesk", "salesforce", "db"])
    if kind == "servicenow":
        s = b.s("ooa.servicenow/create_record", "Create a ServiceNow task", {"instance_url": b.inp("instance_url", "ServiceNow address."),
                                                                             "access_token": b.inp("sn_token", "ServiceNow token.", "Credential"),
                                                                             "table": "sc_task", "fields": {"short_description": f"Follow up {t['one']} " + "${row." + t["col"].replace(" ", "_") + "}"}},
                {"number": "task_number"})
        say = rng.choice([f"for each row create a ServiceNow task to follow it up", "raise a ServiceNow task per row"])
    elif kind == "jira":
        s = b.s("ooa.jira/create_issue", "Create a Jira issue", {"site_url": b.inp("jira_url", "Jira site address."), "email": b.inp("jira_email", "Jira user email."),
                                                                 "api_token": b.inp("jira_token", "Jira API token.", "Credential"), "project": "OPS",
                                                                 "summary": f"Check {t['one']} " + "${row.ID}"}, {"key": "issue_key"})
        say = rng.choice(["open a Jira issue in the OPS project for each row", "for every row, create a Jira ticket in OPS"])
    elif kind == "zendesk":
        s = b.s("ooa.zendesk/create_ticket", "Create a Zendesk ticket", {"site": b.inp("zendesk_site", "Zendesk site."), "email": b.inp("zendesk_email", "Zendesk user email."),
                                                                         "api_token": b.inp("zendesk_token", "Zendesk API token.", "Credential"),
                                                                         "subject": f"{t['subj']} " + "${row.ID}", "description": f"Please review this {t['one']}."},
                {"ticket_id": "ticket_id"})
        say = rng.choice(["create a Zendesk ticket for each one", "open one Zendesk ticket per row"])
    elif kind == "salesforce":
        s = b.s("ooa.salesforce/create_record", "Create a Salesforce task", {"object": "Task", "fields": {"Subject": f"Follow up {t['one']} " + "${row.ID}"},
                                                                             "instance_url": b.inp("sf_instance_url", "Salesforce instance address."),
                                                                             "access_token": b.inp("sf_token", "Salesforce access token.", "Credential")}, {"id": "task_id"})
        say = rng.choice(["create a follow-up Task in Salesforce for each row", "log a Salesforce task per row"])
    else:
        if not getattr(c, "db", False):
            return None
        s = b.s("ooa.database/execute", "Mark the row as processed", {"sql": f"UPDATE {snake(t['items'])} SET status = 'Processed' WHERE id = @id",
                                                                      "parameters": {"id": "$row.ID"}}, {"rows_affected": "updated"})
        say = rng.choice(["set each one's status to Processed in the database", "update every row to Processed in the database"])
    return [rec, b.loop("$records", "row", [s], "For each row", "Dictionary")], say


def u_bulk(b, c, rng):
    t = c.t
    kind = rng.choice(["gsheet", "db", "onedrive", "drive", "slack_file"])
    if kind in ("gsheet", "db") and not c.table:
        return None
    if kind == "gsheet":
        if getattr(c, "gsheet", False):
            return None
        s = b.s("ooa.google/sheets_append", "Append the rows to the Google Sheet", {"access_token": b.inp("google_token", "Google access token.", "Credential"),
                                                                                   "spreadsheet_id": b.inp("log_sheet_id", "Id of the log Google Sheet."),
                                                                                   "range": "Log!A1", "table": "$" + c.table})
        say = rng.choice(["append the rows to our log Google Sheet", "add them to the Log tab of the Google Sheet"])
    elif kind == "db":
        s = b.s("ooa.database/bulk_insert", "Insert the rows into the staging table", {"table": f"stg_{snake(t['items'])}", "data": "$" + c.table},
                {"rows_inserted": "inserted"})
        if not getattr(c, "db", False):
            con = b.s("ooa.database/connect", "Connect to the database", {"database_type": "sqlserver", "server": b.inp("db_server", "Database server."),
                                                                          "database": b.inp("db_name", "Database name."), "user": b.inp("db_user", "Database user."),
                                                                          "password": b.inp("db_password", "Database password.", "Credential")})
            c.closers.append(b.s("ooa.database/disconnect", "Disconnect from the database", {}))
            c.db = True
            return [con, s], rng.choice([f"connect to the database and load the rows into the stg_{snake(t['items'])} table",
                                         f"bulk insert everything into stg_{snake(t['items'])} in SQL Server"])
        say = f"insert the rows into the stg_{snake(t['items'])} table"
    elif kind == "onedrive":
        if not c.report:
            return None
        s = b.s("ooa.microsoft365/upload_file", "Upload the report to SharePoint", {"access_token": b.inp("ms_token", "Microsoft 365 access token.", "Credential"),
                                                                                    "local_path": c.report, "remote_path": f"/Reports/{t['items'].title()}"})
        say = rng.choice([f"upload {c.report_label} to the Reports folder on SharePoint", f"put {c.report_label} on OneDrive under Reports"])
    elif kind == "drive":
        if not c.report:
            return None
        s = b.s("ooa.google/drive_upload", "Upload the report to Google Drive", {"access_token": b.inp("google_token", "Google access token.", "Credential"),
                                                                                 "local_path": c.report, "folder_id": b.inp("drive_folder_id", "Id of the Drive folder.")},
                {"file_id": "drive_file_id"})
        say = rng.choice([f"upload {c.report_label} to our Google Drive folder", f"save {c.report_label} to Google Drive"])
    else:
        if not c.report:
            return None
        s = b.s("ooa.slack/upload_file", "Upload the report to Slack", {"bot_token": b.inp("slack_token", "Slack bot token.", "Credential"),
                                                                        "channel": f"#{snake(t['team'])}", "files": [c.report]})
        say = rng.choice([f"upload {c.report_label} to the #{snake(t['team'])} Slack channel", f"share {c.report_label} in Slack (#{snake(t['team'])})"])
    return [s], say


UPDATES = {"rows": (u_rows, 5), "bulk": (u_bulk, 5), "none": (None, 4)}


# --------------------------------------------------------------------------------------------------------- notify

def n_notify(b, c, rng):
    t = c.t
    who, var, d = person(rng)
    b.inp(var, d)
    count = "${" + c.count + "}" if c.count else None
    body = (f"{count} {t['items']} processed." if count else f"The {t['items']} run has finished.")
    if getattr(c, "total", None):
        body += " Total: ${" + c.total + "}."
    att = [c.report] if c.report and not c.report.startswith("$files") else None
    kind = rng.choice(["smtp", "smtp", "outlook", "gmail", "teams", "slack"])
    if kind in ("teams", "slack") and (getattr(c, "html", None) or (att and rng.random() < 0.5)):
        kind = rng.choice(["smtp", "outlook", "gmail"])            # an HTML table is made for an email body
    if getattr(c, "html", None):
        body = "${html}"
    if kind == "smtp":
        a = {"to": "$" + var, "subject": f"{t['subj']} run summary", "body": body}
        if getattr(c, "html", None):
            a["html"] = True
        if att:
            a["attachments"] = att
        s = b.s("ooa.email/send", f"Email {who} the summary", a)
        say = f"email {who} " + (f"{c.report_label} with " if att else "") + ("the HTML table" if getattr(c, "html", None) else "a summary") + \
            (f" saying how many were done" if count else "")
    elif kind == "outlook":
        a = {"access_token": b.inp("ms_token", "Microsoft 365 access token.", "Credential"), "to": "$" + var, "subject": f"{t['subj']} run summary", "body": body}
        if att:
            a["attachments"] = att
        if getattr(c, "html", None):
            a["html"] = True
        s = b.s("ooa.microsoft365/send_mail", f"Send {who} the summary from Outlook", a)
        say = f"send {who} " + (f"{c.report_label}" if att else "the HTML table" if getattr(c, "html", None) else "a summary") + " from Outlook"
    elif kind == "gmail":
        a = {"access_token": b.inp("google_token", "Google access token.", "Credential"), "to": "$" + var, "subject": f"{t['subj']} run summary", "body": body}
        if att:
            a["attachments"] = att
        if getattr(c, "html", None):
            a["html"] = True
        s = b.s("ooa.google/gmail_send", f"Send {who} the summary from Gmail", a)
        say = f"mail {who} " + (f"{c.report_label}" if att else "the HTML table" if getattr(c, "html", None) else "a summary") + " from Gmail"
    elif kind == "teams":
        s = b.s("ooa.microsoft365/teams_message", "Post the summary in Teams", {"access_token": b.inp("ms_token", "Microsoft 365 access token.", "Credential"),
                                                                                "message": body, "team_name": t["team"].title(), "channel_name": "General"})
        say = f"post a summary in the {t['team'].title()} Teams channel" + (" with the count" if count else "")
    else:
        s = b.s("ooa.slack/post_message", "Post the summary in Slack", {"bot_token": b.inp("slack_token", "Slack bot token.", "Credential"),
                                                                        "channel": f"#{snake(t['team'])}", "text": body})
        say = f"post the summary to #{snake(t['team'])} on Slack"
    if c.count and rng.random() < 0.35:
        node = b.if_(f"${c.count} > 0", [s], f"If there is anything to report", [b.log(f"No {t['items']} today", "Log that there was nothing")])
        return [node], say + rng.choice([" (only if there was at least one; otherwise just log it)", ", but only when the count is above zero - else log 'nothing today'"])
    return [s], say


# --------------------------------------------------------------------------------------------------------- build

def sentence(clause, k, last, rng):
    clause = clause.strip().rstrip(".")
    if k == 0:
        pre = ""
    elif last:
        pre = rng.choice(FINALLY) + " "
    else:
        pre = rng.choice(THEN) + " " if rng.random() < 0.75 else ""
    s = pre + clause
    return s[0].upper() + s[1:] + "."


def compose(B, rng):
    b = B()
    t = rng.choice(THEMES)
    c = Ctx(t, rng)
    names, w = zip(*[(k, v[1]) for k, v in INTAKE.items()])
    intake = rng.choices(names, w)[0]
    parts = []
    steps, say = INTAKE[intake][0](b, c, rng)
    parts.append(say)
    if c.files:
        got = p_files(b, c, rng)
        if got is None:
            return None
        s2, say2 = got
        steps += s2
        parts.append(say2)
    if c.table and (not c.files or rng.random() < 0.5):
        s2, say2 = p_table(b, c, rng)
        steps += s2
        parts.append(say2)
    if c.table or c.files:
        names, w = zip(*[(k, v[1]) for k, v in OUTPUTS.items()])
        o = rng.choices(names, w)[0]
        if (o in ("excel", "word", "csv", "html")) and not c.table:
            o = rng.choice(["merge", "zip"])
        if o == "merge" and (c.file_kind != "pdf" or getattr(c, "moved", False)):
            o = "zip" if c.files else "csv"
        got = OUTPUTS[o][0](b, c, rng)
        if got:
            steps += got[0]
            parts.append(got[1])
    names, w = zip(*[(k, v[1]) for k, v in UPDATES.items()])
    u = rng.choices(names, w)[0]
    if UPDATES[u][0] and (c.table or u == "bulk"):
        got = UPDATES[u][0](b, c, rng)
        if got:
            steps += got[0]
            parts.append(got[1])
    s2, say2 = n_notify(b, c, rng)
    steps += s2
    parts.append(say2)
    steps += c.closers
    if c.closers and rng.random() < 0.5:
        parts.append(rng.choice(["close everything that was opened", "sign out / disconnect at the end"]))
    real = [s for s in R.walk(steps) if not s["action"].startswith("core/")]
    if not 5 <= len(real) <= 10:
        return None
    opener = rng.choice(OPENERS).format(team=t["team"])
    style = rng.random()
    if style < 0.15:
        body = " ".join(f"{k + 1}) {p.strip()[0].upper()}{p.strip()[1:].rstrip('.')}." for k, p in enumerate(parts))
    elif style < 0.3:
        txt = "; ".join(p.strip().rstrip(".") for p in parts)
        body = txt[0].upper() + txt[1:] + "."
    else:
        body = " ".join(sentence(p, k, k == len(parts) - 1, rng) for k, p in enumerate(parts))
    if opener.endswith(", ") and not body[0].isdigit():
        body = body[0].lower() + body[1:]
    req = opener + body
    if rng.random() < 0.08:
        req = typo(req, rng, 2)
    title = f"{t['items'].title()} {intake} process"
    return req, b.flow(title, steps, description=f"{t['items'].capitalize()} process: " + "; ".join(p.strip() for p in parts)[:400])


def main():
    cat_path, schema, out = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    rng = random.Random(int(sys.argv[5]) if len(sys.argv) > 5 else 23)
    checker, guard = Checker(cat_path, schema), LeakGuard()
    from gen_targeted import run_families
    rows, made, why = run_families({"long": (compose, 1)}, n, rng, checker, guard, "long", max_per_seq=20)
    for r in rows:
        r["source"] = "long:" + r["flow"]["name"].split()[-2]
    write_jsonl(out, rows)
    seqs = Counter(tuple(R.actions_of(r["flow"])) for r in rows)
    lens = Counter(sum(1 for a in R.actions_of(r["flow"]) if not a.startswith("core/")) for r in rows)
    print(f"{len(rows)} records | checker pass {checker.ok}/{checker.n} | {len(seqs)} distinct action sequences (max {max(seqs.values())})"
          f" | non-core steps {dict(sorted(lens.items()))} | skipped {dict(why)}")


if __name__ == "__main__":
    main()
