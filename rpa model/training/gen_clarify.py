"""Clarifying questions: requests too vague to automate get ONE short question asking for the missing piece (which
file, which app, what to do with it, who to send it to). Also short requests that ARE clear enough, answered with a
normal flow, so the model does not ask when it can already build the flow.

Writes NEUTRAL records:
  {"kind": "clarify", "request", "question", "source": "clarify:<intent>"}            (vague; about 600)
  {"kind": "flow", "request", "flow", "source": "clarify_counter:<intent>"}          (short but clear; about 300)
Every counter-example flow passes repair + ooa_reward.problems == []; requests close to a test request (difflib
ratio > 0.8) are dropped.

Usage: python gen_clarify.py CATALOG.jsonl SCHEMA.json OUT.jsonl [n_vague=600] [n_clear=300] [seed=31]
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_targeted import Checker, LeakGuard, run_families, typo, write_jsonl   # noqa: E402

# intent: ([vague requests], [one short question each could get])
VAGUE = {
    "invoices": (["automate my invoices", "do the invoice thing", "handle invoices automatically", "invoices ka kuch automate karo",
                  "make a bot for invoices", "can u automate invoice stuff"],
                 ["Where do the invoices come from (email, a folder or a portal), and what should happen to each one?",
                  "What should the bot do with the invoices: save them, read the totals, or enter them somewhere? And where do they arrive?"]),
    "monthly_report": (["do the monthly report thing", "make the monthly report", "automate the month end report", "monthly report bana do",
                        "run the usual report"],
                       ["Which report is it: where does its data come from, and should the result be Excel, PDF or an email?",
                        "What goes into the monthly report (which file or system), and who gets it?"]),
    "fix_excel": (["fix the excel", "fix my spreadsheet", "the excel is broken, fix it", "excel theek kar do", "clean up the excel"],
                  ["Which Excel file is it, and what needs fixing: duplicates, empty rows, formulas or formatting?",
                   "Which workbook should I work on, and what exactly is wrong with it?"]),
    "send_it": (["send it to them", "send this to them", "mail it over", "forward it", "bhej do unko", "send the thing"],
                ["Which file or message should I send, and to whom?", "What should be sent, and to which email address?"]),
    "emails": (["handle my emails", "do something with the emails", "automate my inbox", "sort out the mailbox", "emails manage karo"],
               ["Which emails (folder, sender or subject), and what should happen to them: move, reply, forward or save the attachments?",
                "Which mailbox and which emails should I look at, and what should be done with each one?"]),
    "process_files": (["process the files", "do the files", "handle the documents", "process these", "files ko process karo"],
                      ["Which folder are the files in, and what should be done with each file?",
                       "What kind of files, where are they, and what should processing mean (move, rename, read, convert)?"]),
    "update_sheet": (["update the sheet", "update the tracker", "fill in the sheet", "sheet update kar do", "keep the tracker updated"],
                     ["Which sheet (Excel or Google Sheets), and what should be written into it from where?",
                      "Where does the new data come from, and which sheet and columns should I update?"]),
    "clean_data": (["clean up the data", "clean the data", "data saaf karo", "tidy up the data"],
                   ["Which file or table holds the data, and what does clean mean here: remove duplicates, empty rows or bad values?",
                    "Where is the data (CSV, Excel, database), and what should be removed or fixed?"]),
    "usual": (["do the usual", "do the same as last week", "do what we always do on Fridays", "wahi karo jo har hafte karte hain",
               "repeat yesterday's task"],
              ["I don't have last time's steps. What should the bot do, step by step?",
               "Can you list the steps you usually do, and which files or apps they use?"]),
    "payroll": (["help with payroll", "automate payroll", "payroll stuff automate karo", "do the payroll run"],
                ["Which part of payroll: collecting timesheets, checking hours, or sending payslips? And where is the data?",
                 "What should the payroll bot do, and which files or system does it use?"]),
    "onboarding": (["automate onboarding", "help with new joiners", "do the onboarding tasks", "new joiner process automate karo"],
                   ["Which onboarding steps should the bot do (accounts, emails, documents), and where is the list of new joiners?",
                    "Where do new joiner details come from, and what should happen for each person?"]),
    "reconcile": (["reconcile it", "do the reconciliation", "run the recon", "match the payments", "recon kar do"],
                  ["Which two sources should be matched (for example the bank statement and the ledger), and what should happen to mismatches?",
                   "What should be reconciled against what, and where should the differences go?"]),
    "report_numbers": (["update the numbers", "refresh the numbers", "get me the numbers", "numbers nikal do"],
                       ["Which numbers, from which file or system, and where should they go?",
                        "Which report or sheet do the numbers come from, and what should I do with them?"]),
    "export": (["export it", "export the data", "download the report", "export kar do"],
               ["What should be exported (which file, sheet or system), and to which format: CSV, Excel or PDF?",
                "Where is the data, and where should the export be saved?"]),
    "share": (["share the file with the team", "share it", "send the report to everyone", "team ko file share karo"],
              ["Which file, and how should it reach the team: email, Teams, Slack or a shared folder?",
               "Which file should I share, and with which people or channel?"]),
    "archive": (["archive the old stuff", "clean up old files", "archive everything old", "purani files archive karo"],
                ["Which folder, how old counts as old, and should the files be moved, zipped or deleted?",
                 "Where are the old files, and where should they be archived?"]),
    "portal": (["check the portal", "check the website every day", "portal check karo daily", "look at the site"],
               ["Which website, and what should I check or download there?",
                "What should the bot look for on the portal, and what should happen when it finds it?"]),
    "orders": (["process the orders", "handle new orders", "orders automate karo", "do the order entry"],
               ["Where do the orders come from (email, a file or a website), and where should they be entered?",
                "What should happen to each order, and in which system?"]),
    "reminders": (["send the reminders", "remind them", "send reminder emails", "reminder bhej do sabko"],
                  ["Who should get the reminder, where is their list, and what should it say?",
                   "Which list of people, and what is the reminder about?"]),
    "move_them": (["move them", "move the files", "shift these", "inko move karo"],
                  ["Which files should I move, and from which folder to which folder?",
                   "Which folder are the files in, and where should they go?"]),
    "boring": (["automate the boring parts of my job", "make my work easier", "automate something useful", "kuch automate kar do",
                "build me a bot"],
               ["Which task do you do by hand most often? Describe its steps and the files or apps it uses.",
                "What is the task you'd like automated, step by step?"]),
    "approve": (["do the approvals", "handle approvals", "approve the requests", "approvals ka process automate karo"],
                ["Which approvals (expenses, invoices, leave), where do the requests arrive, and what decides approve or reject?",
                 "Where are the requests and what rule should decide whether each is approved?"]),
    "data_entry": (["do the data entry", "enter the data", "type all this into the system", "data entry automate karo"],
                   ["Where does the data come from, and which application or sheet should it be entered into?",
                    "Which source file and which target system should the data entry use?"]),
    "convert": (["convert them", "convert the files", "convert it", "convert kar do"],
                ["Which files, and to what format (PDF, Excel, CSV)?", "What should be converted, and into which format?"]),
    "ticket": (["raise the tickets", "log the issues", "create tickets for these", "tickets bana do"],
               ["Which system should the tickets go into (ServiceNow, Jira, Zendesk), and where do the issues come from?",
                "Where are the issues listed, and in which ticketing tool should I create them?"]),
    "check_stock": (["check the stock", "check inventory", "stock check karo", "do the stock thing"],
                    ["Where is the stock data (Excel, a database, a website), and what should happen when something is low?",
                     "Which inventory file or system, and what should the bot report?"]),
    "notify": (["let everyone know", "tell the team", "notify them when it's done", "sabko bata do"],
               ["Who should be told, about what, and by email, Teams or Slack?", "What should the message say, and who should get it?"]),
    "vendor": (["sort out the vendor stuff", "handle vendors", "vendor ka kaam automate karo", "do vendor management"],
               ["Which vendor task: onboarding, payments, statements or queries? And where is the vendor data?",
                "What should happen for each vendor, and which file or system holds them?"]),
}
FRAMES = ["{r}", "{r}", "{r} pls", "can you {r}", "{r}?", "i want to {r}", "hey {r}", "{r} asap", "need to {r}",
          "{r} for me", "please {r}", "{r} thanks", "Could you {r}?", "I need a flow to {r}"]


def vague_rows(n, rng, guard):
    rows, seen = [], set()
    intents = list(VAGUE)
    tries = 0
    while len(rows) < n and tries < n * 50:
        tries += 1
        k = rng.choice(intents)
        reqs, qs = VAGUE[k]
        r = rng.choice(reqs)
        if not any(w in r for w in (" karo", " kar do", " bana do", " bhej do", " theek", " nikal do", " bata do", " saaf", " jo har")):
            r = rng.choice(FRAMES).format(r=r)
        if rng.random() < 0.1:
            r = typo(r, rng, 1)
        if rng.random() < 0.3:
            r = r[0].upper() + r[1:]
        if r.lower() in seen or guard.leaks(r):
            continue
        seen.add(r.lower())
        rows.append({"kind": "clarify", "request": r, "question": rng.choice(qs),
                     "source": f"clarify:{k}"})
    return rows


# short but clear: (wordings, builder)
def c_merge(B, rng):
    b = B()
    s = b.s("ooa.pdf/merge", "Merge the PDFs into one file", {"folder": b.inp("folder", "Folder with the PDFs."), "pattern": "*.pdf",
                                                              "target": b.inp("merged_path", "The merged PDF.")})
    return rng.choice(["merge these PDFs", "merge pdfs", "combine the pdfs in a folder", "join all PDFs into one",
                       "pdfs merge kar do", "merge all the pdf files"]), b.flow("Merge PDFs", [s])


def c_list(B, rng):
    b = B()
    s = b.s("ooa.file/list", "List the files in the folder", {"folder": b.inp("folder", "The folder.")}, {"files": "files", "count": "count"})
    b.output("files")
    return rng.choice(["list files in a folder", "show me what's in a folder", "get all file names in a folder", "folder ki files list karo",
                       "list everything in my downloads folder"]), b.flow("List files", [s], outputs=["files"])


def c_zip(B, rng):
    b = B()
    f = rng.choice(["reports", "photos", "downloads", "invoices", "project"])
    s = b.s("ooa.file/zip", f"Zip the {f} folder", {"source": b.inp("folder", f"The {f} folder."), "target": b.inp("zip_path", "The zip to create.")},
            {"path": "zip_file"})
    return rng.choice([f"zip the {f} folder", f"zip {f}", f"compress my {f} folder", f"{f} folder ka zip bana do"]), b.flow("Zip folder", [s])


def c_unzip(B, rng):
    b = B()
    s = b.s("ooa.file/unzip", "Unzip the archive", {"archive": b.inp("zip_path", "The zip file."), "folder": b.inp("folder", "Where to extract.")},
            {"files": "files"})
    return rng.choice(["unzip this file", "extract a zip", "unzip it to a folder", "zip khol do"]), b.flow("Unzip", [s])


def c_word_pdf(B, rng):
    b = B()
    steps = [b.s("ooa.word/open", "Open the document", {"path": b.inp("document_path", "The Word document.")}),
             b.s("ooa.word/save_as", "Save it as PDF", {"path": b.inp("pdf_path", "The PDF to create."), "format": "pdf"}),
             b.s("ooa.word/close", "Close the document", {})]
    return rng.choice(["convert this word doc to pdf", "docx to pdf", "make a pdf of my word file", "word ko pdf bana do"]), b.flow("Word to PDF", steps)


def c_excel_pdf(B, rng):
    b = B()
    s = b.s("ooa.excel_desktop/export_pdf", "Export the workbook to PDF", {"path": b.inp("workbook_path", "The Excel workbook."),
                                                                          "target": b.inp("pdf_path", "The PDF to create.")})
    return rng.choice(["excel to pdf", "export my spreadsheet as pdf", "convert the xlsx to a PDF", "excel ko pdf me convert karo"]), b.flow("Excel to PDF", [s])


def c_ocr(B, rng):
    b = B()
    steps = [b.s("ooa.pdf/read_text_ocr", "Read the scanned PDF with OCR", {"path": b.inp("pdf_path", "The scanned PDF.")}, {"text": "text"}),
             b.s("ooa.file/write_text", "Save the text", {"path": b.inp("text_path", "The text file to write."), "text": "$text"})]
    return rng.choice(["ocr this scanned pdf into a text file", "read a scanned pdf and save the text", "scan wali pdf ka text nikal ke save karo"]), \
        b.flow("OCR PDF", steps)


def c_summary(B, rng):
    b = B()
    steps = [b.s("ooa.pdf/extract_text", "Get the text of the PDF", {"path": b.inp("pdf_path", "The PDF.")}, {"text": "text"}),
             b.s("ooa.genai/summarize", "Summarise it", {"text": "$text", "sentences": 5}, {"summary": "summary"})]
    b.output("summary")
    return rng.choice(["summarize this pdf", "give me a summary of a PDF", "pdf ka summary bana do", "short summary of the report pdf"]), \
        b.flow("Summarise PDF", steps, outputs=["summary"])


def c_translate(B, rng):
    b = B()
    lang = rng.choice(["French", "German", "Hindi", "Spanish", "Japanese"])
    steps = [b.s("ooa.file/read_text", "Read the text file", {"path": b.inp("file_path", "The text file.")}, {"text": "text"}),
             b.s("ooa.genai/translate", f"Translate it to {lang}", {"text": "$text", "target_language": lang}, {"text": "translated"}),
             b.s("ooa.file/write_text", "Save the translation", {"path": b.inp("output_path", "The translated file."), "text": "$translated"})]
    return rng.choice([f"translate a text file to {lang}", f"translate this txt into {lang} and save it", f"{lang} me translate karke save karo"]), \
        b.flow("Translate file", steps)


def c_folder(B, rng):
    b = B()
    name = rng.choice(["Archive", "2026", "Invoices", "Clients", "Backup", "Q4"])
    s = b.s("ooa.folder/create", f"Create the {name} folder", {"path": "${parent_folder}/" + name})
    b.inp("parent_folder", "Where to create the folder.")
    return rng.choice([f"create a folder named {name}", f"new folder {name}", f"make a {name} folder", f"{name} naam ka folder bana do"]), \
        b.flow("Create folder", [s])


def c_copy_folder(B, rng):
    b = B()
    s = b.s("ooa.folder/copy", "Copy the folder", {"source": b.inp("source_folder", "The folder to copy."), "target": b.inp("backup_folder", "Where to copy it.")},
            {"copied": "copied"})
    return rng.choice(["back up a folder", "copy my project folder to the backup drive", "copy a folder somewhere else", "folder ka backup le lo"]), \
        b.flow("Copy folder", [s])


def c_screenshot(B, rng):
    b = B()
    steps = [b.s("ooa.browser/open", "Open the website", {"url": b.inp("url", "Address of the website.")}),
             b.s("ooa.browser/screenshot", "Take a screenshot", {"path": b.inp("screenshot_path", "Where to save it.")}, {"path": "shot"}),
             b.s("ooa.browser/close", "Close the browser", {})]
    return rng.choice(["screenshot a web page", "grab a screenshot of a site", "save a picture of a webpage", "website ka screenshot lo"]), \
        b.flow("Screenshot", steps)


def c_download(B, rng):
    b = B()
    s = b.s("ooa.browser/download_url", "Download the file", {"url": b.inp("file_url", "Address of the file."), "save_to": b.inp("save_path", "Where to save it.")},
            {"path": "path"})
    return rng.choice(["download a file from a link", "download this url", "save the file at this link", "link se file download karo"]), \
        b.flow("Download URL", [s])


def c_send(B, rng):
    b = B()
    s = b.s("ooa.email/send", "Send the email", {"to": b.inp("to", "Recipient email."), "subject": b.inp("subject", "Subject."),
                                                  "body": b.inp("body", "Message.")})
    return rng.choice(["send an email", "email someone a message", "send a quick mail", "ek email bhejo"]), b.flow("Send email", [s])


def c_csv_excel(B, rng):
    b = B()
    steps = [b.s("ooa.csv/read", "Read the CSV", {"path": b.inp("csv_path", "The CSV file.")}, {"table": "table"}),
             b.s("ooa.datatable/write_to_file", "Save it as Excel", {"table": "$table", "path": b.inp("xlsx_path", "The Excel file."), "format": "xlsx"},
                 {"path": "saved"})]
    return rng.choice(["convert a csv to excel", "csv to xlsx", "turn my csv into an excel file", "csv ko excel bana do"]), b.flow("CSV to Excel", steps)


def c_sheets(B, rng):
    b = B()
    steps = [b.s("ooa.excel/open", "Open the workbook", {"path": b.inp("workbook_path", "The Excel workbook.")}),
             b.s("ooa.excel/get_sheet_names", "Get the sheet names", {}, {"names": "names"}),
             b.s("ooa.excel/close", "Close the workbook", {})]
    return rng.choice(["list the sheets in a workbook", "what tabs does this excel have", "get the sheet names of an xlsx"]), \
        b.flow("Sheet names", steps, outputs=["names"])


def c_json(B, rng):
    b = B()
    steps = [b.s("ooa.json/read_file", "Read the JSON file", {"path": b.inp("json_path", "The JSON file.")}, {"value": "value"}),
             b.s("ooa.json/to_table", "Turn it into a table", {"value": "$value"}, {"table": "table"}),
             b.s("ooa.csv/write", "Save it as CSV", {"table": "$table", "path": b.inp("csv_path", "The CSV file.")})]
    return rng.choice(["json to csv", "convert a json file to csv", "turn this json into a csv"]), b.flow("JSON to CSV", steps)


def c_slack(B, rng):
    b = B()
    s = b.s("ooa.slack/post_message", "Post the message", {"bot_token": b.inp("slack_token", "Slack bot token.", "Credential"),
                                                            "channel": b.inp("channel", "Slack channel."), "text": b.inp("message", "Message text.")})
    return rng.choice(["post a message in slack", "send a slack message to a channel", "slack pe message daal do"]), b.flow("Slack message", [s])


def c_ping(B, rng):
    b = B()
    steps = [b.s("ooa.network/ping", "Ping the server", {"host": b.inp("host", "Server name or address.")}, {"reachable": "reachable"}),
             b.log("Reachable: ${reachable}", "Log the result")]
    return rng.choice(["check if a server is up", "ping our server", "is the server reachable? check it", "server ping karo"]), b.flow("Ping", steps)


def c_print(B, rng):
    b = B()
    s = b.s("ooa.printer/print_file", "Print the file", {"path": b.inp("file_path", "The file to print.")})
    return rng.choice(["print a pdf", "print this file", "send a document to the printer", "file print kar do"]), b.flow("Print file", [s])


def c_rename(B, rng):
    b = B()
    s = b.s("ooa.file/move", "Rename the file", {"source": b.inp("file_path", "The file."), "target": b.inp("new_name", "The new name.")})
    return rng.choice(["rename a file", "rename this file", "change a file's name", "file ka naam badlo"]), b.flow("Rename file", [s])


def c_dedupe(B, rng):
    b = B()
    steps = [b.s("ooa.csv/read", "Read the CSV", {"path": b.inp("csv_path", "The CSV file.")}, {"table": "table"}),
             b.s("ooa.datatable/remove_duplicates", "Remove duplicate rows", {"table": "$table"}, {"table": "clean"}),
             b.s("ooa.csv/write", "Write the clean CSV back", {"table": "$clean", "path": "$csv_path"})]
    return rng.choice(["dedupe a csv in place", "remove duplicates from my csv and overwrite it", "csv se duplicates hata do"]), \
        b.flow("Dedupe CSV", steps)


CLEAR = {k[2:]: (f, 1) for k, f in list(globals().items()) if k.startswith("c_") and callable(f)}


def main():
    cat_path, schema, out = sys.argv[1], sys.argv[2], sys.argv[3]
    n_vague = int(sys.argv[4]) if len(sys.argv) > 4 else 600
    n_clear = int(sys.argv[5]) if len(sys.argv) > 5 else 300
    rng = random.Random(int(sys.argv[6]) if len(sys.argv) > 6 else 31)
    checker, guard = Checker(cat_path, schema), LeakGuard()
    vague = vague_rows(n_vague, rng, guard)

    def styled(fn):
        def run(B, r):
            req, flow = fn(B, r)
            if r.random() < 0.5:
                req = r.choice(FRAMES).format(r=req)
            if r.random() < 0.1:
                req = typo(req, r, 1)
            if r.random() < 0.3:
                req = req[0].upper() + req[1:]
            return req, flow
        return run
    clear, made, why = run_families({k: (styled(f), w) for k, (f, w) in CLEAR.items()}, n_clear, rng, checker, guard, "clarify_counter")
    write_jsonl(out, vague + clear)
    print(f"{len(vague)} clarify records over {len(VAGUE)} intents | {len(clear)} clear short flows over {len(made)} intents"
          f" | checker pass {checker.ok}/{checker.n} | test-close dropped {guard.dropped} | skipped {dict(why)}")


if __name__ == "__main__":
    main()
