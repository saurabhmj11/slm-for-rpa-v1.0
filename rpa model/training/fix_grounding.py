"""Deterministic grounding fix for the neutral SFT records in data/extra_sft/ (see grounding_check.py).

    python fix_grounding.py FILE.jsonl ... [--check] [--verbose]

Rewrites each file in place (idempotent: a fixed file comes out unchanged) and DROPS the records it cannot make
faithful cleanly (a made-up Salesforce field, a guessed reply field, a text that still names a made-up detail or
reads badly after the rewrite), printing how many per file and why.

  - a detail the request does not give (a saved connection, an address, a SharePoint site, an AD distinguished name,
    a team / channel / group / sheet / table / procedure / warehouse / mailbox / project / printer name, a database
    type, a currency, a link, a path or the folder of a path) becomes an input variable with a clear name, a
    description and no default. A path whose file name the request gives (or a new file the flow writes) but not
    its folder keeps the name and gets a folder variable ("${download_folder}\\catalog.zip").
  - every copy is replaced the same way: the step that writes a path and every step that reads it (an unzip of the
    downloaded archive, a sharing link of the uploaded item, an attachment of the written report), a DN inside an
    LDAP filter, a path inside a message or a ticket description ("${backup_file} was not found.").
  - a file type filter the request does not ask for ('*.csv' for "the exports", extension 'pdf' for "every file")
    is dropped ('*' / no filter); a made-up company name or link in a subject / body goes; template text
    ("2 copy/copies", "hour(s)") is written out.
  - the record's texts (clear, understand, steps lines, flow / step descriptions) say "the saved Jira connection",
    "the download folder", "the workbook", "the printer" ... instead of the made-up value, and nothing the request
    says is rewritten; facts the request does not give are dropped. A step-builder part's understand stays its
    request.
  - a bare $row.Name inside other text (sent literally by the runner) is written ${row.Name}.
  - clarify records without a "missing" field get the slots their question asks for (MISSING_BY_REQUEST).
--check only reports what would change. Standard library only."""
import argparse, collections, copy, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ooa_format as F
import grounding_check as G

_REF = G._REF

APP_NAMES = {"jira": "Jira", "zendesk": "Zendesk", "google": "Google", "servicenow": "ServiceNow",
             "salesforce": "Salesforce", "microsoft365": "Microsoft 365", "slack": "Slack"}

# key -> (variable name, words used in the texts, description)
NAMES = {
    "drive": ("sharepoint_site", "the SharePoint site", "SharePoint site (host:/sites/Name) to upload to."),
    "site": ("sharepoint_site", "the SharePoint site", "SharePoint site (host:/sites/Name)."),
    "team_name": ("team_name", "the team", "Microsoft Teams team to post in."),
    "channel_name": ("channel_name", "the channel", "Channel of the team to post in."),
    "channel": ("channel", "the channel", "Channel to post in."),
    "group": ("group_name", "the group", "Group to add the users to."),
    "groups": ("group_name", "the group", "Group to add the users to."),
    "assignment_group": ("assignment_group", "the assignment group", "Assignment group of the new record."),
    "warehouse": ("warehouse", "the warehouse", "Warehouse that runs the query."),
    "procedure": ("procedure_name", "the procedure", "Stored procedure to call."),
    "table_name": ("table_name", "the table", "Table to load into."),
    "sheet": ("sheet_name", "the sheet", "Sheet (tab) of the workbook."),
    "sheet_name": ("sheet_name", "the sheet", "Sheet (tab) of the workbook."),
    "mailbox": ("shared_mailbox", "the shared mailbox", "Shared mailbox to read."),
    "project": ("project", "the project", "Project to create the items in."),
    "owner": ("repo_owner", "the repository owner", "Owner (user or organisation) of the repository."),
    "url": ("api_url", "the API address", "Address (URL) to call."),
    "warehouse_id": ("warehouse_id", "the warehouse", "Warehouse id."),
    "pattern": ("file_pattern", "the matching", "Name pattern of the files (for example *.csv)."),
    "from_contains": ("sender_email", "the sender", "Sender address of the emails."),
    "printer": ("printer_name", "the printer", "Printer to print on."),
    "database_type": ("database_type", "the database", "Kind of database (for example sqlserver or postgresql)."),
    "from_currency": ("from_currency", "the source currency", "Currency code the amounts are in (for example USD)."),
    "to_currency": ("to_currency", "the target currency", "Currency code to convert to (for example EUR)."),
    "to_folder": ("mail_folder", "the mail folder", "Mail folder to move the emails to."),
    "folder_id": ("folder_id", "the folder", "Folder id."),
}
# AD distinguished names: key -> (variable, description head)
DN_NAMES = {"target_ou_dn": ("target_ou", "Distinguished name (DN) of the OU to move the user to"),
            "base_dn": ("base_ou", "Distinguished name (DN) of the OU the users are created in"),
            "group_dn": ("group_dn", "Distinguished name (DN) of the group"),
            "filter": ("group_dn", "Distinguished name (DN) of the group"),
            "ou": ("target_ou", "Distinguished name (DN) of the OU"), "ou_dn": ("target_ou", "Distinguished name of the OU"),
            "dn": ("user_dn", "Distinguished name (DN) of the user"),
            "user_dn": ("user_dn", "Distinguished name (DN) of the user")}
EMAIL_KEYS = {"to", "cc", "bcc", "attendees", "recipients", "recipient", "email", "emails", "mailbox", "reply_to"}
PATH_KEYS = {"path", "paths", "local_path", "remote_path", "target", "source", "attachments", "file", "files",
             "archive", "index_file", "item", "public_key_file", "private_key_file", "key_file", "save_to",
             "report_path", "documents", "target_path", "output_path", "template", "template_path", "workbook",
             "new_path", "output", "dest", "destination"}
FOLDER_KEYS = {"folder", "target_folder", "local_folder", "remote_folder", "source_folder", "output_folder",
               "save_to"}

# what a file is called in a text, by its extension
NOUNS = {"xlsx": "workbook", "xls": "workbook", "xlsm": "workbook", "csv": "CSV file", "pdf": "PDF",
         "txt": "text file", "log": "log file", "docx": "document", "doc": "document", "zip": "zip file",
         "json": "JSON file", "xml": "XML file", "idx": "index file", "py": "script", "ps1": "script",
         "asc": "key file", "pptx": "presentation", "jpg": "image", "png": "image", "bak": "backup file",
         "pgp": "encrypted file"}

# clarify records without "missing": the slots their question asks for (ooa_format.MISSING_WORDS / ASK_SLOTS)
MISSING_BY_REQUEST = {
    "update the accounts in salesforce": ["records", "details"],
    "sign in to salesforce and pull the report": ["data", "place"],
    "salesforce wale leads ko assign kar do": ["records", "recipient"],
    "connect to sf and close the old ones": ["records", "condition"],
    "upload the files to the customer in salesforce": ["files", "records"],
    "raise tickets in servicenow for these": ["records", "details"],
    "servicenow login karke update kar do": ["records", "details"],
    "close the servicenow stuff": ["records", "condition"],
    "send me the servicenow list every day": ["list", "place"],
    "get the workday report": ["report", "place"],
    "load the file into workday": ["file", "details"],
    "workday se data chahiye": ["data", "place"],
    "fix the okta accounts": ["details", "records"],
    "add them to the group in okta": ["records", "details"],
    "okta mein naye log daal do": ["list", "details"],
    "offboard him": ["records", "details"],
    "run the databricks job": ["details", "recipient"],
    "query databricks and send it": ["data", "recipient"],
    "export from snowflake": ["data", "place"],
    "load the files to snowflake": ["files", "details"],
    "snowflake mein data update kar do": ["records", "details"],
    "send these to kafka": ["data", "details"],
    "read the kafka messages": ["details", "place"],
    "create the jira tickets": ["list", "details"],
    "move my jira tickets": ["records", "details"],
    "jira wale tickets ka report bhejo": ["records", "recipient"],
    "use the saved jira connection and comment": ["records", "details"],
    "clean up zendesk": ["details", "condition"],
    "make tickets from the emails": ["folder", "condition"],
    "zendesk tickets pe tag laga do": ["records", "details"],
    "sign in to office 365 and send the file": ["file", "recipient"],
    "update the excel on onedrive": ["file", "sheet"],
    "post it on teams": ["details", "place"],
    "move the mails": ["records", "folder"],
    "onedrive excel mein entry kar do": ["file", "details"],
    "disable the user in microsoft": ["records", "details"],
    "add the tasks to planner": ["list", "place"],
    "go through the teams chat": ["details", "condition"],
    "upload the files to the server": ["files", "place"],
    "get the files from the ftp": ["files", "place"],
    "sftp pe purani files hata do": ["folder", "condition"],
    "open the csv and do the needful": ["file", "what"],
    "clean the excel": ["file", "what"],
    "update the google sheet": ["sheet", "details"],
    "save the attachments from the shared mailbox": ["folder", "place"],
    "outlook se mails archive kar do": ["folder", "place"],
    "run my python for each row": ["file", "data"],
    "watch the folder": ["folder", "what"],
    "keep checking the program": ["details", "recipient"],
    "tell me when a new incident comes": ["app", "place"],
}

_BARE = re.compile(r"(?<![\w{$])\$([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)(?!\$)")
_FILE_EXT = {"txt", "csv", "pdf", "xlsx", "xls", "docx", "doc", "json", "xml", "zip", "html", "png", "jpg", "msg", "log"}


# a text noun right before a made-up value already names it ('the policy index D:\\KB\\x.idx')
TEXT_NOUNS = ("index", "file", "workbook", "folder", "sheet", "document", "report", "template", "script", "key",
              "list", "log", "archive", "PDF", "mailbox", "printer", "site", "warehouse", "procedure", "table",
              "project", "channel", "team", "group")


class Unfixable(Exception):
    """A record the fix cannot make faithful cleanly: it is dropped."""


def braced_refs(value):
    """'D:\\out\\$row.Name.txt' -> 'D:\\out\\${row.Name}.txt': a bare dotted reference inside other text is sent
    literally by the runner; ${...} inserts the value. A last part that is a file extension is not a field."""
    if not isinstance(value, str) or "$" not in value or G._FULL_REF.fullmatch(value.strip()):
        return value

    def rep(m):
        parts = m.group(1).split(".")
        tail = ""
        if len(parts) > 2 and parts[-1].lower() in _FILE_EXT:
            tail = "." + parts.pop()
        return "${" + ".".join(parts) + "}" + tail
    keep = G._INTERP_OK.split(value)
    seps = G._INTERP_OK.findall(value)
    out = _BARE.sub(rep, keep[0])
    for sep, rest in zip(seps, keep[1:]):
        out += sep + _BARE.sub(rep, rest)
    return out


def plural_text(text):
    """Template text: '2 copy/copies' -> '2 copies', '1 hour(s)' -> '1 hour', 'ticket(s)' -> 'tickets'."""
    if not isinstance(text, str):
        return text

    def slash(m):
        n, a, b = m.group(1), m.group(2), m.group(3)
        if b.lower() not in (a.lower() + "s", a.lower() + "es", a.lower()[:-1] + "ies"):
            return m.group(0)
        return f"{n} {a if n == '1' else b}"
    t = re.sub(r"\b(\d+) (\w+)/(\w+)\b", slash, text)
    t = re.sub(r"\b(\d+) (\w+)\(s\)", lambda m: f"{m.group(1)} {m.group(2)}" + ("" if m.group(1) == "1" else "s"), t)
    return re.sub(r"\b(\w+)\(s\)", r"\1s", t)


def _ext(name):
    m = re.search(r"\.([A-Za-z0-9]{2,5})$", name or "")
    return m.group(1).lower() if m else ""


def folder_role(action, key):
    """(variable name, words) for the folder of a path a step uses."""
    a = action or ""
    if key in ("remote_path", "remote_folder") or a.startswith("ooa.microsoft365/") and key == "item":
        return "remote_folder", "the remote folder"
    if a.endswith("/download_files") and key == "folder" or a.endswith("/delete_files") or \
            a.startswith(("ooa.file_transfer/", "ooa.sftp/", "ooa.ftp/")) and key == "folder":
        return "remote_folder", "the remote folder"
    if re.search(r"/(?:download_file|download|drive_download|download_files)$", a):
        return "download_folder", "the download folder"
    if a.endswith("/unzip") and key == "folder":
        return "extract_folder", "the extract folder"
    if key == "archive":
        return "download_folder", "the download folder"
    if re.search(r"save_attachments|save_mail_attachments|save_all", a):
        return "attachments_folder", "the attachments folder"
    if G.is_producer(a, key) or key in ("target", "target_folder", "output_path", "output_folder"):
        return "output_folder", "the output folder"
    if key in ("source", "source_folder") or a.startswith("ooa.file/list") or a.startswith("core/loop"):
        return "source_folder", "the source folder"
    return "input_folder", "the input folder"


def file_role(action, key, value):
    """(variable name, words, description head) for a whole made-up file path a step reads."""
    ext = _ext(G.path_parts(value)[1])
    noun = NOUNS.get(ext, "file")
    if key == "index_file":
        return "index_file", "the index file", "Index file of the documents"
    if key in ("public_key_file", "private_key_file", "key_file"):
        return "key_file", "the key file", "Key file"
    if key == "file" and ext in ("py", "ps1", "bat", "vbs", "js"):
        return "script_file", "the script", "Script file"
    base = {"workbook": "workbook_path", "CSV file": "csv_file", "PDF": "pdf_file", "text file": "text_file",
            "document": "document_path", "zip file": "zip_file", "image": "image_file",
            "backup file": "backup_file"}.get(noun, "file_path")
    if key in ("local_path",) and noun == "file":
        base = "local_file"
    return base, f"the {noun}", noun[0].upper() + noun[1:]


class Fixer:
    def __init__(self, rec):
        self.rec = rec
        self.flow = rec.get("flow")
        self.hay = G.haystack(rec)
        self.request = rec.get("request") or ""
        self.vars = {v["name"]: v for v in (self.flow or {}).get("variables") or [] if isinstance(v, dict)}
        self.exact = {}               # literal -> replacement text (a whole value)
        self.folders = {}             # folder literal -> (variable reference, words)
        self.text_subs = []           # (literal or pattern, words for the record's texts)
        self.drop_args = []           # (step, key) arguments to remove
        self.notes = []
        self.used_words = {}          # words -> literal, to keep two values from reading the same in a text

    # ------------------------------------------------------------------ variables
    def new_var(self, base, description):
        name, k = base, 2
        while name in self.vars and self.vars[name].get("description") != description:
            name, k = f"{base}_{k}", k + 1
        if name not in self.vars:
            v = {"name": name, "type": "String", "direction": "input", "required": True, "description": description}
            self.flow.setdefault("variables", [])
            ins = [i for i, x in enumerate(self.flow["variables"]) if x.get("direction") == "input"]
            self.flow["variables"].insert(ins[-1] + 1 if ins else 0, v)
            self.vars[name] = v
        return name

    def words_for(self, literal, words):
        """Two different made-up values must not read the same in one text: the second is not fixable cleanly."""
        old = self.used_words.get(words)
        if old is not None and old != literal:
            raise Unfixable(f"two values read as {words!r}")
        self.used_words[words] = literal
        return words

    def email_name(self, key, value):
        local = value.split("@")[0].lower()
        toks = [t for t in re.split(r"[._+-]", local) if t]
        hay = G.words_hay(self.hay)
        said = [t for t in toks if re.search(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", hay)]
        if said and len(said) == len(toks) or (said and len("".join(said)) >= 4):
            return "_".join(said) + "_email", f"the {' '.join(said)} address"
        if key == "mailbox":
            return "shared_mailbox", "the shared mailbox"
        role = {"attendees": "attendee", "cc": "cc", "from_contains": "sender", "sender": "sender",
                "from": "sender"}.get(key, "recipient")
        return f"{role}_email", f"the {role} address"

    # ------------------------------------------------------------------ deciding one literal
    def decide(self, step, key, value, producer):
        """Record how a made-up literal is replaced (self.exact / self.folders) and how the texts say it."""
        if value in self.exact:
            return
        action = step.get("action") or ""
        desc = (step.get("description") or "").rstrip(".")
        for_desc = f" (for: {desc})." if desc else "."
        if key == "connection":
            args = step.get("args") or {}
            it = args.get("iterator_args") if isinstance(args.get("iterator_args"), dict) else {}
            src = args.get("iterator") if isinstance(args.get("iterator"), str) else it.get("source")
            tk = (src if isinstance(src, str) and "/" in src else action).split("/")[0].split(".")[-1]
            if tk in ("core", "triggerloop"):
                tk = "app"
            app = APP_NAMES.get(tk, tk.capitalize())
            name = self.new_var(f"{tk}_connection", f"Saved {app} connection (under Connections).")
            self.exact[value] = "$" + name
            self.text_subs.append((value, self.words_for(value, f"the saved {app} connection")))
        elif G._EMAIL.fullmatch(value.strip()):
            base, words = self.email_name(key, value.strip())
            name = self.new_var(base, f"Email address{for_desc}" if desc else "Email address.")
            self.exact[value] = "$" + name
            self.text_subs.append((value, self.words_for(value, words)))
        elif G._DN_PART.search(value):
            self.decide_dn(key, value)
        elif key == "extension" or key in ("pattern", "name_pattern") and re.fullmatch(r"\*\.\w+", value.strip()):
            ext = value.strip().lstrip("*.").lower()
            self.exact[value] = None if key == "extension" else "*"
            self.text_subs.append((re.compile(r"\*\." + re.escape(ext) + r"\s+(?=files?\b)", re.I), ""))
            self.text_subs.append((re.compile(r"\b(?:the )?" + re.escape(ext) + r" (files?|attachments?)\b", re.I),
                                   r"the \1"))
            self.text_subs.append((re.compile(r"\*\." + re.escape(ext) + r"\b", re.I), "the files"))
        elif key == "range":
            self.exact[value] = value.split("!", 1)[1]
        elif key == "location":
            self.exact[value] = None
            self.text_subs.append((re.compile(r",? (?:in|at|on) " + re.escape(value) + r"\b(?! ?[\\/])"), ""))
        elif key in ("from_currency", "to_currency"):
            base, words, d = NAMES[key]
            name = self.new_var(base, d)
            self.exact[value] = "$" + name
            self.text_subs.append((re.compile(r"\bfrom " + re.escape(value) + r"\b"), ""))
            self.text_subs.append((re.compile(r"\b" + re.escape(value) + r"\b"), self.words_for(value, words)))
        elif key in ("database_type",):
            base, words, d = NAMES[key]
            name = self.new_var(base, d)
            self.exact[value] = "$" + name
            for w in G.DB_WORDS.get(value.lower(), (value,)):
                self.text_subs.append((re.compile(r"\b" + re.escape(w) + r"\b", re.I), "the"))
        elif key in ("folder", "to_folder") and self.is_mail(action) and not G._is_abs(value):
            toks = self.said_tokens(value)
            base, words = ((f"{toks[-1]}_folder", f"the {toks[-1]} folder") if toks else
                           ("mail_folder", "the mail folder"))
            name = self.new_var(base, "Mail folder" + for_desc)
            self.exact[value] = "$" + name
            self.text_subs.append((value, self.words_for(value, words)))
        elif self.is_path(key, value):
            self.decide_path(step, key, value, producer, for_desc)
        else:
            base, words, d = NAMES.get(key, (key, f"the {key.replace('_', ' ')}", f"{key.replace('_', ' ')}."))
            m = re.search(r"/sites/([\w-]+)", value)
            site_said = bool(m) and G._word_in(m.group(1), G.words_hay(self.hay))
            if key in ("site", "drive") and not site_said:
                self.text_subs += self.site_aliases(value)
            if site_said:
                d = f"The {m.group(1)} SharePoint site (host:/sites/{m.group(1)})."
            if key == "to_folder":
                toks = self.said_tokens(value)
                if toks:
                    base, words = f"{toks[-1]}_folder", f"the {toks[-1]} folder"
            if key == "url":
                mm = re.search(r"\b(?:the |our )?([\w-]+) API\b", self.request, re.I)
                if mm and mm.group(1).lower() not in ("the", "our", "an", "a", "rest", "json"):
                    words = f"the {mm.group(1)} API"
            name = self.new_var(base, d if not desc else d.rstrip(".") + for_desc)
            self.exact[value] = "$" + name
            self.text_subs.append((value, self.words_for(value, words)))
        self.notes.append(f"{key}: {value!r} -> {self.exact.get(value, self.folders.get(value))!r}")

    @staticmethod
    def is_mail(action):
        return action.startswith(("ooa.email/", "ooa.outlook_desktop/", "ooa.outlook/")) or \
            action.startswith("ooa.microsoft365/") and "mail" in action

    @staticmethod
    def is_path(key, value):
        if key in NAMES:
            return False
        return key in PATH_KEYS or bool(re.search(r"[\\/]", G._REF.sub("", value))) and key not in G.DN_KEYS

    def decide_dn(self, key, value):
        parts = G._DN_PART.findall(value)
        hay = G.words_hay(self.hay)
        if key == "filter" or not any(k for k in DN_NAMES if k == key):
            # an LDAP filter: each DN inside it becomes a variable
            for dn in re.findall(r"(?:CN|OU)=[^()]*?DC=[^()]+?(?=\)|$)", value):
                self.decide_dn("group_dn", dn)
            return
        base, head = DN_NAMES[key]
        first = parts[0][1].strip() if parts else ""
        said_first = first and G._word_in(first, hay)
        if said_first:
            slug = re.sub(r"[^a-z0-9]+", "_", first.lower()).strip("_")
            kind = "group" if parts[0][0].upper() == "CN" else "OU"
            base = f"{slug}_{'group_dn' if kind == 'group' else 'ou'}"
            d = f"{head} '{first}' (for example CN={first},OU=...,DC=...)." if kind == "group" else \
                f"{head} '{first}' (for example OU={first},DC=...)."
        else:
            d = head + "."
        name = self.new_var(base, d)
        self.exact[value] = "$" + name

    def decide_path(self, step, key, value, producer, for_desc):
        action = step.get("action") or ""
        folder, last = G.path_parts(value)
        kind = G.judge(value, self.hay, producer=producer)
        is_folder_value = key in FOLDER_KEYS or (not G._EXT.search(last) and key not in ("item", "report_path"))
        lit = folder.rstrip("\\/")
        if key == "report_path":
            new_folder = folder
            if lit and not G._REF.fullmatch(lit) and "${" not in lit and not G._folder_said(lit, self.hay):
                if lit not in self.folders:
                    name = self.new_var("report_owner", "Owner of the custom report (the folder part of its path).")
                    self.folders[lit] = ("${" + name + "}", "the report owner")
                    self.text_subs.append((lit + "/", ""))
                new_folder = self.folders[lit][0] + folder[len(lit):]
            new_last = last
            if not G._word_in(last, G.words_hay(self.hay)):
                name = self.new_var("report_name", "Name of the Workday custom report to run.")
                new_last = "${" + name + "}"
                e = re.escape(last)
                self.text_subs += [(re.compile(r"\b(?:the )?(?:custom )?report " + e + r"\b"), "the custom report"),
                                   (re.compile(r"\b(Run|run) " + e + r"\b"), r"\1 the custom report"),
                                   (re.compile(r"\b" + e + r"\b"), "the report")]
            self.exact[value] = new_folder + new_last
            return
        if is_folder_value:
            ref, words = self.folder_var(action, key, value.rstrip("\\/"), for_desc)
            self.exact[value] = ref
            return
        if kind == "folder" and lit and not G._REF.fullmatch(lit):
            ref, words = self.folder_var(action, key, lit, for_desc)
            self.exact[value] = ref + folder[len(lit):] + last
            return
        if kind == "folder" and G._REF.fullmatch(lit or "-"):
            raise Unfixable(f"made-up name under a variable folder: {value!r}")
        # a whole made-up file the step reads (its name is not said either)
        if G._REF.search(value):
            raise Unfixable(f"made-up name inside a variable path: {value!r}")
        base, words, head = file_role(action, key, value)
        toks = (self.said_tokens(re.split(r"[\\/]", lit)[-1]) or
                self.said_tokens(re.sub(r"\.[A-Za-z0-9]{2,5}$", "", last)))[:1]
        if toks and words.startswith("the "):
            if words.split()[-1].lower() not in toks:
                words = "the " + " ".join(toks) + " " + words[4:]
            else:
                words = "the " + " ".join(toks)
            base = "_".join(toks) + ("_file" if not base.endswith(("_path", "_file")) else "_" + base.split("_")[-1])
        name = self.new_var(base, f"{head}{for_desc}")
        self.exact[value] = "$" + name
        self.text_subs.append((value, self.words_for(value, words)))
        if len(last) >= 5 and not G.grounded(last, self.hay):
            self.text_subs.append((last, words))
        if lit and not G._folder_said(lit, self.hay) and lit not in self.folders:
            self.text_subs.append((lit, "the folder"))

    def folder_var(self, action, key, lit, for_desc):
        if lit in self.folders:
            return self.folders[lit]
        if G._folder_said(lit, self.hay):
            return lit, lit
        base, words = folder_role(action, key)
        toks = [t for t in self.said_tokens(re.split(r"[\\/]", lit)[-1])
                if re.search(r"(?<![a-z])" + re.escape(t) + r"s? folder", G.words_hay(self.hay))][:1]
        if toks:
            base, words = "_".join(toks) + "_folder", "the " + " ".join(toks) + " folder"
        fdesc = {"remote_folder": "Remote folder", "download_folder": "Folder to download into",
                 "extract_folder": "Folder to unzip into", "output_folder": "Folder to save the output in",
                 "attachments_folder": "Folder to save the attachments in",
                 "source_folder": "Folder to read the files from",
                 "input_folder": "Folder of the input file"}.get(base, "Folder")
        if toks:
            fdesc = f"Folder of the {' '.join(toks)}"
        elif base == "remote_folder":
            for w, app in (("onedrive", "OneDrive"), ("sharepoint", "SharePoint"), ("google drive", "Google Drive"),
                           ("drive", "Google Drive"), ("box", "Box")):
                if G._bounded(w, G.words_hay(self.hay)) and (app != "Google Drive" or "google" in action):
                    words, fdesc = f"the {app} folder", f"{app} folder"
                    break
        name = self.new_var(base, fdesc + ".")
        words = self.words_for(lit, words if name == base else words.replace(" folder", f" folder {name[-1]}"))
        if name != base:
            raise Unfixable(f"two made-up folders read the same: {lit!r}")
        self.folders[lit] = ("${" + name + "}", words)
        return self.folders[lit]

    def said_tokens(self, name):
        """The words of a file / folder name the request says ('policy_2027' -> ['policy']), at most two."""
        hay = G.words_hay(self.hay)
        toks = [t.lower() for t in re.split(r"[_\-\s.]+", name or "") if len(t) >= 3 and not t.isdigit()]
        said = []
        for t in toks:
            if G._bounded(t, hay):
                said.append(t)
            elif t.endswith("s") and len(t) > 4 and G._bounded(t[:-1], hay):
                said.append(t[:-1])
        return said[:2]

    @staticmethod
    def site_aliases(value):
        m = re.search(r"/sites/([\w-]+)", value)
        if not m:
            return []
        n = re.escape(m.group(1))
        return [(re.compile(rf"\bthe {n} SharePoint site\b"), "the SharePoint site"),
                (re.compile(rf"\b{n} SharePoint site\b"), "the SharePoint site"),
                (re.compile(rf"\bSharePoint site {n}\b"), "the SharePoint site"),
                (re.compile(rf"\bthe {n} site\b"), "the SharePoint site"),
                (re.compile(rf"\bSharePoint \({n}\)"), "SharePoint")]

    # ------------------------------------------------------------------ the flow
    def plan(self):
        """Walk the steps in order and decide every made-up literal (the detector's view of the flow)."""
        hay = self.hay
        written = set()
        for s, _ in F.walk(self.flow.get("steps")):
            action = s.get("action") or ""
            for c in G.custom_fields(s):
                if not G._word_in(re.sub(r"__c$", "", c), G.words_hay(hay)):
                    raise Unfixable(f"made-up Salesforce field {c}")
            if action.startswith(("ooa.json/", "ooa.rest/")):
                for key, val in G.all_values(s):
                    if key == "path" and G._DOTTED.fullmatch(val.strip()):
                        last = [x for x in re.split(r"[.\[\]]", val) if x and not x.isdigit()][-1]
                        if not G._word_in(last, G.words_hay(hay)):
                            raise Unfixable(f"guessed reply field {val}")
            for key, val in G.all_values(s):
                v = val.strip()
                if not v or G._FULL_REF.fullmatch(v):
                    continue
                if key in G.DETAIL_KEYS and not G.exempt(action, key, val):
                    bad = self.literal_bad(action, key, val, written)
                    prod = G.is_producer(action, key)
                    if bad:
                        self.decide(s, key, val, prod)
                    elif prod:
                        written.add(G.norm(v))
                elif G._DN_PART.search(v) and key not in G.DETAIL_KEYS:
                    if not all(G._word_in(p, G.words_hay(hay)) for _, p in G._DN_PART.findall(v)):
                        self.decide_dn("filter", val)

    def literal_bad(self, action, key, val, written):
        hay, v = self.hay, val.strip()
        if key == "extension" or key in ("pattern", "name_pattern") and v.startswith("*."):
            return G.type_filter_issue(action, key, val, hay)
        if key == "database_type":
            return not G.db_type_said(v, hay)
        if key in ("from_currency", "to_currency"):
            return not G._bounded(v.lower(), G.words_hay(hay))
        if key == "range":
            return "!" in v and not G._word_in(v.split("!")[0].strip("'"), G.words_hay(hay))
        if key == "location":
            return not re.search(r"\b(?:location|room|venue|online|at)\b", hay) or not G._word_in(v, hay)
        if G.norm(v) in written:
            return False
        return G.judge(val, hay, producer=G.is_producer(action, key)) not in ("said", "derived")

    def sub_value(self, key, x):
        """The replacement of one argument string: whole value, folder prefix, or inside free text."""
        if x in self.exact:
            return self.exact[x]
        for lit, (ref, _) in sorted(self.folders.items(), key=lambda p: -len(p[0])):
            if lit and (x == lit or x.startswith(lit + "\\") or x.startswith(lit + "/")):
                return ref + x[len(lit):]
        if key in G.DETAIL_KEYS:
            return x
        if key in G.CODE_KEYS or key in G.EXPRESSION_KEYS:
            if key == "filter":
                for old, new in sorted(self.exact.items(), key=lambda p: -len(p[0])):
                    if new and G._DN_PART.search(old) and old in x:
                        x = x.replace(old, "${" + new.lstrip("$") + "}" if not new.startswith("${") else new)
            return x
        return self.sub_text_value(x)

    def sub_text_value(self, x):
        """A message / subject / description the flow sends: made-up values inside it."""
        if not isinstance(x, str) or not x:
            return x
        t = x
        for old, new in sorted(self.exact.items(), key=lambda p: -len(p[0])):
            if new is None or len(old) < 4 or old not in t:
                continue
            ref = new if new.startswith("${") or not new.startswith("$") else "${" + new[1:] + "}"
            if re.match(r"^\$\{[^}]+\}", new) or new.startswith("$"):
                t = t.replace(old, ref)
        for lit, (ref, _) in sorted(self.folders.items(), key=lambda p: -len(p[0])):
            if lit and lit in t:
                t = t.replace(lit, ref)
        hay = G.words_hay(self.hay)
        keep = {}                                  # addresses and said links are kept as they are

        def mask(m):
            keep[f"\x01{len(keep)}\x01"] = m.group(0)
            return f"\x01{len(keep) - 1}\x01"
        t = G._EMAIL.sub(mask, t)
        # company names the request never gives; links and paths it never gives
        for org in G.ORG_WORDS:
            if re.search(r"(?<![\w$])" + re.escape(org) + r"(?![\w-])", t, re.I) and not G._bounded(org, hay):
                t = re.sub(r"\s+(?:with|at|from|of)\s+" + re.escape(org) + r"(?![\w-])", "", t, flags=re.I)
                t = re.sub(r"(?<![\w$])" + re.escape(org) + r"(?![\w-])\s*", "", t, flags=re.I)
        for k, v in keep.items():
            t = t.replace(k, v)
        for m in list(G._URL.finditer(t)):
            u = m.group(0).rstrip(".,;:")
            if G.norm(u).rstrip("/") not in self.hay:
                name = self.new_var("help_link", "Link to give in the reply.")
                t = t.replace(u, "${" + name + "}")
        for m in list(G._WINPATH.finditer(G._REF.sub(lambda r: "\0" * len(r.group(0)), t))):
            p = m.group(0).rstrip(".,;:").replace("\0", "")
            if p and p in t and not G.grounded(p, self.hay) and "${" not in p:
                noun = NOUNS.get(_ext(p), "file") if _ext(p) else "folder"
                t = t.replace(p, f"the {noun}")
                t = re.sub(r"(^|[.!?]\s+)the ", lambda mm: mm.group(1) + "The ", t)
        for m in list(G._GLOB.finditer(t)):
            if not G.ext_said(m.group(1), self.hay):
                t = re.sub(r"\*\." + re.escape(m.group(1)) + r"\s+(?=files?\b)", "", t)
                t = t.replace(m.group(0), "files")
        made = self.made_names()
        for m in list(G._FILE.finditer(G._REF.sub(lambda r: "\0" * len(r.group(0)), t))):
            name = m.group(0)
            if name in t and not G.grounded(name, self.hay) and G.norm(name) not in made:
                t = re.sub(r"(?<![\w\\/.-])" + re.escape(name), f"the {NOUNS.get(_ext(name), 'file')}", t)
                t = re.sub(r"(^|[.!?]\s+)the ", lambda mm: mm.group(1) + "The ", t)
        return plural_text(re.sub(r"\s{2,}", " ", t).strip())

    def made_names(self):
        """File names the flow writes (or the request gives): a text may name them."""
        out = set()
        for _, k, v in G.flow_constants(self.flow):
            out.add(G.norm(G.path_parts(v)[1]))
        for s, _ in F.walk(self.flow.get("steps")):
            for k, v in G.all_values(s):
                if k in PATH_KEYS:
                    out.add(G.norm(G.path_parts(v)[1]))
        return out

    def apply(self):
        def walk_args(x, key):
            if isinstance(x, dict):
                if set(x) <= {"op", "left", "right"} or "expr" in x or "literal" in x:
                    return x
                out = {}
                for k, v in x.items():
                    kk = k if not isinstance(v, dict) or k not in ("args", "fields", "source_inputs",
                                                                    "iterator_args", "value") else key
                    nv = walk_args(v, kk)
                    if nv is None and isinstance(v, str):
                        continue                      # a made-up filter / location: dropped
                    out[k] = nv
                return out
            if isinstance(x, list):
                return [y for y in (walk_args(v, key) for v in x) if y is not None]
            if not isinstance(x, str):
                return x
            if key not in G.EXPRESSION_KEYS:
                y = braced_refs(x)
                if y != x:
                    self.notes.append(f"{key}: {x!r} -> {y!r}")
                    x = y
            if G._FULL_REF.fullmatch(x.strip()):
                return x
            return self.sub_value(key, x)
        for s, _ in F.walk(self.flow.get("steps")):
            if isinstance(s.get("args"), dict):
                new = {}
                for k, v in s["args"].items():
                    nv = walk_args(v, k)
                    if nv is None and isinstance(v, str):
                        continue
                    new[k] = nv
                if new != s["args"]:
                    s["args"] = new

    # ------------------------------------------------------------------ texts
    def fix_text(self, text, keep_said=True):
        if not isinstance(text, str) or not text:
            return text
        t = text
        req = self.request
        subs = []
        for old, words in self.text_subs:
            subs.append((old, words))
        for lit, (ref, words) in self.folders.items():
            subs.append((("FOLDER", lit), words))
        def key_len(p):
            o = p[0]
            return -len(o[1]) if isinstance(o, tuple) else (-len(o) if isinstance(o, str) else 0)
        for old, words in sorted(subs, key=key_len):
            if isinstance(old, re.Pattern):
                t = old.sub(lambda m: m.expand(words) if m.group(0) not in req else m.group(0), t)
                continue
            if isinstance(old, tuple):
                lit = old[1]
                if lit in req:
                    continue
                e = re.escape(lit)
                # 'to D:\\Temp\\fx.zip' -> 'to fx.zip in the download folder'
                t = re.sub(e + r"[\\/]+([^\s,;:)]+?)(?=[\s,;:)]|\.(?:\s|$)|$)", r"\1 in " + words, t)
                t = re.sub(r"(?:\bthe )?" + e + r"(?![\w\\/])(?:\s+folder\b)?", words, t)
                continue
            if words.startswith("the saved "):
                t = re.sub(r"\s*\((?:with )?(?:the )?saved connection " + re.escape(old) + r"\)", "", t)
                t = re.sub(r"(?:the )?saved connection " + re.escape(old), words, t)
                t = re.sub(r"(?:the )?" + re.escape(old) + r" connection", words, t)
            if old in req and len(old) >= 3:
                continue
            noun = words.split()[-1] if words else ""
            for v in sorted({old, old.replace("/", " / "), old.replace("/", " > ")}, key=len, reverse=True):
                if not v or v not in t:
                    continue
                e = (r"(?<![\w\\/.-])" if re.match(r"\w", v) else "") + re.escape(v) + r"(?![\w@-])"
                if words == "":
                    t = re.sub(e, "", t)
                    continue
                if words in ("the repository owner", "the report owner"):
                    t = re.sub(e + r"[\\/]", "", t)
                    continue
                if noun == "sheet":                      # 'open X on Raw' -> 'open X'; 'the Raw sheet' -> 'the sheet'
                    t = re.sub(r"\s+on (?:the )?(?:sheet )?" + e + r"(?:\s+sheet)?", "", t)
                    t = re.sub(r"\b(?:the )?(?:sheet )?" + e + r"\s+sheet\b", "the sheet", t)
                    t = re.sub(r"\bsheet " + e, "the sheet", t)
                # 'the policy index D:\\KB\\x.idx' -> 'the policy index'
                t = re.sub(r"\b(" + "|".join(TEXT_NOUNS) + r") " + e, r"\1", t)
                t = re.sub(r"(?:\bthe )?" + e + (r"(?:\s+" + re.escape(noun) + r"\b)?" if noun else ""), words, t)
                if noun:
                    t = re.sub(r"\b" + re.escape(noun) + r"\s+" + re.escape(words) + r"\b", words, t)
        t = plural_text(t)
        t = re.sub(r"\b(?:to|on|in) (SharePoint|OneDrive) (?:under|in|at) the \1 folder\b", r"to the \1 folder", t)
        t = re.sub(r"\bthe the\b", "the", t)
        t = re.sub(r"\b(a|an|the) (the|a|an)\b", r"\2", t)
        t = re.sub(r"\(\s*\)", "", t)
        t = re.sub(r" +([,;:])(?=\s|$)", r"\1", t)
        t = re.sub(r" +\.(?=\s|$)", ".", t)
        t = re.sub(r" {2,}", " ", t).strip()
        t = re.sub(r"(^|[.!?]\s+)the ", lambda m: m.group(1) + "The ", t)
        return t

    def fix_texts(self):
        r = self.rec
        for k in ("clear", "understand"):
            if isinstance(r.get(k), str):
                if k == "understand" and r.get("variables") is not None:
                    continue                                   # a step-builder part: its understand is its request
                r[k] = self.fix_text(r[k])
        if isinstance(r.get("steps"), list):
            out = []
            for line in r["steps"]:
                head, sep, tail = str(line).partition(" -> ")
                m = re.match(r"^(\s*\d+\.\s*)(.*)$", head)
                if m:
                    body = self.fix_text(m.group(2))
                    head = m.group(1) + (body[:1].upper() + body[1:] if body else body)
                else:
                    head = self.fix_text(head)
                out.append(head + sep + tail)
            r["steps"] = out
        if isinstance(self.flow, dict):
            self.flow["description"] = self.fix_text(self.flow.get("description"))
            for s, _ in F.walk(self.flow.get("steps")):
                if isinstance(s.get("description"), str):
                    d = self.fix_text(s["description"])
                    s["description"] = d[:1].upper() + d[1:] if d else d
        if isinstance(r.get("facts"), list):
            keep = []
            for p in r["facts"]:
                if isinstance(p, (list, tuple)) and len(p) == 2 and p[0] not in ("schedule", "filter") and \
                        not G.grounded(str(p[1]), self.hay):
                    continue
                keep.append(p)
            if keep != r["facts"]:
                self.notes.append(f"facts dropped: {[p for p in r['facts'] if p not in keep]}")
                r["facts"] = keep

    def run(self):
        before = set(self.vars)
        r = self.rec
        req = plural_text(r.get("request"))                # '1 hour(s)' in a generated request
        if isinstance(req, str) and req != r.get("request"):
            if r.get("understand") == r.get("request"):
                r["understand"] = req
            r["request"] = self.request = req
            self.hay = G.haystack(r)
            self.notes.append("request template text")
        if isinstance(self.flow, dict):
            self.plan()
            self.apply()
        self.fix_texts()
        for name in set(self.vars) - before:         # descriptions of the new variables: no made-up values either
            self.vars[name]["description"] = self.fix_text(self.vars[name]["description"])
        if r.get("kind") == "clarify" and not r.get("missing"):
            m = MISSING_BY_REQUEST.get(" ".join(str(r.get("request")).lower().split()))
            if m:
                r["missing"] = m
                self.notes.append(f"missing = {m}")
        return self.notes


def fix_record(rec):
    """Fix one record in place; returns the notes (empty when nothing changed). Raises Unfixable for a record the
    fix cannot make faithful cleanly."""
    before = json.dumps(rec, sort_keys=True)
    notes = Fixer(rec).run()
    if json.dumps(rec, sort_keys=True) != before and not notes:
        notes = ["texts"]
    left = G.check_record(rec)
    if left:
        raise Unfixable("still: " + "; ".join(f"{k} {w} {v}" for k, w, _, v in left[:3]))
    return notes


def fix_file(path, check=False):
    lines = [l for l in open(path, encoding="utf-8") if l.strip()]
    out, changed, notes, dropped = [], 0, [], []
    for i, line in enumerate(lines, 1):
        rec = json.loads(line)
        try:
            n = fix_record(rec)
        except Unfixable as e:
            dropped.append((i, str(e)))
            continue
        if n:
            changed += 1
            notes.append((i, n))
        out.append(json.dumps(rec, ensure_ascii=False))
    if not check:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
    return changed, notes, dropped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    for p in a.files:
        changed, notes, dropped = fix_file(p, a.check)
        why = collections.Counter(re.sub(r"[:'\"].*", "", d) for _, d in dropped)
        print(f"{os.path.basename(p)}: {changed} records {'would change' if a.check else 'changed'}, "
              f"{len(dropped)} {'would be ' if a.check else ''}dropped {dict(why)}")
        if a.verbose:
            for i, n in notes:
                print(f"  {i}: " + "; ".join(n))
            for i, d in dropped:
                print(f"  DROP {i}: {d}")
