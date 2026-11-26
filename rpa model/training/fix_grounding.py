"""Deterministic grounding fix for the neutral SFT records in data/extra_sft/ (see grounding_check.py).

    python fix_grounding.py FILE.jsonl ... [--catalog ../data/catalog/catalog.jsonl] [--check]

Rewrites each file in place (idempotent: a fixed file comes out unchanged):
  - a detail the request does not give (a saved connection, an address, a SharePoint site, a team / channel / group /
    sheet / table / procedure / warehouse / mailbox / project name, a link, a path or the folder of a path) becomes an
    input variable with a clear name, a description and no default; a path whose file name the request gives but not
    its folder keeps the name and gets a folder variable ("${download_folder}\\catalog.zip").
  - the record's texts (clear, understand, steps lines, flow / step descriptions) say "the saved Jira connection",
    "the SharePoint site", "the team" ... instead of the made-up value; facts the request does not give are dropped.
  - a bare $row.Name inside other text (sent literally by the runner) is written ${row.Name}
    ("D:\\out\\$row.Name.txt" -> "D:\\out\\${row.Name}.txt").
  - clarify records without a "missing" field get the slots their question asks for (MISSING_BY_REQUEST).
--check only reports what would change. Standard library only."""
import argparse, json, os, re, sys

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
    "report_path": ("report_path", "the report", "Path of the report to run."),
    "warehouse_id": ("warehouse_id", "the warehouse", "Warehouse id."),
    "pattern": ("file_pattern", "the matching", "Name pattern of the files (for example *.csv)."),
    "from_contains": ("sender_email", "the sender", "Sender address of the emails."),
}
PATH_NAMES = {   # key -> (whole-value variable, folder variable)
    "path": ("file_path", "folder"), "local_path": ("local_file", "local_folder"),
    "remote_path": ("remote_path", "remote_folder"), "target": ("output_path", "output_folder"),
    "source": ("source_path", "source_folder"), "attachments": ("attachment_path", "attachment_folder"),
    "folder": ("folder", "folder"), "target_folder": ("target_folder", "target_folder"),
    "local_folder": ("local_folder", "local_folder"), "remote_folder": ("remote_folder", "remote_folder"),
    "file": ("script_file", "script_folder"), "public_key_file": ("public_key_file", "key_folder"),
    "report_path": ("report_path", "report_owner"),
}
PATH_WORDS = {"path": "the file", "local_path": "the file", "remote_path": "the remote file", "target": "the output file",
              "source": "the source", "attachments": "the file", "folder": "the folder", "target_folder": "the folder",
              "local_folder": "the folder", "remote_folder": "the remote folder", "file": "the script",
              "public_key_file": "the key file", "report_path": "the report"}
EMAIL_KEYS = {"to", "cc", "bcc", "attendees", "recipients", "recipient", "email", "emails", "mailbox", "reply_to"}

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


class Fixer:
    def __init__(self, rec):
        self.rec = rec
        self.flow = rec.get("flow")
        self.hay = G.haystack(rec)
        self.vars = {v["name"]: v for v in (self.flow or {}).get("variables") or [] if isinstance(v, dict)}
        self.by_value = {}            # original literal -> replacement text in the flow
        self.text_subs = []           # (original text, words for the record's texts)
        self.notes = []
        self.made = set()             # file names the flow uses that the request gives or the flow makes
        if isinstance(self.flow, dict):
            for _, k, v in G.flow_constants(self.flow):
                if G.judge(v, self.hay) in ("said", "derived"):
                    self.made.add(G.norm(G.path_parts(v)[1]))

    # ------------------------------------------------------------------ variables
    def new_var(self, base, description, reuse_desc=None):
        name, k = base, 2
        while name in self.vars and self.vars[name].get("description") != description:
            if self.vars[name].get("direction") == "input" and reuse_desc and reuse_desc(self.vars[name]):
                return name
            name, k = f"{base}_{k}", k + 1
        if name not in self.vars:
            v = {"name": name, "type": "String", "direction": "input", "required": True, "description": description}
            self.flow.setdefault("variables", [])
            ins = [i for i, x in enumerate(self.flow["variables"]) if x.get("direction") == "input"]
            self.flow["variables"].insert(ins[-1] + 1 if ins else 0, v)
            self.vars[name] = v
        return name

    def email_name(self, key, value):
        local = value.split("@")[0].lower()
        toks = [t for t in re.split(r"[._+-]", local) if t]
        said = [t for t in toks if re.search(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", self.hay)]
        if said and len(said) == len(toks) or (said and len("".join(said)) >= 4):
            return "_".join(said) + "_email", f"the {' '.join(said)} address"
        if key == "mailbox":
            return "shared_mailbox", "the shared mailbox"
        role = {"attendees": "attendee", "cc": "cc", "from_contains": "sender", "sender": "sender",
                "from": "sender"}.get(key, "recipient")
        return f"{role}_email", f"the {role} address"

    # ------------------------------------------------------------------ one literal
    def replace(self, step, key, value):
        if value in self.by_value:
            return self.by_value[value]
        action = step.get("action") or ""
        desc = (step.get("description") or "").rstrip(".")
        kind = G.judge(value, self.hay)
        if kind in ("said", "derived"):
            return value
        folder, last = G.path_parts(value)
        if kind == "invented" and key in PATH_NAMES and _REF.sub("", folder).strip(" \\/:") \
                and G.norm(last) in self.made:
            kind = "folder"           # a file the flow makes elsewhere, in a folder nobody named
        if key == "connection":
            args = step.get("args") or {}
            src = (args.get("iterator_args") or {}).get("source") if isinstance(args.get("iterator_args"), dict) else None
            tk = (src if isinstance(src, str) and "/" in src else action).split("/")[0].split(".")[-1]
            if tk in ("core", "triggerloop"):
                tk = "app"
            app = APP_NAMES.get(tk, tk.capitalize())
            name = self.new_var(f"{tk}_connection", f"Saved {app} connection (under Connections).")
            new = "$" + name
            self.text_subs.append((value, f"the saved {app} connection"))
        elif G._EMAIL.fullmatch(value.strip()):
            base, words = self.email_name(key, value.strip())
            name = self.new_var(base, f"Email address for: {desc}." if desc else "Email address.")
            new = "$" + name
            self.text_subs.append((value, words))
        elif key in PATH_NAMES and kind == "folder":
            folder, last = G.path_parts(value)
            whole, fbase = PATH_NAMES[key]
            lit = folder.rstrip("\\/")
            if lit in self.by_value:
                new = self.by_value[lit] + folder[len(lit):] + last
            else:
                fdesc = "Remote folder" if key.startswith("remote") else "Folder"
                if key == "report_path":
                    name = self.new_var("report_owner", "Owner of the custom report (the folder part of its path).")
                else:
                    name = self.new_var(fbase, f"{fdesc} for: {desc}." if desc else f"{fdesc}.")
                self.by_value[lit] = "${" + name + "}"
                new = "${" + name + "}" + folder[len(lit):] + last
                words = "the folder" if key != "report_path" else "the report owner"
                self.text_subs.append((lit, words))
                # a SharePoint path 'Shared Documents/Payroll' is also written with spaces or '>'
        elif key in PATH_NAMES:
            whole, _ = PATH_NAMES[key]
            what = "File" if key not in ("folder", "target_folder", "local_folder", "remote_folder") else "Folder"
            name = self.new_var(whole, f"{what} for: {desc}." if desc else f"{what}.")
            new = "$" + name
            self.text_subs.append((value, PATH_WORDS.get(key, "the file")))
            folder, last = G.path_parts(value)
            if G._EXT.search(last) and len(last) >= 5 and not G.grounded(last, self.hay) \
                    and G.norm(last) not in self.made:
                self.text_subs.append((last, PATH_WORDS.get(key, "the file")))
            lit = folder.rstrip("\\/")
            if lit and not G._REF.search(lit) and not G.grounded(lit, self.hay):
                self.text_subs.append((lit, "the folder"))
        else:
            base, words, d = NAMES.get(key, (key, f"the {key.replace('_', ' ')}", f"{key.replace('_', ' ')}."))
            if key in ("site", "drive"):
                self.text_subs += self.site_aliases(value)
            name = self.new_var(base, d if not desc else d.rstrip(".") + f" (for: {desc}).")
            new = "$" + name
            self.text_subs.append((value, words))
        self.by_value[value] = new
        self.notes.append(f"{key}: {value!r} -> {new}")
        return new

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
    def fix_args(self, step, x, key):
        if isinstance(x, dict):
            if set(x) <= {"op", "left", "right"} or "expr" in x or "literal" in x:
                return x
            return {k: self.fix_args(step, v, k if not isinstance(v, dict) or k not in
                                     ("args", "fields", "source_inputs", "iterator_args", "value") else key)
                    for k, v in x.items()}
        if isinstance(x, list):
            return [self.fix_args(step, v, key) for v in x]
        if not isinstance(x, str):
            return x
        if key not in G.EXPRESSION_KEYS:
            y = braced_refs(x)
            if y != x:
                self.notes.append(f"{key}: {x!r} -> {y!r}")
                x = y
        if key in G.DETAIL_KEYS and x.strip() and not G._FULL_REF.fullmatch(x.strip()) \
                and not G.exempt(step.get("action") or "", key, x):
            return self.replace(step, key, x)
        return x

    def fix_flow(self):
        if not isinstance(self.flow, dict):
            return
        for s, _ in F.walk(self.flow.get("steps")):
            if isinstance(s.get("args"), dict):
                s["args"] = {k: self.fix_args(s, v, k) for k, v in s["args"].items()}

    # ------------------------------------------------------------------ texts
    def fix_text(self, text):
        if not isinstance(text, str) or not text:
            return text
        t = text
        for old, words in sorted(self.text_subs, key=lambda p: -len(p[0]) if isinstance(p[0], str) else 0):
            if isinstance(old, re.Pattern):
                t = old.sub(words, t)
                continue
            if words.startswith("the saved "):
                t = re.sub(r"\s*\((?:with )?(?:the )?saved connection " + re.escape(old) + r"\)", "", t)
                t = re.sub(r"(?:the )?saved connection " + re.escape(old), words, t)
                t = re.sub(r"(?:the )?" + re.escape(old) + r" connection", words, t)
            variants = {old, old.replace("\\\\", "\\"), old.replace("/", " / "), old.replace("/", " > ")}
            noun = words.split()[-1]
            for v in sorted(variants, key=len, reverse=True):
                if not v or v not in t:
                    continue
                e = (r"(?<![\w\\/.-])" if re.match(r"\w", v) else "") + re.escape(v)
                # 'ourfirm/billing-api' -> 'billing-api'; 'D:\\Temp\\fx.zip' -> 'fx.zip in the given folder'
                if words in ("the repository owner", "the report owner"):
                    t = re.sub(e + r"[\\/]", "", t)
                elif words.endswith("folder"):
                    t = re.sub(e + r"[\\/]+([^\s,;:)]+?)(?=[\s,;:)]|\.(?:\s|$)|$)", r"\1 in the given folder", t)
                t = re.sub(r"(?:\bthe )?" + e + r"(?:\s+" + re.escape(noun) + r"\b)?", words, t)
                t = re.sub(r"\b" + re.escape(noun) + r"\s+" + re.escape(words) + r"\b", words, t)
        t = re.sub(r"\bthe the\b", "the", t)
        t = re.sub(r"\(\s*the ([\w ]+)\)", r"(the \1)", t)
        return t

    def fix_texts(self):
        r = self.rec
        for k in ("clear", "understand"):
            if isinstance(r.get(k), str):
                r[k] = self.fix_text(r[k])
        if isinstance(r.get("steps"), list):
            out = []
            for line in r["steps"]:
                head, sep, tail = str(line).partition(" -> ")
                out.append(self.fix_text(head) + sep + tail)
            r["steps"] = out
        if isinstance(self.flow, dict):
            self.flow["description"] = self.fix_text(self.flow.get("description"))
            for s, _ in F.walk(self.flow.get("steps")):
                if isinstance(s.get("description"), str):
                    s["description"] = self.fix_text(s["description"])
        if isinstance(r.get("facts"), list):
            keep = [p for p in r["facts"] if not (isinstance(p, (list, tuple)) and len(p) == 2 and
                                                  p[0] not in ("schedule", "filter") and
                                                  not G.grounded(str(p[1]), self.hay))]
            if keep != r["facts"]:
                self.notes.append(f"facts dropped: {[p for p in r['facts'] if p not in keep]}")
                r["facts"] = keep

    def run(self):
        before = set(self.vars)
        self.fix_flow()
        self.fix_texts()
        for name in set(self.vars) - before:         # descriptions of the new variables: no made-up values either
            self.vars[name]["description"] = self.fix_text(self.vars[name]["description"])
        r = self.rec
        if r.get("kind") == "clarify" and not r.get("missing"):
            m = MISSING_BY_REQUEST.get(" ".join(str(r.get("request")).lower().split()))
            if m:
                r["missing"] = m
                self.notes.append(f"missing = {m}")
        return self.notes


def fix_record(rec):
    """Fix one record in place; returns the notes (empty when nothing changed)."""
    return Fixer(rec).run()


def fix_file(path, check=False):
    lines = [l for l in open(path, encoding="utf-8") if l.strip()]
    out, changed, notes = [], 0, []
    for i, line in enumerate(lines, 1):
        rec = json.loads(line)
        n = fix_record(rec)
        if n:
            changed += 1
            notes.append((i, n))
        out.append(json.dumps(rec, ensure_ascii=False))
    if not check:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
    return changed, notes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    for p in a.files:
        changed, notes = fix_file(p, a.check)
        print(f"{os.path.basename(p)}: {changed} records {'would change' if a.check else 'changed'}")
        if a.verbose:
            for i, n in notes:
                print(f"  {i}: " + "; ".join(n))
