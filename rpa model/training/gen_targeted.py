"""Targeted training data for the failures measured through Studio (no LLM): many wordings of the requests the model
got wrong, each with a hand-shaped flow built from the real catalog.

Measured failures it covers: file vocabulary (rename became delete; rename / move / copy / delete / zip / unzip are
different actions, and nothing is deleted unless the request asks for it), "for each X" (list or read, then core/loop
with the per-item steps inside), conditions ("if the file exists ... else ...", "only if the amount is over 1000", file
type), web tasks (open, search, click, screenshot, table to CSV, download), "... and then email it" (the produced file
is attached), OCR (scans are read with pdf/read_text_ocr, text PDFs with pdf/extract_text), Excel / Word to PDF, and
the source named in the request (Google Sheet vs Excel vs CSV; Outlook vs Gmail vs mail server).

Also holds the small kit the other generators import (gen_long.py, gen_clarify.py): FlowBuilder (declares variables
from the catalog as steps are added), Checker (repair + ooa_reward.problems must be empty), LeakGuard (drops requests
too close to a test request) and wording helpers (casual / formal / short / typos / Hinglish).

Writes NEUTRAL records, one per line: {"kind": "flow", "request": str, "flow": obj, "source": "targeted:<family>"}.

Usage: python gen_targeted.py CATALOG.jsonl SCHEMA.json OUT.jsonl [n=4000] [seed=11]
"""
import difflib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ooa_reward as R   # noqa: E402
import repair as RP      # noqa: E402

TYPE_MAP = {"Text": "String", "File path": "String", "Number": "Number", "Yes/No": "Boolean", "List": "List",
            "Table": "Table", "Dictionary": "Dictionary", "DateTime": "DateTime", "Any": "Any",
            "Credential (vault)": "Credential"}
SECRET = re.compile(r"(password|secret|token|api_key|private_key|passphrase)$")


# ---------------------------------------------------------------------------------------------------------------- kit

def load_catalog(path):
    cat = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return {e["id"]: e for e in cat}


def snake(s):
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return (s if s and not s[0].isdigit() else "v_" + s)[:64]


def words(name):
    return name.replace("_", " ")


class FlowBuilder:
    """Collects variables and steps. Every action, input name and output name is checked against the catalog when the
    step is made, so a typo here fails at once. Outputs become local variables typed from the catalog; flow inputs are
    declared with inp(); any other $name read by a step is declared as a String input at the end (a Credential when
    the name says password / token / secret)."""

    def __init__(self, cat):
        self.cat, self.vars, self.hints = cat, {}, {}

    def inp(self, name, desc, typ="String"):
        if name not in self.vars:
            self.vars[name] = {"name": name, "type": typ, "direction": "input", "required": True, "description": desc}
        return "$" + name

    def local(self, name, typ="Any"):
        if name not in self.vars:
            self.vars[name] = {"name": name, "type": typ, "direction": "local"}
        return name

    def output(self, name):
        if name in self.vars:
            self.vars[name]["direction"] = "output"
            self.vars[name].pop("required", None)

    def s(self, action, desc, args=None, out=None):
        e = self.cat[action]
        args = dict(args or {})
        ins = {i["name"] for i in e["inputs"]}
        bad = set(args) - ins
        assert not bad, f"{action} has no input {bad}"
        if not action.startswith("core/"):
            miss = {i["name"] for i in e["inputs"] if i["required"]} - set(args)
            assert not miss, f"{action} needs {miss}"
        step = {"id": "", "action": action, "description": desc, "args": args}
        if out:
            outs = {o["name"]: o["type"] for o in e["outputs"]}
            for k, v in out.items():
                assert k in outs, f"{action} has no output {k}"
                self.local(v, TYPE_MAP.get(outs[k], "Any"))
            step["out"] = dict(out)
        return step

    def loop(self, over, item, children, desc, item_type="Any", iterator="list"):
        self.local(item, item_type)
        return {"id": "", "action": "core/loop", "description": desc,
                "args": {"iterator": iterator, "over": over, "as": item}, "children": list(children)}

    def if_(self, cond, children, desc, else_=None):
        n = {"id": "", "action": "core/if", "description": desc, "args": {"condition": cond}, "children": list(children)}
        if else_:
            n["else"] = list(else_)
        return n

    def try_(self, children, catch, desc, message_to="error_message"):
        self.local(message_to, "String")
        return {"id": "", "action": "core/try", "description": desc, "args": {}, "children": list(children),
                "catch": [{"message_to": message_to, "children": list(catch)}]}

    def log(self, message, desc="Write a log line", level="info"):
        return self.s("core/log", desc, {"message": message, "level": level})

    def flow(self, name, steps, outputs=(), description=None):
        for o in outputs:
            self.output(o)
        n = [0]

        def renumber(nodes):
            for s in nodes:
                n[0] += 1
                s["id"] = f"s{n[0]}"
                for b in RP.bodies(s):
                    renumber(b)
        renumber(steps)
        f = {"kind": "TaskBot", "id": snake(name)[:40] or "flow", "name": name[:80],
             "description": description or name, "variables": list(self.vars.values()), "steps": steps}
        used = {r for s in R.walk(steps) for r in R.refs(s.get("args") or {})}
        for r in sorted(used - set(self.vars)):
            typ = "Credential" if SECRET.search(r) else "String"
            v = {"name": r, "type": typ, "direction": "input", "required": True,
                 "description": self.hints.get(r) or (words(r).capitalize() + ".")}
            f["variables"].append(v)
            self.vars[r] = v
        return f


class Checker:
    """repair.repair then ooa_reward.problems must be empty; returns the repaired flow or None."""

    def __init__(self, catalog_path, schema_path):
        R.configure(catalog_path, schema_path)
        self.cat = load_catalog(catalog_path)
        self.n, self.ok = 0, 0

    def check(self, flow):
        self.n += 1
        try:
            fixed, _ = RP.repair(flow, self.cat)
        except Exception:
            return None
        if R.problems(json.dumps(fixed, ensure_ascii=False)):
            return None
        self.ok += 1
        return fixed

    def problems(self, flow):
        return R.problems(json.dumps(flow, ensure_ascii=False))


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", t.lower())).strip()


class LeakGuard:
    """Never train on a test request: a request whose difflib ratio with any test prompt is over 0.8 is dropped."""

    def __init__(self, eval_dir=None, threshold=0.8):
        ev = Path(eval_dir) if eval_dir else HERE.parent / "eval"
        self.tests = [norm(json.loads(l)["prompt"]) for p in sorted(ev.glob("plain_requests*.jsonl"))
                      for l in open(p, encoding="utf-8") if l.strip()]
        self.t, self.dropped = threshold, 0

    def leaks(self, request):
        r = norm(request)
        for t in self.tests:
            m = difflib.SequenceMatcher(None, r, t)
            if m.real_quick_ratio() > self.t and m.quick_ratio() > self.t and m.ratio() > self.t:
                self.dropped += 1
                return True
        return False


# wording ------------------------------------------------------------------------------------------------------------

CASUAL = ["can you {r}", "pls {r}", "i need a bot to {r}", "{r} pls", "hey, {r}", "could you {r}", "{r} thanks",
          "need to {r}", "quick one: {r}", "{r}, every morning"]
FORMAL = ["Please {r}.", "I would like an automation that will {r}.", "Create an automation to {r}.",
          "Build a flow that will {r}.", "Kindly {r}.", "We need a process that will {r}."]
KEYS = "qwertyuiopasdfghjklzxcvbnm"


def typo(text, rng, n=None):
    """One or two realistic slips: a swapped pair, a dropped or doubled letter."""
    w = text.split(" ")
    for _ in range(n or rng.choice([1, 1, 2])):
        cand = [k for k, x in enumerate(w) if len(x) > 4 and x.isalpha()]
        if not cand:
            break
        k = rng.choice(cand)
        x, j = w[k], rng.randrange(1, len(w[k]) - 2)
        op = rng.random()
        if op < 0.4:
            x = x[:j] + x[j + 1] + x[j] + x[j + 2:]
        elif op < 0.7:
            x = x[:j] + x[j + 1:]
        else:
            x = x[:j] + x[j] + x[j:]
        w[k] = x
    return " ".join(w)


def style(core, rng, hinglish=None, p_hing=0.125):
    """core: an imperative clause without final punctuation ("zip the Reports folder and email it to my manager").
    Returns one request in a random register; Hinglish when given and drawn (about 1 in 8)."""
    if hinglish and rng.random() < p_hing:
        r = rng.choice(hinglish) if isinstance(hinglish, list) else hinglish
        return typo(r, rng) if rng.random() < 0.1 else r
    x = rng.random()
    if x < 0.35:
        r = core[0].upper() + core[1:]
        r = r if r.endswith((".", "?", "!")) else r + "."
    elif x < 0.6:
        r = rng.choice(CASUAL).format(r=core[0].lower() + core[1:]).rstrip(".")
    elif x < 0.85:
        r = rng.choice(FORMAL).format(r=core[0].lower() + core[1:].rstrip("."))
    else:
        r = core.lower().rstrip(".")
    if rng.random() < 0.12:
        r = typo(r, rng)
    return r


def pick(rng, items):
    return rng.choice(items)


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")


def flow_record(request, flow, source):
    return {"kind": "flow", "request": request, "flow": flow, "source": source}


def run_families(families, n, rng, checker, guard, prefix, max_per_seq=None, dedupe=True):
    """families: {name: (fn, weight)}; fn(fb_factory, rng) -> (request, flow) or None. Returns records + stats."""
    names, weights = zip(*[(k, w) for k, (_, w) in families.items()])
    quota = {k: max(1, round(n * w / sum(weights))) for k, w in zip(names, weights)}
    made, rows, seen, why, per_seq = Counter(), [], set(), Counter(), Counter()
    tries = 0
    while len(rows) < n and tries < n * 40:
        tries += 1
        fam = rng.choices(names, weights)[0]
        if made[fam] >= quota[fam] and any(made[k] < quota[k] for k in names):
            continue
        try:
            got = families[fam][0](lambda: FlowBuilder(checker.cat), rng)
        except AssertionError as e:
            raise SystemExit(f"{fam}: {e}")
        if not got:
            why["no flow"] += 1
            continue
        req, flow = got
        req = re.sub(r"\s+", " ", req).strip()
        if dedupe and req.lower() in seen:
            why["same request"] += 1
            continue
        if guard.leaks(req):
            why["close to a test request"] += 1
            continue
        fixed = checker.check(flow)
        if fixed is None:
            why["checker"] += 1
            continue
        if max_per_seq:
            key = tuple(R.actions_of(fixed))
            if per_seq[key] >= max_per_seq:
                why["sequence quota"] += 1
                continue
            per_seq[key] += 1
        seen.add(req.lower())
        made[fam] += 1
        rows.append(flow_record(req, fixed, f"{prefix}:{fam}"))
    return rows, made, why


# ------------------------------------------------------------------------------------------------------------ slots

FOLDERS = ["Downloads", "Reports", "Invoices", "Scans", "Archive", "Exports", "Temp", "HR Documents", "Purchase Orders",
           "Receipts", "Contracts", "Statements", "Payroll", "Shared", "Incoming", "Processed", "Backups", "Quotes",
           "Delivery Notes", "Claims", "Timesheets", "Vendor Docs", "Month End", "Audit"]
RENAMES = [("txt", "bak"), ("log", "old"), ("csv", "txt"), ("jpeg", "jpg"), ("htm", "html"), ("tmp", "bak"),
           ("dat", "csv"), ("txt", "md"), ("xml", "bak"), ("json", "bak")]
FTYPES = [("pdf", "PDF", "PDFs"), ("xlsx", "Excel", "Excel files"), ("csv", "CSV", "CSV files"),
          ("docx", "Word", "Word documents"), ("txt", "text", "text files"), ("png", "PNG", "PNG images"),
          ("jpg", "JPG", "photos"), ("zip", "zip", "zip files"), ("xml", "XML", "XML files"), ("log", "log", "log files")]
PEOPLE = [("my manager", "Email address of your manager."), ("the finance team", "Finance team email address."),
          ("the HR team", "HR team email address."), ("my boss", "Your boss's email address."),
          ("the accounts team", "Accounts team email address."), ("the client", "Client email address."),
          ("the auditors", "Auditors' email address."), ("the sales team", "Sales team email address."),
          ("Priya", "Priya's email address."), ("Rahul", "Rahul's email address."), ("John", "John's email address."),
          ("the operations lead", "Operations lead email address."), ("the IT helpdesk", "IT helpdesk email address."),
          ("procurement", "Procurement team email address."), ("the payroll team", "Payroll team email address.")]
SITES = ["the supplier portal", "our intranet", "the vendor site", "the bank portal", "the tax portal",
         "the shipping tracker site", "the competitor's pricing page", "the government tenders site",
         "the courier website", "our CRM web page", "the exchange rates page", "the job board"]
THINGS = [("orders", "Order"), ("customers", "Customer"), ("invoices", "Invoice"), ("employees", "Employee"),
          ("vendors", "Vendor"), ("products", "Product"), ("tickets", "Ticket"), ("students", "Student"),
          ("members", "Member"), ("suppliers", "Supplier"), ("leads", "Lead"), ("claims", "Claim")]
COLS = ["Amount", "Total", "Balance", "Price", "Quantity", "Salary", "Score", "Value", "Outstanding"]
DAYS = [7, 14, 30, 60, 90, 180, 365]


def person(rng):
    p, d = rng.choice(PEOPLE)
    return p, snake(p.replace("the ", "").replace("my ", "")) + "_email", d


def fsnake(folder):
    return snake(folder) + "_folder"


# ------------------------------------------------------------------------------------------------- file vocabulary

def f_rename_one(B, rng):
    b = B()
    olds = ["report.xlsx", "draft.docx", "invoice_scan.pdf", "data.csv", "notes.txt", "photo.jpg", "summary.pdf",
            "budget.xlsx", "minutes.docx", "export.csv"]
    old = rng.choice(olds)
    stem, ext = old.rsplit(".", 1)
    new = rng.choice([f"{stem}_final.{ext}", f"{stem}_2026.{ext}", f"{stem}_old.{ext}", f"final_{stem}.{ext}",
                      f"{stem}_v2.{ext}", f"approved_{stem}.{ext}"])
    folder = rng.choice(FOLDERS)
    path = b.inp("file_path", f"Full path of {old}.")
    step = b.s("ooa.file/move", f"Rename {old} to {new}", {"source": path, "target": new})
    core = pick(rng, [f"rename {old} in the {folder} folder to {new}", f"rename the file {old} to {new}",
                      f"change the name of {old} to {new}", f"{old} ka naam badal ke {new} kar do",
                      f"give {old} the new name {new}", f"rename {old} as {new}, don't delete anything"])
    req = style(core, rng, hinglish=[f"{old} ko rename karke {new} kar do", f"{folder} folder me {old} ka naam {new} kar do"])
    return req, b.flow(f"Rename {old}", [step])


def f_rename_ext(B, rng):
    b = B()
    a, z = rng.choice(RENAMES)
    folder = rng.choice(FOLDERS)
    fv = b.inp("folder", f"The {folder} folder.")
    lst = b.s("ooa.file/list", f"List the .{a} files in the folder", {"folder": fv, "pattern": f"*.{a}"}, {"files": "files"})
    name = b.s("ooa.file/get_name", "Get the file name without the extension", {"path": "$file"},
               {"name_without_extension": "base_name"})
    mv = b.s("ooa.file/move", f"Rename the file to .{z}", {"source": "$file", "target": "${folder}/${base_name}." + z})
    body = [name, mv]
    if rng.random() < 0.3:
        body.append(b.log("Renamed ${file}"))
    loop = b.loop("$files", "file", body, f"For each .{a} file", "String")
    core = pick(rng, [
        f"rename every .{a} file in the {folder} folder to .{z}",
        f"change the extension of all {a} files in {folder} to {z}",
        f"in the {folder} folder, rename all the .{a} files so they end in .{z}",
        f"go through the {folder} folder and rename each .{a} file to .{z}",
        f"for each .{a} file in {folder}, rename it to .{z} (keep the same name)",
        f"rename all *.{a} in a folder to *.{z}",
        f"make every .{a} in the {folder} folder a .{z} file by renaming it",
        f"bulk rename the .{a} files in {folder} to .{z}, don't delete them"])
    hing = [f"{folder} folder ki saari .{a} files ko .{z} me rename kar do",
            f"har .{a} file ka extension .{z} kar do {folder} folder me"]
    return style(core, rng, hing), b.flow(f"Rename {a} files to {z}", [lst, loop])


def f_rename_prefix(B, rng):
    b = B()
    folder = rng.choice(FOLDERS)
    ext, _, plural = rng.choice(FTYPES[:6])
    pre = rng.choice(["Processed_", "OLD_", "2026_", "Done_", "Checked_", "Archived_", "Q3_"])
    fv = b.inp("folder", f"The {folder} folder.")
    lst = b.s("ooa.file/list", f"List the {plural} in the folder", {"folder": fv, "pattern": f"*.{ext}"}, {"files": "files"})
    name = b.s("ooa.file/get_name", "Get the file name", {"path": "$file"}, {"name": "file_name"})
    mv = b.s("ooa.file/move", f"Rename the file with the {pre} prefix",
             {"source": "$file", "target": "${folder}/" + pre + "${file_name}"})
    loop = b.loop("$files", "file", [name, mv], f"For each {ext} file", "String")
    core = pick(rng, [f"add {pre} to the start of every {ext} file name in the {folder} folder",
                      f"rename all {plural} in {folder} so their names start with {pre}",
                      f"put the prefix {pre} in front of each {plural[:-1] if plural.endswith('s') else plural} name in the {folder} folder",
                      f"for each {ext} in {folder}, rename it to {pre} plus the old name"])
    return style(core, rng, [f"{folder} ki har {ext} file ke naam ke aage {pre} laga do"]), \
        b.flow(f"Prefix {ext} files", [lst, loop])


def f_move_type(B, rng):
    b = B()
    src, dst = rng.sample(FOLDERS, 2)
    ext, label, plural = rng.choice(FTYPES)
    s1 = b.inp(fsnake(src), f"The {src} folder.")
    s2 = b.inp(fsnake(dst), f"The {dst} folder.")
    step = b.s("ooa.file/move", f"Move the {plural} to {dst}", {"source": s1, "target": s2, "pattern": f"*.{ext}"},
               {"count": "moved_count"})
    steps = [step]
    if rng.random() < 0.35:
        steps.append(b.log("Moved ${moved_count} files"))
    core = pick(rng, [f"move all the {plural} from {src} to {dst}", f"move every .{ext} file from the {src} folder into {dst}",
                      f"shift the {label} files out of {src} and into {dst}", f"move {ext} files from {src} to the {dst} folder",
                      f"take all {plural} in {src} and move them to {dst}"])
    return style(core, rng, [f"{src} se saari {ext} files {dst} me move kar do"]), b.flow(f"Move {ext} files", steps)


def f_copy(B, rng):
    b = B()
    src, dst = rng.sample(FOLDERS, 2)
    ext, label, plural = rng.choice(FTYPES)
    if rng.random() < 0.5:
        step = b.s("ooa.file/copy", f"Copy the {plural} to {dst}",
                   {"source": b.inp(fsnake(src), f"The {src} folder."), "target": b.inp(fsnake(dst), f"The {dst} folder."),
                    "pattern": f"*.{ext}"}, {"count": "copied_count"})
        core = pick(rng, [f"copy all the {plural} from {src} into {dst}", f"make a copy of every .{ext} in {src} in the {dst} folder",
                          f"copy {label} files from {src} to {dst}, keep the originals", f"duplicate the {plural} from {src} into {dst}"])
        hing = [f"{src} ki saari {ext} files ko {dst} me copy kar do"]
    else:
        f = rng.choice(["the budget workbook", "the contract", "today's report", "the price list", "the template"])
        step = b.s("ooa.file/copy", f"Copy {f} to {dst}",
                   {"source": b.inp("file_path", f"Path of {f}."), "target": b.inp(fsnake(dst), f"The {dst} folder.")},
                   {"path": "copy_path"})
        core = pick(rng, [f"copy {f} into the {dst} folder", f"put a copy of {f} in {dst}",
                          f"back up {f} to {dst} by copying it"])
        hing = [f"{f} ki ek copy {dst} folder me daal do"]
    return style(core, rng, hing), b.flow("Copy files", [step])


def f_delete(B, rng):
    b = B()
    folder = rng.choice(FOLDERS)
    ext, label, plural = rng.choice(FTYPES)
    d = rng.choice(DAYS)
    x = rng.random()
    if x < 0.45:
        step = b.s("ooa.file/delete", f"Delete files older than {d} days",
                   {"path": b.inp("folder", f"The {folder} folder."), "pattern": "*", "older_than_days": d}, {"deleted": "deleted_count"})
        core = pick(rng, [f"delete files older than {d} days in the {folder} folder", f"clean up {folder}: remove anything older than {d} days",
                          f"remove files in {folder} that are more than {d} days old"])
        hing = [f"{folder} folder se {d} din se purani files delete kar do"]
    elif x < 0.8:
        step = b.s("ooa.file/delete", f"Delete the {plural}", {"path": b.inp("folder", f"The {folder} folder."), "pattern": f"*.{ext}"},
                   {"deleted": "deleted_count"})
        core = pick(rng, [f"delete all {plural} in {folder}", f"remove every .{ext} file from the {folder} folder",
                          f"get rid of the {label} files in {folder}"])
        hing = [f"{folder} ki saari {ext} files delete kar do"]
    else:
        f = rng.choice(["the old export", "yesterday's temp file", "the duplicate report", "the draft copy"])
        step = b.s("ooa.file/delete", f"Delete {f}", {"path": b.inp("file_path", f"Path of {f}.")}, {"deleted": "deleted_count"})
        core = pick(rng, [f"delete {f}", f"remove {f} from disk", f"permanently delete {f}"])
        hing = [f"{f} ko delete kar do"]
    steps = [step]
    if rng.random() < 0.3:
        steps.append(b.log("Deleted ${deleted_count} files"))
    return style(core, rng, hing), b.flow("Delete files", steps)


def f_zip(B, rng):
    b = B()
    folder = rng.choice(FOLDERS)
    if rng.random() < 0.5:
        ext, label, plural = rng.choice(FTYPES[:7])
        step = b.s("ooa.file/zip", f"Zip the {plural}", {"source": b.inp("folder", f"The {folder} folder."), "pattern": f"*.{ext}",
                                                         "target": b.inp("zip_path", "Where to save the zip file.")}, {"path": "zip_file"})
        core = pick(rng, [f"zip only the {plural} in {folder}", f"compress the {label} files of the {folder} folder into one zip",
                          f"make a zip of all .{ext} files in {folder}"])
        hing = [f"{folder} ki {ext} files ka zip bana do"]
    else:
        step = b.s("ooa.file/zip", f"Zip the {folder} folder", {"source": b.inp("folder", f"The {folder} folder."),
                                                                "target": b.inp("zip_path", "Where to save the zip file.")}, {"path": "zip_file"})
        core = pick(rng, [f"zip up the {folder} folder", f"compress the {folder} folder", f"create a zip archive of {folder}",
                          f"put the whole {folder} folder into a zip file"])
        hing = [f"{folder} folder ko zip kar do"]
    return style(core, rng, hing), b.flow("Zip files", [step], outputs=["zip_file"])


def f_unzip(B, rng):
    b = B()
    folder = rng.choice(FOLDERS)
    what = rng.choice(["the invoices zip", "the archive from the client", "the backup zip", "the payroll zip",
                       "the zip we got from the bank", "the photos archive", "the downloaded zip"])
    step = b.s("ooa.file/unzip", f"Unzip {what}", {"archive": b.inp("zip_path", f"Path of {what}."),
                                                   "folder": b.inp(fsnake(folder), f"The {folder} folder.")}, {"files": "files"})
    steps = [step]
    if rng.random() < 0.5:
        steps.append(b.loop("$files", "file", [b.log("Extracted ${file}")], "For each extracted file", "String"))
        core = pick(rng, [f"unzip {what} into {folder} and log every file that came out",
                          f"extract {what} to the {folder} folder, then write each file name to the log"])
    else:
        core = pick(rng, [f"unzip {what} into the {folder} folder", f"extract {what} to {folder}",
                          f"unpack {what} into {folder}"])
    return style(core, rng, [f"{what} ko {folder} me unzip kar do"]), b.flow("Unzip archive", steps)


def f_backup_then_delete(B, rng):
    """copy + delete original (explicitly asked) vs move."""
    b = B()
    src, dst = rng.sample(FOLDERS, 2)
    ext, label, plural = rng.choice(FTYPES)
    sv, dv = b.inp(fsnake(src), f"The {src} folder."), b.inp(fsnake(dst), f"The {dst} folder.")
    s1 = b.s("ooa.file/copy", f"Copy the {plural} to {dst}", {"source": sv, "target": dv, "pattern": f"*.{ext}"}, {"count": "copied"})
    s2 = b.s("ooa.file/delete", f"Delete the {plural} from {src}", {"path": sv, "pattern": f"*.{ext}"}, {"deleted": "deleted"})
    core = pick(rng, [f"copy the {plural} from {src} to {dst} and after that delete them from {src}",
                      f"back up every .{ext} in {src} to {dst}, then remove the originals",
                      f"first copy all {label} files from {src} into {dst}, then delete them in {src}"])
    return style(core, rng, [f"{src} ki {ext} files {dst} me copy karo, phir {src} se delete kar do"]), \
        b.flow("Back up then delete", [s1, s2])


# ------------------------------------------------------------------------------------------------------- for each

def f_each_csv_rows(B, rng):
    b = B()
    folder = rng.choice(FOLDERS)
    lst = b.s("ooa.file/list", "List the CSV files", {"folder": b.inp("folder", f"The {folder} folder."), "pattern": "*.csv"},
              {"files": "files"})
    rd = b.s("ooa.csv/read", "Read the CSV", {"path": "$file"}, {"table": "table", "row_count": "row_count"})
    lg = b.log("${file}: ${row_count} rows", "Log the file name and its row count")
    body = [rd, lg]
    if rng.random() < 0.3:
        body = [b.try_([rd, lg], [b.log("Could not read ${file}: ${error_message}", "Log the file that failed", "error")],
                       "Try to read the file")]
    core = pick(rng, [f"for each CSV in the {folder} folder, count the rows and log it",
                      f"count how many rows every CSV file in {folder} has and write the counts to the log",
                      f"go through all the csv files in {folder} and log the number of rows in each",
                      f"log the row count of each CSV file in the {folder} folder",
                      f"for every csv in {folder}: read it, count rows, log file name + count"])
    return style(core, rng, [f"{folder} folder ki har CSV ke rows gino aur log me likho"]), \
        b.flow("Row count per CSV", [lst, b.loop("$files", "file", body, "For each CSV file", "String")])


def f_each_customer_email(B, rng):
    b = B()
    things, one = rng.choice(THINGS)
    src = rng.choice(["csv", "csv", "excel", "gsheet"])
    steps = []
    if src == "csv":
        steps.append(b.s("ooa.csv/read", f"Read the {things} CSV", {"path": b.inp("csv_path", f"The {things} CSV file.")}, {"table": "table"}))
        where = rng.choice([f"the {things} CSV", f"a CSV of {things}", f"{things}.csv"])
    elif src == "excel":
        steps.append(b.s("ooa.excel/open", f"Open the {things} workbook", {"path": b.inp("workbook_path", f"The {things} Excel file.")}))
        steps.append(b.s("ooa.excel/get_all_cells", "Read the sheet as a table", {}, {"table": "table"}))
        steps.append(b.s("ooa.excel/close", "Close the workbook", {}))
        where = rng.choice([f"the {things} Excel sheet", f"the {things} workbook", f"our {things} spreadsheet (xlsx)"])
    else:
        steps.append(b.s("ooa.google/sheets_read", f"Read the {things} Google Sheet",
                         {"access_token": b.inp("access_token", "Google access token.", "Credential"),
                          "spreadsheet_id": b.inp("spreadsheet_id", f"Id of the {things} Google Sheet.")}, {"table": "table"}))
        where = rng.choice([f"the {things} Google Sheet", f"our {things} sheet in Google Sheets"])
    steps.append(b.s("ooa.datatable/to_records", "Turn the rows into records", {"table": "$table"}, {"records": "rows"}))
    topic = rng.choice(["a payment reminder", "a thank-you note", "their monthly statement notice", "a renewal reminder",
                        "the holiday schedule", "a meeting invite note", "a welcome message", "an outstanding balance reminder"])
    subj = topic.split(" ", 1)[1].capitalize()
    send = b.s("ooa.email/send", f"Email {topic} to the {one.lower()}",
               {"to": "$row.Email", "subject": subj, "body": "Dear ${row.Name}, this is " + topic + "."})
    steps.append(b.loop("$rows", "row", [send], f"For each {one.lower()}", "Dictionary"))
    core = pick(rng, [f"send each of the {things} in {where} {topic} by email",
                      f"for every row in {where}, email that person {topic}",
                      f"read {where} and email every {one.lower()} {topic}",
                      f"go through {where} and send {topic} to each {one.lower()}'s email address",
                      f"email {topic} to all {things} listed in {where}, one email per person"])
    return style(core, rng, [f"{where} me jitne {things} hain sabko {topic} email karo"]), b.flow(f"Email each {one}", steps)


def f_each_file_action(B, rng):
    b = B()
    folder = rng.choice(FOLDERS)
    kind = rng.choice(["pages", "info", "text", "print", "extract"])
    lst_args = {"folder": b.inp("folder", f"The {folder} folder.")}
    if kind == "pages":
        lst_args["pattern"] = "*.pdf"
        body = [b.s("ooa.pdf/page_count", "Count the pages", {"path": "$file"}, {"count": "pages"}),
                b.log("${file} has ${pages} pages", "Log the page count")]
        core = pick(rng, [f"for each PDF in {folder}, count its pages and log the number",
                          f"log how many pages every pdf in the {folder} folder has",
                          f"go through the PDFs in {folder} one by one and write their page counts to the log"])
    elif kind == "info":
        body = [b.s("ooa.file/get_info", "Get the file details", {"path": "$file"}, {"info": "info"}),
                b.log("${file}: ${info.size} bytes", "Log the size")]
        core = pick(rng, [f"for each file in {folder}, look up its size and log it", f"log the size of every file in the {folder} folder"])
    elif kind == "text":
        lst_args["pattern"] = "*.txt"
        body = [b.s("ooa.file/read_text", "Read the text file", {"path": "$file"}, {"text": "text"}),
                b.s("ooa.file/write_text", "Add it to the combined file", {"path": b.inp("combined_path", "The combined text file."),
                                                                          "text": "$text", "append": True})]
        core = pick(rng, [f"combine all the text files in {folder} into one file, one after another",
                          f"for every .txt in {folder}, read it and append it to a single combined file"])
    elif kind == "print":
        lst_args["pattern"] = "*.pdf"
        body = [b.s("ooa.printer/print_file", "Print the PDF", {"path": "$file"})]
        core = pick(rng, [f"print every PDF in the {folder} folder", f"for each pdf in {folder}, send it to the printer"])
    else:
        lst_args["pattern"] = "*.pdf"
        body = [b.s("ooa.pdf/extract_text", "Get the text out of the PDF", {"path": "$file"}, {"text": "text"}),
                b.s("ooa.file/get_name", "Get the file name", {"path": "$file"}, {"name_without_extension": "base_name"}),
                b.s("ooa.file/write_text", "Save the text next to it", {"path": "${folder}/${base_name}.txt", "text": "$text"})]
        core = pick(rng, [f"for each pdf in {folder}, pull out the text and save it as a .txt with the same name",
                          f"convert every PDF in the {folder} folder to a text file with the same name"])
    lst = b.s("ooa.file/list", "List the files", lst_args, {"files": "files"})
    return style(core, rng, [f"{folder} ki har file ke liye ye karo: {core}"] if rng.random() < 0.5 else None), \
        b.flow(f"Each file in {folder}", [lst, b.loop("$files", "file", body, "For each file", "String")])


def f_each_email(B, rng):
    b = B()
    folder = rng.choice(["Inbox", "Invoices", "Supplier Mail", "Orders", "Support", "Resumes", "Claims"])
    target = rng.choice(FOLDERS)
    con = b.s("ooa.email/connect", "Sign in to the mailbox", {"imap_host": b.inp("imap_host", "Incoming mail server."),
                                                             "user": b.inp("mail_user", "Mailbox user name."),
                                                             "password": b.inp("mail_password", "Mailbox password.", "Credential")})
    args = {"folder": folder, "unread_only": True}
    if rng.random() < 0.5:
        args["has_attachments_only"] = True
    lst = b.s("ooa.email/list", f"Find the unread emails in {folder}", args, {"table": "emails", "count": "email_count"})
    rec = b.s("ooa.datatable/to_records", "Turn the list into records", {"table": "$emails"}, {"records": "mails"})
    save = b.s("ooa.email/save_attachments", "Save the attachments", {"id": "$mail.id", "target_folder": b.inp(fsnake(target), f"The {target} folder."),
                                                                      "folder": folder}, {"files": "saved"})
    mark = b.s("ooa.email/mark", "Mark the email as read", {"id": "$mail.id", "read": True, "folder": folder})
    body = [save, mark] if rng.random() < 0.6 else [save]
    core = pick(rng, [f"for each unread email in the {folder} folder, save its attachments to {target}" + (" and mark it read" if len(body) > 1 else ""),
                      f"go through every unread mail in {folder} and download the attachments into the {target} folder" + (", then mark each one as read" if len(body) > 1 else ""),
                      f"save the attachments of each new email in {folder} to {target}" + (" and mark them read" if len(body) > 1 else "")])
    steps = [con, lst, rec, b.loop("$mails", "mail", body, "For each email", "Dictionary")]
    return style(core, rng), b.flow("Save attachments per email", steps)


def f_each_list_items(B, rng):
    b = B()
    things, one = rng.choice(THINGS)
    f = rng.choice(["a text file", "the list file", "a notepad file"])
    rd = b.s("ooa.file/read_text", "Read the list file", {"path": b.inp("list_path", f"The text file with one {one.lower()} per line.")}, {"text": "text"})
    sp = b.s("ooa.string/split", "Split it into lines", {"text": "$text", "delimiter": "\n", "remove_empty": True}, {"items": "names"})
    folder = rng.choice(FOLDERS)
    mk = b.s("ooa.folder/create", f"Create a folder for the {one.lower()}", {"path": "${base_folder}/${name}"})
    b.inp("base_folder", f"The {folder} folder.")
    core = pick(rng, [f"I have {f} with one {one.lower()} name per line; for each name create a folder in {folder}",
                      f"read the {things} names from {f} and make a folder for each one inside {folder}",
                      f"create one folder per {one.lower()} in {folder}, using the names listed in {f}"])
    return style(core, rng, [f"{f} me jo {things} ke naam hain, har ek ke liye {folder} me folder bana do"]), \
        b.flow(f"Folder per {one}", [rd, sp, b.loop("$names", "name", [mk], f"For each {one.lower()}", "String")])


# ------------------------------------------------------------------------------------------------------ conditions

def f_if_exists(B, rng):
    b = B()
    f = rng.choice(["today's sales report", "the bank statement", "the price file", "the export", "the payroll file",
                    "the stock file", "the daily orders file"])
    p = b.inp("file_path", f"Path of {f}.")
    cond = {"condition": "ooa.file/exists", "args": {"path": p}}
    kind = rng.choice(["move", "copy", "email", "read"])
    if kind == "move":
        dst = rng.choice(FOLDERS)
        then = [b.s("ooa.file/move", f"Move it to {dst}", {"source": p, "target": b.inp(fsnake(dst), f"The {dst} folder.")})]
        what = f"move it to {dst}"
    elif kind == "copy":
        dst = rng.choice(FOLDERS)
        then = [b.s("ooa.file/copy", f"Copy it to {dst}", {"source": p, "target": b.inp(fsnake(dst), f"The {dst} folder.")})]
        what = f"copy it into {dst}"
    elif kind == "email":
        who, var, d = person(rng)
        b.inp(var, d)
        then = [b.s("ooa.email/send", f"Email it to {who}", {"to": "$" + var, "subject": f.capitalize(), "attachments": [p]})]
        what = f"email it to {who}"
    else:
        then = [b.s("ooa.csv/read", "Read it", {"path": p}, {"table": "table", "row_count": "row_count"}),
                b.log("The file has ${row_count} rows", "Log the row count")]
        what = "read it and log how many rows it has"
    other = rng.choice(["log", "email", "stop"])
    if other == "log":
        else_ = [b.log(f"{f.capitalize()} is missing", "Log that the file is missing", "warning")]
        oth = rng.choice(["otherwise log that it's missing", "else write a warning to the log", "if not, log it"])
    elif other == "email":
        who, var, d = person(rng)
        b.inp(var, d)
        else_ = [b.s("ooa.email/send", f"Tell {who} it is missing", {"to": "$" + var, "subject": f"{f.capitalize()} is missing",
                                                                      "body": f"{f.capitalize()} was not found today."})]
        oth = rng.choice([f"otherwise email {who} that it's missing", f"if it isn't there, let {who} know by email"])
    else:
        else_ = [b.s("core/stop", "Stop with a message", {"message": f"{f.capitalize()} not found", "status": "failed"})]
        oth = rng.choice(["otherwise stop the bot", "else stop with an error"])
    node = b.if_(cond, then, "If the file exists", else_)
    core = pick(rng, [f"if {f} exists, {what}, {oth}", f"check whether {f} is there; if it is, {what}; {oth}",
                      f"when {f} exists {what}, {oth}"])
    return style(core, rng, [f"agar {f} hai to {what}, warna log kar do"]), b.flow("If file exists", [node])


def f_if_amount(B, rng):
    b = B()
    things, one = rng.choice(THINGS)
    col = rng.choice(COLS)
    lim = rng.choice([500, 1000, 2500, 5000, 10000, 25000, 50000, 100000])
    op_word, op = rng.choice([("over", ">"), ("more than", ">"), ("above", ">"), ("at least", ">="), ("below", "<"), ("under", "<")])
    rd = b.s("ooa.csv/read", f"Read the {things} file", {"path": b.inp("csv_path", f"The {things} CSV file.")}, {"table": "table"})
    if rng.random() < 0.5:
        opn = {">": "gt", ">=": "ge", "<": "lt"}[op]
        flt = b.s("ooa.datatable/filter", f"Keep rows where {col} is {op_word} {lim}",
                  {"table": "$table", "column": col, "operator": opn, "value": lim}, {"table": "matches", "count": "match_count"})
        out = rng.choice(["csv", "excel"])
        if out == "csv":
            wr = b.s("ooa.csv/write", "Save them to a new CSV", {"table": "$matches", "path": b.inp("output_path", "The new CSV file.")})
            tail = "save them to a new CSV"
        else:
            wr = b.s("ooa.datatable/write_to_file", "Save them to an Excel file", {"table": "$matches", "path": b.inp("output_path", "The new Excel file."),
                                                                                    "format": "xlsx"}, {"path": "saved_path"})
            tail = "save them as an Excel file"
        core = pick(rng, [f"from the {things} CSV keep only the rows where {col} is {op_word} {lim} and {tail}",
                          f"filter the {things} file to {col.lower()} {op_word} {lim}, then {tail}",
                          f"only {things} with {col.lower()} {op_word} {lim} please, {tail}"])
        return style(core, rng, [f"{things} csv me sirf {col.lower()} {lim} se {'zyada' if op != '<' else 'kam'} wale rakho aur {tail}"]), \
            b.flow(f"Filter {things}", [rd, flt, wr])
    rec = b.s("ooa.datatable/to_records", "Turn the rows into records", {"table": "$table"}, {"records": "rows"})
    who, var, d = person(rng)
    b.inp(var, d)
    send = b.s("ooa.email/send", f"Email {who} about this {one.lower()}",
               {"to": "$" + var, "subject": f"{one} " + "${row.ID}" + f" {col.lower()} alert",
                "body": f"{one} " + "${row.ID}" + f" has {col.lower()} " + "${row." + col + "}."})
    cond = f"$row.{col} {op} {lim}"
    node = b.if_(cond, [send], f"Only if {col} is {op_word} {lim}")
    core = pick(rng, [f"go through the {things} CSV and, only if the {col.lower()} is {op_word} {lim}, email {who} about that {one.lower()}",
                      f"for each {one.lower()} in the CSV, if {col.lower()} {op} {lim} send {who} an email",
                      f"check every row of the {things} file and email {who} only for the ones with {col.lower()} {op_word} {lim}"])
    return style(core, rng, [f"har {one.lower()} ke liye, agar {col.lower()} {lim} se {'zyada' if op != '<' else 'kam'} hai to {who} ko mail karo"]), \
        b.flow(f"{one} {col.lower()} alert", [rd, rec, b.loop("$rows", "row", [node], f"For each {one.lower()}", "Dictionary")])


def f_if_ext(B, rng):
    b = B()
    folder, dst = rng.sample(FOLDERS, 2)
    (e1, l1, p1), (e2, l2, p2) = rng.sample(FTYPES, 2)
    lst = b.s("ooa.file/list", "List the files", {"folder": b.inp("folder", f"The {folder} folder.")}, {"files": "files"})
    d1, d2 = b.inp(snake(l1) + "_folder", f"Folder for the {p1}."), b.inp(snake(l2) + "_folder", f"Folder for the {p2}.")
    mv1 = b.s("ooa.file/move", f"Move it to the {l1} folder", {"source": "$file", "target": d1})
    mv2 = b.s("ooa.file/move", f"Move it to the {l2} folder", {"source": "$file", "target": d2})
    inner = b.if_({"condition": "ooa.file/has_extension", "args": {"path": "$file", "extensions": e2}}, [mv2], f"If it is a .{e2} file")
    node = b.if_({"condition": "ooa.file/has_extension", "args": {"path": "$file", "extensions": e1}}, [mv1], f"If it is a .{e1} file", [inner])
    core = pick(rng, [f"sort the files in {folder}: {p1} go to one folder and {p2} to another",
                      f"for each file in {folder}, if it's a {l1} file move it to the {l1} folder, if it's {l2} move it to the {l2} folder",
                      f"go through {folder} and move .{e1} files into their folder and .{e2} files into theirs"])
    return style(core, rng, [f"{folder} me {e1} files ek folder me aur {e2} files dusre folder me daal do"]), \
        b.flow("Sort files by type", [lst, b.loop("$files", "file", [node], "For each file", "String")])


def f_if_empty(B, rng):
    b = B()
    things, one = rng.choice(THINGS)
    who, var, d = person(rng)
    b.inp(var, d)
    rd = b.s("ooa.csv/read", f"Read the {things} file", {"path": b.inp("csv_path", f"The {things} CSV file.")}, {"table": "table"})
    flt = b.s("ooa.datatable/filter", f"Keep the overdue {things}", {"table": "$table", "column": "Status", "operator": "eq", "value": "Overdue"},
              {"table": "overdue", "count": "overdue_count"})
    wr = b.s("ooa.csv/write", "Save them to a CSV", {"table": "$overdue", "path": b.inp("output_path", "The CSV file to send.")})
    send = b.s("ooa.email/send", f"Email the list to {who}", {"to": "$" + var, "subject": f"Overdue {things}", "attachments": ["$output_path"]})
    nothing = b.s("ooa.email/send", f"Tell {who} there are none", {"to": "$" + var, "subject": f"No overdue {things} today"})
    node = b.if_("$overdue_count > 0", [wr, send], f"If there are overdue {things}", [nothing])
    core = pick(rng, [f"find the overdue {things} in the CSV; if there are any, save them to a file and email it to {who}, otherwise email {who} that there are none",
                      f"check the {things} file for status Overdue. if any, send {who} the list as an attachment, else just say nothing is overdue",
                      f"email {who} the overdue {things} as a CSV, or a 'none today' email when there aren't any"])
    return style(core, rng, [f"overdue {things} nikalo, agar hai to csv bana ke {who} ko bhejo, nahi to bolo kuch nahi hai"]), \
        b.flow(f"Overdue {things}", [rd, flt, node])


def f_if_folder_empty(B, rng):
    b = B()
    folder = rng.choice(FOLDERS)
    fv = b.inp("folder", f"The {folder} folder.")
    z = b.s("ooa.file/zip", "Zip the folder", {"source": fv, "target": b.inp("zip_path", "Where to save the zip.")}, {"path": "zip_file"})
    who, var, d = person(rng)
    b.inp(var, d)
    send = b.s("ooa.email/send", f"Email the zip to {who}", {"to": "$" + var, "subject": f"{folder} files", "attachments": ["$zip_file"]})
    node = b.if_({"condition": "ooa.folder/is_empty", "args": {"path": fv}}, [b.log(f"{folder} is empty, nothing to send", "Log that there is nothing")],
                 "If the folder is empty", [z, send])
    core = pick(rng, [f"if the {folder} folder is empty just log it, otherwise zip it and email the zip to {who}",
                      f"zip {folder} and send it to {who}, but only when the folder has something in it (log it if it's empty)"])
    return style(core, rng), b.flow("Zip and send if not empty", [node])


def f_if_text(B, rng):
    b = B()
    word = rng.choice(["overdue", "urgent", "final notice", "cancelled", "approved", "rejected", "past due", "legal action"])
    src = rng.choice(["pdf", "word", "txt"])
    if src == "pdf":
        rd = [b.s("ooa.pdf/extract_text", "Read the PDF text", {"path": b.inp("pdf_path", "The PDF file.")}, {"text": "text"})]
        what = rng.choice(["the PDF", "the letter PDF", "the statement PDF"])
    elif src == "word":
        rd = [b.s("ooa.word/open", "Open the document", {"path": b.inp("document_path", "The Word document.")}),
              b.s("ooa.word/read_text", "Read its text", {}, {"text": "text"}), b.s("ooa.word/close", "Close the document", {})]
        what = rng.choice(["the Word document", "the contract docx"])
    else:
        rd = [b.s("ooa.file/read_text", "Read the text file", {"path": b.inp("file_path", "The text file.")}, {"text": "text"})]
        what = rng.choice(["the text file", "the log file"])
    dst = rng.choice(FOLDERS)
    p = list(rd[0]["args"].values())[0]
    mv = b.s("ooa.file/move", f"Move it to {dst}", {"source": p, "target": b.inp(fsnake(dst), f"The {dst} folder.")})
    node = b.if_(f"contains(lower($text), '{word}')", [mv], f"If it mentions {word}")
    core = pick(rng, [f"read {what} and if it says '{word}' move it to {dst}", f"move {what} to {dst} only if the text contains {word}",
                      f"check {what} for the words {word}; when found, move the file to the {dst} folder"])
    return style(core, rng), b.flow(f"Move if {word}", rd + [node])


# ----------------------------------------------------------------------------------------------------------- web

def _open(b, desc="Open the website"):
    return b.s("ooa.browser/open", desc, {"url": b.inp("url", "Address of the website.")})


def f_web_search(B, rng):
    b = B()
    site = rng.choice(SITES)
    term = rng.choice(["the order number", "our company name", "the invoice number", "the tracking number", "the part number",
                       "the employee id", "the PAN number", "the GST number"])
    steps = [_open(b, f"Open {site}"),
             b.s("ooa.browser/type_text", f"Type {term} in the search box", {"text": b.inp("search_text", f"{term.capitalize()} to search for."),
                                                                               "selector": "input[type=search]", "press_enter": True})]
    end = rng.choice(["shot", "text", "table"])
    if end == "shot":
        steps.append(b.s("ooa.browser/screenshot", "Take a screenshot of the results", {"path": b.inp("screenshot_path", "Where to save the screenshot.")},
                         {"path": "shot"}))
        tail = "take a screenshot of the result"
    elif end == "text":
        steps.append(b.s("ooa.browser/get_text", "Read the status text", {"selector": ".status"}, {"text": "status"}))
        steps.append(b.log("Status: ${status}", "Log the status"))
        tail = "read the status shown and log it"
    else:
        steps.append(b.s("ooa.browser/extract_table", "Read the results table", {}, {"table": "results"}))
        steps.append(b.s("ooa.csv/write", "Save the results to CSV", {"table": "$results", "path": b.inp("csv_path", "The CSV file to write.")}))
        tail = "save the results table to a CSV"
    steps.append(b.s("ooa.browser/close", "Close the browser", {}))
    core = pick(rng, [f"open {site}, search for {term}, {tail}, then close the browser",
                      f"go to {site}, type {term} into the search box and press enter, {tail}",
                      f"look up {term} on {site} and {tail}"])
    return style(core, rng, [f"{site} kholo, {term} search karo aur {tail}"]), b.flow("Web search", steps)


def f_web_table(B, rng):
    b = B()
    site = rng.choice(SITES)
    what = rng.choice(["rates table", "price table", "results table", "tender list", "shipment table", "holiday table"])
    steps = [_open(b, f"Open {site}"), b.s("ooa.browser/extract_table", f"Read the {what}", {}, {"table": "table"})]
    out = rng.choice(["csv", "excel", "gsheet"])
    if out == "csv":
        steps.append(b.s("ooa.csv/write", "Write it to a CSV file", {"table": "$table", "path": b.inp("csv_path", "The CSV file to write.")}))
        tail = rng.choice(["save it as a CSV", "write it to a csv file"])
    elif out == "excel":
        steps += [b.s("ooa.excel/create_workbook", "Create the workbook", {"path": b.inp("workbook_path", "The new Excel file.")}),
                  b.s("ooa.excel/write_table", "Write the table to the sheet", {"table": "$table"}),
                  b.s("ooa.excel/save", "Save the workbook", {}), b.s("ooa.excel/close", "Close the workbook", {})]
        tail = rng.choice(["put it into a new Excel workbook", "save it in a fresh xlsx"])
    else:
        steps.append(b.s("ooa.google/sheets_write_data_table", "Write it to the Google Sheet",
                         {"access_token": b.inp("access_token", "Google access token.", "Credential"),
                          "spreadsheet_id": b.inp("spreadsheet_id", "Id of the Google Sheet."), "table": "$table", "clear_sheet_first": True}))
        tail = rng.choice(["write it into our Google Sheet", "paste it into the Google Sheet (replace what's there)"])
    steps.append(b.s("ooa.browser/close", "Close the browser", {}))
    core = pick(rng, [f"grab the {what} from {site} and {tail}", f"open {site}, copy the {what} and {tail}",
                      f"scrape the {what} on {site}, {tail}"])
    return style(core, rng, [f"{site} se {what} utha ke {tail}"]), b.flow("Web table export", steps)


def f_web_download(B, rng):
    b = B()
    site = rng.choice(SITES)
    what = rng.choice(["monthly statement", "latest invoice", "GST report", "payslip", "price list", "delivery report"])
    folder = rng.choice(FOLDERS)
    steps = [_open(b, f"Open {site}")]
    login = rng.random() < 0.5
    if login:
        steps += [b.s("ooa.browser/type_text", "Type the user name", {"text": b.inp("username", "Login user name."), "selector": "#username"}),
                  b.s("ooa.browser/type_text", "Type the password", {"text": b.inp("password", "Login password.", "Credential"), "selector": "#password"}),
                  b.s("ooa.browser/click", "Click Sign in", {"selector": "button[type=submit]"})]
    steps.append(b.s("ooa.browser/click", f"Click the {what} link", {"selector": f"text={what.title()}"}))
    steps.append(b.s("ooa.browser/download", f"Download the {what}", {"folder": b.inp(fsnake(folder), f"The {folder} folder.")}, {"path": "downloaded"}))
    steps.append(b.s("ooa.browser/close", "Close the browser", {}))
    core = pick(rng, [f"{'log in to' if login else 'open'} {site}, click on the {what} and download it to {folder}",
                      f"download the {what} from {site}{' (needs login)' if login else ''} into the {folder} folder",
                      f"go to {site}{', sign in' if login else ''}, find the {what} link and save the file in {folder}"])
    return style(core, rng, [f"{site} se {what} download karke {folder} me rakh do"]), b.flow("Download from website", steps)


def f_web_screenshot(B, rng):
    b = B()
    site = rng.choice(SITES)
    steps = [_open(b, f"Open {site}"), b.s("ooa.browser/screenshot", "Take a full-page screenshot",
                                           {"path": b.inp("screenshot_path", "Where to save the screenshot."), "full_page": True}, {"path": "shot"})]
    mail = rng.random() < 0.4
    if mail:
        who, var, d = person(rng)
        b.inp(var, d)
        steps.append(b.s("ooa.email/send", f"Email the screenshot to {who}", {"to": "$" + var, "subject": f"Screenshot of {site}", "attachments": ["$shot"]}))
    steps.append(b.s("ooa.browser/close", "Close the browser", {}))
    core = pick(rng, [f"take a full page screenshot of {site}" + (f" and email it to {who}" if mail else ""),
                      f"capture {site} as an image" + (f", send it to {who}" if mail else ""),
                      f"screenshot {site} every morning" + (f" and mail the picture to {who}" if mail else "")])
    return style(core, rng), b.flow("Website screenshot", steps)


# ---------------------------------------------------------------------------------------------- produce + email

def _sender(b, rng, to_var, subject, attach, body=None):
    """Plain mail server, Outlook (Microsoft 365) or Gmail, as the request names it."""
    kind = rng.choice(["smtp", "smtp", "smtp", "outlook", "gmail"])
    if kind == "smtp":
        a = {"to": to_var, "subject": subject, "attachments": attach}
        if body:
            a["body"] = body
        return b.s("ooa.email/send", "Send the file by email", a), ""
    if kind == "outlook":
        a = {"access_token": b.inp("ms_token", "Microsoft 365 access token.", "Credential"), "to": to_var, "subject": subject, "attachments": attach}
        if body:
            a["body"] = body
        return b.s("ooa.microsoft365/send_mail", "Send the file from Outlook", a), rng.choice([" from Outlook", " via outlook", " using our Office 365 mail"])
    a = {"access_token": b.inp("google_token", "Google access token.", "Credential"), "to": to_var, "subject": subject, "attachments": attach}
    if body:
        a["body"] = body
    return b.s("ooa.google/gmail_send", "Send the file from Gmail", a), rng.choice([" from Gmail", " through gmail", " with our Gmail account"])


def f_then_email(B, rng):
    b = B()
    who, var, d = person(rng)
    b.inp(var, d)
    folder = rng.choice(FOLDERS)
    attach_var = None
    kind = rng.choice(["zip", "merge", "csv", "excel_pdf", "word_pdf", "download", "filter", "zip_pattern", "saveas"])
    if kind == "zip":
        steps = [b.s("ooa.file/zip", f"Zip the {folder} folder", {"source": b.inp("folder", f"The {folder} folder."),
                                                                  "target": b.inp("zip_path", "Where to save the zip.")}, {"path": "result_file"})]
        made = rng.choice([f"zip the {folder} folder", f"compress {folder} into a zip", f"zip up everything in {folder}"])
        it = "the zip"
    elif kind == "zip_pattern":
        ext, label, plural = rng.choice(FTYPES[:5])
        steps = [b.s("ooa.file/zip", f"Zip the {plural}", {"source": b.inp("folder", f"The {folder} folder."), "pattern": f"*.{ext}",
                                                           "target": b.inp("zip_path", "Where to save the zip.")}, {"path": "result_file"})]
        made = rng.choice([f"zip all the {plural} in {folder}", f"put the {label} files from {folder} in a zip"])
        it = "the zip"
    elif kind == "merge":
        steps = [b.s("ooa.pdf/merge", "Merge the PDFs into one", {"folder": b.inp("folder", f"The {folder} folder."), "pattern": "*.pdf",
                                                                  "target": b.inp("merged_path", "The merged PDF file.")})]
        made = rng.choice([f"merge all PDFs in {folder} into one file", f"combine the {folder} PDFs into a single pdf"])
        it = "the merged PDF"
        attach_var = "$merged_path"
    elif kind == "csv":
        steps = [b.s("ooa.csv/read", "Read the CSV", {"path": b.inp("csv_path", "The source CSV.")}, {"table": "table"}),
                 b.s("ooa.datatable/sort", "Sort by date", {"table": "$table", "column": "Date", "descending": True}, {"table": "sorted"}),
                 b.s("ooa.datatable/write_to_file", "Save it as a new file", {"table": "$sorted", "path": b.inp("output_path", "The sorted file.")},
                     {"path": "result_file"})]
        made = rng.choice(["sort the CSV by date (newest first) and save it as a new file", "sort the csv newest first into a new file"])
        it = "that file"
    elif kind == "excel_pdf":
        steps = [b.s("ooa.excel_desktop/export_pdf", "Export the workbook to PDF", {"path": b.inp("workbook_path", "The Excel workbook."),
                                                                                   "target": b.inp("pdf_path", "The PDF file to create.")})]
        made = rng.choice(["export the Excel report to PDF", "convert the workbook into a PDF", "save the excel sheet as pdf"])
        it = "the PDF"
        attach_var = "$pdf_path"
    elif kind == "word_pdf":
        steps = [b.s("ooa.word/open", "Open the document", {"path": b.inp("document_path", "The Word document.")}),
                 b.s("ooa.word/save_as", "Save it as PDF", {"path": b.inp("pdf_path", "The PDF file to create."), "format": "pdf"}),
                 b.s("ooa.word/close", "Close the document", {})]
        made = rng.choice(["turn the Word letter into a PDF", "save the docx as a pdf", "convert the Word document to PDF"])
        it = "the PDF"
        attach_var = "$pdf_path"
    elif kind == "download":
        site = rng.choice(SITES)
        steps = [b.s("ooa.browser/download_url", "Download the file", {"url": b.inp("file_url", "Address of the file."),
                                                                       "save_to": b.inp("save_path", "Where to save it.")}, {"path": "result_file"})]
        made = rng.choice([f"download the report file from {site}", f"download the file at the link from {site}"])
        it = "it"
    elif kind == "filter":
        things, one = rng.choice(THINGS)
        steps = [b.s("ooa.csv/read", f"Read the {things} CSV", {"path": b.inp("csv_path", f"The {things} CSV.")}, {"table": "table"}),
                 b.s("ooa.datatable/remove_duplicates", "Remove duplicate rows", {"table": "$table"}, {"table": "clean"}),
                 b.s("ooa.csv/write", "Write the clean CSV", {"table": "$clean", "path": b.inp("output_path", "The cleaned CSV.")})]
        made = rng.choice([f"remove duplicates from the {things} CSV into a clean copy", f"dedupe the {things} csv and save it"])
        it = "the clean file"
        attach_var = "$output_path"
    else:
        steps = [b.s("ooa.excel/open", "Open the workbook", {"path": b.inp("workbook_path", "The Excel workbook.")}),
                 b.s("ooa.excel/set_cell", "Write today's date in A1", {"cell": "A1", "value": "${sys.date}"}),
                 b.s("ooa.excel/save_as", "Save a copy with a new name", {"path": b.inp("copy_path", "The new copy.")}),
                 b.s("ooa.excel/close", "Close the workbook", {})]
        made = rng.choice(["put today's date in A1 of the workbook and save a copy under a new name", "stamp the date in cell A1 and save as a new file"])
        it = "the copy"
        attach_var = "$copy_path"
    att = attach_var or "$result_file"
    send, via = _sender(b, rng, "$" + var, "Here is " + it, [att])
    send["description"] = f"Email {it} to {who}"
    steps.append(send)
    core = pick(rng, [f"{made} and email {it} to {who}{via}", f"{made}, then send {it} to {who}{via}",
                      f"{made} and then mail {it} to {who} as an attachment{via}", f"{made}; after that email it to {who}{via}"])
    return style(core, rng, [f"{made} aur phir {who} ko email kar do{via}"]), b.flow("Make a file and email it", steps)


# --------------------------------------------------------------------------------------------------------- OCR

def f_ocr(B, rng):
    b = B()
    scanned = rng.random() < 0.6
    what = rng.choice(["invoice", "bill", "receipt", "delivery note", "contract", "ID card", "statement", "purchase order"])
    steps = []
    if scanned:
        src = rng.choice([f"scanned {what} PDF", f"{what} scan (PDF)", f"image-only {what} PDF", f"photographed {what} saved as pdf"])
        rd = b.s("ooa.pdf/read_text_ocr", "Read the scanned PDF with OCR", {"path": b.inp("pdf_path", f"The scanned {what} PDF.")}, {"text": "text"})
    elif rng.random() < 0.5:
        src = rng.choice([f"{what} PDF", f"digital {what} pdf", f"{what} PDF we got by email"])
        rd = b.s("ooa.pdf/extract_text", "Get the text out of the PDF", {"path": b.inp("pdf_path", f"The {what} PDF.")}, {"text": "text"})
    else:
        src = rng.choice([f"photo of the {what} (jpg)", f"{what} image", f"picture of the {what}"])
        rd = b.s("ooa.ocr/read_image", "Read the text in the picture", {"path": b.inp("image_path", f"The {what} image.")}, {"text": "text"})
    steps.append(rd)
    end = rng.choice(["save", "extract", "find"])
    if end == "save":
        steps.append(b.s("ooa.file/write_text", "Save the text to a file", {"path": b.inp("text_path", "The text file to write."), "text": "$text"}))
        tail = rng.choice(["save the text in a .txt file", "write what it says into a text file"])
    elif end == "extract":
        steps.append(b.s("ooa.genai/extract", "Pull out the number, date and total", {"text": "$text", "fields": {
            "number": f"the {what} number", "date": f"the {what} date", "total": "the total amount"}}, {"fields": "fields"}))
        steps.append(b.log("Number ${fields.number}, total ${fields.total}", "Log what was found"))
        tail = rng.choice([f"pick out the {what} number, date and total and log them", "get the number, date and total amount from it"])
    else:
        steps.append(b.s("ooa.string/extract_text", "Take the text after 'Total'", {"text": "$text", "after": "Total", "before": "\n"}, {"result": "total"}))
        steps.append(b.log("Total: ${total}", "Log the total"))
        tail = rng.choice(["find the amount after the word Total and log it", "grab the value after 'Total' and write it to the log"])
    core = pick(rng, [f"read the {src} and {tail}", f"get the text from the {src}, then {tail}", f"take the {src}, read it, {tail}"])
    return style(core, rng), b.flow(f"Read {what}", steps)


def f_office_pdf(B, rng):
    b = B()
    x = rng.choice(["excel", "excel_sheet", "word", "gdoc", "folder_excel"])
    if x == "excel":
        steps = [b.s("ooa.excel_desktop/export_pdf", "Export the workbook to PDF", {"path": b.inp("workbook_path", "The Excel workbook."),
                                                                                   "target": b.inp("pdf_path", "The PDF to create.")})]
        core = pick(rng, ["export the sales workbook to PDF", "convert our Excel report to a pdf", "save the xlsx as a PDF file",
                          "make a PDF out of the budget spreadsheet"])
    elif x == "excel_sheet":
        sh = rng.choice(["Summary", "Invoice", "Dashboard", "Totals", "Report"])
        steps = [b.s("ooa.excel_desktop/export_pdf", f"Export the {sh} sheet to PDF", {"path": b.inp("workbook_path", "The Excel workbook."),
                                                                                       "target": b.inp("pdf_path", "The PDF to create."), "sheet": sh})]
        core = pick(rng, [f"export only the {sh} sheet of the workbook to pdf", f"turn the {sh} tab of the excel file into a PDF"])
    elif x == "word":
        steps = [b.s("ooa.word/open", "Open the document", {"path": b.inp("document_path", "The Word document.")}),
                 b.s("ooa.word/save_as", "Save it as PDF", {"path": b.inp("pdf_path", "The PDF to create."), "format": "pdf"}),
                 b.s("ooa.word/close", "Close the document", {})]
        core = pick(rng, ["convert the offer letter docx to PDF", "save the Word report as PDF", "turn the Word file into a pdf"])
    elif x == "gdoc":
        steps = [b.s("ooa.google/drive_download_export", "Export the Google file as PDF", {"access_token": b.inp("access_token", "Google access token.", "Credential"),
                                                                                         "file_id": b.inp("file_id", "Id of the Google Doc or Sheet."),
                                                                                         "local_path": b.inp("pdf_path", "Where to save the PDF."), "format": "pdf"},
                     {"path": "pdf_file"})]
        core = pick(rng, ["download our Google Doc as a PDF", "export the Google Sheet to pdf on my PC", "save the google doc as pdf locally"])
    else:
        folder = rng.choice(FOLDERS)
        lst = b.s("ooa.file/list", "List the workbooks", {"folder": b.inp("folder", f"The {folder} folder."), "pattern": "*.xlsx"}, {"files": "files"})
        nm = b.s("ooa.file/get_name", "Get the workbook name", {"path": "$file"}, {"name_without_extension": "base_name"})
        ex = b.s("ooa.excel_desktop/export_pdf", "Export it to PDF", {"path": "$file", "target": "${folder}/${base_name}.pdf"})
        steps = [lst, b.loop("$files", "file", [nm, ex], "For each workbook", "String")]
        core = pick(rng, [f"convert every Excel file in {folder} to PDF with the same name", f"for each xlsx in {folder}, export a PDF copy next to it"])
    return style(core, rng, ["excel/word file ko pdf me convert kar do"] if x in ("excel", "word") else None), b.flow("Export to PDF", steps)


# -------------------------------------------------------------------------------------------- named sources

def f_named_source(B, rng):
    b = B()
    things, one = rng.choice(THINGS)
    col = rng.choice(COLS)
    src = rng.choice(["gsheet", "excel", "csv"])
    if src == "gsheet":
        steps = [b.s("ooa.google/sheets_read", f"Read the {things} Google Sheet", {"access_token": b.inp("access_token", "Google access token.", "Credential"),
                                                                                  "spreadsheet_id": b.inp("spreadsheet_id", f"Id of the {things} Google Sheet.")},
                     {"table": "table"})]
        name = rng.choice([f"the {things} Google Sheet", f"our Google Sheets tracker of {things}", f"the {things} gsheet"])
    elif src == "excel":
        steps = [b.s("ooa.excel/open", f"Open the {things} workbook", {"path": b.inp("workbook_path", f"The {things} Excel file.")}),
                 b.s("ooa.excel/get_all_cells", "Read the sheet as a table", {}, {"table": "table"}),
                 b.s("ooa.excel/close", "Close the workbook", {})]
        name = rng.choice([f"the {things} Excel file", f"{things}.xlsx", f"the {things} workbook on the shared drive"])
    else:
        steps = [b.s("ooa.csv/read", f"Read the {things} CSV", {"path": b.inp("csv_path", f"The {things} CSV file.")}, {"table": "table"})]
        name = rng.choice([f"the {things} CSV", f"{things}.csv", f"the exported {things} csv"])
    op = rng.choice(["sum", "count", "sort", "top"])
    if op == "sum":
        steps += [b.s("ooa.datatable/sum_column", f"Add up the {col} column", {"table": "$table", "column": col}, {"total": "total"}),
                  b.log(f"Total {col.lower()}: " + "${total}", "Log the total")]
        tail = rng.choice([f"add up the {col.lower()} column and log the total", f"give me the total of {col}"])
    elif op == "count":
        steps += [b.s("ooa.datatable/row_count", "Count the rows", {"table": "$table"}, {"count": "rows"}),
                  b.log(f"{things.capitalize()}: " + "${rows}", "Log the count")]
        tail = rng.choice([f"count how many {things} there are", "log the number of rows"])
    elif op == "sort":
        steps += [b.s("ooa.datatable/sort", f"Sort by {col}", {"table": "$table", "column": col, "descending": True}, {"table": "sorted"}),
                  b.s("ooa.csv/write", "Write the sorted CSV", {"table": "$sorted", "path": b.inp("output_path", "The sorted CSV file.")})]
        tail = rng.choice([f"sort it by {col.lower()} (highest first) and save as CSV", f"sort by {col} descending and write a csv"])
    else:
        steps += [b.s("ooa.datatable/filter", f"Keep rows with {col} above zero", {"table": "$table", "column": col, "operator": "gt", "value": 0},
                      {"table": "nonzero", "count": "n"}),
                  b.s("ooa.datatable/write_to_file", "Save them as Excel", {"table": "$nonzero", "path": b.inp("output_path", "The Excel file to write."), "format": "xlsx"},
                      {"path": "saved"})]
        tail = rng.choice([f"keep the {things} with {col.lower()} above zero and save them to an Excel file", f"filter {col} > 0 and write it out as xlsx"])
    core = pick(rng, [f"from {name}, {tail}", f"read {name} and {tail}", f"open {name}, {tail}"])
    return style(core, rng, [f"{name} padh ke {tail}"]), b.flow(f"{things.capitalize()} from {src}", steps)


FAMILIES = {
    "rename_one": (f_rename_one, 5), "rename_ext": (f_rename_ext, 7), "rename_prefix": (f_rename_prefix, 4),
    "move_type": (f_move_type, 4), "copy": (f_copy, 4), "delete": (f_delete, 3), "zip": (f_zip, 3), "unzip": (f_unzip, 3),
    "backup_then_delete": (f_backup_then_delete, 2),
    "each_csv_rows": (f_each_csv_rows, 5), "each_customer_email": (f_each_customer_email, 7), "each_file": (f_each_file_action, 6),
    "each_email": (f_each_email, 4), "each_list_item": (f_each_list_items, 2),
    "if_exists": (f_if_exists, 6), "if_amount": (f_if_amount, 6), "if_ext": (f_if_ext, 3), "if_empty": (f_if_empty, 3),
    "if_folder_empty": (f_if_folder_empty, 2), "if_text": (f_if_text, 3),
    "web_search": (f_web_search, 4), "web_table": (f_web_table, 4), "web_download": (f_web_download, 4), "web_screenshot": (f_web_screenshot, 3),
    "then_email": (f_then_email, 12), "ocr": (f_ocr, 6), "office_pdf": (f_office_pdf, 5), "named_source": (f_named_source, 9)}


def main():
    cat_path, schema, out = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
    rng = random.Random(int(sys.argv[5]) if len(sys.argv) > 5 else 11)
    checker, guard = Checker(cat_path, schema), LeakGuard()
    rows, made, why = run_families(FAMILIES, n, rng, checker, guard, "targeted")
    write_jsonl(out, rows)
    print(f"{len(rows)} records | checker pass {checker.ok}/{checker.n} | families {dict(made)} | skipped {dict(why)}")


if __name__ == "__main__":
    main()
