"""Beginner training data: requests written the way people with no IT background talk, with the SAME flows as the
clear requests they come from, plus friendly clarifying questions when a beginner leaves out a detail the bot cannot
know.

For a broad sample of existing flow records (smart/data long, business2, targeted, the short clear flows of clarify, and
the chat rows with flow answers in ft/clean, turned into neutral flow records), it writes 3-5 beginner requests per flow:
  vague          the request wrapped in a goal ("sort out the invoices - ..."), vague / business words
  business       tool phrases replaced by the business verbs beginners use (PHRASEBOOK: chase / remind -> email,
                 tidy / sort out -> move / rename, tally / reconcile -> compare / filter, ...)
  casual         lower case, short forms (pls, u, thx), no punctuation
  typo           realistic slips: keyboard neighbours, dropped / doubled / swapped letters, phonetic misspellings; an
                 action-critical noun (pdf, excel, email, folder ...) gets at most one slip
  hinglish       Hindi-English mix: verbs moved to the end of each clause and said in Hindi, postpositions (se, me, ko),
                 connectors (aur, phir, agar ... to, nahi to), time words (roz subah, har hafte); HINGLISH_* tables
  indian_english Kindly / do the needful / itself / only / I am having
  voice          dictated speech: fillers (um, like, you know), no punctuation, "dot csv", "at" for @
  story          first person, the steps told as what I do ("First I open ..., then I ...") and a question at the end
Each is a "rewrite" record (ooa_format.render_chat): the answer first restates the request in tool words
("Clear request: <clear>", clear = ooa_format.clear_request(flow)), then the usual plan and flow.

Clarify records (beginner tone): a request whose key detail was taken out and replaced by a reference only the user
can resolve ("the usual folder", "the usual people", "the big ones", "our system", "that file"); the answer is one
short, friendly question about that detail (SLOTS: folder, recipient, condition, app, file). A detail that is merely
not given (no folder named at all) stays a flow input, as in the other data; only the vague reference is asked about.

Every flow passes the repo checker (gen_targeted.Checker: repair + ooa_reward.problems == []). Requests are
deduplicated (normalised text) and nothing is close to a test prompt (any eval/*.jsonl prompt: difflib ratio over 0.8
or word overlap over 0.6, beginner_score.word_overlap).

Writes NEUTRAL records, one per line:
  {"kind": "rewrite", "request", "clear", "flow", "style": [...], "source": "beginner:<style>:<origin>"}
  {"kind": "clarify", "request", "question", "missing": <slot>, "style": [..., "missing_detail"],
   "source": "beginner_clarify:<slot>"}
and OUT's stem + "_report.json" with the counts per style, kind and origin.

Usage: python gen_beginner.py CATALOG.jsonl SCHEMA.json OUT.jsonl --smart SMART_DATA_DIR --chat CHAT_DIR
                              [--flows 2300] [--clarify 1800] [--seed 41]
"""
import argparse
import difflib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))
import ooa_format as F                      # noqa: E402
from gen_targeted import Checker, norm     # noqa: E402
import beginner_score as BS                 # noqa: E402

STYLES = ("vague", "business", "casual", "typo", "hinglish", "indian_english", "voice", "story")
STYLE_WEIGHTS = {"vague": 1.0, "business": 1.3, "casual": 1.0, "typo": 1.0, "hinglish": 1.3, "indian_english": 0.8,
                 "voice": 1.0, "story": 1.0}
# how many flows of each origin (share of --flows); the rest of each file is not used
ORIGIN_SHARE = {"long": 0.17, "business2": 0.15, "targeted": 0.24, "clarify_counter": 0.04, "chains": 0.08,
                "combos": 0.11, "tools": 0.10, "scenarios": 0.05, "train_flows": 0.02, "synth_pattern": 0.04}
CHAT_FILES = ("chains_train.jsonl", "combos_train.jsonl", "tools_train.jsonl", "scenarios_train.jsonl",
              "train_flows.jsonl", "synth_pattern.jsonl")
MAX_REQUEST = 900

# ------------------------------------------------------------------------------------------------ cleaning
_LEADS = re.compile(r"^(?:create an ooa flow for this:|build an automation that does this\.?|build a flow that will|"
                    r"i would like an automation that will|create an automation to|we need a process that will|"
                    r"i need a flow to|i need a bot to|could you|can you|please|kindly|pls)\s*[:,]?\s*", re.I)
PROTECT = re.compile(r"(https?://\S+|[\w.+-]+@[\w-]+\.[\w.]+|[A-Za-z]:\\[^\s,;]*|#[\w-]+|\$\w+\$?|'[^']*'|\"[^\"]*\"|"
                     r"\{\{[^}]*\}\}|\b[\w-]+\.(?:csv|xlsx|xls|pdf|docx|doc|txt|json|xml|zip|png|jpg|jpeg|msg)\b)")
CRITICAL = {"pdf", "pdfs", "csv", "excel", "xlsx", "email", "emails", "mail", "folder", "zip", "unzip", "word", "slack",
            "teams", "outlook", "gmail", "jira", "salesforce", "servicenow", "sheet", "workbook", "website", "browser",
            "invoice", "invoices", "attachment", "attachments", "ticket", "tickets", "table", "rows", "column", "file",
            "files", "folders", "zendesk", "asana", "sharepoint", "drive", "database", "report", "screenshot", "print",
            "download", "upload", "rename", "delete", "move", "copy", "merge", "filter", "sort", "duplicates"}


def clean_request(r):
    r = " ".join(str(r).split())
    for _ in range(2):
        r = _LEADS.sub("", r)
    return r.strip().rstrip(".") if r else r


def _lower_keep(text):
    """Lower case except the protected tokens (file names, paths, addresses, quoted text)."""
    out, last = [], 0
    for m in PROTECT.finditer(text):
        out.append(text[last:m.start()].lower())
        out.append(m.group(0))
        last = m.end()
    out.append(text[last:].lower())
    return "".join(out)


def _protected_spans(text):
    return [(m.start(), m.end()) for m in PROTECT.finditer(text)]


def _sub_outside(pattern, repl, text, count=0, flags=re.I):
    """re.sub that never touches protected tokens. repl: str or callable(match)."""
    spans = _protected_spans(text)
    n = [0]

    def f(m):
        if any(a < m.end() and m.start() < b for a, b in spans) or (count and n[0] >= count):
            return m.group(0)
        n[0] += 1
        return repl(m) if callable(repl) else m.expand(repl)
    return re.sub(pattern, f, text, flags=flags), n[0]


# ------------------------------------------------------------------------------------------------ typos
KEYBOARD = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij", "i": "uok", "o": "ipl", "p": "ol",
    "a": "qsz", "s": "adwx", "d": "sfe", "f": "dgr", "g": "fht", "h": "gjy", "j": "hku", "k": "jli", "l": "kp",
    "z": "xa", "x": "zcs", "c": "xvd", "v": "cbf", "b": "vng", "n": "bmh", "m": "nj"}
# common phonetic / learned misspellings (never the exact misspellings of the beginner test set)
PHONETIC = {
    "receive": ["recieve", "receve"], "separate": ["seperate", "separete"], "address": ["adress", "addres"],
    "attachment": ["attachement", "attachmnt"], "attachments": ["attachements", "atachments"],
    "excel": ["exel", "excell"], "folder": ["fodler", "folde"], "schedule": ["shedule", "schedual"],
    "delete": ["delet", "dlete"], "duplicate": ["duplicat", "dublicate"], "duplicates": ["dublicates", "duplicats"],
    "invoice": ["invoce", "invioce"], "invoices": ["invoces", "invioces"], "message": ["mesage", "massage"],
    "column": ["colum", "coloumn"], "every": ["evey", "evry"], "which": ["wich", "whcih"], "then": ["than", "thn"],
    "download": ["donwload", "downlod"], "upload": ["uplod", "uplaod"], "convert": ["covert", "convet"],
    "report": ["reoprt", "repot"], "customer": ["costumer", "custumer"], "customers": ["costumers", "custumers"],
    "account": ["acount", "accout"], "accounts": ["acounts", "accouts"], "payment": ["payement", "paymnet"],
    "payments": ["payements", "paymnets"], "balance": ["balence", "ballance"], "employee": ["employe", "emplyee"],
    "employees": ["employes", "emplyees"], "vendor": ["vender", "vendr"], "vendors": ["venders", "vendrs"],
    "supplier": ["suplier", "supplyer"], "suppliers": ["supliers", "supplyers"], "spreadsheet": ["spredsheet", "spreadshet"],
    "document": ["documnet", "docment"], "documents": ["documnets", "docments"], "summary": ["summery", "sumary"],
    "calendar": ["calender", "calandar"], "necessary": ["neccessary", "necesary"], "tomorrow": ["tommorow", "tomorow"],
    "today": ["todays", "tody"], "website": ["webiste", "websit"], "password": ["pasword", "passwrod"],
    "monthly": ["montly", "monthy"], "weekly": ["weakly", "weekley"], "manager": ["manger", "maneger"],
    "approval": ["aproval", "approvel"], "amount": ["amout", "ammount"], "total": ["totel", "totl"],
    "rename": ["renam", "re name"], "screenshot": ["screen shot", "screenshoot"], "inbox": ["inbx", "in box"],
    "status": ["staus", "stauts"], "update": ["updte", "upadte"], "record": ["recrod", "reccord"],
    "records": ["recrods", "reccords"], "sheet": ["shet", "sheeet"], "table": ["tabel", "tabl"],
    "please": ["plese", "pleas"], "should": ["shoud", "shuold"], "folders": ["foldres", "fodlers"],
    "zendesk": ["zen desk", "zendsk"], "salesforce": ["sales force", "salesforse"], "outlook": ["out look", "outlok"],
}


def _slip(w, rng):
    """One mechanical slip in a word of 5+ letters."""
    j = rng.randrange(1, len(w) - 2)
    op = rng.random()
    if op < 0.3:
        return w[:j] + w[j + 1] + w[j] + w[j + 2:]                     # swapped pair
    if op < 0.55:
        return w[:j] + w[j + 1:]                                        # dropped letter
    if op < 0.75:
        return w[:j] + w[j] + w[j:]                                     # doubled letter
    c = w[j].lower()
    if c in KEYBOARD:
        return w[:j] + rng.choice(KEYBOARD[c]) + w[j + 1:]              # keyboard neighbour
    return w[:j] + w[j + 1:]


def add_typos(text, rng, n=None):
    """1-3 realistic slips outside protected tokens; an action-critical noun gets at most one slip."""
    n = n or rng.choice([1, 1, 2, 2, 3])
    spans = _protected_spans(text)
    words = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"[A-Za-z]{5,}", text)
             if not any(a < m.end() and m.start() < b for a, b in spans)]
    rng.shuffle(words)
    done, critical_used, out = 0, False, text
    edits = []
    for s, e, w in words:
        if done >= n:
            break
        crit = w.lower() in CRITICAL
        if crit and critical_used:
            continue
        lw = w.lower()
        if lw in PHONETIC and rng.random() < 0.7:
            nw = rng.choice(PHONETIC[lw])
            nw = nw.capitalize() if w[0].isupper() else nw
        else:
            nw = _slip(w, rng)
        if nw == w:
            continue
        edits.append((s, e, nw))
        critical_used |= crit
        done += 1
    for s, e, nw in sorted(edits, reverse=True):
        out = out[:s] + nw + out[e:]
    return out if done else None


# ------------------------------------------------------------------------------------------------ phrasebook
# business verbs beginners use -> the tool steps they mean (documentation and the reverse rewrite below)
BUSINESS_VERBS = {
    "chase / remind / nudge / follow up with": "email send (one per person: loop over the list)",
    "tidy / sort out / clean up (files)": "move / rename files into folders",
    "tally / reconcile / match up": "read both sources, compare / filter / join the tables",
    "total up / add up": "sum a column", "keep count of / tally how many": "count rows",
    "weed out / throw out the repeats": "remove duplicate rows", "pick out / pull out only": "filter rows",
    "raise / log an issue": "create a ticket", "let the team know / give a heads-up": "post a chat message or email",
    "pack up / bundle": "zip", "grab / pull down": "download", "put up on / share to": "upload",
    "bin / get rid of / clear out": "delete", "make a spare copy": "copy", "file away / shift": "move",
    "jot down / note": "log or write text", "turn into / make a pdf of": "convert / export",
    "go through each / one by one": "loop", "only when / provided": "if",
}
# (regex on the clear request, replacements): tool phrases -> beginner / business words with the same meaning
PHRASEBOOK = [
    (r"\bsend (?:an? |the )?(?:polite |friendly )?reminders?(?: e-?mails?)? to\b", ["chase", "nudge", "follow up with"]),
    (r"\be-?mail (?:an? |the )?reminders? to\b", ["chase", "nudge", "follow up with"]),
    (r"\bsend (?:an? )?(?:e-?mail|mail) to\b", ["drop a mail to", "shoot a mail to", "write to", "mail"]),
    (r"\bemail (?=(?:the|my|our|each|every|all|them|him|her|it)\b)", ["mail ", "send over to ", "drop a mail to "]),
    (r"\bremove (?:the )?duplicate(?: rows)?\b", ["weed out the repeats", "throw out the doubles", "keep only one of each",
                                                  "get rid of the repeated rows"]),
    (r"\bdrop (?:the )?duplicates\b", ["clear out the repeats", "weed out the doubles"]),
    (r"\bdedupe\b", ["clean out the repeats"]),
    (r"\bmove (?=(?:the|all|every|each|them|it)\b)", ["shift ", "file away ", "put ", "tidy "]),
    (r"\bmove\b", ["shift", "put", "file"]),
    (r"\brename\b", ["fix the names of", "change the name of", "give a proper name to"]),
    (r"\bdelete\b", ["get rid of", "bin", "clear out", "throw away"]),
    (r"\bcopy\b", ["make a spare copy of", "duplicate", "back up"]),
    (r"\bcompare\b", ["tally", "match up", "reconcile", "check against each other"]),
    (r"\bfilter (?:the )?rows?\b", ["pick out only the rows", "pull out just the lines"]),
    (r"\bkeep only (?:the )?rows\b", ["pull out only the lines", "pick out the lines"]),
    (r"\bfilter\b", ["pick out", "pull out only", "weed through"]),
    (r"\bcount the rows\b", ["tally how many lines there are", "see how many entries there are"]),
    (r"\bcount\b", ["tally", "keep count of", "see how many"]),
    (r"\bsum (?:up )?(?:the )?\b", ["total up the ", "add up the "]),
    (r"\bcalculate the total\b", ["work out the total", "add it all up"]),
    (r"\bcreate an? (?:servicenow |jira |zendesk )?(?:ticket|incident|issue)\b", ["raise a ticket", "log an issue", "open a ticket"]),
    (r"\bpost (?:a |the )?(?:message|summary|note) (?:to|in|on)\b", ["give a heads-up in", "let them know in", "drop a note in"]),
    (r"\bpost\b", ["share", "put up", "drop"]),
    (r"\bnotify\b", ["give a heads-up to", "let know", "tell"]),
    (r"\balert\b", ["warn", "give a heads-up to"]),
    (r"\bzip\b", ["pack into a zip", "bundle into a zip", "squeeze into a zip"]),
    (r"\bcompress\b", ["shrink into a zip", "pack up"]),
    (r"\bunzip\b", ["open up the zip", "extract the zip", "take out what's in the zip"]),
    (r"\bdownload\b", ["grab", "pull down", "get hold of"]),
    (r"\bupload\b", ["put up", "push up", "share"]),
    (r"\bextract (?:the )?text\b", ["pull the words out", "get the writing out"]),
    (r"\bextract\b", ["pull out", "fish out", "pick out"]),
    (r"\bread (?:the |all )?(?:rows|data)\b", ["go through the lines", "look at all the entries"]),
    (r"\bread\b", ["go through", "look at"]),
    (r"\bwrite (?:it|them|the (?:table|result|results|rows))\b", ["put it", "jot it", "keep it"]),
    (r"\bsave (?:it |them )?as\b", ["keep it as", "store it as", "turn it into"]),
    (r"\bconvert\b", ["turn", "change", "make"]),
    (r"\bexport\b", ["turn out", "make", "save out"]),
    (r"\blog (?=(?:the|a|that|it|how)\b)", ["note down ", "jot down ", "write down "]),
    (r"\bfor each\b", ["for every", "one by one for each", "go through each"]),
    (r"\bif there are none\b", ["if nothing turns up", "if there's nothing"]),
    (r"\botherwise\b", ["if not", "or else"]),
    (r"\bupdate\b", ["refresh", "bring up to date", "fix up"]),
    (r"\bapprove\b", ["okay", "sign off"]),
    (r"\bschedule\b", ["book", "set up"]),
    (r"\bscreenshot\b", ["picture of the screen", "screen grab"]),
    (r"\bprint\b", ["get a printout of", "print out"]),
    (r"\bmerge\b", ["join together", "put together into one", "combine"]),
    (r"\bsort (?:it |them )?(?:by|on)\b", ["arrange by", "put in order of"]),
    (r"\bcreate (?:a |the )?folder\b", ["make a new folder", "set up a folder"]),
    (r"\bcreate\b", ["make", "set up"]),
    (r"\bthe recipient\b", ["the person"]),
    (r"\battachments?\b", ["attached files", "files that came with it"]),
]
GOALS = [  # (keywords, goal phrases): vague goal said first
    (("invoice", "bill"), ["sort out my invoices", "keep the bills under control", "stop chasing invoices by hand"]),
    (("unpaid", "overdue", "payment", "outstanding"), ["chase the money people owe us", "get paid on time",
                                                         "stay on top of payments"]),
    (("report", "summary", "dashboard"), ["get the report done without me", "make the reporting easier",
                                          "stop doing this report by hand"]),
    (("ticket", "incident", "support", "helpdesk"), ["keep on top of the tickets", "help the support desk"]),
    (("customer", "client"), ["look after our customers better", "keep the customers in the loop"]),
    (("employee", "joiner", "payroll", "payslip", "hr"), ["make the HR work easier", "help with the staff admin"]),
    (("vendor", "supplier", "purchase"), ["handle the vendor paperwork", "keep the suppliers sorted"]),
    (("zip", "archive", "backup", "back up"), ["keep my files safe", "clean up old stuff"]),
    (("move", "rename", "folder", "copy", "delete"), ["tidy up my files", "sort out my folders", "keep the folders neat"]),
    (("email", "mail", "inbox"), ["keep on top of my emails", "sort out my mailbox stuff"]),
    (("excel", "csv", "sheet", "table"), ["sort out this spreadsheet", "clean up the data"]),
    (("pdf", "scan", "ocr"), ["deal with all these pdfs", "get through the scanned papers"]),
    (("website", "portal", "browser", "site"), ["stop checking the site by hand", "save time on the portal"]),
]
GENERIC_GOALS = ["save some time every day", "stop doing this by hand", "make my day easier", "automate a boring job"]
VAGUE_FRAMES = ["{g}. basically {r}", "i want to {g} - {r}", "help me {g}: {r}", "{g}, so {r}", "trying to {g}. {r}",
                "need to {g}... {r}", "{r}. just want to {g}"]
TIMES = ["every morning", "each morning", "every day", "daily", "every week", "every monday", "every friday",
         "every month", "at month end", "every evening"]

# ------------------------------------------------------------------------------------------------ Hinglish
# English verb (start of a clause) -> Hindi verb said at the end of the clause
HINGLISH_VERBS = {
    "send": ["bhej do", "bhej dena", "send kar do"], "email": ["mail kar do", "email kar do", "mail bhej do"],
    "mail": ["mail kar do", "mail bhej do"], "forward": ["forward kar do", "aage bhej do"],
    "reply": ["reply kar do", "jawab bhej do"], "read": ["padh lo", "padh ke dekho", "read karo"],
    "open": ["khol lo", "kholo", "open karo"], "close": ["band kar do", "close karo"],
    "save": ["save kar do", "save karna", "rakh do"], "write": ["likh do", "daal do", "write karo"],
    "delete": ["delete kar do", "hata do", "mita do"], "remove": ["hata do", "nikal do", "remove karo"],
    "drop": ["hata do", "nikal do"], "move": ["move kar do", "shift kar do", "daal do"],
    "copy": ["copy kar lo", "copy kar do"], "create": ["bana do", "create karo", "banao"],
    "make": ["bana do", "banao"], "check": ["check karo", "dekh lo", "check kar lena"],
    "find": ["dhundh lo", "dhundo", "find karo"], "search": ["search karo", "dhundo"],
    "download": ["download kar lo", "download karo", "utar lo"], "upload": ["upload kar do", "chadha do"],
    "print": ["print nikal do", "print kar do"], "rename": ["naam badal do", "rename kar do"],
    "merge": ["jod do", "merge kar do", "ek me mila do"], "filter": ["filter karo", "chhaant lo", "alag kar lo"],
    "keep": ["rakh lo", "rakho"], "count": ["gin lo", "count karo", "ginti karo"], "sort": ["sort karo", "line se lagao"],
    "add": ["jod do", "add karo", "daal do"], "post": ["post kar do", "daal do"], "log": ["log me likh do", "log karo"],
    "extract": ["nikal lo", "nikaalo", "extract karo"], "compare": ["match karo", "milao", "compare karo"],
    "zip": ["zip kar do", "zip bana do"], "unzip": ["unzip kar do", "khol do"], "convert": ["convert karo", "badal do"],
    "export": ["export kar do", "nikal do"], "update": ["update kar do", "update karo"],
    "connect": ["connect karo", "se jud jao"], "sign": ["sign in karo", "login karo"], "login": ["login karo"],
    "log in": ["login karo", "log in karo"], "list": ["list nikalo", "list bana do"], "get": ["le lo", "nikal lo", "le aao"],
    "fetch": ["le aao", "nikal lo"], "pull": ["nikal lo", "kheench lo"], "take": ["le lo", "utha lo"],
    "grab": ["utha lo", "le lo"], "insert": ["daal do", "insert karo"], "fill": ["bhar do", "fill karo"],
    "type": ["type karo", "likho"], "click": ["click karo", "daba do"], "go": ["jao", "chale jao"],
    "screenshot": ["screenshot le lo", "photo le lo"], "take a screenshot": ["screenshot le lo"],
    "summarise": ["summary bana do", "chhota karke likho"], "summarize": ["summary bana do", "short me likho"],
    "translate": ["translate karo", "anuvaad karo"], "tell": ["bata do", "bol do"], "notify": ["bata do", "inform kar do"],
    "inform": ["bata do", "inform karo"], "assign": ["assign kar do", "de do"], "approve": ["approve kar do", "pass kar do"],
    "reject": ["reject kar do", "mana kar do"], "mark": ["mark kar do", "nishaan laga do"], "archive": ["archive kar do"],
    "attach": ["attach kar do", "laga do"], "schedule": ["schedule karo", "time fix karo"], "set": ["set kar do"],
    "calculate": ["calculate karo", "hisaab lagao"], "sum": ["total karo", "jod do"], "total": ["total nikal do"],
    "run": ["chala do", "run karo"], "start": ["shuru karo", "start karo"], "stop": ["rok do", "band karo"],
    "wait": ["ruko", "wait karo"], "split": ["tod do", "split karo"], "replace": ["badal do", "replace karo"],
    "encrypt": ["lock kar do", "encrypt karo"], "share": ["share kar do", "bhej do"], "clean": ["saaf kar do"],
    "loop": ["ek ek karke dekho"], "add up": ["total karo", "jod do"], "get rid of": ["hata do", "nikal phenko"],
    "work out": ["nikal lo", "hisaab lagao"], "look at": ["dekh lo"], "look inside": ["andar dekho"],
    "go through": ["dekh lo", "ek ek karke padho"], "set up": ["set up kar do", "bana do"], "load": ["load karo"],
    "query": ["query chalao", "nikal lo"], "scrape": ["scrape karo", "nikal lo"], "ping": ["ping karo"],
    "keep only": ["sirf rakh lo", "baaki hata do"], "turn": ["badal do", "bana do"], "combine": ["mila do", "jod do"],
    "clear": ["saaf kar do", "khaali kar do"], "back up": ["backup le lo"], "transfer": ["transfer kar do"], "process": ["process karo", "nipta do"], "book": ["book kar do"],
    "raise": ["raise karo", "bana do"], "store": ["store karo", "rakh do"], "look": ["dekho", "dekh lo"],
}
# phrases replaced anywhere in the clause (English -> Hinglish); order matters (longer first)
HINGLISH_WORDS = [
    (r"\bevery morning\b", ["roz subah", "har subah"]), (r"\beach morning\b", ["har subah"]),
    (r"\bevery evening\b", ["roz shaam", "har shaam"]), (r"\bevery day\b", ["roz", "har din"]), (r"\bdaily\b", ["roz"]),
    (r"\bevery week\b", ["har hafte"]), (r"\bweekly\b", ["har hafte"]), (r"\bevery month\b", ["har mahine"]),
    (r"\bmonthly\b", ["har mahine"]), (r"\bevery monday\b", ["har somvaar", "har Monday"]),
    (r"\bevery friday\b", ["har shukravaar", "har Friday"]), (r"\btoday's\b", ["aaj ka", "aaj ki"]),
    (r"\byesterday's\b", ["kal ka", "kal ki"]), (r"\btoday\b", ["aaj"]), (r"\byesterday\b", ["kal"]),
    (r"\btomorrow\b", ["kal"]), (r"\bat month end\b", ["mahine ke end me"]), (r"\blast month\b", ["pichle mahine"]),
    (r"\bafter that\b", ["uske baad", "phir"]), (r"\band then\b", ["aur phir", "phir"]), (r"\bthen\b", ["phir"]),
    (r"\bnext\b", ["uske baad"]), (r"\bfinally\b", ["last me", "aakhir me"]), (r"\bfirst\b", ["pehle"]),
    (r"\botherwise\b", ["nahi to", "warna"]), (r"\belse\b", ["warna"]), (r"\bonly\b", ["sirf"]),
    (r"\bif there are none\b", ["agar kuch na ho"]), (r"\bif\b", ["agar"]), (r"\bwhen\b", ["jab"]),
    (r"\bfor each\b", ["har ek", "har"]), (r"\bevery\b", ["har"]), (r"\beach\b", ["har"]),
    (r"\ball (?:of )?the\b", ["saare", "saari"]), (r"\ball\b", ["saare", "sab"]), (r"\band\b", ["aur"]),
    (r"\bwith\b", ["ke saath"]), (r"\bwithout\b", ["bina"]), (r"\bmy\b", ["mera", "mere", "meri"]),
    (r"\bour\b", ["hamara", "hamare", "apna"]), (r"\bme\b", ["mujhe"]), (r"\bthem\b", ["unko", "un sabko"]),
    (r"\bit\b", ["usko", "ise"]), (r"\bthis\b", ["ye", "is"]), (r"\bthat\b", ["wo", "jo"]), (r"\bthese\b", ["ye sab"]),
    (r"\bnew\b", ["naya", "nayi"]), (r"\bold\b", ["purana", "purani"]), (r"\bbig\b", ["bade"]), (r"\bsmall\b", ["chhote"]),
    (r"\bunpaid\b", ["unpaid wale", "jinka payment baaki hai"]), (r"\boverdue\b", ["overdue wale", "late wale"]),
    (r"\bthe team\b", ["team", "poori team"]), (r"\bthe manager\b", ["manager"]), (r"\bmanager\b", ["manager", "sir"]),
    (r"\bthe summary\b", ["summary"]), (r"\bnumber of\b", ["kitne"]), (r"\bhow many\b", ["kitne"]),
    (r"\bthere\b", ["wahan"]), (r"\bhere\b", ["yahan"]), (r"\bnot\b", ["nahi"]), (r"\bis empty\b", ["khaali hai"]),
    (r"\bempty\b", ["khaali"]), (r"\bmore than\b", ["se zyada"]), (r"\bless than\b", ["se kam"]),
]
# postpositions: "<prep> the X folder" -> "X folder <postposition>"
_NAME = r"((?:[\w&'.-]+ ){0,2}?[\w&'.-]+)"        # one to three words naming a folder, sheet, team ...
HINGLISH_POST = [(r"\bfrom (?:the |my |our )?" + _NAME + r" (folder|sheet|file|workbook|inbox|mailbox|site|portal|website|table|list|database|drive)\b", ["{0} {1} SE__"]),
                 (r"\b(?:in|into|inside|under) (?:the |my |our )?" + _NAME + r" (folder|sheet|file|workbook|inbox|mailbox|channel|table|list|database|drive)\b", ["{0} {1} ME__", "{0} {1} KE__ANDAR__"]),
                 (r"\bto (the |my |our )?((?:\w+ ){0,2}(?:team|manager|boss|client|clients|customer|customers|auditors|lead|head|desk|approver|owner|vendor|vendors|supplier|suppliers|staff|employee|employees))\b", ["{1} KO__"]),
                 (r"\bon (slack|teams|whatsapp)\b", ["{0} PE__", "{0} PAR__"]),
                 (r"\bto (?:each|every) ([\w']+)", ["HAR__ {0} KO__"]),
                 (r"\bfor (?:each|every) ([\w']+)", ["HAR__ {0} KE__LIYE__", "HAR__ EK__ {0} KE__LIYE__"]),
                 (r"(?-i:\bfrom (?:the )?([A-Z][\w]*(?: [A-Z][\w]*){0,2}))\b", ["{0} SE__"]),
                 (r"(?-i:\b(?:in|into|on|inside) (?:the )?([A-Z][\w]*(?: [A-Z][\w]*){0,2}))\b", ["{0} ME__", "{0} PE__"])]
_HINGLISH_ONE = "|".join(f"(?P<g{i}>{p})" for i, (p, _) in enumerate(HINGLISH_WORDS))
HINGLISH_JOINS = [" aur ", ", phir ", ", uske baad ", " aur phir ", ", fir "]
HINGLISH_ENDS = ["", "", " please", " yaar", " jaldi", " bhai", " ok?", " karna hai"]


def _hinglish_clause(c, rng):
    c = c.strip()
    if not c:
        return c, 0
    n = 0
    lc = c.lower()
    verb = None
    for v in sorted(HINGLISH_VERBS, key=len, reverse=True):
        if lc.startswith(v + " ") or lc == v:
            verb = v
            break
    rest = c[len(verb):].strip() if verb else c
    for pat, outs in HINGLISH_POST:
        def f(m, outs=outs):
            g = [x.strip() if x else "" for x in m.groups()]
            g = [x for x in g if x]
            return rng.choice(outs).format(*(g + [""] * 3))
        rest, k = _sub_outside(pat, f, rest)
        n += k
    def word(m):                      # one pass, so a Hindi word is never replaced again ("me" -> "mujhe")
        return rng.choice(HINGLISH_WORDS[int(m.lastgroup[1:])][1])
    rest, k = _sub_outside(_HINGLISH_ONE, word, rest)
    n += k
    for ph, hi in (("KE__ANDAR__", "ke andar"), ("KE__LIYE__", "ke liye"), ("EK__", "ek")):
        rest = rest.replace(ph, hi)
    rest = re.sub(r"\b([A-Z]+)__", lambda m: m.group(1).lower(), rest)
    rest = re.sub(r"^(?:the|a|an)\s+", "", rest, flags=re.I)
    rest = re.sub(r"\s+(?:the|a|an)\s+", " ", rest, flags=re.I)
    if verb:
        n += 1
        rest = (rest + " " + rng.choice(HINGLISH_VERBS[verb])).strip()
    return rest, n


_SPLIT = re.compile(r"\s*(?:;|\.\s+|,\s*(?:and\s+)?then\s+|\s+and\s+then\s+|,\s+after that,?\s+|\s+then\s+|,\s+(?=\w+ )|"
                    r"\s+and\s+(?=(?:" + "|".join(sorted((v for v in HINGLISH_VERBS if " " not in v), key=len, reverse=True))
                    + r")\b))\s*", re.I)
_NUMBERED = re.compile(r"(?:^|\s)\d+[.)]\s+")


_LEAD_WORDS = re.compile(r"^(?:after that|and then|then|next|lastly|finally|also|and|once that's done|once done|"
                         r"when everything is done|when that's done|after which|afterwards)\b\s*,?\s*", re.I)
WHEN = re.compile(r"(?i)^(every \w+|each \w+|daily|weekly|monthly|at month end|at the end of the month|on mondays|"
                  r"every \w+ morning)\s*[:,]?\s*(.*)$")


def clauses_of(r):
    r = _NUMBERED.sub(" ", r)
    out = []
    for c in _SPLIT.split(r):
        c = (c or "").strip(" ,.")
        for _ in range(2):
            c = _LEAD_WORDS.sub("", c).strip(" ,.")
        if c:
            out.append(c)
    return out


def to_hinglish(r, rng):
    parts, n = [], 0
    for c in clauses_of(r):
        h, k = _hinglish_clause(c, rng)
        parts.append(h)
        n += k
    if n < 3 or len(parts) == 0:
        return None
    out = parts[0]
    for p in parts[1:]:
        out += rng.choice(HINGLISH_JOINS) + p
    return _lower_keep(out) + rng.choice(HINGLISH_ENDS)


# ------------------------------------------------------------------------------------------------ the other styles
def to_business(r, rng):
    n, out = 0, r
    pats = PHRASEBOOK[:]
    rng.shuffle(pats)
    for pat, outs in pats:
        if n >= 3:
            break
        out, k = _sub_outside(pat, lambda m, outs=outs: rng.choice(outs), out, count=1)
        n += k
    if not n:
        return None
    return out[:1].lower() + out[1:] if rng.random() < 0.6 else out


def goal_for(r, rng):
    lr = r.lower()
    hits = [g for keys, gs in GOALS if any(k in lr for k in keys) for g in gs]
    return rng.choice(hits) if hits else rng.choice(GENERIC_GOALS)


def to_vague(r, rng):
    base = to_business(r, rng) if rng.random() < 0.5 and not is_hinglish(r) else None
    base = base or r
    base = base[:1].lower() + base[1:]
    return rng.choice(VAGUE_FRAMES).format(g=goal_for(r, rng), r=base)


CASUAL_SUBS = [(r"\bplease\b", ["pls", "plz"]), (r"\byou\b", ["u"]), (r"\byour\b", ["ur"]), (r"\bthanks\b", ["thx"]),
               (r"\bdo not\b", ["dont"]), (r"\bdoesn't\b", ["doesnt"]), (r"\bit is\b", ["its"]), (r"\bthere is\b", ["theres"]),
               (r"\bthere are\b", ["theres"]), (r"\bthat is\b", ["thats"]), (r"\bwith\b", ["w/", "with"]),
               (r"\band\b", ["and", "n", "&"]), (r"\bbecause\b", ["cuz", "coz"]), (r"\bmessage\b", ["msg"]),
               (r"\binformation\b", ["info"]), (r"\bdocuments?\b", ["docs"]), (r"\bspreadsheet\b", ["sheet"]),
               (r"\btomorrow\b", ["tmrw"]), (r"\bto be\b", ["2 be"]), (r"\bfor\b", ["for", "4"])]
CASUAL_HEADS = ["", "", "hey ", "pls ", "can u ", "quick one - ", "need help: ", "yo ", "ok so ", "hi, "]
CASUAL_TAILS = ["", "", " pls", " thx", " asap", " :)", " thanks!!", " ty", "?", " plz"]


def to_casual(r, rng):
    out = _lower_keep(r)
    for pat, outs in CASUAL_SUBS:
        if rng.random() < 0.6:
            out, _ = _sub_outside(pat, lambda m, outs=outs: rng.choice(outs), out)
    out = re.sub(r"[.;]\s+", " ", out)
    out = re.sub(r"\s*:\s+", " - ", out)
    return (rng.choice(CASUAL_HEADS) + out.rstrip(".!") + rng.choice(CASUAL_TAILS)).strip()


INDIAN_HEADS = ["Kindly ", "Please do the needful and ", "Request you to ", "Please do one thing, ", "Dear team, kindly ",
                "Hi, kindly ", "Need your help, please ", "As discussed, kindly "]
INDIAN_TAILS = ["", " itself", " only", ", thanks in advance", " at the earliest", " ASAP please", ". Thanks and regards",
                " on priority basis", " without fail"]
INDIAN_SUBS = [(r"\bI have\b", ["I am having"]), (r"\bwe have\b", ["we are having"]), (r"\btoday\b(?!')", ["today itself"]),
               (r"\bnow\b", ["now only"]), (r"\btell\b", ["intimate"]), (r"\binform\b", ["intimate", "update"]),
               (r"\bemail\b", ["mail"]), (r"\bsend\b", ["send", "share"]), (r"\bthe details\b", ["the details"]),
               (r"\bat once\b", ["immediately"]), (r"\bcheck\b", ["cross-check", "verify"]), (r"\bfile\b", ["file", "sheet"])]


def to_indian(r, rng):
    out = r[:1].lower() + r[1:]
    for pat, outs in INDIAN_SUBS:
        if rng.random() < 0.5 and not (pat == r"\bfile\b"):
            out, _ = _sub_outside(pat, lambda m, outs=outs: rng.choice(outs), out)
    return (rng.choice(INDIAN_HEADS) + out.rstrip(".") + rng.choice(INDIAN_TAILS)).strip()


VOICE_HEADS = ["ok so ", "so um ", "right so basically ", "hey so the thing is ", "okay what i need is ", "um so like ",
               "so yeah ", "alright so ", "so basically what happens is ", "ok um "]
VOICE_FILLERS = [" um", " uh", " like", " you know", " basically", " so yeah", " i mean", " sort of"]
VOICE_TAILS = ["", " thats it", " you know", " if that makes sense", " yeah", " that's basically it", " and done",
               " can you do that"]


def to_voice(r, rng):
    out = r
    if rng.random() < 0.6:     # how dictation writes file names and addresses
        out = re.sub(r"\b([\w-]+)\.(csv|xlsx|pdf|docx|txt|zip|json)\b", lambda m: f"{m.group(1)} dot {m.group(2)}", out)
        out = re.sub(r"([\w.+-]+)@([\w-]+)\.(com|in|org|net)\b", lambda m: f"{m.group(1)} at {m.group(2)} dot {m.group(3)}", out)
    out = _lower_keep(out)
    cl = clauses_of(out)
    if not cl:
        return None
    pieces = [cl[0]]
    for c in cl[1:]:
        pieces.append(rng.choice([" and then", " and", " then", " and after that", " and then like"]) +
                      (rng.choice(VOICE_FILLERS) if rng.random() < 0.4 else "") + " " + c)
    out = "".join(pieces)
    out, _ = _sub_outside(r"[,;:!?]", "", out, flags=0)
    out, _ = _sub_outside(r"\.(?=\s|$)", "", out, flags=0)
    return (rng.choice(VOICE_HEADS) + out + rng.choice(VOICE_TAILS)).strip()


_IMPERATIVE = set(HINGLISH_VERBS) | {"go", "click", "type", "sign", "log", "pick", "grab", "put", "drop", "collect",
                                      "build", "fill", "attach", "look", "turn", "export", "sum", "total", "remove"}
STORY_HEADS = ["Here's what I do: ", "Every day I do this by hand. ", "My process is like this. ", "So what I do is, ",
               "This is my routine: ", "Let me explain my task. ", "I do this manually right now: "]
STORY_JOINS = [", then I ", ". After that I ", ". Then I ", ", and then I ", ". Next I ", ", after which I "]
STORY_TAILS = [" Can the bot do this?", " Can you automate that?", " I want a bot to do it for me.", "",
               " Is that possible?", " Please make it automatic."]


def _first_person(c):
    w = c.split(" ", 1)
    if w[0].lower() in _IMPERATIVE:
        rest = re.sub(r"\bme\b", "myself", w[1]) if len(w) > 1 else ""
        return w[0].lower() + (" " + rest if rest else "")
    if re.match(r"(?i)if\b|for each\b|when\b|only\b", c):
        return None
    return None


def to_story(r, rng):
    cl = clauses_of(r)
    when = ""
    if cl:
        m = WHEN.match(cl[0])
        if m:
            when = m.group(1)
            cl = ([m.group(2)] if m.group(2) else []) + cl[1:]
    if len(cl) < 2:
        return None
    parts, made = [], 0
    for c in cl:
        fp = _first_person(c)
        if fp:
            made += 1
            parts.append(("I", fp))
        else:
            parts.append(("", c[:1].lower() + c[1:]))
    if made < 2:
        return None
    text = ""
    for i, (who, c) in enumerate(parts):
        if i == 0:
            text = ("first I " if who else "first, ") + c
        elif who:
            text += rng.choice(STORY_JOINS) + c
        else:
            text += ", " + c
    head = rng.choice(STORY_HEADS)
    if when:
        head = f"{when[:1].upper() + when[1:]} " + rng.choice(["I sit down and do this. ", "this is my job. ", "I do the same thing. "])
    text = head + (text[:1].upper() + text[1:] if head.endswith((". ", ": ")) else text)
    return text.rstrip(".") + "." + rng.choice(STORY_TAILS)


_HINDI = re.compile(r"\b(?:karo|kar|do|ko|ke|ki|ka|me|se|aur|phir|wala|wali|wale|hai|hain|nikal|bana|bhej|padh|sab|saare|har|"
                    r"sirf|agar|nahi|jo|uske|unko|mujhe|dena|lo|daal|hata)\b", re.I)


def is_hinglish(text):
    return len(_HINDI.findall(text)) >= 3


HINGLISH_OK = ("vague", "casual", "typo", "voice")     # styles that still read well on a request already in Hinglish

TRANSFORMS = {"vague": to_vague, "business": to_business, "casual": to_casual, "typo": lambda r, rng: add_typos(r, rng),
              "hinglish": to_hinglish, "indian_english": to_indian, "voice": to_voice, "story": to_story}


def beginner_versions(request, rng, k):
    """Up to k beginner requests for one clear request: [(style list, text)], distinct styles, ~1 in 4 also gets a
    typo (then "typo" is in its style list)."""
    base = clean_request(request)
    if not base:
        return []
    styles = list(STYLES) if not is_hinglish(base) else list(HINGLISH_OK)
    rng.shuffle(styles)
    styles.sort(key=lambda s: -STYLE_WEIGHTS[s] * rng.random())
    out = []
    for st in styles:
        if len(out) >= k:
            break
        try:
            t = TRANSFORMS[st](base, rng)
        except (IndexError, ValueError):
            t = None
        if not t:
            continue
        tags = [st]
        if st not in ("typo",) and rng.random() < 0.25:
            t2 = add_typos(t, rng, 1)
            if t2:
                t, tags = t2, tags + ["typo"]
        t = " ".join(t.split())
        if norm(t) == norm(base) or len(t) > MAX_REQUEST:
            continue
        out.append((tags, t))
    return out


# ------------------------------------------------------------------------------------------------ clarify (missing detail)
# slot: (regex finding the detail, replacements with a vague reference, friendly questions)
SLOTS = {
    "folder": (r"\b(from|in|into|inside|to|under) (?:the |my |our )?(?:[A-Z][\w&'-]*(?: [A-Z][\w&'-]*){0,2}|[A-Za-z]:\\[^\s,;]+) folder\b",
               {"from": ["from the usual folder", "from the folder we always use"],
                "in": ["in the usual folder", "in our normal folder"], "inside": ["inside the usual folder"],
                "into": ["into the right folder", "into the usual place"], "to": ["to the right folder", "to the usual place"],
                "under": ["under the usual folder"]},
               ["Sure! Which folder do you mean? Please tell me its name or path (for example C:\\Invoices).",
                "Happy to help. Which folder should I use? A name or a path like D:\\Reports is fine.",
                "Got it. What is the folder called, and where is it on your computer?",
                "Which folder is the usual one? Please give me its name or full path."]),
    "recipient": (r"\b((?i:e-?mail|mail|send|forward|share)\b[^.;:]{0,40}?) (?i:to) (?:[\w.+-]+@[\w-]+\.[\w.]+|(?:the |my |our )?(?:(?:\w+ ){0,2}(?:team|manager|boss|auditors|client|lead|head|approver|desk)|[A-Z][a-z]+))\b",
                  {"to": ["to the usual people", "to the right person", "to the people who need it", "to the usual guys"]},
                  ["Sure! Who should get it? Please give me their name or email address.",
                   "Who are the usual people? Please share their email addresses.",
                   "Happy to do that. Who exactly should I send it to (email address)?",
                   "Who is the right person here? An email address is best."]),
    "condition": (r"\b(?i:over|above|more than|greater than|bigger than|of at least) [$₹€£]?\d[\d,.]*(?i: ?(?:rupees|rs|dollars|k|lakh))?\b(?!(?i: ?(?:days|hours|minutes|weeks|months|mb|kb|gb|rows|pages)))|"
                  r"\b(?i:where|with|whose) (?:the )?[A-Z]\w* (?:is|=|equals) '?[A-Z]\w*'?|\bto [A-Z]\w* = [A-Z]\w*",
                  {"": ["that are big", "that matter", "that are the important ones", "that look right"]},
                  ["What counts as big here? Please give me the amount (for example over 5,000).",
                   "How do I tell which ones you mean? Tell me the rule, for example amount above 10,000.",
                   "Which ones should I pick? Please give me the limit or the condition.",
                   "Where is the cut-off? Tell me the number I should compare with."]),
    "app": (r"\b(?:in|on|to|via|through|from|into) (?:the )?(Slack|Teams|Microsoft Teams|Jira|ServiceNow|Salesforce|Zendesk|Asana|Google Sheets?|SharePoint|Outlook|Gmail|Google Drive|OneDrive|Box|HubSpot|Freshdesk)\b",
            {"": ["in our system", "in the app we use", "in the usual tool", "through our usual app"]},
            ["Sure! Which app do you mean: Slack, Teams, Jira, ServiceNow or something else?",
             "Which system should I use for this? Please tell me the app's name.",
             "Happy to help. Which tool is 'our system' - for example Outlook, Gmail, Jira or Salesforce?",
             "What is the name of the app you use for this?"]),
    "file": (r"\b[\w-]+\.(?:csv|xlsx|xls|docx|pdf|txt|json)\b",
             {"": ["that file", "the usual file", "the sheet I always use", "my file"]},
             ["Which file do you mean? Please tell me its name and which folder it is in.",
              "Sure! What is the file called, and where is it saved?",
              "Which file should I use? A full path like C:\\Data\\orders.xlsx is perfect.",
              "Can you tell me the name of the file, and where it is?"]),
}
SLOT_STYLES = ["casual", "casual", "plain", "plain", "hinglish", "indian_english", "voice", "typo"]


def take_out(request, slot, rng):
    """The request with one detail of this slot replaced by a vague reference, or None when it has none."""
    pat, repl, _ = SLOTS[slot]
    m = re.search(pat, request, flags=re.I if slot == "file" else 0)
    if not m:
        return None
    word = m.group(1).lower() if slot == "folder" else ("to" if slot == "recipient" else "")
    if slot == "app":
        word = ""
    choices = repl.get(word) or repl.get("") or ["the usual one"]
    keep = m.group(1) if slot == "recipient" else ""       # the words between the verb and "to"
    pick = rng.choice(choices)
    if slot == "condition" and re.search(r"\b(?:is|are|was)\s*$", request[:m.start()]):
        pick = rng.choice(["big", "high enough", "too high", "large"])       # "the balance is over 5000"
    if slot == "recipient":
        pick = pick[3:] if pick.startswith("to ") else pick
        out = request[:m.start()] + keep + " to " + pick + request[m.end():]
    else:
        out = request[:m.start()] + pick + request[m.end():]
    if slot == "file" and re.search(r"\b[\w-]+\.(?:csv|xlsx|xls|docx|pdf|txt|json)\b", out):
        return None                       # still names another file: not really missing
    return " ".join(out.split())


def clarify_records(sources, n, rng, guard, seen):
    out, per_slot = [], Counter()
    order = list(range(len(sources)))
    rng.shuffle(order)
    slots = list(SLOTS)
    for i in order:
        if len(out) >= n:
            break
        base = clean_request(sources[i])
        if not base or len(base) > 220 or is_hinglish(base):
            continue
        rng.shuffle(slots)
        slots.sort(key=lambda s: per_slot[s])        # keep the slots balanced
        for slot in slots:
            vague = take_out(base, slot, rng)
            if not vague:
                continue
            st = rng.choice(SLOT_STYLES)
            text = vague
            if st != "plain":
                t = TRANSFORMS[st](vague, rng) if st != "typo" else add_typos(vague, rng, 1)
                if t and all(w in t.lower() for w in ("usual",) if "usual" in vague.lower()):
                    text = t
                else:
                    st = "plain"
            text = " ".join(text.split())
            key = norm(text)
            if key in seen or guard.leaks(text):
                continue
            seen.add(key)
            q = rng.choice(SLOTS[slot][2])
            styles = ([] if st == "plain" else [st]) + ["missing_detail"]
            out.append({"kind": "clarify", "request": text, "question": q, "missing": slot, "style": styles,
                        "source": f"beginner_clarify:{slot}"})
            per_slot[slot] += 1
            break
    return out


# ------------------------------------------------------------------------------------------------ sources and guard
class Guard:
    """Never close to a test prompt: any eval/*.jsonl prompt with difflib ratio over 0.8 or word overlap over 0.6."""

    def __init__(self, eval_dir=None, ratio=0.8, overlap=BS.MAX_OVERLAP):
        ev = Path(eval_dir) if eval_dir else HERE.parent / "eval"
        self.raw = [json.loads(l)["prompt"] for p in sorted(ev.glob("*.jsonl")) for l in open(p, encoding="utf-8")
                    if l.strip() and "prompt" in json.loads(l)]
        self.tests = [norm(t) for t in self.raw]
        self.index = BS.overlap_index(self.raw)
        self.ratio, self.overlap, self.dropped = ratio, overlap, 0

    def leaks(self, request):
        r = norm(request)
        for t in self.tests:
            m = difflib.SequenceMatcher(None, r, t)
            if m.real_quick_ratio() > self.ratio and m.quick_ratio() > self.ratio and m.ratio() > self.ratio:
                self.dropped += 1
                return True
        o, _ = BS.closest(request, self.raw, self.index)
        if o > self.overlap:
            self.dropped += 1
            return True
        return False


def _flow_of(text):
    t = (text or "").strip()
    if not t.startswith("{"):
        return None
    try:
        obj = json.loads(t)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) and isinstance(obj.get("steps"), list) and obj["steps"] else None


def load_sources(smart, chat):
    """{origin: [(request, flow)]} from the neutral files and the chat rows whose answer is a flow."""
    src = defaultdict(list)
    smart = Path(smart)
    for name, origin in (("long_train.jsonl", "long"), ("business2_train.jsonl", "business2"),
                         ("targeted_train.jsonl", "targeted"), ("clarify_train.jsonl", "clarify_counter")):
        p = smart / name
        if not p.exists():
            continue
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r.get("kind") == "flow" and isinstance(r.get("flow"), dict):
                src[origin].append((r["request"], r["flow"]))
    if chat:
        for name in CHAT_FILES:
            p = Path(chat) / name
            if not p.exists():
                continue
            for l in open(p, encoding="utf-8"):
                if not l.strip():
                    continue
                msgs = [m for m in json.loads(l).get("messages", []) if m.get("role") != "system"]
                if len(msgs) != 2:
                    continue
                flow = _flow_of(msgs[1].get("content"))
                req, _ = F.split_user(msgs[0].get("content", ""))
                if flow is not None and req:
                    src[name.split("_train")[0].replace(".jsonl", "")].append((req, flow))
    return src


def clarify_sources(src):
    return [r for origin in ("targeted", "chains", "combos", "tools", "clarify_counter", "business2", "long")
            for r, _ in src.get(origin, [])]


def described(flow):
    """True when every step (closing steps aside) has a short plain description, so clear_request reads well."""
    return all(F._plain(s.get("description")) for s, _ in F.walk(flow.get("steps"))
               if not F._HOUSEKEEPING.search(str(s.get("action") or "")))


def clear_for(request, flow, cat):
    """The "Clear request" line: the flow's steps in plain words (ooa_format.clear_request) when every step is
    described; else the original request when it is short, clear English (catalog names make poor sentences)."""
    if described(flow):
        return F.clear_request(flow, cat)
    base = clean_request(request)
    if base and len(base) <= 240 and not is_hinglish(base):
        base = base[:1].upper() + base[1:]
        return base if base.endswith((".", "?", "!")) else base + "."
    return F.clear_request(flow, cat)


def generate(src, catalog_path, schema_path, n_flows=2300, n_clarify=1800, seed=41, guard=None, log=print):
    rng = random.Random(seed)
    checker = Checker(catalog_path, schema_path)
    cat = checker.cat
    guard = guard or Guard()
    seen, rows = set(), []
    for origin in src:
        for r, _ in src[origin]:
            seen.add(norm(clean_request(r)))      # never repeat an existing training request
    counts = Counter()
    total_share = sum(ORIGIN_SHARE.get(o, 0) for o in src)
    for origin in sorted(src):
        share = ORIGIN_SHARE.get(origin, 0)
        if not share:
            continue
        want = round(n_flows * share / total_share)
        pool = src[origin][:]
        rng.shuffle(pool)
        used = 0
        for req, flow in pool:
            if used >= want:
                break
            fixed = checker.check(flow)
            if fixed is None:
                counts["checker_dropped"] += 1
                continue
            clear = clear_for(req, fixed, cat)
            if not clear:
                continue
            versions = beginner_versions(req, rng, rng.choice([3, 4, 4, 5]))
            made = 0
            for tags, text in versions:
                key = norm(text)
                if key in seen or guard.leaks(text):
                    continue
                seen.add(key)
                rows.append({"kind": "rewrite", "request": text, "clear": clear, "flow": fixed, "style": tags,
                             "source": f"beginner:{tags[0]}:{origin}"})
                made += 1
            if made:
                used += 1
                counts["flows:" + origin] += 1
    clar = clarify_records(clarify_sources(src), n_clarify, rng, guard, seen)
    rows += clar
    report = {"records": len(rows), "rewrite": sum(r["kind"] == "rewrite" for r in rows), "clarify": len(clar),
              "flows_used": {k[6:]: v for k, v in counts.items() if k.startswith("flows:")},
              "checker_dropped": counts["checker_dropped"], "close_to_test_dropped": guard.dropped,
              "style": dict(Counter(s for r in rows for s in r["style"])),
              "first_style": dict(Counter(r["style"][0] for r in rows if r["kind"] == "rewrite")),
              "missing": dict(Counter(r["missing"] for r in clar)), "checker": {"ok": checker.ok, "n": checker.n}}
    log(json.dumps(report, indent=1))
    return rows, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("catalog")
    ap.add_argument("schema")
    ap.add_argument("out")
    ap.add_argument("--smart", required=True, help="folder with long / business2 / targeted / clarify _train.jsonl")
    ap.add_argument("--chat", default=None, help="folder with the chat files (chains, combos, tools ...)")
    ap.add_argument("--flows", type=int, default=2300)
    ap.add_argument("--clarify", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=41)
    a = ap.parse_args(argv)
    src = load_sources(a.smart, a.chat)
    rows, report = generate(src, a.catalog, a.schema, a.flows, a.clarify, a.seed)
    rng = random.Random(a.seed + 1)
    rng.shuffle(rows)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rp = Path(a.out).with_name(Path(a.out).stem + "_report.json")
    rp.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {len(rows)} records -> {a.out} (report {rp})")


if __name__ == "__main__":
    main()
