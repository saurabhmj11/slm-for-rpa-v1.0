"""Grounding check for SFT records: which details does a record's flow or text hold that its request never gave?

A flow constant in a detail argument (an email address, a path / folder / file name, an archive / index / item path,
a URL, a saved connection, an AD distinguished name, a team / channel / sheet / table / site / group / project name,
an id, a file type filter, a database type) that the request (and, for a step-builder part, the listed known
variables) does not name teaches the model to invent details. Such a value belongs in an input variable. The same
holds for the free text the flow sends (subjects, bodies, messages, descriptions: no made-up link, path, address,
company or AD name) and for the record's texts ("clear", "understand", "facts", the flow and step descriptions and
the "steps" lines): they may only repeat what the request says, and they must read cleanly (no "the given folder"
when the request gives none, no "2 copy/copies" template text).

Paths: a path is "said" when the request writes it (or its folder, for a file the flow writes there). A new file
name is fine only where the flow itself writes it ("derived": a csv/write, save_as, download, zip ... target inside
a folder the request names); a step that reads a file reads one the request names or one an earlier step wrote --
the same text, so a value turned into a variable must be turned into it in every step that uses it.

Also flagged: a bare `$name.field` written inside other text ("D:\\out\\$row.Name.txt"), which the runner sends
literally (only `${...}` or `$path$` is replaced inside text, plat docs 01-agents §3; runner/expressions._INTERP).

Run: python grounding_check.py FILE.jsonl ...   -> counts per file and kind, a few examples; exit 1 when any issue.
Standard library only (ooa_format for the step walk)."""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ooa_format as F

# arguments whose literal value is a detail only the user can give
DETAIL_KEYS = {
    "to", "cc", "bcc", "attendees", "recipients", "recipient", "email", "emails", "from_address", "reply_to",
    "path", "paths", "folder", "file", "files", "target", "target_folder", "target_path", "source", "source_folder",
    "destination", "local_path", "remote_path", "remote_folder", "local_folder", "attachments", "output_path",
    "template", "template_path", "workbook", "public_key_file", "private_key_file", "key_file", "save_to",
    "archive", "index_file", "item", "to_folder", "new_path", "output", "output_folder", "dest",
    "url", "site_url", "base_url", "endpoint", "drive", "site", "site_id", "instance_url", "org_url", "account_url",
    "connection", "team_name", "channel_name", "channel", "team", "sheet", "sheet_name", "table_name", "procedure",
    "group", "groups", "assignment_group", "project", "queue", "list", "list_name", "database", "schema", "bucket",
    "container", "repo", "owner", "topic", "host", "username", "mailbox", "shared_mailbox", "plan", "plan_id",
    "warehouse", "warehouse_id", "job_id", "spreadsheet_id", "document_id", "calendar_id", "folder_id", "drive_id",
    "board", "board_id", "workspace", "space", "space_key", "report", "report_path", "integration", "eib",
    "consumer_group", "server", "domain", "ou", "manager", "assignee", "approver", "user", "login",
    "from_contains", "sender", "from", "pattern", "location", "printer",
    "target_ou_dn", "base_dn", "group_dn", "ou_dn", "dn", "user_dn", "extension", "database_type",
    "from_currency", "to_currency", "range", "documents",
}
DN_KEYS = {"target_ou_dn", "base_dn", "group_dn", "ou_dn", "dn", "user_dn", "ou"}
# free text the flow sends or stores: checked for made-up links, paths, addresses, company and AD names
TEXT_KEYS = {"subject", "body", "message", "text", "description", "short_description", "summary", "notes", "comment",
             "title", "name", "close_notes", "work_notes", "html", "content", "caption", "reply", "comments",
             "email_subject", "note", "prompt", "question", "details", "value"}
# platform constants that translate a request word, not user details
SYSTEM_VALUES = {
    "incident", "change_request", "sc_req_item", "sc_task", "problem", "sys_user", "sc_request", "task",
    "inbox", "sent items", "sent", "drafts", "archive", "me", "primary", "sheet1", "localhost", "root",
}
# fixed parts of a platform address, not user details (Workday custom report paths)
SYSTEM_PREFIXES = ("/ccx/service/customreport2/${wd_tenant}/", "/ccx/service/customreport2/")
# arguments read as an expression, where a bare $name.field is a reference
EXPRESSION_KEYS = {"condition", "expr", "expression", "formula", "where"}
# code / query text: not checked for details (its table and column names are the query's)
CODE_KEYS = {"sql", "soql", "jql", "query", "script", "code", "command", "xpath", "selector", "filter", "regex"}
# words a request uses for a file type
EXT_WORDS = {
    "xlsx": ("xlsx", "excel", "workbook", "spreadsheet"), "xls": ("xls", "excel", "workbook", "spreadsheet"),
    "xlsm": ("xlsm", "excel", "workbook"), "csv": ("csv",), "pdf": ("pdf",), "txt": ("txt", "text file"),
    "docx": ("docx", "word"), "doc": ("doc", "word"), "jpg": ("jpg", "jpeg", "photo", "image", "picture"),
    "jpeg": ("jpg", "jpeg", "photo", "image"), "png": ("png", "image", "screenshot"), "zip": ("zip",),
    "xml": ("xml",), "json": ("json",), "msg": ("msg",), "eml": ("eml",), "log": ("log",), "tif": ("tif", "tiff"),
}
DB_WORDS = {"sqlserver": ("sql server", "sqlserver", "mssql", "ms sql"), "postgresql": ("postgres", "postgresql"),
            "mysql": ("mysql",), "oracle": ("oracle",), "sqlite": ("sqlite",), "mariadb": ("mariadb",),
            "db2": ("db2",), "snowflake": ("snowflake",)}
# company / domain names the generators wrote; a text may name one only when the request does
ORG_WORDS = ("ourfirm", "acme", "contoso", "clientco", "ourpayrollvendor", "adsplatform", "acme-corp", "globex",
             "initech", "northwind", "fabrikam")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"https?://[^\s\"'<>)]+")
_WINPATH = re.compile(r"(?:[A-Za-z]:\\|\\\\)[^\s\"'<>|,;()]*")
_FILE = re.compile(r"(?<![\w$.{\\/-])[\w][\w.-]*?\.(?:csv|xlsx|xls|xlsm|docx|doc|pdf|txt|json|xml|zip|msg|eml|log|asc|"
                   r"pgp|gpg|png|jpg|html|py|ps1|bat|sql|tsv|dat|ini|bak|sha256)\b", re.I)
_GLOB = re.compile(r"(?<![\w$])\*\.([A-Za-z0-9]{2,5})\b")
_HASH = re.compile(r"(?<![\w&])#[a-z][\w-]+")
_IDENT = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")        # UPPER_SNAKE names (tables, procedures)
_SAVED_CONN = re.compile(r"saved connection[s]? ['\"]?([\w.-]+)", re.I)
_DN_PART = re.compile(r"\b(DC|OU|CN)=([^,()]+)", re.I)
_ORG = re.compile(r"(?<![\w$])(" + "|".join(re.escape(o) for o in ORG_WORDS) + r")(?![\w-])", re.I)
_DBNAME = re.compile(r"\b(SQL Server|PostgreSQL|Postgres|MySQL|Oracle|MariaDB)\b", re.I)
_TEMPLATE = re.compile(r"\b\w+\(s\)|\b\d+ (\w+)/(\w+)\b")
_REF = re.compile(r"\$\{[^}]*\}|\$[A-Za-z_][\w]*(?:\.[\w]+|\[[^\]]*\])*\$?")
_BARE_IN_TEXT = re.compile(r"\$(?!\{)[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
_FULL_REF = re.compile(r"\$[A-Za-z_]\w*(?:\.[A-Za-z0-9_]+|\[[^\]]+\])*\$?")
_INTERP_OK = re.compile(r"\$\{[^}]*\}|\$[A-Za-z_]\w*(?:\.[A-Za-z0-9_]+|\[[^\]$]+\])*\$(?![0-9])")
_EXT = re.compile(r"\.[a-z0-9]{2,6}$", re.I)

# (action pattern, key) where a step writes the path it is given: a new file name there is the flow's own output
PRODUCERS = [
    (r"/(?:write|write_text|append|save_as|save|create_workbook|export_to_file|export|export_to_pdf)$", "path"),
    (r"/(?:copy|move|zip|encrypt|merge|rename|convert)$", "target"),
    (r"^ooa\.(?:pdf|pgp)/", "target"), (r"/download_file$", "path"), (r"/(?:download|drive_download)$", "local_path"),
    (r"/(?:upload|upload_file|upload_files)$", "remote_path"), (r"/end_session$", "save_to"),
    (r"^ooa\.log/write$", "path"), (r"/export_to_pdf$", "local_path"), (r"/save_all$", "save_to"),
    (r"^ooa\.folder/create$", "path"),
]


def norm(text):
    return " ".join(str(text or "").replace("\\\\", "\\").lower().split())


def haystack(rec):
    """What the record's answer may use: the request, plus the known variables of a step-builder part."""
    parts = [rec.get("request") or ""]
    for v in rec.get("variables") or []:
        if isinstance(v, dict) and v.get("name"):
            parts.append(str(v["name"]))
    return norm(" ".join(parts))


def words_hay(hay):
    """The haystack without its links and addresses: a folder name is said only as a word or path of the request,
    not because a link happens to hold it ('rates' of https://x.org/rates/fx.zip)."""
    return _EMAIL.sub(" ", _URL.sub(" ", hay))


def _bounded(w, hay):
    return re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", hay) is not None


def _word_in(word, hay):
    """Is one name / path segment said? Whole words only (no '<3 letters are fine', no 'rate' for 'Rates')."""
    w = word.strip(" ._-'\"").lower()
    if not w:
        return True
    if w.isdigit():
        return True
    for c in {w, w.replace("_", " "), w.replace("-", " "), w.replace("_", "-"), w.replace("-", "_"),
              w.replace("_", ""), w.replace(" ", "_")}:
        if c and (_bounded(c, hay) or _bounded(c + "s", hay) or _bounded(c + "es", hay)):
            return True
    stem = re.sub(r"\.[a-z0-9]{2,5}$", "", w)
    if stem != w and len(stem) >= 5:                     # 'date_merchant.jpg' for 'rename it date_merchant'
        return _word_in(stem, hay)
    return False


def path_parts(value):
    """A path / link value -> (folder part, last part) of its literal text ('' folder when it has none); the
    variables written in it (${x}, $x$) are kept as they are."""
    v = str(value).rstrip("\\/")
    m = re.match(r"^(.*[\\/])([^\\/]*)$", v)
    return (m.group(1), m.group(2)) if m else ("", v)


def _is_abs(text):
    return bool(re.match(r"^(?:[a-z]:\\|\\\\)", norm(text)))


def _folder_said(folder, hay):
    """A drive / network folder is said when the request writes it (or a path inside it); a relative folder
    ('Shared Documents/Payroll', 'Reports/Sales') when the request writes it too (a single name: as a word)."""
    f = norm(_REF.sub("", folder)).rstrip("\\/ ")
    if not f:
        return True
    if _is_abs(f):
        hay_paths = [norm(m.group(0)).rstrip(".,;:\\") for m in _WINPATH.finditer(hay)]
        return any(p == f or p.startswith(f + "\\") or p.startswith(f + " ") for p in hay_paths) or f in hay
    g = f.strip("/")
    for c in (f, g, g.replace("/", " / "), g.replace("/", " > "), g.replace("/", "\\"), g.replace("/", " ")):
        if c and re.search(r"(?<![\w/\\])" + re.escape(c) + r"(?![\w])", hay):
            return True
    if "/" in g or "\\" in g:
        return False                       # a folder path of two or more names: the request writes it
    return _parts_said(g, words_hay(hay))


def _parts_said(text, hay):
    lit = _REF.sub(" ", text)
    parts = [p for p in re.split(r"[\\/]+|\*|\?|\s+-\s+", norm(lit)) if p.strip(" ._-:")]
    parts = [p for p in parts if p not in ("https:", "http:")]
    return all(_word_in(p, hay) for p in parts)


def judge(value, hay, producer=False):
    """'said' (the request names it), 'derived' (a new file the flow writes inside a folder the request names, or
    inside a folder held in a variable), 'folder' (the folder part is not said, the name is), else 'invented'."""
    for p in SYSTEM_PREFIXES:
        if value.startswith(p):
            value = value[len(p):]
    v = norm(value)
    if not v or v in hay or v in SYSTEM_VALUES:
        return "said"
    if re.fullmatch(r"[\d.,:%\s-]+", v):
        return "said"                                        # numbers / times: not checked here
    if _EMAIL.fullmatch(v):
        return "said" if v in hay else "invented"
    if v.startswith("#") and _bounded(v[1:], hay):
        return "said"
    lit = _REF.sub(" ", value)                               # literal parts around ${...} / $x
    if "$" in value and not lit.strip(" \\/._-:"):
        return "said"
    if re.fullmatch(r"\*(?:\.\*)?", v):
        return "said"                                        # every file
    if _DN_PART.search(value):
        return "said" if all(_word_in(p, words_hay(hay)) for _, p in _DN_PART.findall(value)) else "invented"
    if _URL.match(value.strip()):
        return "said" if v.rstrip("/") in hay else "invented"
    if not re.search(r"[\\/]", lit):
        parts = [p for p in re.split(r"\*|\?|\s+-\s+", norm(lit)) if p.strip(" ._-:")]
        return "said" if all(_word_in(p, words_hay(hay)) for p in parts or [v]) else "invented"
    folder, last = path_parts(value)
    folder_lit = _REF.sub("", folder).strip(" \\/:")
    folder_ok = _folder_said(folder, hay) if folder_lit else bool(folder)
    last_lit = _REF.sub("", last)
    last_ok = (not _EXT.sub("", last_lit).strip(" ._-")) or _word_in(last_lit, words_hay(hay)) or \
        bool(_EXT.search(last)) and _bounded(norm(last), hay)          # 'fx_rates.zip' of the link
    if folder_ok and last_ok:
        return "said"
    if not folder and last_ok:
        return "said"
    if folder and folder_ok and producer and _EXT.search(last_lit or last) and "*" not in last:
        return "derived"
    if folder and not folder_ok and (last_ok or producer and _EXT.search(last_lit or last)):
        return "folder"
    return "invented"


def grounded(value, hay):
    """Is a literal value said by the haystack (or a file the flow makes in a folder it names)?"""
    return judge(value, hay) in ("said", "derived")


def is_producer(action, key):
    return any(key == k and re.search(a, action or "") for a, k in PRODUCERS)


def _values(x, key):
    if isinstance(x, dict):
        if set(x) <= {"op", "left", "right"} or "expr" in x or "literal" in x:
            return
        for k, v in x.items():
            yield from _values(v, k if k not in ("args", "fields", "source_inputs", "iterator_args", "value") or
                               not isinstance(v, dict) else key)
    elif isinstance(x, list):
        for v in x:
            yield from _values(v, key)
    elif isinstance(x, str):
        yield key, x


def all_values(step):
    """(key, value) of every string a step's arguments hold, the key being the innermost argument name."""
    for k, v in (step.get("args") or {}).items():
        yield from _values(v, k)


_DOTTED = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*|\[\d+\])+")
_ACTION_ID = re.compile(r"(?:core|ooa\.[a-z0-9_]+)/[a-z0-9_]+")
FILE_TOOLKITS = ("ooa.file/", "ooa.file_transfer/", "ooa.sftp/", "core/loop", "ooa.ftp/", "ooa.salesforce/",
                 "ooa.email/", "ooa.outlook_desktop/", "ooa.microsoft365/", "ooa.google/", "ooa.pdf/", "ooa.box/")
DATA_TOOLKITS = ("ooa.json/", "ooa.rest/", "ooa.xml/", "ooa.dictionary/", "core/")


def ext_said(ext, hay):
    ext = ext.lower().lstrip(".")
    if any(_bounded(w, hay) or _bounded(w + "s", hay) for w in EXT_WORDS.get(ext, (ext,))):
        return True
    return re.search(r"\." + re.escape(ext) + r"(?![a-z0-9])", hay) is not None


def exempt(action, key, value):
    """A value that is no user detail: a JSON / data path ('shipment.status'), an action id ('ooa.servicenow/query'),
    a date pattern."""
    v = value.strip()
    if _ACTION_ID.fullmatch(v):
        return True
    if key == "pattern" and not action.startswith(FILE_TOOLKITS):
        return True                                          # a date / text pattern
    if key == "path" and action.startswith(DATA_TOOLKITS) and bool(_DOTTED.fullmatch(v)):
        return True
    return key == "item" and not re.search(r"[\\/.]", v)     # an item id / name, not a path


def type_filter_issue(action, key, value, hay):
    """A file type filter ('*.pdf', extension 'jpg') the request does not ask for ('all files', 'the exports')."""
    if key == "extension":
        exts = [e for e in re.split(r"[,;\s]+", value.lower().strip(" .")) if e]
        return not all(ext_said(e, hay) for e in exts)
    if key in ("pattern", "name_pattern") and action.startswith(FILE_TOOLKITS):
        m = re.fullmatch(r"\*\.(\w+)", value.strip())
        if m:
            return not ext_said(m.group(1), hay)
    return False


def db_type_said(value, hay):
    return any(_bounded(w, hay) for w in DB_WORDS.get(value.lower(), (value.lower(),)))


def flow_constants(flow):
    """[(step id, key, value)] for every detail argument literal in the flow (nested args, iterator_args, fields)."""
    out = []
    for s, _ in F.walk(flow.get("steps") if isinstance(flow, dict) else []):
        for key, val in all_values(s):
            if key in DETAIL_KEYS and val.strip() and not _FULL_REF.fullmatch(val.strip()) \
                    and not exempt(s.get("action") or "", key, val):
                out.append((s.get("id"), key, val))
    return out


def bare_refs(flow):
    """[(step id, key, value)] where a $name.field sits inside other text (sent literally by the runner)."""
    out = []
    for s, _ in F.walk(flow.get("steps") if isinstance(flow, dict) else []):
        for key, val in all_values(s):
            if key in EXPRESSION_KEYS or "$" not in val or _FULL_REF.fullmatch(val.strip()):
                continue
            rest = _INTERP_OK.sub(" ", val)
            if _BARE_IN_TEXT.search(rest):
                out.append((s.get("id"), key, val))
    return out


_EXAMPLE = re.compile(r"\b(?:e\.g\.|for example)\s*:?\s*[^;)]*?(?=[;)]|\.\s|\.?$)", re.I)


def text_details(text):
    """The detail tokens written in a text: addresses, links, paths, file names, file type globs, #channels,
    UPPER_SNAKE names, saved connection names, AD name parts, company names, database names. An example ("e.g.
    ourfirm.zendesk.com", "for example D:\\AP\\Invoices") is not a detail of the flow."""
    t = _EXAMPLE.sub(" ", _REF.sub(" ", str(text or "")))
    found = []
    for rx in (_EMAIL, _URL, _WINPATH):
        for m in rx.finditer(t):
            found.append(m.group(0).rstrip(".,;:"))
    t2 = _WINPATH.sub(" ", _URL.sub(" ", _EMAIL.sub(" ", t)))
    for rx in (_FILE, _HASH, _IDENT):
        found += [m.group(0).strip() for m in rx.finditer(t2)]
    found += ["*." + m.group(1) for m in _GLOB.finditer(t2)]
    found += [m.group(1) for m in _SAVED_CONN.finditer(t)]
    found += [m.group(0) for m in _DN_PART.finditer(t2)]
    found += [m.group(1) for m in _ORG.finditer(t2)]
    found += ["db:" + m.group(1) for m in _DBNAME.finditer(t2)]
    return [f for f in found if f]


def detail_said(tok, hay, made=()):
    """Is a detail token of a text said by the request (or a file the flow writes)?"""
    n = norm(tok)
    if n in made or norm(path_parts(tok)[1]) in made:
        return True
    if tok.startswith("db:"):
        return _bounded(tok[3:].lower(), hay)
    if tok.startswith("*."):
        return ext_said(tok[2:], hay)
    m = _DN_PART.fullmatch(tok)
    if m:
        return _word_in(m.group(2), words_hay(hay))
    if tok.lower() in ORG_WORDS:
        return _bounded(tok.lower(), words_hay(hay))
    if _URL.match(tok):
        return n.rstrip("/.") in hay
    return grounded(tok, hay)


def record_texts(rec):
    """(where, text) for the texts a record trains on besides the flow's arguments."""
    out = []
    for k in ("clear", "understand"):
        if rec.get(k):
            out.append((k, rec[k]))
    for i, line in enumerate(rec.get("steps") or []):
        out.append(("steps", str(line).split(" -> ")[0]))
    fl = rec.get("flow")
    if isinstance(fl, dict):
        out.append(("flow.description", fl.get("description") or ""))
        for s, _ in F.walk(fl.get("steps")):
            out.append(("step.description", s.get("description") or ""))
        for v in fl.get("variables") or []:
            if isinstance(v, dict) and v.get("description"):
                out.append(("variable.description", v["description"]))
    return out


def site_aliases(value):
    """How a text names a SharePoint site value 'host:/sites/HR': 'the HR SharePoint site', 'the HR site', ..."""
    m = re.search(r"/sites/([\w-]+)", str(value))
    if not m:
        return []
    n = re.escape(m.group(1))
    return [re.compile(rf"\b{n} SharePoint site\b"), re.compile(rf"\bSharePoint site {n}\b"),
            re.compile(rf"\bthe {n} site\b"), re.compile(rf"\bSharePoint \({n}\)")]


_GIVEN = re.compile(r"\bthe given (?:folder|file|path|site|team|channel)\b", re.I)
_AWKWARD = re.compile(r"\bthe the\b|\bon the sheet[,.]|\bsave it as the file\b|\bas the file\b|"
                      r"\bto the file\b to\b|\bfolder in the folder\b|\bin the folder in\b|\binto the folder folder\b|"
                      r"\bon the sheet\b(?! (?-i:[A-Z]))|\b(?:index|file|workbook|folder|document|report|template|script) the [\w ]{0,20}(?:file|folder|workbook|document|report|script|PDF)\b|\bthe file file\b|\bthe folder folder\b|\(\s*\)|\s[,.](?=\s|$)", re.I)


def text_style_issues(text, hay):
    """Wording a rewrite or a template left behind: 'the given folder' when the request gives none, '2 copy/copies',
    doubled words."""
    out = []
    if _GIVEN.search(text) and not _WINPATH.search(hay) and "given" not in hay:
        out.append(_GIVEN.search(text).group(0))
    for m in _TEMPLATE.finditer(text):
        a, b = (m.group(1) or "").lower(), (m.group(2) or "").lower()
        if a and b not in (a + "s", a + "es", a[:-1] + "ies"):
            continue                                          # '2 Reports/Sales' is a path, not a template
        out.append(m.group(0))
    m = _AWKWARD.search(text)
    if m:
        out.append(m.group(0))
    return out


def flow_issues(fl, hay):
    """[(kind, step id, key, value)] of a flow's arguments, the step order kept: a path a step reads is said, or
    written by an earlier step (the same text)."""
    issues, invented, made, written = [], [], set(), set()
    for s, _ in F.walk(fl.get("steps")):
        action = s.get("action") or ""
        for key, val in all_values(s):
            v = val.strip()
            if not v or _FULL_REF.fullmatch(v):
                continue
            if key in DETAIL_KEYS and not exempt(action, key, val):
                if key == "extension" or key == "pattern" and v.startswith("*."):
                    if type_filter_issue(action, key, val, hay):
                        issues.append(("type_filter", s.get("id"), key, val))
                    continue
                if key == "database_type":
                    if not db_type_said(v, hay):
                        issues.append(("flow_constant", s.get("id"), key, val))
                    continue
                if key in ("from_currency", "to_currency"):
                    if not _bounded(v.lower(), words_hay(hay)):
                        issues.append(("flow_constant", s.get("id"), key, val))
                    continue
                if key == "range":
                    if "!" in v and not _word_in(v.split("!")[0].strip("'"), words_hay(hay)):
                        issues.append(("flow_constant", s.get("id"), key, val))
                    continue
                if key == "location":
                    if not re.search(r"\b(?:location|room|venue|online|at)\b", hay) or not _word_in(v, hay):
                        issues.append(("flow_constant", s.get("id"), key, val))
                    continue
                prod = is_producer(action, key)
                if norm(v) in written:
                    j = "said"
                else:
                    j = judge(val, hay, producer=prod)
                if j in ("said", "derived"):
                    made.add(norm(path_parts(v)[1]))
                    made.add(norm(v))
                    if prod:
                        written.add(norm(v))
                else:
                    issues.append(("flow_constant", s.get("id"), key, val))
                    invented.append(val)
            elif _DN_PART.search(v) and key not in DETAIL_KEYS:
                if not all(_word_in(p, words_hay(hay)) for _, p in _DN_PART.findall(v)):
                    issues.append(("ad_name", s.get("id"), key, val))
                    invented.append(val)
            elif key in TEXT_KEYS or key not in CODE_KEYS | EXPRESSION_KEYS and re.search(r"[\\@]|https?:", v):
                for tok in text_details(val):
                    if tok.startswith("#") or _IDENT.fullmatch(tok):
                        continue                              # ticket tags / codes inside a message
                    if not detail_said(tok, hay, made | written):
                        issues.append(("flow_text", s.get("id"), key, tok))
        if action.startswith(("ooa.json/", "ooa.rest/")):
            for key, val in all_values(s):
                if key == "path" and _DOTTED.fullmatch(val.strip()):
                    last = [x for x in re.split(r"[.\[\]]", val) if x and not x.isdigit()][-1]
                    if not _word_in(last, words_hay(hay)):
                        issues.append(("data_path", s.get("id"), key, val))    # a guessed field of the reply
        for k in custom_fields(s):
            if not _word_in(re.sub(r"__c$", "", k), words_hay(hay)):
                issues.append(("flow_constant", s.get("id"), "field", k))
    return issues, invented, made | written


def custom_fields(step):
    """Salesforce custom field names (Name__c) a step writes or reads: a field the request does not name is made
    up."""
    out = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str) and k.endswith("__c"):
                    out.append(k)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            out.extend(re.findall(r"\b[A-Za-z]\w*__c\b", x))
    walk(step.get("args") or {})
    return out


def check_record(rec):
    """[(kind, where, key, value)] grounding issues of one record (empty when every detail is the user's)."""
    hay = haystack(rec)
    issues = []
    fl = rec.get("flow")
    made, invented = set(), []                 # file names the flow makes; values the request does not give
    if isinstance(fl, dict):
        fi, invented, made = flow_issues(fl, hay)
        issues += fi
        for sid, k, v in bare_refs(fl):
            issues.append(("bare_ref_in_text", sid, k, v))
    for where, text in record_texts(rec):
        for tok in text_details(text):
            if not detail_said(tok, hay, made):
                issues.append(("text_detail", where, "", tok))
        for v in invented:                       # a made-up flow value repeated in the text
            m = re.search(r"/sites/([\w-]+)", v)
            said_site = bool(m) and _word_in(m.group(1), words_hay(hay))    # 'SharePoint (Legal)': name said
            pats = ([] if said_site else site_aliases(v)) + ([re.compile(r"(?<![\w@.-])" + re.escape(v) + r"(?![\w@-])")]
                                      if len(v) >= 3 else [])
            if any(p.search(text) for p in pats) and v not in text_details(text):
                issues.append(("text_detail", where, "", v))
        for bad in text_style_issues(text, hay):
            issues.append(("text_style", where, "", bad))
    if rec.get("variables") is not None and rec.get("understand") and \
            norm(rec["understand"]) != norm(rec.get("request")):
        issues.append(("text_changed", "understand", "", rec["understand"]))   # a part's understand is its request
    for bad in text_style_issues(str(rec.get("request") or ""), hay):
        if _TEMPLATE.search(bad):
            issues.append(("text_style", "request", "", bad))
    for pair in rec.get("facts") or []:
        if isinstance(pair, (list, tuple)) and len(pair) == 2 and pair[0] != "schedule" and pair[0] != "filter":
            if not grounded(str(pair[1]), hay) and not detail_said(str(pair[1]), hay, made):
                issues.append(("fact", "facts", pair[0], str(pair[1])))
    for k in ("question",):
        if rec.get(k):
            for tok in text_details(rec[k]):
                if not detail_said(tok, hay):
                    issues.append(("text_detail", k, "", tok))
    if rec.get("kind") == "clarify" and not rec.get("missing"):
        issues.append(("clarify_without_missing", "missing", "", ""))
    return issues


def scan(paths, examples=3):
    import collections
    report = {}
    for p in paths:
        c, recs, ex = collections.Counter(), 0, collections.defaultdict(list)
        n = 0
        for i, line in enumerate(open(p, encoding="utf-8"), 1):
            if not line.strip():
                continue
            n += 1
            iss = check_record(json.loads(line))
            if iss:
                recs += 1
            for kind, where, k, v in iss:
                c[kind] += 1
                if len(ex[kind]) < examples:
                    ex[kind].append((i, where, k, v))
        report[os.path.basename(p)] = {"records": n, "records_with_issues": recs, "issues": dict(c),
                                       "examples": dict(ex)}
    return report


if __name__ == "__main__":
    rep = scan(sys.argv[1:])
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    sys.exit(1 if any(r["records_with_issues"] for r in rep.values()) else 0)
