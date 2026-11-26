"""Grounding check for SFT records: which details does a record's flow or text hold that its request never gave?

A flow constant in a detail argument (an email address, a path / folder / file name, a URL, a saved connection, a
team / channel / sheet / table / site / group / project name, an id) that the request (and, for a step-builder part,
the listed known variables) does not name teaches the model to invent details. Such a value belongs in an input
variable. The same holds for the record's texts ("clear", "understand", "facts", the flow and step descriptions and
the "steps" lines): they may only repeat what the request says.

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
    "template", "template_path", "workbook", "public_key_file", "private_key_file", "key_file",
    "url", "site_url", "base_url", "endpoint", "drive", "site", "site_id", "instance_url", "org_url", "account_url",
    "connection", "team_name", "channel_name", "channel", "team", "sheet", "sheet_name", "table_name", "procedure",
    "group", "groups", "assignment_group", "project", "queue", "list", "list_name", "database", "schema", "bucket",
    "container", "repo", "owner", "topic", "host", "username", "mailbox", "shared_mailbox", "plan", "plan_id",
    "warehouse", "warehouse_id", "job_id", "spreadsheet_id", "document_id", "calendar_id", "folder_id", "drive_id",
    "board", "board_id", "workspace", "space", "space_key", "report", "report_path", "integration", "eib",
    "consumer_group", "server", "domain", "ou", "manager", "assignee", "approver", "user", "login",
    "from_contains", "sender", "from", "pattern",
}
# platform constants that translate a request word, not user details
SYSTEM_VALUES = {
    "incident", "change_request", "sc_req_item", "sc_task", "problem", "sys_user", "sc_request", "task",
    "inbox", "sent items", "sent", "drafts", "archive", "me", "primary", "sheet1", "localhost", "root",
}
# arguments read as an expression, where a bare $name.field is a reference
EXPRESSION_KEYS = {"condition", "expr", "expression", "formula", "where"}
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"https?://[^\s\"'<>)]+")
_WINPATH = re.compile(r"(?:[A-Za-z]:\\|\\\\)[^\s\"'<>|,;()]*")
_FILE = re.compile(r"(?<![\w$.{\\/-])[\w][\w.-]*?\.(?:csv|xlsx|xls|xlsm|docx|doc|pdf|txt|json|xml|zip|msg|eml|log|asc|"
                   r"pgp|gpg|png|jpg|html|py|ps1|bat|sql|tsv|dat|ini)\b", re.I)
_HASH = re.compile(r"(?<![\w&])#[a-z][\w-]+")
_IDENT = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")        # UPPER_SNAKE names (tables, procedures)
_SAVED_CONN = re.compile(r"saved connection[s]? ['\"]?([\w.-]+)", re.I)
_REF = re.compile(r"\$\{[^}]*\}|\$[A-Za-z_][\w]*(?:\.[\w]+|\[[^\]]*\])*\$?")
_BARE_IN_TEXT = re.compile(r"\$(?!\{)[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
_FULL_REF = re.compile(r"\$[A-Za-z_]\w*(?:\.[A-Za-z0-9_]+|\[[^\]]+\])*\$?")
_INTERP_OK = re.compile(r"\$\{[^}]*\}|\$[A-Za-z_]\w*(?:\.[A-Za-z0-9_]+|\[[^\]$]+\])*\$(?![0-9])")


def norm(text):
    return " ".join(str(text or "").replace("\\\\", "\\").lower().split())


def haystack(rec):
    """What the record's answer may use: the request, plus the known variables of a step-builder part."""
    parts = [rec.get("request") or ""]
    for v in rec.get("variables") or []:
        if isinstance(v, dict) and v.get("name"):
            parts.append(str(v["name"]))
    return norm(" ".join(parts))


def _word_in(word, hay):
    w = word.strip(" ._-'\"").lower()
    if len(w) < 3 and not w.isdigit():
        return True
    if w in hay:
        return True
    stem = re.sub(r"\.[a-z0-9]{2,5}$", "", w)
    if stem != w and len(stem) >= 5 and stem in hay:        # 'date_merchant.jpg' for 'rename it date_merchant'
        return True
    # "statement" in "statements", "#ap-alerts" given as "ap-alerts", "Open Items" given as "open items"
    return w.rstrip("s") in hay or w.replace("_", " ") in hay or w.replace("-", " ") in hay


_EXT = re.compile(r"\.[a-z0-9]{2,6}$", re.I)


def path_parts(value):
    """A path / link value -> (folder part, last part) of its literal text ('' folder when it has none); the
    variables written in it (${x}, $x$) are kept as they are."""
    v = str(value).rstrip("\\/")
    m = re.match(r"^(.*[\\/])([^\\/]*)$", v)
    return (m.group(1), m.group(2)) if m else ("", v)


def _parts_said(text, hay):
    lit = _REF.sub(" ", text)
    parts = [p for p in re.split(r"[\\/]+|\*|\?|\s+-\s+", norm(lit)) if p.strip(" ._-:")]
    parts = [p for p in parts if p not in ("https:", "http:")]
    return all(_word_in(p, hay) for p in parts)


def judge(value, hay):
    """'said' (the request names it), 'derived' (a new file named inside a folder the request names, or one held in
    a variable: an output the flow makes), 'folder' (the folder part is not said, the name is), else 'invented'."""
    v = norm(value)
    if not v or v in hay or v in SYSTEM_VALUES:
        return "said"
    if re.fullmatch(r"[\d.,:%\s-]+", v):
        return "said"                                        # numbers / times: not checked here
    if _EMAIL.fullmatch(v):
        return "said" if v in hay else "invented"
    if v.startswith("#") and v[1:] in hay:
        return "said"
    lit = _REF.sub(" ", value)                               # literal parts around ${...} / $x
    if "$" in value and not lit.strip(" \\/._-:"):
        return "said"
    if not re.search(r"[\\/*?]", lit) or re.search(r"\s-\s", lit) and not re.search(r"[\\/]", lit):
        parts = [p for p in re.split(r"\*|\?|\s+-\s+", norm(lit)) if p.strip(" ._-:")]
        return "said" if all(_word_in(p, hay) for p in parts or [v]) else "invented"
    if _parts_said(value, hay):
        return "said"
    folder, last = path_parts(value)
    folder_ok = _parts_said(folder, hay) if _REF.sub("", folder).strip(" \\/:") else bool(folder)
    last_ok = _parts_said(last, hay)
    if folder and folder_ok and _EXT.search(_REF.sub("", last) or last):
        return "derived"
    if folder and not folder_ok and last_ok and _REF.sub("", folder).strip(" \\/:"):
        return "folder"
    return "invented"


def grounded(value, hay):
    """Is a literal value said by the haystack (or a file the flow makes in a folder it names)?"""
    return judge(value, hay) in ("said", "derived")


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


_DOTTED = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*|\[\d+\])+")
_ACTION_ID = re.compile(r"(?:core|ooa\.[a-z0-9_]+)/[a-z0-9_]+")
FILE_TOOLKITS = ("ooa.file/", "ooa.file_transfer/", "ooa.sftp/", "core/loop", "ooa.ftp/")
DATA_TOOLKITS = ("ooa.json/", "ooa.rest/", "ooa.xml/", "ooa.dictionary/", "core/")


def exempt(action, key, value):
    """A value that is no user detail: a JSON / data path ('shipment.status'), an action id ('ooa.servicenow/query')."""
    v = value.strip()
    if _ACTION_ID.fullmatch(v):
        return True
    if key == "pattern" and (not action.startswith(FILE_TOOLKITS) or re.fullmatch(r"\*(?:\.\w+)?", v)):
        return True                                          # a date pattern, or only a file type ('*.pdf')
    return key == "path" and action.startswith(DATA_TOOLKITS) and bool(_DOTTED.fullmatch(v))


def flow_constants(flow):
    """[(step id, key, value)] for every detail argument literal in the flow (nested args, iterator_args, fields)."""
    out = []
    for s, _ in F.walk(flow.get("steps") if isinstance(flow, dict) else []):
        for k, v in (s.get("args") or {}).items():
            for key, val in _values(v, k):
                if key in DETAIL_KEYS and val.strip() and not _FULL_REF.fullmatch(val.strip()) \
                        and not exempt(s.get("action") or "", key, val):
                    out.append((s.get("id"), key, val))
    return out


def bare_refs(flow):
    """[(step id, key, value)] where a $name.field sits inside other text (sent literally by the runner)."""
    out = []
    for s, _ in F.walk(flow.get("steps") if isinstance(flow, dict) else []):
        for k, v in (s.get("args") or {}).items():
            for key, val in _values(v, k):
                if key in EXPRESSION_KEYS or "$" not in val or _FULL_REF.fullmatch(val.strip()):
                    continue
                rest = _INTERP_OK.sub(" ", val)
                if _BARE_IN_TEXT.search(rest):
                    out.append((s.get("id"), key, val))
    return out


def text_details(text):
    """The detail tokens written in a text: addresses, links, paths, file names, #channels, UPPER_SNAKE names,
    saved connection names."""
    t = _REF.sub(" ", str(text or ""))
    found = []
    for rx in (_EMAIL, _URL, _WINPATH):
        for m in rx.finditer(t):
            found.append(m.group(0).rstrip(".,;:"))
    t2 = _WINPATH.sub(" ", _URL.sub(" ", _EMAIL.sub(" ", t)))
    for rx in (_FILE, _HASH, _IDENT):
        found += [m.group(0).strip() for m in rx.finditer(t2)]
    found += [m.group(1) for m in _SAVED_CONN.finditer(t)]
    return [f for f in found if f]


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
    return out


def site_aliases(value):
    """How a text names a SharePoint site value 'host:/sites/HR': 'the HR SharePoint site', 'the HR site', ..."""
    m = re.search(r"/sites/([\w-]+)", str(value))
    if not m:
        return []
    n = re.escape(m.group(1))
    return [re.compile(rf"\b{n} SharePoint site\b"), re.compile(rf"\bSharePoint site {n}\b"),
            re.compile(rf"\bthe {n} site\b"), re.compile(rf"\bSharePoint \({n}\)")]


def check_record(rec):
    """[(kind, where, key, value)] grounding issues of one record (empty when every detail is the user's)."""
    hay = haystack(rec)
    issues = []
    fl = rec.get("flow")
    made, invented = set(), []                 # file names the flow makes; values the request does not give
    if isinstance(fl, dict):
        for sid, k, v in flow_constants(fl):
            j = judge(v, hay)
            if j in ("said", "derived"):
                made.add(norm(path_parts(v)[1]))
                made.add(norm(v))
            else:
                issues.append(("flow_constant", sid, k, v))
                invented.append(v)
        for sid, k, v in bare_refs(fl):
            issues.append(("bare_ref_in_text", sid, k, v))
    for where, text in record_texts(rec):
        for tok in text_details(text):
            if not grounded(tok, hay) and norm(tok) not in made and norm(path_parts(tok)[1]) not in made:
                issues.append(("text_detail", where, "", tok))
        for v in invented:                       # a made-up flow value repeated in the text
            pats = site_aliases(v) + ([re.compile(r"(?<![\w@.-])" + re.escape(v) + r"(?![\w@-])")]
                                      if len(v) >= 3 else [])
            if any(p.search(text) for p in pats) and v not in text_details(text):
                issues.append(("text_detail", where, "", v))
    for pair in rec.get("facts") or []:
        if isinstance(pair, (list, tuple)) and len(pair) == 2 and pair[0] != "schedule" and pair[0] != "filter":
            if not grounded(str(pair[1]), hay):
                issues.append(("fact", "facts", pair[0], str(pair[1])))
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
