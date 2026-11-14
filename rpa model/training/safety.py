"""Plain warnings for flow steps that destroy or change things that cannot easily be undone.

    warnings(flow, catalog) -> ["Check before running: step s3 deletes files or records (Delete file, ooa.box/delete_file).", ...]

A step is flagged when a word of its action id or catalog name is on the list below (delete, remove, purge, terminate,
wipe, drop, overwrite, empty trash / recycle bin, send to all, bulk update / delete, revoke, expire, ...), when it is
given overwrite: true, or when a database statement in its arguments deletes, drops, truncates or updates rows.
Actions that only change data in memory (tables, lists, dictionaries, text, JSON / XML values) are not flagged: nothing
outside the bot changes. A flagged step inside a loop says so (it runs once per item). Pure standard library.
"""
import re

# word in the action id / catalog name -> what the step does, in plain words
WORDS = [
    (("delete", "remove", "erase", "destroy"), "deletes files or records"),
    (("purge",), "purges data"),
    (("wipe",), "wipes data"),
    (("drop",), "drops data"),
    (("truncate",), "truncates data"),
    (("overwrite",), "overwrites existing data"),
    (("terminate", "kill"), "terminates running resources"),
    (("shutdown", "reboot", "restart"), "restarts or shuts down a machine or service"),
    (("uninstall",), "uninstalls software"),
    (("revoke",), "revokes access"),
    (("expire",), "expires passwords or access"),
    (("disable", "deactivate", "suspend"), "disables accounts or features"),
    (("reset",), "resets settings or passwords"),
    (("void", "cancel"), "cancels or voids items"),
    (("clear",), "clears data"),
    (("bulk", "mass"), "changes many items at once"),
    (("broadcast",), "sends a message to many people"),
]
PHRASES = [  # several words together
    (re.compile(r"\bempty\b.*\b(trash|recycle|bin)\b"), "empties the trash"),
    (re.compile(r"\bsend\b.*\ball\b|\ball\b.*\b(users|recipients|contacts)\b"), "sends to everyone"),
    (re.compile(r"\bdelete\b.*\ball\b"), "deletes everything that matches"),
]
# toolkits whose actions only change values inside the running bot
IN_MEMORY = {"ooa.datatable", "ooa.list", "ooa.dictionary", "ooa.json", "ooa.xml", "ooa.string", "ooa.text",
             "ooa.number", "ooa.datetime", "ooa.regex", "ooa.variables", "ooa.forms", "ooa.interrupts", "ooa.log",
             "ooa.locks", "ooa.longrunning", "core"}
HARMLESS = {"ooa.desktop/clipboard_clear", "ooa.terminal/clear_screen", "ooa.datetime/format", "ooa.number/format"}
SQL = re.compile(r"\b(delete\s+from|drop\s+(table|database|schema|view)|truncate(\s+table)?|update\s+\w+\s+set|"
                 r"alter\s+table)\b", re.I)
SQL_KEYS = {"sql", "query", "statement", "command", "commands", "script"}
SQL_TOOLKITS = {"ooa.database", "ooa.snowflake", "ooa.databricks"}


def _words(text):
    return [w for w in re.split(r"[^a-z0-9]+", str(text).lower()) if w]


def _strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from _strings(x)
    elif isinstance(v, list):
        for x in v:
            yield from _strings(x)


def classify(step, catalog):
    """What a destructive step does (plain words) or None."""
    a = step.get("action")
    if not isinstance(a, str):
        return None
    toolkit = a.split("/", 1)[0]
    args = step.get("args") if isinstance(step.get("args"), dict) else {}
    literal_sql = False
    for k, v in args.items():
        if k.lower() in SQL_KEYS:
            for s in _strings(v):
                m = SQL.search(s)
                if m:
                    return f"runs a {' '.join(m.group(0).split()).upper()} database statement"
                literal_sql = literal_sql or not s.lstrip().startswith("$")
    if toolkit in SQL_TOOLKITS and any(k.lower() in SQL_KEYS for k in args):
        if literal_sql:
            return None      # the statement is written out and does not change or remove rows
        return "runs a database statement built at run time that may change or remove rows"
    if args.get("overwrite") is True:
        return "overwrites existing files"
    if toolkit in IN_MEMORY or a in HARMLESS:
        return None
    e = (catalog or {}).get(a) or {}
    words = _words(a.split("/", 1)[-1]) + _words(e.get("name", ""))
    text = " ".join(words)
    for rx, phrase in PHRASES:
        if rx.search(text):
            return phrase
    for keys, phrase in WORDS:
        if any(w in keys for w in words):
            return phrase
    return None


def _walk(steps, in_loop=False):
    for s in steps if isinstance(steps, list) else []:
        if not isinstance(s, dict):
            continue
        yield s, in_loop
        loop = in_loop or s.get("action") == "core/loop"
        for k in ("children", "else", "finally", "default"):
            if isinstance(s.get(k), list):
                yield from _walk(s[k], loop)
        for k in ("elseIf", "catch", "cases"):
            for b in s.get(k) if isinstance(s.get(k), list) else []:
                if isinstance(b, dict):
                    yield from _walk(b.get("children"), loop)
        for st in s.get("states") if isinstance(s.get("states"), list) else []:
            if isinstance(st, dict):
                yield from _walk(st.get("entry"), loop)
                for t in st.get("transitions") if isinstance(st.get("transitions"), list) else []:
                    if isinstance(t, dict):
                        yield from _walk(t.get("steps"), loop)


def warnings(flow, catalog=None):
    """Plain warnings, one per destructive step, in flow order. catalog: dict id -> entry or a list of entries."""
    if isinstance(catalog, list):
        catalog = {e["id"]: e for e in catalog}
    out = []
    for s, in_loop in _walk(flow.get("steps") if isinstance(flow, dict) else flow):
        what = classify(s, catalog)
        if not what:
            continue
        a = s.get("action")
        name = ((catalog or {}).get(a) or {}).get("name")
        where = f"{name}, {a}" if name else a
        loop = ", once for every item of a loop" if in_loop else ""
        out.append(f"Check before running: step {s.get('id') or '?'} {what} ({where}){loop}.")
    return out
