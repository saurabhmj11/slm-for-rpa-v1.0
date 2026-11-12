"""Generate valid flows that teach COMBINING actions, not specific chains (round 2 after gen_chains.py).

gen_chains.py builds a fixed list of 47 chains, and the model learned those chains rather than how to combine
actions: on test set 2 (whose chains are never in training) it got 5/15 right. This builds many different,
type-correct combinations from the ~65 everyday actions, in four shapes:

  random  : 2-4 actions where each step's input comes from an earlier output (type-matched), across toolkits;
            Excel / Word / browser steps get their open (or create) step first, as a person would do it
  folder  : list a folder's files -> loop over them -> act on each file (copy / move / delete / read / count ...),
            sometimes only for one file type (core/if with ooa.file/has_extension)
  content : read some text (PDF, text file, web page, Word, email) -> core/if on what it says -> act
  pattern : a file action with a file-type filter (pattern "*.pdf") alone or followed by another step

Honest test sets: a flow is skipped when its actions (ignoring core/ steps) are the same set as a test request's
needed actions and it would count as right for that request (eval/plain_requests.jsonl and plain_requests_2.jsonl).
So the model never trains on an answer to a test request.

Usage: python gen_combos.py CATALOG.jsonl PLATFORM_REPO OUT.jsonl [n_flows=600] [seed=7]
Writes {"flow": {...}, "shape": "...", "actions": [...]} per line; everyday requests are written separately.
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_flows as G
import gen_chains as GC

O = "ooa."
POOL = ["browser/click", "browser/close", "browser/download", "browser/extract_table", "browser/get_text",
        "browser/navigate", "browser/open", "browser/screenshot", "browser/type_text", "csv/read", "csv/write",
        "datatable/filter", "datatable/remove_duplicates", "datatable/row_count", "datatable/sort",
        "datatable/sum_column", "datatable/write_to_file", "email/forward", "email/get", "email/list", "email/move",
        "email/reply", "email/save_attachments", "email/send", "excel/close", "excel/create_workbook",
        "excel/get_all_cells", "excel/get_cell", "excel/get_sheet_names", "excel/open", "excel/read_column",
        "excel/read_row", "excel/row_count", "excel/save", "excel/save_as", "excel/set_cell", "excel/write_table",
        "file/copy", "file/delete", "file/get_info", "file/list", "file/move", "file/read_text", "file/unzip",
        "file/write_text", "file/zip", "folder/copy", "folder/create", "folder/delete", "folder/list_subfolders",
        "folder/rename_folder", "pdf/extract_text", "pdf/get_properties", "pdf/merge", "pdf/page_count", "pdf/split",
        "pdf/to_images", "word/add_paragraph", "word/add_table", "word/close", "word/create_document", "word/open",
        "word/read_text", "word/replace_text", "word/save"]
OPENERS = {"excel": ["excel/open", "excel/create_workbook"], "word": ["word/open", "word/create_document"],
           "browser": ["browser/open"]}
CLOSERS = {"excel": ["excel/save", "excel/close"], "word": ["word/save", "word/close"], "browser": ["browser/close"]}
SESSION_TK = set(OPENERS)
PER_FILE = ["file/copy", "file/move", "file/delete", "pdf/page_count", "pdf/extract_text", "pdf/split",
            "pdf/get_properties", "file/read_text", "file/get_info", "pdf/to_images", "file/zip"]
EXT_FOR = {"pdf/page_count": "pdf", "pdf/extract_text": "pdf", "pdf/split": "pdf", "pdf/get_properties": "pdf",
           "pdf/to_images": "pdf", "file/read_text": "txt"}
TEXT_SOURCES = {"pdf/extract_text": None, "file/read_text": None, "browser/get_text": "browser/open",
                "word/read_text": "word/open", "email/get": None}
ACTS_ON_TEXT = ["email/send", "file/move", "file/copy", "file/write_text", "email/forward", "email/move",
                "file/delete", "email/reply"]
WORDS = ["overdue", "urgent", "invoice", "error", "approved", "rejected", "payment due", "confidential",
         "cancelled", "refund", "failed", "paid"]
TEXT_IN = {"text", "body", "content", "message"}
TEXTY = {"text", "body", "content", "value", "result", "html", "summary"}
WIRABLE = {"File path", "Text", "Table", "List", "Dictionary", "DateTime"}
EXTS = ["pdf", "xlsx", "csv", "docx", "txt", "png", "zip", "xml", "jpg"]


# What kind of file a path / file-list output holds, and which kinds a path input can take.
# Path inputs not listed (save-to paths, targets, screenshots, new workbooks) are never wired: they name a new file.
OUT_KIND = {("file/zip", "path"): "zip", ("pdf/merge", "path"): "pdf", ("folder/create", "path"): "folder",
            ("folder/rename_folder", "path"): "folder", ("folder/copy", "path"): "folder",
            ("datatable/write_to_file", "path"): "table", ("browser/download", "path"): "file",
            ("browser/screenshot", "path"): "image", ("file/copy", "path"): "file", ("file/move", "path"): "file",
            ("pdf/split", "files"): "pdf", ("email/save_attachments", "files"): "file", ("file/list", "files"): "file",
            ("pdf/to_images", "files"): "image", ("file/unzip", "files"): "file", ("file/copy", "files"): "file",
            ("file/move", "files"): "file", ("folder/list_subfolders", "folders"): "folder"}
ANY_FILE = {"file", "pdf", "zip", "table", "image"}
IN_KIND = {("csv/read", "path"): {"table", "file"}, ("excel/open", "path"): {"table", "file"},
           ("word/open", "path"): {"file"}, ("file/unzip", "path"): {"zip", "file"},
           ("file/read_text", "path"): {"file"}, ("file/list", "folder"): {"folder"},
           ("file/copy", "source"): ANY_FILE, ("file/move", "source"): ANY_FILE, ("file/delete", "path"): ANY_FILE,
           ("file/zip", "source"): {"folder"} | ANY_FILE, ("file/get_info", "path"): ANY_FILE,
           ("folder/copy", "source"): {"folder"}, ("folder/delete", "path"): {"folder"},
           ("folder/list_subfolders", "folder"): {"folder"}, ("folder/rename_folder", "path"): {"folder"},
           ("pdf/merge", "files"): {"pdf"}, ("email/send", "attachments"): ANY_FILE}
for _a in ("pdf/extract_text", "pdf/page_count", "pdf/split", "pdf/get_properties", "pdf/to_images"):
    IN_KIND[(_a, "path")] = {"pdf", "file"}


def in_kinds(aid, name):
    a = aid[len(O):] if aid.startswith(O) else aid
    if name in ("target_folder", "folder") and (a, name) not in IN_KIND:
        return {"folder"}
    return IN_KIND.get((a, name))


def tk(aid):
    return aid.split("/")[0]


class Builder:
    """Collects variables and steps; '$name' args become flow inputs unless an earlier step produced them."""

    def __init__(self, byid, rng):
        self.byid, self.rng = byid, rng
        self.variables, self.names, self.avail, self.descs, self.n, self.used = [], set(), [], [], 0, 0

    def new_name(self, base):
        base = G.snake(base)[:40]
        n, k = base, 2
        while n in self.names:
            n, k = f"{base}_{k}", k + 1
        self.names.add(n)
        return n

    def input(self, name, typ, desc=None, path=False):
        alts = ["my_" + name, name] + ([name + "_path"] if path and not name.endswith("path") else [])
        vn = self.new_name(name if self.rng.random() < 0.75 else self.rng.choice(alts))
        v = {"name": vn, "type": typ, "direction": "input", "required": True}
        if desc:
            v["description"] = desc[:120]
        self.variables.append(v)
        return vn

    def local(self, name, typ="Any"):
        vn = self.new_name(name)
        self.variables.append({"name": vn, "type": typ, "direction": "local"})
        return vn

    def step(self, aid, fixed=None, wire=True):
        """One step for catalog action aid. fixed: {input: value} set by the caller; other required inputs are
        wired from earlier outputs of a matching type (wire=True) or become flow inputs."""
        e = self.byid[aid]
        args = dict(fixed or {})
        for i in [i for i in e["inputs"] if i["required"] and i["name"] not in args]:
            t = G.vtype(i["type"]) or "Any"
            src = None
            if wire and i["type"] in WIRABLE and not i["name"].lower().endswith("id"):
                if i["type"] in ("File path", "List") and in_kinds(aid, i["name"]) is not None:
                    ok = in_kinds(aid, i["name"])
                    src = next((a for a in reversed(self.avail) if a[1] == i["type"] and a[2] in ok), None)
                elif i["type"] in ("File path", "List"):
                    src = None                         # a save-to path or a list it does not know: new input
                elif i["type"] == "Text" and i["name"] not in TEXT_IN:
                    src = None                         # text only flows into text-like inputs (not a column, url, name)
                else:
                    src = next((a for a in reversed(self.avail) if a[1] == i["type"]), None)
            if src:
                args[i["name"]] = "$" + src[0]
                self.avail.remove(src)
                self.used += 1
            else:
                args[i["name"]] = "$" + self.input(i["name"], t, i.get("description"), path=i["type"] == "File path")
        out = {}
        for o in e["outputs"][:2]:
            vn = self.local(o["name"], G.vtype(o["type"]) or "Any")
            out[o["name"]] = vn
            if not o["name"].lower().endswith("id") and not (o["type"] == "Text" and o["name"] not in TEXTY):
                a = aid[len(O):] if aid.startswith(O) else aid
                kind = OUT_KIND.get((a, o["name"])) if o["type"] in ("File path", "List") else None
                if o["type"] not in ("File path", "List") or kind:
                    self.avail.append((vn, o["type"], kind))   # ids, counts and status words never feed other steps
        self.n += 1
        s = {"id": f"s{self.n}", "action": aid, "description": G.clean_desc(e["description"])[:140], "args": args}
        if out:
            s["out"] = out
        self.descs.append(G.clean_desc(e["description"]))
        return s

    def node(self, aid, args, children, desc):
        self.n += 1
        self.descs.append(desc)
        return {"id": f"s{self.n}", "action": aid, "description": desc, "args": args, "children": children}

    def flow(self, steps, outputs=()):
        for vn in outputs:
            for v in self.variables:
                if v["name"] == vn:
                    v["direction"] = "output"
        counter = [0]

        def renumber(nodes):
            for n in nodes:
                counter[0] += 1
                n["id"] = f"s{counter[0]}"
                for k in ("children", "else", "finally"):
                    if isinstance(n.get(k), list):
                        renumber(n[k])
        renumber(steps)
        first = self.descs[0]
        return {"kind": "TaskBot", "id": G.snake(first)[:40] or "flow", "name": " ".join(first.split()[:6]).title(),
                "description": "; then ".join(self.descs), "variables": self.variables, "steps": steps}


def with_session(seq, rng):
    """Put an opener before the first Excel/Word/browser step that needs one, and maybe a closer after."""
    out, opened = [], set()
    for a in seq:
        t = tk(a)
        if t in SESSION_TK and t not in opened:
            if a not in OPENERS[t]:
                out.append(rng.choice(OPENERS[t]) if t != "browser" else "browser/open")
            opened.add(t)
        out.append(a)
    for t in opened:
        if rng.random() < 0.5 and out[-1] not in CLOSERS[t]:
            out.append(rng.choice(CLOSERS[t]))
    return out


def shape_random(byid, rng):
    """2-4 actions; every non-session step after the first uses an output of an earlier step (exact type)."""
    def outs(a):
        return {o["type"] for o in byid[O + a]["outputs"] if not o["name"].lower().endswith("id")
                and not (o["type"] == "Text" and o["name"] not in TEXTY)} & WIRABLE
    def needs(a):
        return [i for i in byid[O + a]["inputs"] if i["required"] and i["type"] in WIRABLE
                and not i["name"].lower().endswith("id")
                and (i["type"] not in ("File path", "List") or in_kinds(O + a, i["name"]) is not None)
                and (i["type"] != "Text" or i["name"] in TEXT_IN)]
    session_steps = set(sum(OPENERS.values(), []) + sum(CLOSERS.values(), []))
    starters = [a for a in POOL if outs(a) and a not in sum(CLOSERS.values(), [])]
    seq = [rng.choice(starters)]
    have = set(outs(seq[0]))
    for _ in range(rng.choice([1, 1, 2, 2, 3])):
        nxt = [a for a in POOL if a not in seq and a not in session_steps and any(i["type"] in have for i in needs(a))]
        if not nxt:
            break
        cross = [a for a in nxt if tk(a) != tk(seq[-1])]
        a = rng.choice(cross if cross and rng.random() < 0.6 else nxt)
        seq.append(a)
        have |= outs(a)
    if len(seq) < 2:
        return None
    seq = with_session(seq, rng)
    b = Builder(byid, rng)
    steps = []
    for a in seq:
        before = b.used
        steps.append(b.step(O + a))
        if a == seq[0] or a in sum(OPENERS.values(), []) or a in sum(CLOSERS.values(), []):
            continue
        if b.used == before:                       # this step did not use anything produced before: not a chain
            return None
    last_out = list((steps[-1].get("out") or {}).values())
    return b.flow(steps, outputs=last_out[:1])


def shape_folder(byid, rng):
    b = Builder(byid, rng)
    ext = None
    act = rng.choice(PER_FILE)
    use_filter = rng.random() < 0.5
    pattern = rng.random() < 0.3 and not use_filter
    want = EXT_FOR.get(act) or rng.choice(EXTS)
    fixed = {}
    if pattern:
        fixed["pattern"] = f"*.{want}"
    lst = b.step(O + "file/list", fixed)
    files = lst["out"]["files"]
    b.avail.clear()
    items, item = files, b.local("file", "String")
    target = None
    inner_fixed = {}
    e = byid[O + act]
    first_req = next(i["name"] for i in e["inputs"] if i["required"])
    inner_fixed[first_req] = "$" + item
    if "target" in {i["name"] for i in e["inputs"] if i["required"]}:
        target = b.input(rng.choice(["target", "archive_folder", "destination"]), "String", "Folder to put the files in.")
        inner_fixed["target"] = "$" + target
    inner = b.step(O + act, inner_fixed, wire=False)
    body = [inner]
    if rng.random() < 0.35:
        body.append(_log(b, item))
    if use_filter:
        ext = want
        cond = {"condition": "ooa.file/has_extension", "args": {"path": "$" + item, "extensions": ext}}
        body = [b.node("core/if", {"condition": cond}, body, f"Only for .{ext} files")]
    loop = b.node("core/loop", {"iterator": "list", "over": "$" + items, "as": item}, body, "Repeat for each file")
    return b.flow([lst, loop])


def _log(b, item):
    b.n += 1
    b.descs.append("Write a log line")
    return {"id": f"s{b.n}", "action": "core/log", "description": "Write a log line",
            "args": {"message": "Processed ${" + item + "}", "level": "info"}}


def shape_content(byid, rng):
    """Read text -> if it mentions a word -> act. File sources act on that same file or send an email;
    an email source acts on that email (forward / move / reply / save its attachments) or sends a new one."""
    b = Builder(byid, rng)
    src = rng.choice(list(TEXT_SOURCES))
    steps = []
    if TEXT_SOURCES[src]:
        steps.append(b.step(O + TEXT_SOURCES[src]))
    s = b.step(O + src)
    steps.append(s)
    text_var = next((v for k, v in (s.get("out") or {}).items() if k in TEXTY), None)
    if text_var is None:
        return None
    b.avail.clear()
    the_file = s["args"].get("path") or (steps[0]["args"].get("path") if len(steps) > 1 else None)
    word = rng.choice(WORDS)
    cond = rng.choice([f"contains(lower(${text_var}), '{word}')", f"contains(${text_var}, '{word.title()}')"])
    if src == "email/get":
        act = rng.choice(["email/forward", "email/move", "email/reply", "email/save_attachments", "email/send"])
        fixed = {"id": s["args"]["id"]} if act != "email/send" else {}
    elif src == "browser/get_text":
        act, fixed = rng.choice(["email/send", "file/write_text", "browser/screenshot"]), {}
    else:
        act = rng.choice(["file/move", "file/copy", "file/delete", "email/send", "email/send"])
        fixed = {}
        if act in ("file/move", "file/copy") and the_file:
            fixed["source"] = the_file
        elif act == "file/delete" and the_file:
            fixed["path"] = the_file
        elif act == "file/delete":
            return None
    if act == "file/write_text":
        fixed["text"] = "$" + text_var
    inner = [b.step(O + act, fixed)]
    node = b.node("core/if", {"condition": cond}, inner, f"Only when the text mentions {word}")
    if rng.random() < 0.25:
        b.n += 1
        b.descs.append("Otherwise write a log line")
        node["else"] = [{"id": f"s{b.n}", "action": "core/log", "description": "Otherwise write a log line",
                         "args": {"message": f"No mention of {word}", "level": "info"}}]
    steps.append(node)
    return b.flow(steps)


def shape_pattern(byid, rng):
    b = Builder(byid, rng)
    act = rng.choice(["file/copy", "file/move", "file/delete", "file/zip", "file/list"])
    ext = rng.choice(EXTS)
    first = b.step(O + act, {"pattern": f"*.{ext}"})
    steps = [first]
    if rng.random() < 0.6:
        follow = rng.choice(["email/send", "file/write_text", "core_log", "file/zip", "file/list"])
        if follow == "core_log":
            b.n += 1
            b.descs.append("Write a log line")
            steps.append({"id": f"s{b.n}", "action": "core/log", "description": "Write a log line",
                          "args": {"message": f"Finished the .{ext} files", "level": "info"}})
        elif follow != act:
            steps.append(b.step(O + follow))
    return b.flow(steps)


SHAPES = {"random": (shape_random, 40), "folder": (shape_folder, 22), "content": (shape_content, 23),
          "pattern": (shape_pattern, 15)}


def actions_of(steps):
    out = []
    for s in steps:
        out.append(s["action"])
        for k in ("children", "else", "finally"):
            out += actions_of(s.get(k) or [])
    return out


def in_order(must, got):
    it = iter(got)
    return all(any(a == m for a in it) for m in must)


def main():
    cat_path, repo, out = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 600
    rng = random.Random(int(sys.argv[5]) if len(sys.argv) > 5 else 7)
    catalog = [json.loads(l) for l in open(cat_path, encoding="utf-8") if l.strip()]
    byid = {e["id"]: e for e in catalog}
    missing = [a for a in POOL if O + a not in byid] + [a for a in ("core/if", "core/loop", "core/log",
                                                                    "ooa.file/has_extension") if a not in byid]
    if missing:
        raise SystemExit(f"catalog lacks: {missing}")
    validate = G.make_validator(repo, catalog)
    ev = Path(__file__).resolve().parent.parent / "eval"
    tests = [json.loads(l)["must"] for p in ("plain_requests.jsonl", "plain_requests_2.jsonl")
             for l in open(ev / p, encoding="utf-8") if l.strip()]
    names, weights = zip(*[(k, w) for k, (_, w) in SHAPES.items()])
    rows, seen, why = [], set(), Counter()
    per_seq, made = Counter(), Counter()
    quota = {k: round(n * w / sum(weights)) + 1 for k, w in zip(names, weights)}
    tries = 0
    while len(rows) < n and tries < n * 300:
        tries += 1
        shape = rng.choices(names, weights)[0]
        try:
            f = SHAPES[shape][0](byid, rng)
        except StopIteration:
            f = None
        if f is None:
            why["no flow"] += 1
            continue
        try:
            validate(f)
        except Exception:
            why["invalid"] += 1
            continue
        acts = actions_of(f["steps"])
        plain = Counter(a for a in acts if not a.startswith("core/"))
        if any(Counter(a for a in m if not a.startswith("core/")) == plain and in_order(m, acts) for m in tests):
            why["answers a test request"] += 1
            continue
        detail = sorted(set(re.findall(r"'([a-z ]+)'|\"extensions\": \"(\w+)\"|\"pattern\": \"\*\.(\w+)\"",
                                       json.dumps(f["steps"]).lower())))
        key = (tuple(acts), str(detail))            # a different trigger word / file type is a different example
        if made[shape] >= quota[shape]:
            continue
        if per_seq[key] >= 2:                       # at most 3 variants of one action sequence: breadth, not repeats
            why["sequence seen 3x"] += 1
            continue
        m = json.dumps(f, separators=(",", ":"), ensure_ascii=False)
        if m in seen:
            continue
        seen.add(m)
        per_seq[key] += 1
        made[shape] += 1
        rows.append({"flow": f, "shape": shape, "actions": acts})
    with open(out, "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} flows | {len(per_seq)} distinct action sequences | shapes {Counter(r['shape'] for r in rows)}"
          f" | skipped {dict(why)}")


if __name__ == "__main__":
    main()
