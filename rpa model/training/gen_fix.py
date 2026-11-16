"""Self-correction training data: a valid flow, a corrupted copy of it (the kind of mistake a small model makes) and the
problems the server reports for the corrupted copy. The model learns to read "### Previous flow" + "### Problems" and
write the corrected flow.

Corruptions (one per record, sometimes two of the checker-visible ones together):
  drop_required    a required input is left out                      "... is missing required input ..."
  wrong_input      an input name is a plausible wrong one             "... has no input ..."
  wrong_output     an output name is a plausible wrong one            "... has no output ..."
  undeclared_var   a step reads a variable under a wrong name          (kept only when repair cannot declare it away)
  invented_action  an action id that does not exist (file/rename)     "Unknown action ..."
  swap_action      a nearby real action of the same toolkit (move -> delete, copy -> move)
  drop_last        the last real step is missing (zip ... and never emails)
  drop_loop        the per-item steps are not inside their loop
  drop_session     the open / create step or the save step around Excel / Word edits is missing
The problems are exactly what serving/ooa_server.py reports for that answer: the previous flow is the raw answer; the
server repairs it (repair.repair), runs ooa_reward.problems, and adds "The plan names <action> but no step of the flow
uses it." for every action of the answer's plan (ooa_format.plan_from_flow of the correct flow) that the repaired flow
lacks. A corruption that repair fixes on its own leaves no problems, the server would not ask for a fix, and no
record is written for it.

Sources: chat rows whose assistant content is a flow (combos, tools, chains, scenarios), and neutral "flow" records
(gen_targeted.py / gen_long.py output, which have the loops, Excel / Word sessions and "... then email it" endings).
Every original flow is first checked (repair + ooa_reward.problems == []); requests close to a test request (difflib
ratio > 0.8) are dropped.

Writes NEUTRAL records: {"kind": "fix", "request", "previous", "problems", "flow", "source": "fix:<corruption>:<file>"}.
Usage: python gen_fix.py CATALOG.jsonl SCHEMA.json OUT.jsonl SRC.jsonl [SRC.jsonl ...] [--n 6000] [--seed 5]
"""
import copy
import difflib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ooa_format as F                                   # noqa: E402
import ooa_reward as R                                   # noqa: E402
import repair as RP                                      # noqa: E402
from gen_targeted import Checker, LeakGuard, write_jsonl   # noqa: E402

IN_WRONG = {"path": ["file_path", "filepath", "file", "filename"], "folder": ["folder_path", "directory", "dir", "path"],
            "target": ["destination", "target_folder", "dest", "to_path"], "source": ["source_path", "src", "from", "file"],
            "to": ["recipient", "email", "to_address", "recipients"], "subject": ["title", "subject_line"],
            "text": ["content", "message", "value"], "table": ["data", "datatable", "rows", "input_table"],
            "url": ["link", "address", "website"], "column": ["column_name", "field", "col"], "id": ["email_id", "message_id", "mail_id"],
            "sql": ["query", "statement"], "value": ["val", "compare_to"], "files": ["file_list", "paths"],
            "pattern": ["filter", "file_pattern", "mask"], "body": ["message", "content", "text"], "attachments": ["attachment", "files"],
            "archive": ["zip_file", "zip_path", "path"], "selector": ["element", "css", "xpath"], "message": ["text", "msg"],
            "fields": ["field_list", "keys"], "operator": ["op", "condition"], "target_folder": ["folder", "save_to", "destination"],
            "spreadsheet_id": ["sheet_id", "spreadsheet"], "access_token": ["token", "auth_token"], "save_to": ["path", "destination"],
            "labels": ["categories", "classes"], "prompt": ["question", "text"], "channel": ["channel_name", "channel_id"],
            "records": ["rows", "list"], "delimiter": ["separator", "sep"], "new_name": ["name", "rename_to"]}
OUT_WRONG = {"table": ["data", "result", "rows", "datatable"], "text": ["content", "result", "output"],
             "files": ["file_list", "paths", "result"], "path": ["file_path", "output_path", "file"], "count": ["total", "num", "n"],
             "row_count": ["rows", "count"], "records": ["rows", "list"], "value": ["result", "cell_value"], "list": ["items", "values"],
             "total": ["sum", "result"], "email": ["message", "mail"], "fields": ["data", "values", "result"],
             "label": ["category", "class"], "result": ["output", "value"], "items": ["list", "parts"]}
VERB_WRONG = {"move": ["rename", "move_file"], "copy": ["copy_file", "duplicate"], "read": ["load", "read_file", "open"],
              "write": ["save", "write_file", "export"], "list": ["list_files", "get_files", "find"], "send": ["send_email", "send_mail"],
              "extract_text": ["read_text", "get_text", "read"], "get_all_cells": ["read_sheet", "get_data", "read"],
              "write_table": ["write", "write_data"], "open": ["open_file", "load"], "zip": ["compress", "create_zip"],
              "unzip": ["extract", "decompress"], "filter": ["filter_rows", "where"], "sort": ["sort_by", "order_by"],
              "merge": ["combine", "join_pdfs"], "save_attachments": ["download_attachments", "get_attachments"],
              "get_text": ["read_text", "text"], "extract_table": ["get_table", "scrape_table"], "type_text": ["type", "fill"],
              "row_count": ["count_rows", "count"], "sum_column": ["sum", "total"], "delete": ["remove", "delete_file"],
              "query": ["select", "run_query"], "create_record": ["create", "insert"], "post_message": ["send_message", "post"]}
SWAPS = {"ooa.file/move": ["ooa.file/delete", "ooa.file/copy"], "ooa.file/copy": ["ooa.file/move", "ooa.file/delete"],
         "ooa.file/zip": ["ooa.file/unzip", "ooa.file/copy"], "ooa.file/unzip": ["ooa.file/zip"],
         "ooa.pdf/extract_text": ["ooa.pdf/read_text_ocr", "ooa.pdf/page_count"], "ooa.pdf/read_text_ocr": ["ooa.pdf/extract_text"],
         "ooa.csv/read": ["ooa.csv/write"], "ooa.csv/write": ["ooa.csv/read"], "ooa.excel/get_all_cells": ["ooa.excel/write_table"],
         "ooa.excel/write_table": ["ooa.excel/get_all_cells"], "ooa.email/send": ["ooa.email/forward", "ooa.email/reply"],
         "ooa.email/forward": ["ooa.email/send", "ooa.email/move"], "ooa.email/move": ["ooa.email/delete"],
         "ooa.datatable/filter": ["ooa.datatable/sort", "ooa.datatable/remove_duplicates"], "ooa.datatable/sort": ["ooa.datatable/filter"],
         "ooa.browser/get_text": ["ooa.browser/type_text"], "ooa.browser/type_text": ["ooa.browser/get_text", "ooa.browser/click"],
         "ooa.word/replace_text": ["ooa.word/read_text"], "ooa.folder/create": ["ooa.folder/delete"],
         "ooa.file/write_text": ["ooa.file/read_text"], "ooa.file/read_text": ["ooa.file/write_text"]}
CLOSERS = {"ooa.excel/close", "ooa.word/close", "ooa.browser/close", "ooa.email/disconnect", "ooa.database/disconnect",
           "ooa.file_transfer/disconnect", "ooa.powerpoint/close"}
OPEN_SAVE = {"ooa.excel/open", "ooa.excel/create_workbook", "ooa.word/open", "ooa.word/create_document", "ooa.excel/save",
             "ooa.word/save", "ooa.excel/save_as", "ooa.word/save_as"}
def real_steps(flow, cat):
    """Non-core steps whose action is in the catalog (an earlier corruption may have invented one)."""
    return [s for s in R.walk(flow.get("steps")) if s.get("action") in cat and not s["action"].startswith("core/")]


def parent_list(nodes, target):
    """The step list that holds target (by identity) and its index."""
    for k, n in enumerate(nodes):
        if n is target:
            return nodes, k
        for b in R.bodies(n):
            got = parent_list(b, target)
            if got:
                return got
    return None


def wrong_name(name, table, valid, rng):
    cands = [c for c in table.get(name, []) if c not in valid]
    if not cands:
        cands = [c for c in (name + "_name", name + "_path", name + "_value", "input_" + name) if c not in valid]
    return rng.choice(cands)


# corruptions: each changes the flow copy in place and returns True, or None when it does not apply to this flow.

def c_drop_required(f, cat, rng):
    cand = [(s, i["name"]) for s in real_steps(f, cat) for i in cat[s["action"]]["inputs"]
            if i["required"] and i["name"] in (s.get("args") or {})]
    if not cand:
        return None
    s, k = rng.choice(cand)
    s["args"].pop(k)
    return True


def c_wrong_input(f, cat, rng):
    cand = [s for s in real_steps(f, cat) if s.get("args")]
    if not cand:
        return None
    s = rng.choice(cand)
    k = rng.choice(list(s["args"]))
    valid = {i["name"] for i in cat[s["action"]]["inputs"]}
    new = wrong_name(k, IN_WRONG, valid | set(s["args"]), rng)
    s["args"] = {(new if x == k else x): v for x, v in s["args"].items()}
    return True


def c_wrong_output(f, cat, rng):
    cand = [s for s in real_steps(f, cat) if s.get("out")]
    if not cand:
        return None
    s = rng.choice(cand)
    k = rng.choice(list(s["out"]))
    valid = {o["name"] for o in cat[s["action"]]["outputs"]}
    new = wrong_name(k, OUT_WRONG, valid | set(s["out"]), rng)
    s["out"] = {(new if x == k else x): v for x, v in s["out"].items()}
    return True


def c_undeclared(f, cat, rng):
    declared = {v["name"] for v in f.get("variables", [])}
    cand = [(s, k, r) for s in R.walk(f.get("steps")) for k, v in (s.get("args") or {}).items()
            if isinstance(v, str) for r in R.refs(v)]
    if not cand:
        return None
    s, k, r = rng.choice(cand)
    opts = [x for x in (r + "s", r + "_list", "my_" + r, r[:-1] if r.endswith("s") else r + "_1", r + "_value", "result", "data",
                        "input_" + r) if x not in declared and x != r]
    new = rng.choice(opts)
    v = s["args"][k]
    s["args"][k] = re.sub(r"\$\{\s*" + re.escape(r) + r"\b", "${" + new, re.sub(r"\$" + re.escape(r) + r"\b", "$" + new, v))
    return True if s["args"][k] != v else None


def c_invented(f, cat, rng):
    cand = [s for s in real_steps(f, cat) if s["action"].split("/")[1] in VERB_WRONG]
    if not cand:
        return None
    s = rng.choice(cand)
    tk, verb = s["action"].split("/")
    opts = [f"{tk}/{w}" for w in VERB_WRONG[verb] if f"{tk}/{w}" not in cat
            and not difflib.get_close_matches(f"{tk}/{w}", list(cat), n=1, cutoff=0.88)]
    if not opts:
        return None
    s["action"] = rng.choice(opts)
    return True


def c_swap(f, cat, rng):
    cand = [s for s in real_steps(f, cat) if s["action"] in SWAPS and any(x in cat for x in SWAPS[s["action"]])]
    if not cand:
        return None
    s = rng.choice(cand)
    s["action"] = rng.choice([x for x in SWAPS[s["action"]] if x in cat])
    s["description"] = cat[s["action"]]["description"].split(";")[0].rstrip(".")[:140]
    return True


def c_drop_last(f, cat, rng):
    top = f.get("steps") or []
    idx = [k for k, s in enumerate(top) if s.get("action") not in CLOSERS]
    if len(top) < 2 or not idx:
        return None
    k = idx[-1]
    if k == 0:
        return None
    top.pop(k)
    return True


def c_drop_loop(f, cat, rng):
    loops = [s for s in R.walk(f.get("steps")) if s.get("action") == "core/loop" and s.get("children")]
    if not loops:
        return None
    lp = rng.choice(loops)
    got = parent_list(f["steps"], lp)
    if not got:
        return None
    lst, k = got
    lst[k:k + 1] = lp["children"]
    return True


def c_drop_session(f, cat, rng):
    cand = [s for s in R.walk(f.get("steps")) if s.get("action") in OPEN_SAVE]
    if not cand:
        return None
    s = rng.choice(cand)
    lst, k = parent_list(f["steps"], s)
    lst.pop(k)
    return True


CORRUPT = {"drop_required": (c_drop_required, 14), "wrong_input": (c_wrong_input, 14), "wrong_output": (c_wrong_output, 8),
           "undeclared_var": (c_undeclared, 0), "invented_action": (c_invented, 10), "swap_action": (c_swap, 10),
           "drop_last": (c_drop_last, 16), "drop_loop": (c_drop_loop, 16), "drop_session": (c_drop_session, 12)}
# undeclared_var only goes with another corruption: repair declares a misnamed variable, so alone the server sees nothing
VISIBLE = ["drop_required", "wrong_input", "wrong_output", "undeclared_var", "invented_action"]


def server_problems(previous, flow, cat):
    """What serving/ooa_server.py reports for the raw answer `previous` whose plan is the correct flow's plan."""
    fixed, _ = RP.repair(previous, cat)
    probs = R.problems(json.dumps(fixed, separators=(",", ":"), ensure_ascii=False))
    probs += [f"The plan names {a} but no step of the flow uses it."
              for a in F.plan_coverage(F.plan_from_flow(flow, cat), fixed)]
    return probs


def corrupt(flow, kinds, cat, rng):
    f = copy.deepcopy(flow)
    for kind in kinds:
        if CORRUPT[kind][0](f, cat, rng) is None:
            return None, None
    if not f.get("steps") or f == flow:
        return None, None
    probs = server_problems(f, flow, cat)
    if not probs:                                  # repair fixes it, or nothing the server can see: no retry happens
        return None, None
    return f, probs


def main():
    args = sys.argv[1:]
    n, seed = 6000, 5
    if "--n" in args:
        k = args.index("--n"); n = int(args[k + 1]); del args[k:k + 2]
    if "--seed" in args:
        k = args.index("--seed"); seed = int(args[k + 1]); del args[k:k + 2]
    cat_path, schema, out, srcs = args[0], args[1], args[2], args[3:]
    rng = random.Random(seed)
    checker, guard = Checker(cat_path, schema), LeakGuard()
    cat = checker.cat
    pool, why = [], Counter()
    for p in srcs:
        name = Path(p).stem
        for l in open(p, encoding="utf-8"):
            if not l.strip():
                continue
            row = json.loads(l)
            if row.get("kind") == "flow":                  # a neutral record (gen_targeted.py / gen_long.py output)
                user, ans = row.get("request"), json.dumps(row.get("flow"))
            else:
                m = row.get("messages") or []
                user = next((x["content"] for x in m if x["role"] == "user"), None)
                ans = next((x["content"] for x in m if x["role"] == "assistant"), None)
            try:
                flow = json.loads(ans)
                assert isinstance(flow, dict) and flow.get("steps")
            except Exception:
                why["not a flow"] += 1
                continue
            if not user or guard.leaks(user):
                why["close to a test request"] += 1
                continue
            good = checker.check(flow)
            if good is None:
                why["source flow fails the checker"] += 1
                continue
            pool.append((user.strip(), good, name))
    rng.shuffle(pool)
    names, weights = zip(*[(k, w) for k, (_, w) in CORRUPT.items()])
    rows, made, seen, used = [], Counter(), set(), Counter()
    tries = 0
    while len(rows) < n and tries < n * 30:
        tries += 1
        kinds = sorted(rng.sample(VISIBLE, 2)) if rng.random() < 0.12 else [rng.choices(names, weights)[0]]
        for _ in range(40):                        # find a flow this corruption applies to (loops and sessions are rarer)
            k = rng.randrange(len(pool))
            if used[k] >= 2:                       # at most two corrupted copies of one source flow
                continue
            req, flow, src = pool[k]
            prev, probs = corrupt(flow, kinds, cat, rng)
            if prev is not None:
                break
        else:
            why["corruption does not apply"] += 1
            continue
        key = json.dumps(prev, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        used[k] += 1
        label = "+".join(kinds)
        made[label] += 1
        rows.append({"kind": "fix", "request": req, "previous": prev, "problems": probs, "flow": flow,
                     "source": f"fix:{label}:{src}"})
    write_jsonl(out, rows)
    print(f"{len(rows)} records from {len(pool)} valid source flows | source checker pass {checker.ok}/{checker.n}"
          f" | kinds {dict(made.most_common())} | skipped {dict(why)}")


if __name__ == "__main__":
    main()
