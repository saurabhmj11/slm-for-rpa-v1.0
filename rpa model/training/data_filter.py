"""Clean the training data before a big training run: every row passes a series of filters, the rest are dropped.

Usage: python data_filter.py SFT_DIR OUT_DIR PLATFORM_REPO [EVAL_DIR] [--cap 60]
Reads every *.jsonl in SFT_DIR (catalog.jsonl is copied as it is; train.jsonl, a combined copy of train_flows and
train_catalog, and the eval*.jsonl test files are skipped), writes the kept rows to OUT_DIR with the same
file names, and OUT_DIR/filter_report.json with what each filter removed per file. Files of neutral records
(fix_train, targeted_train, clarify_train, long_train, business2_train: {"kind": ...} rows, checked when they were
made) are left out; render_dataset.py reads them with --extra.

Filters, in order (a row is dropped by the first one it fails):
  structure      messages are user/assistant turns in order, all text, nothing empty
  too_long       longer than the model's window (2048 tokens, about 6,500 characters): it would be skipped anyway
  request_leak   a flow's request contains $variables, action ids, JSON or step ids (people never type those)
  request_size   a flow's request shorter than 6 or longer than 1,500 characters
  repetitive     the request repeats the same clause 3 or more times (generated noise: "read a JSON file, then read
                 a JSON file, then read a JSON file ...")
  bad_json       the answer starts like a flow but is not valid JSON
  fails_rules    the flow (after adding missing open / save steps) breaks the platform's rules: schema, unknown
                 action or input, undeclared variable, ...  (the same check the assistant applies)
  stale_action   a catalog answer names an action id that is not in the current catalog
  wrong_app      the request names an app (Slack, Asana, GitHub, SAP, ...) that no step of the flow uses
  bad_reply      a not-a-task answer that is JSON, a code block or longer than 60 words
  test_leak      the request is the same or almost the same as a test request (keeps the tests honest)
  duplicate      the same request with the same answer was already kept (in any file)
  conflict       the same request with a different flow: only the most common flow is kept
  over_cap       more than --cap rows with the same action sequence (keeps variety; generated data repeats)
"""
import argparse
import difflib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ooa_reward as R
import repair as RP

MAX_CHARS = 6500
NOT_TRAINING = {"train.jsonl", "hard_queue.jsonl", "selfimprove_log.jsonl"}
DRIVE_COPY = re.compile(r" \(\d+\)\.jsonl$")
LEAK = re.compile(r"\$\{?[A-Za-z_]|\booa\.[a-z_]+/|\bcore/[a-z_]+|[{}]|\bpath_\d\b|\bs\d+\b(?= step)")
APPS = {"slack": "ooa.slack/", "asana": "ooa.asana/", "github": "ooa.github/", "docusign": "ooa.docusign/",
        "salesforce": "ooa.salesforce/", "snowflake": "ooa.snowflake/", "databricks": "ooa.databricks/",
        "jira": "ooa.jira/", "servicenow": "ooa.servicenow/", "zendesk": "ooa.zendesk/", "okta": "ooa.okta/",
        "workday": "ooa.workday/", "azure devops": "ooa.azure_devops/", "sap": "ooa.sap", "kafka": "ooa.kafka/"}
ACTION_ID = re.compile(r"\b((?:ooa\.[a-z0-9_]+|core)/[a-z0-9_]+)\b")


def actions(flow):
    out = []
    def walk(nodes):
        for n in nodes or []:
            if isinstance(n, dict):
                if isinstance(n.get("action"), str):
                    out.append(n["action"])
                for k in ("children", "else", "finally"):
                    walk(n.get(k) if isinstance(n.get(k), list) else None)
                for c in n.get("catch") or []:
                    if isinstance(c, dict):
                        walk(c.get("children"))
    walk(flow.get("steps"))
    return out


def norm(text):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def is_neutral(path):
    """A file of neutral records ({"kind": "flow"|"fix"|"clarify", ...}, gen_targeted.py and friends), not chat rows."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    row = json.loads(line)
                except ValueError:
                    return False
                return isinstance(row, dict) and "kind" in row and "messages" not in row
    return False


def main():
    global MAX_CHARS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sft")
    ap.add_argument("out")
    ap.add_argument("platform")
    ap.add_argument("eval_dir", nargs="?", default=str(Path(__file__).resolve().parent.parent / "eval"))
    ap.add_argument("--cap", type=int, default=60, help="most rows kept per action sequence")
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS, help="longest row in characters (6500 for a 2048-token window)")
    a = ap.parse_args()
    MAX_CHARS = a.max_chars
    sft, out = Path(a.sft), Path(a.out)
    catalog_path = sft / "catalog.jsonl"
    if not catalog_path.exists():
        raise SystemExit(f"no catalog.jsonl in {sft}")
    R.configure(str(catalog_path), str(Path(a.platform) / "contracts" / "flow-schema" / "flow.v1.json"))
    cat = {e["id"]: e for e in (json.loads(l) for l in open(catalog_path, encoding="utf-8") if l.strip())}
    tests = []
    for p in sorted(Path(a.eval_dir).glob("plain_requests*.jsonl")):
        tests += [norm(json.loads(l)["prompt"]) for l in open(p, encoding="utf-8") if l.strip()]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(catalog_path, out / "catalog.jsonl")

    report, kept = {}, {}
    seen_pair = set()
    by_request = defaultdict(Counter)             # normalised request -> Counter(flow text) over all files
    staged = []                                   # (file, row_text, request_norm, answer, seq) after per-row filters
    neutral = []
    for f in sorted(sft.glob("*.jsonl")):
        if (f.name == "catalog.jsonl" or f.name in NOT_TRAINING or f.name.startswith("eval")
                or DRIVE_COPY.search(f.name)):     # "synth_train (1).jsonl": a second upload Google Drive renamed;
                                                   # read first, it would take the rows and leave the real file empty
            continue          # not training data: train.jsonl repeats train_flows + train_catalog (it would make them
                              # "duplicates" and empty them); eval*.jsonl are tests
        if is_neutral(f):
            neutral.append(f.name)     # left for render_dataset.py --extra (checked by their generator)
            continue
        why = Counter()
        n = 0
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            n += 1
            try:
                row = json.loads(line)
                msgs = row["messages"]
                turns = msgs[1:] if msgs and msgs[0].get("role") == "system" else msgs
                ok = len(turns) >= 2 and all(m.get("role") == ("user" if i % 2 == 0 else "assistant")
                                             and isinstance(m.get("content"), str) and m["content"].strip()
                                             for i, m in enumerate(turns))
            except Exception:
                ok = False
            if not ok:
                why["structure"] += 1
                continue
            req, ans = turns[-2]["content"].strip(), turns[-1]["content"].strip()
            if sum(len(m["content"]) for m in msgs) > MAX_CHARS:
                why["too_long"] += 1
                continue
            is_flow = ans.startswith("{")
            seq = ()
            if is_flow:
                if LEAK.search(req):
                    why["request_leak"] += 1
                    continue
                if not 6 <= len(req) <= 1500:
                    why["request_size"] += 1
                    continue
                clauses = [norm(c) for c in re.split(r",? then |; |\. ", req) if len(c.split()) >= 3]
                if clauses and Counter(clauses).most_common(1)[0][1] >= 3:
                    why["repetitive"] += 1
                    continue
                try:
                    flow = json.loads(ans)
                except Exception:
                    why["bad_json"] += 1
                    continue
                if not isinstance(flow, dict):
                    why["bad_json"] += 1
                    continue
                RP.session_fix(flow, cat)
                ans = json.dumps(flow, separators=(",", ":"), ensure_ascii=False)
                if R.problems(ans):
                    why["fails_rules"] += 1
                    continue
                acts = actions(flow)
                low = req.lower()
                if any(re.search(rf"\b{re.escape(k)}\b", low) and not any(x.startswith(pre) for x in acts)
                       for k, pre in APPS.items()):
                    why["wrong_app"] += 1
                    continue
                seq = tuple(acts)
                turns[-1]["content"] = ans
            else:
                named = ACTION_ID.findall(ans)
                if any(x not in cat for x in named):
                    why["stale_action"] += 1
                    continue
                if f.name.startswith("notask") and (ans.startswith(("[", "`")) or len(ans.split()) > 60):
                    why["bad_reply"] += 1
                    continue
            rn = norm(req)
            if f.name != "val.jsonl" and tests and any(
                    t == rn or (difflib.SequenceMatcher(None, rn, t).quick_ratio() > 0.85
                                and difflib.SequenceMatcher(None, rn, t).ratio() > 0.85) for t in tests):
                why["test_leak"] += 1
                continue
            if is_flow:
                by_request[rn][ans] += 1
            staged.append((f.name, json.dumps(row, ensure_ascii=False), rn, ans, seq))
        report[f.name] = {"rows": n, "dropped": dict(why)}

    per_seq = Counter()
    for name, text, rn, ans, seq in staged:
        r = report[name]["dropped"]
        if name != "val.jsonl":
            if (rn, ans) in seen_pair:
                r["duplicate"] = r.get("duplicate", 0) + 1
                continue
            if seq and len(by_request[rn]) > 1 and ans != by_request[rn].most_common(1)[0][0]:
                r["conflict"] = r.get("conflict", 0) + 1
                continue
            if seq and per_seq[seq] >= a.cap:
                r["over_cap"] = r.get("over_cap", 0) + 1
                continue
            seen_pair.add((rn, ans))
            if seq:
                per_seq[seq] += 1
        kept.setdefault(name, []).append(text)
    for name, rows in kept.items():
        (out / name).write_text("".join(t + "\n" for t in rows), encoding="utf-8")
    for name in ("train_flows.jsonl", "train_catalog.jsonl", "synth_train.jsonl", "synth_pattern.jsonl", "val.jsonl"):
        if not (out / name).exists() and name not in kept:
            (out / name).write_text("", encoding="utf-8")       # the training scripts expect these names
    for name in report:
        report[name]["kept"] = len(kept.get(name, []))
        if name not in kept:
            (out / name).write_text("", encoding="utf-8")
    total = {"rows": sum(r["rows"] for r in report.values()), "kept": sum(r["kept"] for r in report.values())}
    drops = Counter()
    for r in report.values():
        drops.update(r["dropped"])
    report["_total"] = {**total, "dropped": dict(drops), "neutral_files_not_filtered": neutral}
    (out / "filter_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"kept {total['kept']} of {total['rows']} rows | dropped: {dict(drops.most_common())}")
    for name, r in sorted(report.items()):
        if name != "_total":
            print(f"  {name:24s} {r['kept']:6d} of {r['rows']:6d}  {r['dropped'] or ''}")
    if neutral:
        print("  neutral records, not filtered (render_dataset.py --extra reads them):", ", ".join(neutral))


if __name__ == "__main__":
    main()
