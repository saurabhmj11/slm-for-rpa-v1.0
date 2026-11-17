"""Render training data into the plan-first format with retrieval context (see ooa_format.py).

    python render_dataset.py SRC_DIR OUT_DIR --catalog CATALOG.jsonl [--k 12] [--seed 7] [--extra FILE ...]
                             [--variables-rate 0.3] [--reply-context 0.5] [--max-chars 6500] [--max-tokens 2020]
                             [--no-parts] [--answer-format compact|full] [--gold-dropout 0.5] [--dropout-frac 0.2,0.45]

Reads every chat file SRC_DIR/*.jsonl ({"messages": [...]} rows, e.g. sft_clean/) and every neutral-record file given
with --extra ({"kind": "flow"|"fix"|"clarify", ...}), and writes OUT_DIR/<same name>.jsonl in chat format:
  - a row whose answer is a flow: the user turn gets the context (ooa_format.context_actions, the same function the
    server uses: BM25 top-k actions for the request, per-clause candidates for a long request, united with the
    gold flow's actions, gold always kept, order shuffled; 0-2 similar training requests with their action outline,
    never the row itself or a row with the same flow; the flow's input variables for --variables-rate of the rows,
    because Studio sends the open agent's variables) and the answer becomes plan + flow. The flow first gets
    repair.session_fix (the missing open / save step), so the plan matches the flow that is trained on.
  - flow records from --extra files (flow / fix / clarify / rewrite) are rendered the same way; a rewrite record (a
    beginner's request, gen_beginner.py) gets an answer that opens with "Clear request: <the request restated>"
    (ooa_format.render_flow_answer), and its request joins the example index like a flow record's.
  - other rows (chat replies, catalog questions) are kept; --reply-context of the single-turn ones also get the
    context block, because at serving every request carries one and a chat request must still get a chat reply.
  - a flow row that would not fit the 2048-token window (user + assistant over --max-chars, or over --max-tokens by
    ooa_format.approx_tokens) gets a smaller context, down to its own actions only; if it still does not fit it is
    written as its parts (ooa_format.split_flow: the part requests the server writes when it builds a long flow in
    parts). What still does not fit is skipped and counted.
  - --answer-format compact (default): the flow after "Flow:" is ooa_format.compact_flow (no variables list, no step
    ids / descriptions / notes; the server rebuilds them with expand_flow). full: the flow as stored (plan-v1).
  - --gold-dropout (default 0.5) of the flow / fix rows get a context that misses --dropout-frac (20-45%) of the
    needed actions (never core/loop, core/if; ooa_format.drop_gold), so the model learns to write an action it knows
    when retrieval did not offer it. Plan and flow still use the right action.
Files named val*.jsonl / eval*.jsonl are held out: rendered the same way, never used as examples.
Also writes OUT_DIR/examples.jsonl (the example index the server loads), OUT_DIR/catalog.jsonl and
OUT_DIR/render_report.json. Standard library only; one pass to load, one to render (about 60k rows in a few minutes).
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ooa_format as F       # noqa: E402
import retrieval             # noqa: E402
import repair as RP          # noqa: E402

SKIP_FILES = {"catalog.jsonl", "examples.jsonl", "train_catalog.jsonl"}


def held_out(name):
    return name.startswith("val") or name.startswith("eval")


def flow_of(text):
    """The flow dict when an assistant answer is only flow JSON, else None."""
    t = (text or "").strip()
    if not t.startswith("{"):
        return None
    try:
        obj = json.loads(t)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) and isinstance(obj.get("steps"), list) and obj["steps"] else None


def read_jsonl(path):
    rows, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                bad += 1
    return rows, bad


def load_items(path, extra, cat, fixes):
    """One file -> list of items {"type": "record"|"chat"|"bad", ...} and the count of unreadable lines."""
    rows, bad = read_jsonl(path)
    items = []
    for r in rows:
        if not isinstance(r, dict):
            items.append({"type": "bad"})
            continue
        if extra or ("kind" in r and "messages" not in r):
            rec = dict(r)
            for key in ("flow", "previous"):
                if isinstance(rec.get(key), str):
                    try:
                        rec[key] = json.loads(rec[key])
                    except ValueError:
                        pass
            if isinstance(rec.get("flow"), dict) and rec.get("kind") in ("flow", "fix", "rewrite"):
                if RP.session_fix(rec["flow"], cat):
                    fixes[0] += 1
            items.append({"type": "record", "record": rec})
            continue
        msgs = r.get("messages")
        if not isinstance(msgs, list) or not msgs:
            items.append({"type": "bad"})
            continue
        rest = [m for m in msgs if isinstance(m, dict) and m.get("role") != "system"]
        system = [m for m in msgs if isinstance(m, dict) and m.get("role") == "system"]
        if len(rest) == 2 and rest[0].get("role") == "user" and rest[1].get("role") == "assistant":
            flow = flow_of(rest[1].get("content"))
            request, variables = F.split_user(rest[0].get("content", ""))
            if flow is not None and request:
                if RP.session_fix(flow, cat):
                    fixes[0] += 1
                items.append({"type": "record", "record": {"kind": "flow", "request": request, "flow": flow,
                                                           "variables": variables}, "system": system})
                continue
            items.append({"type": "chat", "row": r, "single": True, "request": request})
            continue
        items.append({"type": "chat", "row": r, "single": False})
    return items, bad


def render(src, out, catalog, k=F.DEFAULT_K, seed=7, extra=(), variables_rate=0.3, reply_context=0.5,
           max_chars=F.MAX_ROW_CHARS, parts=True, log=print, max_tokens=F.MAX_ROW_TOKENS, answer_format=F.ANSWER_COMPACT,
           gold_dropout=F.GOLD_DROPOUT, dropout_frac=F.GOLD_DROPOUT_FRAC):
    t0 = time.time()
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    entries = [json.loads(l) for l in open(catalog, encoding="utf-8") if l.strip()]
    cat = {e["id"]: e for e in entries}
    index = retrieval.ActionIndex(entries)
    files = []
    extras = {Path(e).resolve() for e in extra or []}
    for p in sorted(src.glob("*.jsonl")):
        if p.name in SKIP_FILES or p.name.startswith("render_") or p.resolve() in extras:
            continue
        files.append((p, False, p.name))
    names = {n for _, _, n in files}
    for e in extra or []:
        p = Path(e)
        name = p.name if p.name not in names else p.stem + "_extra.jsonl"
        names.add(name)
        files.append((p, True, name))

    opts = dict(answer_format=answer_format, gold_dropout=gold_dropout, dropout_frac=tuple(dropout_frac))
    fixes = [0]
    loaded = []
    for p, is_extra, name in files:
        items, bad = load_items(p, is_extra, cat, fixes)
        loaded.append((name, items, bad))
    log(f"loaded {sum(len(i) for _, i, _ in loaded)} rows from {len(loaded)} files "
        f"({fixes[0]} flows given their missing open / save step) in {time.time() - t0:.0f}s")

    pairs = []
    for name, items, _ in loaded:
        if held_out(name):
            continue
        for it in items:
            rec = it.get("record")
            if rec and rec.get("kind") in ("flow", "rewrite") and isinstance(rec.get("flow"), dict):
                pairs.append((rec["request"], rec["flow"]))
    ex_index = F.ExampleIndex.from_flows(pairs)
    ex_index.save(out / "examples.jsonl")
    log(f"example index: {len(ex_index)} requests ({time.time() - t0:.0f}s)")

    report = {"settings": {"k": k, "seed": seed, "variables_rate": variables_rate, "reply_context": reply_context,
                           "max_chars": max_chars, "max_tokens": max_tokens,
                           "format": F.FORMAT_COMPACT if answer_format == F.ANSWER_COMPACT else F.FORMAT_VERSION,
                           "answer_format": answer_format, "gold_dropout": gold_dropout,
                           "dropout_frac": list(dropout_frac),
                           "max_context_chars": F.context_for(F.window_of(max_tokens)),
                           "clauses": {"min": F.CLAUSE_MIN, "k": F.CLAUSE_K, "votes": F.CLAUSE_VOTES,
                                       "whole_votes": F.WHOLE_VOTES, "max_actions": F.MAX_CONTEXT_ACTIONS}},
              "files": {}, "examples": len(ex_index), "session_fixes": fixes[0]}
    for name, items, bad in loaded:
        rng = random.Random(f"{seed}:{name}")
        c = {"rows": len(items), "written": 0, "flow": 0, "fix": 0, "clarify": 0, "rewrite": 0, "kept": 0, "reply_with_context": 0,
             "part": 0, "split_long": 0,
             "skipped_too_long": 0, "skipped_unusable": bad, "max_chars": 0, "chars": 0}
        with open(out / name, "w", encoding="utf-8") as f:
            for it in items:
                row = None
                if it["type"] == "record":
                    rec = it["record"]
                    row = F.render_chat(rec, cat, index, ex_index, rng, k, variables_rate, max_chars, max_tokens,
                                        **opts)
                    if row is None:
                        c["skipped_unusable"] += 1
                        continue
                    if it.get("system"):
                        row["messages"] = it["system"] + row["messages"]
                    kind = rec.get("kind")
                elif it["type"] == "chat":
                    row, kind = it["row"], "kept"
                    if it.get("single") and it.get("request") and rng.random() < reply_context:
                        msgs = [dict(m) for m in row["messages"]]
                        u = next(m for m in msgs if m.get("role") == "user")
                        if F.CONTEXT_HEADER not in str(u.get("content")):
                            exs = ex_index.search(it["request"], rng.choice((0, 1, 2)))     # as at serving
                            acts, required = F.context_actions(it["request"], (), cat, index, k, rng, exs, ex_index)
                            u["content"] = F.build_user(it["request"], acts, exs, (), required=required,
                                                        context_limit=F.context_for(F.window_of(max_tokens)))
                            row, kind = {**row, "messages": msgs}, "reply_with_context"
                else:
                    c["skipped_unusable"] += 1
                    continue
                rows = [(row, kind)]
                if not F.row_fits(row, max_chars, max_tokens) and kind in ("flow", "rewrite") and parts:
                    # too long for the window: teach it in parts, the way the server builds a long flow
                    rows = []
                    for req, part, carried in F.split_flow(rec["flow"], cat):
                        prow = F.render_chat({"kind": "flow", "request": req, "flow": part, "variables": carried},
                                             cat, index, ex_index, rng, k, variables_rate, max_chars, max_tokens,
                                             **opts)
                        if prow is not None:
                            rows.append((prow, "part"))
                    c["split_long"] += 1 if rows else 0
                    rows = rows or [(row, kind)]
                for row, kind in rows:
                    n = F.row_chars(row)
                    if not F.row_fits(row, max_chars, max_tokens):
                        c["skipped_too_long"] += 1
                        continue
                    c[kind] += 1
                    c["written"] += 1
                    c["chars"] += n
                    c["max_chars"] = max(c["max_chars"], n)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        c["avg_chars"] = round(c.pop("chars") / max(1, c["written"]))
        report["files"][name] = c
        log(f"{name}: {c['written']}/{c['rows']} written (flow {c['flow']}, fix {c['fix']}, clarify {c['clarify']}, "
            f"rewrite {c['rewrite']}, "
            f"kept {c['kept']}, reply+context {c['reply_with_context']}, part {c['part']} from {c['split_long']} long flows), "
            f"{c['skipped_too_long']} too long, "
            f"{c['skipped_unusable']} unusable")
    cat_out = out / "catalog.jsonl"
    if Path(catalog).resolve() != cat_out.resolve():
        shutil.copyfile(catalog, cat_out)
    report["totals"] = {k2: sum(f[k2] for f in report["files"].values())
                        for k2 in ("rows", "written", "flow", "fix", "clarify", "rewrite", "kept", "reply_with_context", "part", "split_long",
                                   "skipped_too_long", "skipped_unusable")}
    report["seconds"] = round(time.time() - t0, 1)
    with open(out / "render_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    t = report["totals"]
    log(f"done: {t['written']}/{t['rows']} rows written, {t['skipped_too_long']} too long, "
        f"{t['skipped_unusable']} unusable, {report['seconds']}s -> {out}")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--k", type=int, default=F.DEFAULT_K)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--extra", nargs="*", default=[], help="neutral-record files ({\"kind\": \"flow\"|\"fix\"|\"clarify\"})")
    ap.add_argument("--variables-rate", type=float, default=0.3)
    ap.add_argument("--reply-context", type=float, default=0.5)
    ap.add_argument("--max-chars", type=int, default=F.MAX_ROW_CHARS)
    ap.add_argument("--max-tokens", type=int, default=F.MAX_ROW_TOKENS, help="estimated tokens (ooa_format.approx_tokens)")
    ap.add_argument("--no-parts", action="store_true", help="skip flows that do not fit instead of training their parts")
    ap.add_argument("--answer-format", choices=(F.ANSWER_COMPACT, F.ANSWER_FULL), default=F.ANSWER_COMPACT,
                    help="compact: the flow without variables list / step ids / descriptions (expanded at serving)")
    ap.add_argument("--gold-dropout", type=float, default=F.GOLD_DROPOUT,
                    help="share of flow rows whose context misses some needed actions (0 = never)")
    ap.add_argument("--dropout-frac", default=",".join(str(x) for x in F.GOLD_DROPOUT_FRAC),
                    help="low,high share of the needed actions such a row misses")
    a = ap.parse_args(argv)
    frac = tuple(float(x) for x in a.dropout_frac.split(","))
    render(a.src, a.out, a.catalog, a.k, a.seed, a.extra, a.variables_rate, a.reply_context, a.max_chars, not a.no_parts,
           max_tokens=a.max_tokens, answer_format=a.answer_format, gold_dropout=a.gold_dropout, dropout_frac=frac)


if __name__ == "__main__":
    main()
