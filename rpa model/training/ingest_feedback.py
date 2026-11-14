"""Turn real usage into training data (the long-run self-learning signal).

The assistant (page Accept / Edit / Reject, and Studio's POST /v1/feedback) appends one JSON line per suggestion to
feedback.jsonl:
  {"prompt": "...", "flow": {...}, "verdict": "accept"|"edit"|"reject", "edited_flow": {...optional...}}
  (older rows: {"prompt", "flow", "accepted": true|false, "edited_flow"}; Studio's verdicts inserted / edited /
  discarded are stored as accept / edit / reject)
Rules (humans are a signal, the verifier is the gate):
  accepted / inserted          -> a flow record with the flow
  edited (edited_flow given)   -> a flow record with the edited flow (the human's correction), and a fix record:
                                  the model's flow + what the person changed, as problems -> the edited flow
  discarded with no edit       -> not trained on; listed in rejected_prompts.jsonl as hard cases to write answers for
  every flow trained on must pass the verifier (schema, actions exist, args, variables) or it is dropped
Flows in Studio's shape ({variables, steps}) are wrapped as a TaskBot first.
Usage: python ingest_feedback.py FEEDBACK.jsonl OUT_DIR CATALOG.jsonl FLOW_SCHEMA.json
Writes OUT_DIR/feedback_train.jsonl (nanochat chat rows, request -> flow JSON),
       OUT_DIR/feedback_records.jsonl (neutral records for training/render_dataset.py --extra) and
       OUT_DIR/rejected_prompts.jsonl
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ACCEPT = {"accept", "inserted"}
EDIT = {"edit", "edited"}


def as_flow(obj):
    """A flow dict (Studio's {variables, steps} wrapped as a TaskBot), or None."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except ValueError:
            return None
    if not isinstance(obj, dict) or not isinstance(obj.get("steps"), list):
        return None
    if "kind" in obj:
        return obj
    return {"kind": "TaskBot", "id": "feedback_flow", "name": "feedback flow",
            "variables": list(obj.get("variables") or []), "steps": obj["steps"]}


def verdict_of(r):
    v = r.get("verdict")
    if v in ACCEPT:
        return "accept"
    if v in EDIT and r.get("edited_flow"):
        return "edit"
    if v is None:
        if r.get("edited_flow"):
            return "edit"
        if r.get("accepted"):
            return "accept"
    return "reject"


def changes(before, after, cat):
    """What a person changed in a flow, as plain problems for a fix record."""
    import ooa_format as F
    a, b = F.actions_of(before), F.actions_of(after)
    name = lambda x: (cat.get(x) or {}).get("name") or x
    out = []
    for x in b:
        if x not in a:
            out.append(f"The flow needs a step that uses {x} ({name(x)}).")
    for x in a:
        if x not in b:
            out.append(f"The step that uses {x} ({name(x)}) should not be there.")
    if not out and a != b:
        out.append("The steps are in the wrong order: " + F.outline(after)[:200] + ".")
    if not out:
        steps_a = {s.get("action"): s for s, _ in F.walk(before.get("steps"))}
        for s, _ in F.walk(after.get("steps")):
            old = steps_a.get(s.get("action"))
            if old is not None and (old.get("args") != s.get("args") or old.get("out") != s.get("out")):
                out.append(f"Check the settings of the step that uses {s.get('action')}.")
    if not out:
        out.append("The person changed the variables or descriptions of this flow.")
    return list(dict.fromkeys(out))[:6]


def ingest(rows, cat, valid):
    """rows: feedback rows; valid(flow) -> bool. Returns (chat rows, neutral records, rejected, dropped)."""
    good, records, rejected, dropped = [], [], [], 0
    for r in rows:
        prompt = r.get("prompt") or r.get("request")
        if not isinstance(prompt, str) or not prompt.strip():
            dropped += 1
            continue
        v = verdict_of(r)
        if v == "reject":
            rejected.append({"prompt": prompt})
            continue
        flow = as_flow(r.get("edited_flow") if v == "edit" else r.get("flow"))
        if flow is None or not valid(flow):
            dropped += 1
            continue
        mini = json.dumps(flow, separators=(",", ":"), ensure_ascii=False)
        good.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": mini}]})
        records.append({"kind": "flow", "request": prompt, "flow": flow, "source": "feedback"})
        before = as_flow(r.get("flow"))
        if v == "edit" and before is not None:
            records.append({"kind": "fix", "request": prompt, "previous": before,
                            "problems": changes(before, flow, cat), "flow": flow, "source": "feedback"})
    return good, records, rejected, dropped


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for g in rows:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")


def main(argv):
    fb, out, cat_path, schema = argv[1:5]
    os.environ["OOA_CATALOG"], os.environ["OOA_SCHEMA"] = cat_path, schema
    import ooa_reward as R
    cat = {e["id"]: e for e in (json.loads(l) for l in open(cat_path, encoding="utf-8") if l.strip())}
    rows = []
    for l in open(fb, encoding="utf-8"):
        if l.strip():
            try:
                rows.append(json.loads(l))
            except ValueError:
                continue
    valid = lambda flow: R.score(json.dumps(flow, separators=(",", ":"), ensure_ascii=False))["valid"]
    good, records, rejected, dropped = ingest(rows, cat, valid)
    os.makedirs(out, exist_ok=True)
    write_jsonl(os.path.join(out, "feedback_train.jsonl"), good)
    write_jsonl(os.path.join(out, "feedback_records.jsonl"), records)
    write_jsonl(os.path.join(out, "rejected_prompts.jsonl"), rejected)
    fixes = sum(1 for r in records if r["kind"] == "fix")
    print(f"{len(good)} usable, {dropped} dropped (failed verifier), {len(rejected)} rejected-without-edit, "
          f"{fixes} fix records")


if __name__ == "__main__":
    main(sys.argv)
