"""Notebook cell: run the whole assistant (model -> repair -> verify) on the trained model (imported by make_colab_notebook.py)."""


def build(md, code):
    md("""
## Try the whole assistant (local only)

Runs the same router the server uses: our model writes the flow, typo-level repair is applied, and the platform's rules verify it.
Cloud escalation is **off** here (no key needed), so this shows what our own model can do alone. It also runs the server's
local-model code on a GPU for the first time.
""")
    code('''
#@title 13. The whole assistant, local only: model -> repair -> verify
import json, os, sys
sys.path[:0] = ["/content/ooa", "/content/ooa/serving", "/content/nanochat"]
os.chdir("/content/nanochat")
import ooa_server as S
import ooa_reward as R
from backends import NanochatBackend

policy = S.Policy(allow_escalation=False, max_clauses_local=99, local_retries=2)    # local only; do not skip the local model
local = NanochatBackend("sft", max_new_tokens=1024)
router = S.Router(f"{DRIVE_DIR}/sft/catalog.jsonl", "/content/platform", policy, local, None)
rows = [json.loads(l) for l in open(f"{DRIVE_DIR}/sft/eval_synth.jsonl")]
verified = exact = repaired = 0
for r in rows:
    res = router.run(r["prompt"])
    if res["verified"]:
        verified += 1
        repaired += bool(res["repairs"])
        got = R.actions_of(json.loads(res["flow"]))
        exact += got == r["gold_actions"]
    print(("OK   " if res["verified"] else "FAIL ") + f"{res['seconds']:>5}s attempts={len(res['attempts'])} repairs={len(res.get('repairs', []))}"
          + ("" if res["verified"] else " | " + "; ".join(res.get("problems", [])[:2])[:110]))
print(f"\\nverified (passed every rule): {verified}/{len(rows)} | fixed by repair: {repaired} | exact right actions: {exact}/{len(rows)}")
''')
