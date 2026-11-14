"""Regression tests for the verifier reward. Run: python test_reward.py PLATFORM_REPO
(needs jsonschema). Checks that real mistakes cost points and that nothing can win by cheating."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
repo = sys.argv[1]
os.environ["OOA_CATALOG"] = os.path.join(HERE, "../data/catalog/catalog.jsonl")
os.environ["OOA_SCHEMA"] = os.path.join(repo, "contracts/flow-schema/flow.v1.json")
sys.path.insert(0, HERE)
import ooa_reward as R, gen_flows as G

cat = [json.loads(l) for l in open(os.environ["OOA_CATALOG"], encoding="utf-8")]
flows = G.generate(cat, repo, 200, 5)
# a linear flow with >=2 steps whose first step has args
q, f0, txt = next(x for x in flows if len(x[1]["steps"]) >= 2 and not x[1]["steps"][0]["action"].startswith("core/") and x[1]["steps"][0].get("args"))
gold = R.gold_actions(txt)

def sc(flow, g=gold):
    return R.score(flow if isinstance(flow, str) else json.dumps(flow), g)

def mut(fn):
    f = json.loads(txt); fn(f); return f

assert sc(txt)["solved"] and sc(txt)["total"] == 1.0, "perfect answer must score 1.0"
assert sc(txt + "<|assistant_end|>")["solved"], "end marker must be ignored"
assert sc("Here is a paragraph")["total"] == 0.0, "prose scores 0"
assert sc(txt[:300])["total"] < 0.1, "truncated JSON scores ~0"
assert not sc(mut(lambda f: f.update(steps=[])))["solved"] and sc(mut(lambda f: f.update(steps=[])))["total"] <= 0.25, "empty flow cannot win"
assert not sc(mut(lambda f: f["steps"][0].update(action="ooa.fake/x")))["solved"], "invented action fails"
assert not sc(mut(lambda f: f["steps"][0].update(args={})))["solved"], "missing required input fails"
assert not sc(mut(lambda f: f["steps"][0].update(args={k: "$nope" for k in f["steps"][0]["args"]})))["solved"], "undeclared variable fails"
assert not sc(mut(lambda f: f["steps"][0].update(id=f["steps"][1]["id"])))["solved"], "duplicate ids fail"
assert not sc(mut(lambda f: f["steps"].reverse()))["solved"], "wrong order is not solved"
assert sc(mut(lambda f: f.update(steps=f["steps"][:1])))["total"] < 1.0, "cut-off flow loses points"
other = next(x for x in flows if R.gold_actions(x[2]) != gold)[2]
r = sc(other); assert r["valid"] and not r["solved"], "valid flow for a different task is not solved"
assert sc(txt, None)["total"] <= 0.75, "without gold the total is capped at .75"
# variable references: templates and member access are references to their ROOT variable (docs/llms.txt)
assert R.refs("${folder}/data/x.csv") == {"folder"} and R.refs("$row.Name") == {"row"} and R.refs("$n - 1") == {"n"}
assert R.refs("${len($xs)}") == {"xs"} and R.refs("${sys.date}") == set() and R.refs({"literal": "$x"}) == set()
assert not sc(mut(lambda f: f["steps"][0].update(args={k: "${nope}/x" for k in f["steps"][0]["args"]})))["solved"], "undeclared template variable fails"
# newer containers (flow.v1.json): core/switch cases[].children + default, core/state_machine states[].entry and
# states[].transitions[].steps. Steps inside them are checked like any other step.
_s0 = json.loads(txt)["steps"][0]
_cont = {"kind": "TaskBot", "id": "containers", "name": "containers", "variables": json.loads(txt)["variables"] + [
    {"name": "mode", "type": "String", "direction": "input"}], "steps": [
    {"id": "sw", "action": "core/switch", "args": {"value": "$mode"},
     "cases": [{"values": ["a"], "children": [dict(_s0, id="c1")]}], "default": [dict(_s0, id="d1")]},
    {"id": "sm", "action": "core/state_machine", "args": {"initial": "start"}, "states": [
        {"name": "start", "entry": [dict(_s0, id="e1")], "transitions": [{"to": "end", "steps": [dict(_s0, id="t1")]}]},
        {"name": "end", "final": True}]}]}
assert [s["id"] for s in R.walk(_cont["steps"])] == ["sw", "c1", "d1", "sm", "e1", "t1"], list(R.walk(_cont["steps"]))
assert sc(_cont, None)["valid"], R.problems(json.dumps(_cont))
_bad_if = {"id": "i", "action": "core/if", "args": {"condition": "$mode"}, "children": [],
           "elseIf": [{"condition": "$mode", "children": [{"id": "x", "action": "ooa.fake/x"}]}]}
for _path, _put in [("case", lambda f, s: f["steps"][0]["cases"][0]["children"].append(s)),
                    ("default", lambda f, s: f["steps"][0]["default"].append(s)),
                    ("entry", lambda f, s: f["steps"][1]["states"][0]["entry"].append(s)),
                    ("transition", lambda f, s: f["steps"][1]["states"][0]["transitions"][0]["steps"].append(s))]:
    _f = json.loads(json.dumps(_cont))
    _put(_f, {"id": "bad", "action": "ooa.fake/x", "args": {}})
    assert not sc(_f, None)["valid"] and "Unknown action 'ooa.fake/x'." in R.problems(json.dumps(_f)), _path
    _f = json.loads(json.dumps(_cont))
    _put(_f, dict(_s0, id="dup", args={k: "$nope" for k in _s0["args"]}))
    assert not sc(_f, None)["valid"], "undeclared variable inside " + _path
    _f = json.loads(json.dumps(_cont))
    _put(_f, dict(_bad_if))
    assert "Unknown action 'ooa.fake/x'." in R.problems(json.dumps(_f)), "elseIf inside " + _path
# the real platform example flows are the ground truth: the verifier must accept them (all but the stepless agent file)
import glob as _g
_real = [json.load(open(p)) for p in sorted(_g.glob(os.path.join(repo, "examples", "**", "*.flow.json"), recursive=True))]
_ok = sum(R.score(json.dumps(f))["valid"] for f in _real if f.get("steps"))
assert _ok == sum(1 for f in _real if f.get("steps")), f"verifier rejects real example flows: {_ok}"
print("all reward tests passed")
