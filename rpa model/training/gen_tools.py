"""Generate valid flows for the actions the training data barely covers (cloud and business apps first).

Coverage check (training/tool_coverage.py): about a third of the catalog had no everyday-wording example at all,
mostly the apps whose actions need a vault credential (Google Workspace, Microsoft 365, Azure DevOps, Asana, GitHub,
DocuSign, Box, Databricks, Snowflake, Salesforce, Slack, ...): gen_flows.py skips credential inputs. This builds,
for every action with fewer than --min everyday examples:
  single : the action on its own; credential inputs become Credential flow inputs (from the vault)
  lookup : a find/list/get step of the same app feeding this action by name (channel_id -> channel, issue_key ...),
           the way people chain them ("find the channel by name, then post to it"); one credential shared
Every flow passes the same checks as the other generators.

Usage: python gen_tools.py CATALOG.jsonl PLATFORM_REPO EVERYDAY_TRAIN_FILES(comma) OUT.jsonl [min=3] [seed=5]
Writes {"flow": {...}, "shape": "...", "toolkit": "...", "target": "..."} per line.
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_flows as G

TYPE = dict(G.TYPE_MAP, **{"Credential (vault)": "Credential", "Window": "Window", "UI element": "UIElement",
                           "Record": "Record", "File": "File"})
SKIP_KINDS = {"loop type", "condition"}


def vt(t):
    return TYPE.get(t) or "Any"


def snake(s):
    return G.snake(s)


def actions_in(answer):
    out = []
    try:
        f = json.loads(answer)
    except Exception:
        return out
    def walk(nodes):
        for n in nodes or []:
            if isinstance(n, dict):
                if isinstance(n.get("action"), str):
                    out.append(n["action"])
                for k in ("children", "else", "finally"):
                    walk(n.get(k) if isinstance(n.get(k), list) else None)
    walk(f.get("steps") if isinstance(f, dict) else None)
    return out


def links(out_name, in_name):
    """Does an output (by name) naturally feed an input (by name)? channel_id -> channel, issue -> issue_key ..."""
    o, i = out_name.lower(), in_name.lower()
    if o == i:
        return True
    base_o = re.sub(r"_(id|key|ids|keys|url|name)$", "", o)
    base_i = re.sub(r"_(id|key|ids|keys|url|name)$", "", i)
    return base_o == base_i and (o.endswith(("_id", "_key")) or i.endswith(("_id", "_key")) or base_o == o or base_i == i)


VAGUE = {"id", "number", "item_id", "name", "path", "value", "result", "text", "count", "url", "key", "status"}
LOOKUP = re.compile(r"^(get|list|find|search|lookup|query|resolve|read)_|_by_(name|email|title|key)$")
DESTROY = re.compile(r"^(delete|cancel|remove|archive|void|purge|close)")


def good_feeder(x, target_verb):
    """A step people put before another one: a lookup (find the channel id, list the issues) or, before a
    non-destructive step, a create whose new id the next step uses (create a folder, then upload to it)."""
    v = x["id"].split("/")[1]
    if LOOKUP.search(v):
        return True
    return v.startswith(("create", "add", "upload", "new")) and not DESTROY.match(target_verb)


class Flow:
    def __init__(self, rng):
        self.rng, self.variables, self.names, self.descs, self.steps = rng, [], set(), [], []
        self.avail = []          # (variable, output name, catalog type)
        self.creds = {}          # input name -> credential variable (shared by the steps of one app)

    def name(self, base):
        base = snake(base)[:40] or "value"
        n, k = base, 2
        while n in self.names:
            n, k = f"{base}_{k}", k + 1
        self.names.add(n)
        return n

    def add(self, e, wire=True):
        args = {}
        for i in [i for i in e["inputs"] if i["required"]]:
            if i["type"] == "Credential (vault)":
                if i["name"] not in self.creds:
                    v = self.name(i["name"])
                    self.variables.append({"name": v, "type": "Credential", "direction": "input", "required": True,
                                           "description": (i.get("description") or "From the vault.")[:120]})
                    self.creds[i["name"]] = v
                args[i["name"]] = "$" + self.creds[i["name"]]
                continue
            src = next((a for a in reversed(self.avail) if wire and links(a[1], i["name"])
                        and (a[2] == i["type"] or "Any" in (a[2], i["type"]))), None)
            if src:
                args[i["name"]] = "$" + src[0]
                continue
            v = self.name(i["name"] if self.rng.random() < 0.8 else "my_" + i["name"])
            var = {"name": v, "type": vt(i["type"]), "direction": "input", "required": True}
            if i.get("description"):
                var["description"] = i["description"][:120]
            self.variables.append(var)
            args[i["name"]] = "$" + v
        out = {}
        for o in e["outputs"][:3]:
            v = self.name(o["name"])
            out[o["name"]] = v
            self.variables.append({"name": v, "type": vt(o["type"]), "direction": "local"})
            self.avail.append((v, o["name"], o["type"]))
        d = G.clean_desc(e["description"])
        s = {"id": f"s{len(self.steps) + 1}", "action": e["id"], "description": d[:140], "args": args}
        if out:
            s["out"] = out
        self.steps.append(s)
        self.descs.append(d)
        return s

    def done(self):
        last = self.steps[-1].get("out") or {}
        for v in list(last.values())[:2]:
            for var in self.variables:
                if var["name"] == v:
                    var["direction"] = "output"
        first = self.descs[-1]
        return {"kind": "TaskBot", "id": snake(first)[:40] or "flow", "name": " ".join(first.split()[:6]).title(),
                "description": "; then ".join(self.descs), "variables": self.variables, "steps": self.steps}


def main():
    cat_path, repo, train_files, out = sys.argv[1:5]
    min_ex = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    rng = random.Random(int(sys.argv[6]) if len(sys.argv) > 6 else 5)
    catalog = [json.loads(l) for l in open(cat_path, encoding="utf-8") if l.strip()]
    byid = {e["id"]: e for e in catalog}
    seen = Counter()
    for f in train_files.split(","):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                seen.update(set(actions_in(json.loads(l)["messages"][-1]["content"])))
    validate = G.make_validator(repo, catalog)
    by_tk = defaultdict(list)
    for e in catalog:
        if e["kind"] == "action" and not e["id"].startswith("core/"):
            by_tk[e["toolkit_id"]].append(e)
    targets = [e for e in catalog if e["kind"] == "action" and not e["id"].startswith("core/")
               and e["kind"] not in SKIP_KINDS and seen[e["id"]] < min_ex]
    rows, why = [], Counter()
    for e in targets:
        made = []
        f = Flow(rng)
        f.add(e)
        made.append(("single", f.done()))
        # lookups of the same app whose outputs feed this action's required inputs by name
        need = [i for i in e["inputs"] if i["required"] and i["type"] != "Credential (vault)"]
        verb = e["id"].split("/")[1]
        feeders = [x for x in by_tk[e["toolkit_id"]] if x["id"] != e["id"] and good_feeder(x, verb)
                   and any(links(o["name"], i["name"]) and o["name"].lower() not in VAGUE for o in x["outputs"] for i in need)]
        rng.shuffle(feeders)
        for x in feeders[:2]:
            f = Flow(rng)
            f.add(x)
            s = f.add(e)
            if any(isinstance(v, str) and v[1:] in {a[0] for a in f.avail} for v in s["args"].values()):
                made.append(("lookup", f.done()))
        for shape, flow in made:
            try:
                validate(flow)
            except Exception as ex:
                why[f"invalid: {str(ex)[:60]}"] += 1
                continue
            rows.append({"flow": flow, "shape": shape, "toolkit": e["toolkit"], "target": e["id"]})
    with open(out, "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(targets)} actions with < {min_ex} everyday examples -> {len(rows)} flows "
          f"{dict(Counter(r['shape'] for r in rows))} | toolkits {len({r['toolkit'] for r in rows})} | skipped {dict(why)}")


if __name__ == "__main__":
    main()
