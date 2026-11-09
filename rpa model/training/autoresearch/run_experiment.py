"""Run ONE fixed-budget experiment: build data from recipe.py -> SFT for exactly token_budget tokens -> frozen eval -> result.
READ-ONLY for agents (like Karpathy's prepare.py). Usage:
  python run_experiment.py --platform PLATFORM_REPO --catalog CATALOG.jsonl --workdir DIR --nanochat NANOCHAT_DIR \
      --base-dir NANOCHAT_BASE_DIR [--recipe recipe.py] [--dry-run]
Prints a summary block (grep-able) and writes WORKDIR/result.json. --dry-run builds data, checks leakage and prints the
commands without training (works on CPU)."""
import argparse, json, math, os, re, runpy, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from constants import FIXED_TOKEN_BUDGET
import gen_flows as G

ALLOWED = {"token_budget", "synth_train", "flow_repeat", "smoltalk_rows", "include_catalog", "step_weights", "wrap_weights",
           "seq_len", "device_batch", "grad_accum", "init_lr_frac", "warmdown_ratio", "warmup_ratio", "final_lr_frac"}

ap = argparse.ArgumentParser()
ap.add_argument("--platform", required=True); ap.add_argument("--catalog", required=True)
ap.add_argument("--workdir", required=True); ap.add_argument("--nanochat", default="/content/nanochat")
ap.add_argument("--base-dir", default=os.environ.get("NANOCHAT_BASE_DIR", ""))
ap.add_argument("--recipe", default=str(HERE / "recipe.py")); ap.add_argument("--frozen-dir", default=str(HERE / "frozen"))
ap.add_argument("--ooa-dir", default=str(HERE.parent)); ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()


def load_recipe(path):
    r = runpy.run_path(path)["RECIPE"]
    extra = set(r) - ALLOWED
    assert not extra, f"unknown recipe keys: {sorted(extra)}"
    assert r["token_budget"] == FIXED_TOKEN_BUDGET, "token_budget is fixed (constants.py); experiments must stay comparable"
    assert r["seq_len"] in (512, 1024, 2048, 4096) and r["device_batch"] >= 1 and r["grad_accum"] >= 1
    assert len(r["step_weights"]) == 4 and len(r["wrap_weights"]) == 4 and sum(r["step_weights"]) > 0 and sum(r["wrap_weights"]) > 0
    assert 0 < r["init_lr_frac"] <= 3 and 0 <= r["warmdown_ratio"] <= 1
    return r


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


R = load_recipe(a.recipe)
work = Path(a.workdir); sft = work / "sft"; sft.mkdir(parents=True, exist_ok=True)
t_start = time.time()

# ---- 1. data ---------------------------------------------------------------------------------------------------------
subprocess.run([sys.executable, str(a.ooa_dir + "/build_sft_data.py"), a.platform, str(sft), "9000"], check=True,
               stdout=subprocess.DEVNULL)
catalog = read(sft / "catalog.jsonl")
synth = [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": m}]}
         for q, _, m in G.generate(catalog, a.platform, R["synth_train"], 1, cfg=R)]
write(sft / "synth_train.jsonl", synth)
# the recipe may not train on the frozen eval: drop any row whose request or answer appears in it
frozen = read(Path(a.frozen_dir) / "frozen_eval.jsonl")
bad_prompts = {r["prompt"] for r in frozen}; bad_flows = {r["gold_flow"] for r in frozen}
dropped = 0
for name in ("train_flows", "train_catalog", "synth_train", "train"):
    rows = read(sft / f"{name}.jsonl")
    keep = [r for r in rows if r["messages"][0]["content"] not in bad_prompts and r["messages"][1]["content"] not in bad_flows]
    dropped += len(rows) - len(keep)
    write(sft / f"{name}.jsonl", keep)
# unseen validation rows: held-out catalog Q&A only
val = [r for r in read(sft / "val.jsonl") if r["messages"][1]["content"].startswith(("Use ", "`"))]
write(sft / "val.jsonl", val)

# ---- 2. how many passes does the token budget need? (estimate: 3.3 characters per token) --------------------------------
def est_tokens(rows):
    return sum(len(r["messages"][0]["content"]) + len(r["messages"][1]["content"]) for r in rows) / 3.3

flows, cats, syn = read(sft / "train_flows.jsonl"), read(sft / "train_catalog.jsonl"), read(sft / "synth_train.jsonl")
one_pass = est_tokens(flows) * R["flow_repeat"] + (est_tokens(cats) if R["include_catalog"] else 0) + est_tokens(syn) + R["smoltalk_rows"] * 330
epochs = max(1, min(30, math.ceil(R["token_budget"] / max(1, one_pass))))
tokens_per_step = R["seq_len"] * R["device_batch"] * R["grad_accum"]
iters = R["token_budget"] // tokens_per_step
print(f"data: flows {len(flows)}x{R['flow_repeat']} catalog {len(cats) if R['include_catalog'] else 0} synth {len(syn)} "
      f"smoltalk {R['smoltalk_rows']} | leaked rows dropped: {dropped} | ~{one_pass/1e6:.1f}M tokens/pass -> {epochs} pass(es) | {iters} steps")

sft_cmd = [sys.executable, "-m", "scripts.chat_sft", f"--max-seq-len={R['seq_len']}", f"--device-batch-size={R['device_batch']}",
           f"--total-batch-size={tokens_per_step}", f"--num-iterations={iters}", f"--init-lr-frac={R['init_lr_frac']}",
           f"--warmdown-ratio={R['warmdown_ratio']}", f"--warmup-ratio={R['warmup_ratio']}", f"--final-lr-frac={R['final_lr_frac']}",
           "--eval-every=-1", "--chatcore-every=-1", f"--eval-tokens={tokens_per_step * 2}", "--run=dummy"]
eval_cmd = [sys.executable, "-m", "scripts.ooa_eval_frozen", f"--frozen-dir={a.frozen_dir}", f"--out={work / 'eval.json'}",
            f"--platform={a.platform}", f"--ooa-dir={a.ooa_dir}"]
if a.dry_run:
    print("DRY RUN, nothing trained.\n  sft :", " ".join(sft_cmd[1:]), "\n  eval:", " ".join(eval_cmd[1:]))
    sys.exit(0)

# ---- 3. patch nanochat's SFT script to use this recipe, train, evaluate --------------------------------------------------
nc = Path(a.nanochat)
subprocess.run(["git", "checkout", "-q", "--", "scripts/chat_sft.py"], cwd=nc, check=True)
subprocess.run(["cp", str(Path(a.ooa_dir) / "ooa_task.py"), str(nc / "tasks/ooa.py")], check=True)
subprocess.run(["cp", str(HERE / "ooa_eval_frozen.py"), str(nc / "scripts/ooa_eval_frozen.py")], check=True)
p = nc / "scripts/chat_sft.py"; s = p.read_text()
s = s.replace("from tasks.smoltalk import SmolTalk", "from tasks.smoltalk import SmolTalk\nfrom tasks.ooa import OOADialogues")
i = s.index("train_tasks = ["); j = s.index("\n", s.index("# total: 24K"))
parts = [f'OOADialogues("{sft}/train_flows.jsonl")' for _ in range(R["flow_repeat"])]
if R["include_catalog"]:
    parts.append(f'OOADialogues("{sft}/train_catalog.jsonl")')
parts.append(f'OOADialogues("{sft}/synth_train.jsonl")')
new = (f"train_tasks = [{', '.join(parts)}] * {epochs}\n"
       + (f'train_tasks.append(SmolTalk(split="train", stop={R["smoltalk_rows"]}))\n' if R["smoltalk_rows"] > 0 else "")
       + f'train_dataset = TaskMixture(train_tasks)\nval_dataset = TaskMixture([OOADialogues("{sft}/val.jsonl")])')
p.write_text(s[:i] + new + s[j:])
env = {**os.environ, "NANOCHAT_BASE_DIR": a.base_dir}
subprocess.run(["rm", "-rf", f"{a.base_dir}/chatsft_checkpoints"], check=True)
t0 = time.time()
log = subprocess.run(sft_cmd, cwd=nc, env=env, capture_output=True, text=True)
(work / "sft.log").write_text(log.stdout + log.stderr)
if log.returncode != 0:
    print((log.stdout + log.stderr)[-1500:]); sys.exit(1)
bpb = re.findall(r"Minimum validation bpb: ([\d.]+)", log.stdout)
ev = subprocess.run(eval_cmd, cwd=nc, env=env, capture_output=True, text=True)
(work / "eval.log").write_text(ev.stdout + ev.stderr)
if ev.returncode != 0:
    print((ev.stdout + ev.stderr)[-1500:]); sys.exit(1)
res = json.load(open(work / "eval.json"))
res.update({"val_bpb": float(bpb[-1]) if bpb else None, "minutes": round((time.time() - t_start) / 60, 1),
            "train_tokens_M": round(iters * tokens_per_step / 1e6, 1), "leaked_rows_dropped": dropped})
json.dump(res, open(work / "result.json", "w"))
print("---")
for k in ("score", "real_score", "solved_rate", "valid_rate", "val_bpb", "minutes", "train_tokens_M"):
    print(f"{k + ':':16s}{res.get(k)}")
