"""Train the OOA model on a PC with an NVIDIA GPU: the same SFT run as Colab cell 7, in one command.

Run inside WSL (Ubuntu) or Linux, after setup_wsl.sh. See README.md in this folder.

    python train_local.py --work ~/ooa_train                 # full run (2000 optimizer steps, like Colab)
    python train_local.py --work ~/ooa_train --check         # only check that everything is in place
    python train_local.py --work ~/ooa_train --device-batch 2   # GPU with 6-8 GB memory

--work must contain what you copied from Google Drive (ooa-nanochat):
    nanochat_base/tokenizer/                tokenizer.pkl, token_bytes.pt
    nanochat_base/base_checkpoints/d8/      model_<step>.pt, meta_<step>.json, optim_<step>_rank0.pt (the newest step)
    sft/                                    train_flows.jsonl, train_catalog.jsonl, synth_train.jsonl,
                                            synth_pattern.jsonl, val.jsonl (+ natural_train.jsonl and the other
                                            optional files if you have them)
The trained model lands in nanochat_base/chatsft_checkpoints/d8/. An older one there is renamed (kept), never deleted.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RPA = HERE.parent.parent                                   # "rpa model"
REQUIRED_SFT = ["train_flows.jsonl", "train_catalog.jsonl", "synth_train.jsonl", "synth_pattern.jsonl", "val.jsonl"]
OPTIONAL_SFT = ["self_improved.jsonl", "hard_fixed.jsonl", "feedback_train.jsonl", "natural_train.jsonl", "core_train.jsonl"]
NANOCHAT_COMMIT = "92d63d4"


def patch_chat_sft(src, sft_dir, a):
    """Same data mix as Colab cell 7 (make_colab_notebook.py); keep the two in step."""
    s = src
    s = s.replace("from tasks.smoltalk import SmolTalk", "from tasks.smoltalk import SmolTalk\nfrom tasks.ooa import OOADialogues")
    i, j = s.index("train_tasks = ["), s.index("\n", s.index("# total: 24K"))
    d = Path(sft_dir).as_posix()      # forward slashes: a Windows path pasted into Python code has no escapes
    new = f"""train_tasks = ([OOADialogues("{d}/train_flows.jsonl") for _ in range({a.flow_repeat})]
               + [OOADialogues("{d}/train_catalog.jsonl"), OOADialogues("{d}/synth_train.jsonl"),
                  OOADialogues("{d}/synth_pattern.jsonl")]) * {a.epochs}
import os as _os
_extra = [OOADialogues("{d}/" + _f) for _f in ("self_improved.jsonl", "hard_fixed.jsonl", "feedback_train.jsonl")
          if _os.path.exists("{d}/" + _f)]
train_tasks += _extra * {a.epochs}
print0(f"extra self-learning files in the mix: {{len(_extra)}}")
_nat = "{d}/natural_train.jsonl"
if _os.path.exists(_nat):
    train_tasks += [OOADialogues(_nat)] * ({a.natural_repeat} * {a.epochs})
print0(f"natural-wording file in the mix: {{_os.path.exists(_nat)}}")
_core = "{d}/core_train.jsonl"
if _os.path.exists(_core):
    train_tasks += [OOADialogues(_core)] * ({a.core_repeat} * {a.epochs})
print0(f"core-actions file in the mix: {{_os.path.exists(_core)}}")
if {a.smoltalk} > 0:
    train_tasks.append(SmolTalk(split="train", stop={a.smoltalk}))
train_dataset = TaskMixture(train_tasks)
print0(f"OOA training mixture: {{len(train_dataset):,}} rows")
val_dataset = TaskMixture([OOADialogues("{d}/val.jsonl")])"""
    s = s[:i] + new + s[j:]
    old_r = "            ids, mask = tokenizer.render_conversation(conversation)\n            conv_buffer.append((ids, mask))"
    if old_r not in s:
        raise SystemExit("nanochat's chat_sft.py is not the expected version (commit " + NANOCHAT_COMMIT + ")")
    s = s.replace(old_r, "            ids, mask = tokenizer.render_conversation(conversation, max_tokens=10**9)\n"
                         "            if len(ids) <= row_capacity:\n"
                         "                conv_buffer.append((ids, mask))\n"
                         "            else:\n"
                         "                skipped_long[0] += 1")
    s = s.replace("last_step = False # we will toggle this", "skipped_long = [0]\nlast_step = False # we will toggle this")
    if a.no_compile:   # plain Windows has no C++ compiler for torch.compile; the model then runs uncompiled
        old_c = "model = torch.compile(model, dynamic=False)"
        if old_c not in s:
            raise SystemExit("nanochat's chat_sft.py is not the expected version (torch.compile line)")
        s = s.replace(old_c, "pass  # torch.compile switched off (--no-compile)")
        # nanochat's optimizer (nanochat/optim.py) is also @torch.compile'd: make torch.compile a no-op before it is
        # imported (disabling dynamo instead fails on its fullgraph=True functions)
        s = ("import torch as _torch_nc\n"
             "_torch_nc.compile = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)\n") + s
    s = s.replace('print0(f"Total training time:',
                  'print0(f"Conversations skipped as longer than the window (not cut): {skipped_long[0]}")\nprint0(f"Total training time:')
    return s


def newest_step(folder, prefix="model_"):
    steps = [int(m.group(1)) for f in glob.glob(str(folder / f"{prefix}*.pt")) if (m := re.search(r"_(\d+)", Path(f).name))]
    return max(steps) if steps else None


def main():
    try:                                    # a Windows console / log file may not take every character: never crash on it
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="folder with nanochat_base/ and sft/ copied from Drive")
    ap.add_argument("--nanochat", default=str(RPA / "nanochat"), help="nanochat code (pinned version, bundled)")
    ap.add_argument("--steps", type=int, default=2500, help="optimizer steps of 32,768 tokens each (same as Colab: 2500)")
    ap.add_argument("--device-batch", type=int, default=4, choices=[1, 2, 4, 8, 16],
                    help="rows per GPU pass: 4 needs ~10+ GB, use 2 for 6-8 GB, 1 for 4 GB (same result, slower)")
    ap.add_argument("--seq", type=int, default=2048, choices=[512, 1024, 2048])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--flow-repeat", type=int, default=5)
    ap.add_argument("--natural-repeat", type=int, default=5)
    ap.add_argument("--core-repeat", type=int, default=5)
    ap.add_argument("--smoltalk", type=int, default=3000, help="general chat rows (downloads once); 0 = none")
    ap.add_argument("--init-lr-frac", type=float, default=0.8, help="learning rate as a fraction of the base one "
                    "(0.8 = Colab; a top-up from an already trained model uses less)")
    ap.add_argument("--eval-every", type=int, default=100, help="validation every N steps (costly on a CPU)")
    ap.add_argument("--no-compile", action="store_true", default=(os.name == "nt"),
                    help="run without torch.compile (automatic on Windows, which has no compiler for it)")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for training (0 = all; fewer keeps the PC usable)")
    ap.add_argument("--low-priority", action="store_true", help="run training at low priority so the PC stays responsive")
    ap.add_argument("--check", action="store_true", help="only check files and the GPU, do not train")
    ap.add_argument("--allow-cpu", action="store_true", help="train without a GPU (days; only for testing)")
    a = ap.parse_args()

    work = Path(a.work).expanduser().resolve()
    base, sft, nano = work / "nanochat_base", work / "sft", Path(a.nanochat).resolve()
    problems = []
    for f in ("tokenizer/tokenizer.pkl", "tokenizer/token_bytes.pt"):
        if not (base / f).exists():
            problems.append(f"missing {base / f}")
    bdir = base / "base_checkpoints" / "d8"
    step = newest_step(bdir)
    if step is None:
        problems.append(f"no model_<step>.pt in {bdir} (copy base_checkpoints/d8 from Drive)")
    else:
        for f in (f"meta_{step:06d}.json", f"optim_{step:06d}_rank0.pt"):
            if not (bdir / f).exists():
                problems.append(f"missing {bdir / f}" + (" (optional, but training starts slightly worse)" if f.startswith("optim") else ""))
    for f in REQUIRED_SFT:
        if not (sft / f).exists():
            problems.append(f"missing {sft / f}")
    if not (nano / "scripts" / "chat_sft.py").exists():
        problems.append(f"nanochat code not found at {nano}. Unzip the NEW ooa_assistant.zip (it has rpa model/nanochat), "
                        "or: cd \"<rpa model folder>\" && git clone https://github.com/karpathy/nanochat.git && "
                        "cd nanochat && git checkout 92d63d4e8bb4df75c3b71618f31ddde2378b2bcd")
    other = sorted(p.name for p in (base / "base_checkpoints").glob("d*") if p.is_dir() and p.name != "d8")
    if other:
        print("note: base_checkpoints also has", ", ".join(other), "- training uses d8 only (the one checked here)")

    import torch
    gpu = torch.cuda.is_available()
    print("work folder :", work)
    print("base model  :", f"{bdir} step {step}" if step is not None else "-")
    print("GPU         :", torch.cuda.get_device_name(0) if gpu else "NONE")
    if gpu:
        cap = torch.cuda.get_device_capability()
        if cap < (7, 5):
            problems.append(f"GPU {torch.cuda.get_device_name(0)} (compute capability {cap[0]}.{cap[1]}) is too old for the "
                            "training code (torch.compile); it needs an RTX 20xx / GTX 16xx or newer")
        mem = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"GPU memory  : {mem:.1f} GB", "(use --device-batch 2)" if mem < 10 and a.device_batch > 2 else "")
    print("data        :", ", ".join(f"{f} {'yes' if (sft / f).exists() else 'NO'}" for f in REQUIRED_SFT + OPTIONAL_SFT))
    hard = [p for p in problems if "optional" not in p]
    for p in problems:
        print("  !", p)
    if not gpu and not a.allow_cpu:
        hard.append("no NVIDIA GPU visible (in WSL: install the NVIDIA Windows driver, then check `nvidia-smi` in Ubuntu)")
        print("  !", hard[-1])
    if hard:
        raise SystemExit("\nNot ready: fix the lines marked ! above.")
    if a.check:
        print("\nREADY. Run again without --check to train.")
        return

    # nanochat code: add our task, write a patched copy of chat_sft (the original file stays untouched)
    shutil.copy(RPA / "training" / "ooa_task.py", nano / "tasks" / "ooa.py")
    (nano / "scripts" / "chat_sft_ooa.py").write_text(
        patch_chat_sft((nano / "scripts" / "chat_sft.py").read_text(encoding="utf-8"), sft, a), encoding="utf-8")

    out = base / "chatsft_checkpoints"
    if out.exists():                                        # nanochat loads the highest step: move an old run aside
        keep = base / f"chatsft_checkpoints_before_{time.strftime('%Y%m%d_%H%M%S')}"
        out.rename(keep)
        print("previous model kept in", keep)

    # one CPU thread is right next to a GPU, but on a CPU-only run it would leave all other cores idle
    env = dict(os.environ, NANOCHAT_BASE_DIR=str(base), PYTHONUTF8="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS=("1" if gpu else str(a.threads or os.cpu_count() or 4)),
               NANOCHAT_DTYPE=("bfloat16" if gpu and torch.cuda.get_device_capability() >= (8, 0) else
                               "float16" if gpu else "float32"))
    total = 32768                                           # tokens per optimizer step, same as Colab
    accum = max(1, total // (a.seq * a.device_batch))       # nanochat counts --num-iterations in micro-batches
    cmd = [sys.executable, "-m", "scripts.chat_sft_ooa", f"--max-seq-len={a.seq}", f"--device-batch-size={a.device_batch}",
           f"--total-batch-size={a.seq * a.device_batch * accum}", f"--num-iterations={a.steps * accum}",
           f"--eval-every={a.eval_every}", f"--eval-tokens={a.seq * a.device_batch * 8}", "--chatcore-every=-1", "--run=dummy",
           f"--init-lr-frac={a.init_lr_frac}",
           "--model-tag=d8", f"--model-step={step}"]       # exactly the base checkpoint checked above
    if not gpu:
        cmd.append("--device-type=cpu")

    log = work / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"
    print("dtype:", env["NANOCHAT_DTYPE"], "| log:", log, "\n" + " ".join(cmd) + "\n")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as lf:
        prio = {}
        if a.low_priority:                                   # the PC stays usable while it trains
            if os.name == "nt":
                prio["creationflags"] = 0x00004000           # BELOW_NORMAL_PRIORITY_CLASS
            else:
                prio["preexec_fn"] = lambda: os.nice(10)
        p = subprocess.Popen(cmd, cwd=nano, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             encoding="utf-8", errors="replace", **prio)   # nanochat prints characters like the square-root sign
        for line in p.stdout:
            sys.stdout.write(line)
            lf.write(line)
        rc = p.wait()
    mins = (time.time() - t0) / 60
    if rc != 0:
        raise SystemExit(f"\ntraining stopped with an error after {mins:.0f} min (see {log}). Send the last 30 lines.")
    d8 = out / "d8"
    s = newest_step(d8)
    if s is None:
        raise SystemExit(f"\nTraining finished but no model_*.pt is in {d8}. Contents of {out}: "
                         f"{sorted(str(p.relative_to(out)) for p in out.rglob('*'))}")
    print(f"\nDONE in {mins:.0f} min. New model files: {d8}/model_{s:06d}.pt and {d8}/meta_{s:06d}.json")
    print("Use it on the PC: in  rpa model\\nanochat_base\\chatsft_checkpoints\\d8\\  first move the old model_*.pt and "
          "meta_*.json out, then copy these two in, restart start_assistant.bat and run  python basic_test.py")


if __name__ == "__main__":
    main()
