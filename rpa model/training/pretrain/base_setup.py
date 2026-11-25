"""The clean base model (v6): profiles, time / disk estimates, the patch of nanochat's scripts/base_train.py, the
tokenizer card and the base model pack. Shared by the PC (training/local/pretrain_pc.py) and Colab cells 3-4
(PROFILE "v6"), so both train the same model the same way. Pure standard library (torch / tiktoken only inside the
functions that need them).

Data: pretrain_data.py (FineWeb-Edu ODC-By + the owner's domain text, no ClimbMix). Tokenizer: nanochat's
scripts/tok_train.py on those shards (16,384 tokens; measured 4.2 characters per token on the training mix, 3.9-4.2 on
rendered flows vs 2.1-2.7 for the old web-only tokenizer). Model: nanochat d8 as before (dim 512, 8 heads of 64, full
attention "L", value embeddings; 75.5M parameters with a 16,384 vocabulary), pretrained at 2048 tokens.

Training horizon: nanochat's own rules (--target-param-data-ratio: tokens = ratio x scaling params, the batch size,
learning-rate and weight-decay scaling follow from it). "Scaling params" are nanochat's (transformer matrices +
lm_head: 33.6M at d8), so ratio 45 = 1.51B tokens = 20 tokens per parameter of the whole model (Chinchilla counted all
of them). The old base saw 0.3B tokens at 1024.

Patch (patch_base_train, anchors checked against the pinned commit; it refuses to run on another version):
  - NaN guard: a training loss that is not a number stops the run (bfloat16 / float32 at the first such step: the
    weights already took the step; float16 with its GradScaler after NAN_STEPS in a row, a skipped step leaves the
    weights alone), so does a validation bpb that is not a number; before every save all weights are checked and
    nothing broken is ever written. Exit code NAN_EXIT, first line NAN_MARK (the same as the SFT guard).
  - only the newest OOA_KEEP_CHECKPOINTS checkpoints are kept (each is about 0.8 GB at d8 with the optimizer state).
  - every checkpoint records the sha1 of the tokenizer it was trained with ("tokenizer_sha1"): a model pack must carry
    that tokenizer (assistant/model_backend.tokenizer_problem).
  - optional: torch.compile off (Windows has no compiler for it).
"""
import hashlib
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

NAN_STEPS = 5                        # float16 (GradScaler skips such a step): this many in a row -> stop
NAN_EXIT = 86                        # the exit code (sft_setup.NAN_EXIT: train_pc / pretrain_pc / Colab know it)
NAN_MARK = "OOA NAN STOP"            # the first line it prints (sft_setup.NAN_MARK)
KEEP_CHECKPOINTS = 2
VOCAB = 16384
TOK_MAX_CHARS = 2_000_000_000        # tok_train --max-chars (nanochat's default)
CHARS_PER_TOKEN_MIX = 4.18           # measured: the 16,384 mix tokenizer on the training mix (sizes data before the
                                     # tokenizer exists; the real value is measured once it does)
SHARD_BYTES_PER_CHAR = 0.407         # measured: zstd parquet shards, 0.407 GB per 1B characters
DATA_MARGIN = 1.08                   # build this much more text than the run reads (packing, the tokenizer's own c/t)
MODEL_TAG = "d8"                     # the folder every OOA tool reads (chatsft_checkpoints/d8); kept for any depth
VERSION_PREFIX = "ooa-base-v6"

# The measured anchor for the estimates: SFT of d8 at a 4096 window, device batch 1, bfloat16, no torch.compile on
# the owner's RTX 4060 Laptop (8 GB): 1.1 s per 32,768-token step = 29.6k tokens/s. Other shapes scale by nanochat's
# FLOPs per token (GPT.estimate_flops); small matrices run a little less efficiently, so the real speed can be
# 10-30% lower (thermals of a laptop too). The Colab T4 (float16, torch.compile) ran the same SFT at about 27k tokens/s.
ANCHOR = {"rtx4060": 29_600, "t4": 27_000}
ANCHOR_FLOPS = 4.027e8               # FLOPs per token of d8 at 4096 (the anchor's shape)

PROFILES = {
    # depth, window, rows per GPU pass, tokens per optimizer step (nanochat's own optimum for this horizon: auto_batch),
    # horizon (ratio or explicit iterations), checkpoint / validation spacing in steps
    "pc":   dict(depth=8, seq=2048, dbs=2, total_batch=262144, ratio=45.0, save_every=200, eval_every=500,
                 eval_tokens=2 * 2**20, sample_every=1000, repeat_scale=2.0, vocab=VOCAB),
    "t4":   dict(depth=8, seq=2048, dbs=4, total_batch=262144, ratio=45.0, save_every=100, eval_every=500,
                 eval_tokens=2 * 2**20, sample_every=1000, repeat_scale=2.0, vocab=VOCAB),
    # the CPU pipeline test: a few steps of a 2-layer model on a few MB
    "tiny": dict(depth=2, seq=128, dbs=2, total_batch=1024, iterations=6, save_every=2, eval_every=3,
                 eval_tokens=2048, sample_every=-1, repeat_scale=1.0, vocab=2048, shard_chars=1_000_000),
}


def model_shape(depth, aspect=64, head_dim=64):
    dim = ((depth * aspect + head_dim - 1) // head_dim) * head_dim
    return dim, dim // head_dim


def param_counts(depth, vocab=VOCAB, aspect=64, head_dim=64):
    """nanochat GPT parameter counts without torch: wte, value_embeds, lm_head, transformer_matrices, total,
    scaling (transformer matrices + lm_head, what --target-param-data-ratio multiplies). Value embeddings sit on
    every other layer (gpt.py has_ve: the last layer and every second one before it)."""
    dim, _ = model_shape(depth, aspect, head_dim)
    ve_layers = sum(1 for i in range(depth) if i % 2 == (depth - 1) % 2)
    gate = 12 * (dim // head_dim)                               # value-embedding gates (ve_gate) on those layers
    mats = depth * 12 * dim * dim + ve_layers * gate
    out = {"wte": vocab * dim, "value_embeds": ve_layers * vocab * dim, "lm_head": vocab * dim,
           "transformer_matrices": mats}
    out["total"] = sum(out.values()) + 2 * depth                 # + resid / x0 lambdas
    out["scaling"] = out["transformer_matrices"] + out["lm_head"]
    return out


def flops_per_token(depth, seq, vocab=VOCAB, aspect=64, head_dim=64):
    """GPT.estimate_flops: 6 x (matrix params + lm_head) + 12 x layers x heads x head_dim x seq (full attention)."""
    dim, heads = model_shape(depth, aspect, head_dim)
    pc = param_counts(depth, vocab, aspect, head_dim)
    return 6 * (pc["transformer_matrices"] + pc["lm_head"]) + 12 * depth * heads * head_dim * seq


def auto_batch(target_tokens, ratio):
    """nanochat's optimal total batch (Power Lines, B ~ D^0.383 around 2^19 tokens at the d12 reference)."""
    d_ref = ratio * param_counts(12)["scaling"]
    return 2 ** round(math.log2(2**19 * (target_tokens / d_ref) ** 0.383))


def horizon(profile, vocab=VOCAB):
    """(target tokens, iterations, tokens actually trained) of a profile, as base_train computes them."""
    p = profile
    vocab = p.get("vocab", vocab)
    if p.get("iterations"):
        it = int(p["iterations"])
        return it * p["total_batch"], it, it * p["total_batch"]
    target = int(p["ratio"] * param_counts(p["depth"], vocab)["scaling"])
    it = target // p["total_batch"]
    return target, it, it * p["total_batch"]


def estimate(profile, gpu="rtx4060", vocab=VOCAB, chars_per_token=CHARS_PER_TOKEN_MIX):
    """Hours and disk of a profile on a GPU (ANCHOR, scaled by FLOPs per token). Disk: shards for the text read
    (with DATA_MARGIN) + KEEP_CHECKPOINTS checkpoints + the final pack; checkpoint = fp32 weights + optimizer state
    (Muon momentum for the matrices, AdamW's two moments for embeddings / lm_head), about 12 bytes per embedding
    parameter and 8 per matrix parameter."""
    p = profile
    vocab = p.get("vocab", vocab)
    _, it, tokens = horizon(p, vocab)
    tps = ANCHOR[gpu] * ANCHOR_FLOPS / flops_per_token(p["depth"], p["seq"], vocab)
    evals = (it // max(1, p.get("eval_every", 250)) + 1) * p.get("eval_tokens", 0) / (3 * tps)
    hours = (tokens / tps + evals) / 3600
    pc = param_counts(p["depth"], vocab)
    emb = pc["total"] - pc["transformer_matrices"]
    ckpt_gb = (emb * 12 + pc["transformer_matrices"] * 8) / 1e9
    data_gb = tokens * chars_per_token * DATA_MARGIN * SHARD_BYTES_PER_CHAR / 1e9
    return {"params_M": round(pc["total"] / 1e6, 1), "scaling_M": round(pc["scaling"] / 1e6, 1),
            "tokens_B": round(tokens / 1e9, 3), "iterations": it, "tokens_per_s": int(tps),
            "hours": round(hours, 1), "hours_range": (round(hours, 1), round(hours / 0.75, 1)),
            "seconds_per_step": round(p["total_batch"] / tps, 1),
            "data_gb": round(data_gb, 2), "checkpoint_gb": round(ckpt_gb, 2),
            "disk_gb": round(data_gb + ckpt_gb * (KEEP_CHECKPOINTS + 1), 1),
            "tokens_per_param": round(tokens / pc["total"], 1)}


def train_chars_for(profile, chars_per_token=CHARS_PER_TOKEN_MIX, vocab=VOCAB):
    """Characters of training text to build so the run never repeats an epoch (and the tokenizer has its 2B)."""
    _, _, tokens = horizon(profile, vocab)
    return int(max(tokens * chars_per_token * DATA_MARGIN, min(TOK_MAX_CHARS, tokens * chars_per_token * 3)))


def base_train_args(profile, resume_step=None, device_type=None, model_tag=MODEL_TAG):
    """scripts.base_train flags for a profile (the same on the PC and Colab)."""
    p = profile
    args = [f"--depth={p['depth']}", "--head-dim=64", "--window-pattern=L", f"--max-seq-len={p['seq']}",
            f"--device-batch-size={p['dbs']}", f"--total-batch-size={p['total_batch']}",
            f"--eval-every={p['eval_every']}", f"--eval-tokens={p['eval_tokens']}", "--core-metric-every=-1",
            f"--sample-every={p['sample_every']}", f"--save-every={p['save_every']}", "--run=dummy",
            f"--model-tag={model_tag}"]
    if p.get("iterations"):
        args.append(f"--num-iterations={int(p['iterations'])}")
    else:
        args.append(f"--target-param-data-ratio={p['ratio']:g}")
    if resume_step:
        args.append(f"--resume-from-step={int(resume_step)}")
    if device_type:
        args.append(f"--device-type={device_type}")
    return args


def device_batch(mem_gb, seq):
    """Rows per GPU pass at a window: about 8 GB holds 2 x 2048 at d8 (SFT measured 1 x 4096 on the RTX 4060 8 GB);
    the T4's 15 GB holds 4 x 2048. Tokens per step stay the profile's (the rest is gradient accumulation)."""
    rows = 8 if mem_gb >= 30 else 4 if mem_gb >= 14 else 2 if mem_gb >= 7 else 1
    return max(1, rows * 2048 // seq)


# ------------------------------------------------------------------------------------------- base_train.py patch
COMPILE_LINE = ("model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so "
                "dynamic=False is safe\n")
LOOP_ANCHOR = "# Go!\nwhile True:\n"
LOSS_ANCHOR = "    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point\n"
VAL_ANCHOR = '        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")\n'
SAVE_ANCHOR = ("    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and "
               "step % args.save_every == 0):\n        save_checkpoint(\n")
SAVE_END = "            rank=ddp_rank,\n        )\n"
META_ANCHOR = '                "user_config": user_config, # inputs to the training script\n'
TOKENIZER_ANCHOR = "vocab_size = tokenizer.get_vocab_size()\n"

PRE_LOOP = f'''# OOA: stop when the training goes unstable, before anything broken is saved (base_setup.py)
import math as _ooa_math
_ooa_bad = [0]
_ooa_nan_steps = {NAN_STEPS} if scaler is not None else 1   # float16: GradScaler skips a nan step; else the weights took it
def _ooa_stop(why):
    print0(f"{NAN_MARK}: {{why}}. Stopped; nothing of it was saved (the last checkpoint is good).")
    raise SystemExit({NAN_EXIT})
'''
LOSS_GUARD = '''    _ooa_bad[0] = 0 if _ooa_math.isfinite(train_loss_f) else _ooa_bad[0] + 1
    if _ooa_bad[0] >= _ooa_nan_steps:
        _ooa_stop(f"the training loss was not a number (nan/inf) for {_ooa_bad[0]} step(s) in a row (step {step}): "
                  "the training went unstable")
'''
VAL_GUARD = '''        if not _ooa_math.isfinite(val_bpb):
            _ooa_stop(f"Validation bpb is not a number ({val_bpb}) at step {step}")
'''
SAVE_GUARD = '''        _ooa_nf = [n for n, p_ in orig_model.named_parameters() if not bool(torch.isfinite(p_).all())]
        if _ooa_nf:
            _ooa_stop(f"weights with nan/inf at step {step}: {', '.join(_ooa_nf[:6])}")
'''
PRUNE = '''        # OOA: keep only the newest OOA_KEEP_CHECKPOINTS checkpoints (disk)
        _ooa_keep = int(os.environ.get("OOA_KEEP_CHECKPOINTS", "0"))
        if master_process and _ooa_keep > 0:
            import re as _ooa_re
            _ooa_steps = sorted({int(m.group(1)) for f in os.listdir(checkpoint_dir)
                                 if (m := _ooa_re.match(r"(?:model|meta|optim)_(\\d+)", f))})
            for _s in _ooa_steps[:-_ooa_keep]:
                for f in os.listdir(checkpoint_dir):
                    if _ooa_re.match(rf"(?:model|meta|optim)_{_s:06d}[._]", f):
                        os.remove(os.path.join(checkpoint_dir, f))
'''
TOKENIZER_SHA = '''import hashlib as _ooa_hashlib  # OOA: the tokenizer this model trains with, recorded in every checkpoint
with open(os.path.join(get_base_dir(), "tokenizer", "tokenizer.pkl"), "rb") as _f:
    _ooa_tok_sha1 = _ooa_hashlib.sha1(_f.read()).hexdigest()
print0(f"Tokenizer sha1: {_ooa_tok_sha1}")
'''
META_FIX = (META_ANCHOR +
            '                "tokenizer_sha1": _ooa_tok_sha1, # OOA: the model pack must carry this tokenizer\n'
            '                "compute_dtype": str(COMPUTE_DTYPE).replace("torch.", ""), # OOA\n')
NO_COMPILE_PRELUDE = ("import torch as _torch_nc  # OOA: torch.compile switched off (no compiler on Windows)\n"
                      "_torch_nc.compile = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)\n")
ANCHORS = {"compile": COMPILE_LINE, "loop": LOOP_ANCHOR, "loss": LOSS_ANCHOR, "validation": VAL_ANCHOR,
           "save": SAVE_ANCHOR, "save end": SAVE_END, "meta": META_ANCHOR, "tokenizer": TOKENIZER_ANCHOR}


def missing_anchors(s):
    return [k for k, a in ANCHORS.items() if s.count(a) != 1]


def patch_base_train(s, compile_ok=True):
    """scripts/base_train.py (pinned nanochat) -> the patched text (see the module docstring). Raises SystemExit
    naming the missing anchors when the file is another version."""
    s = s.replace("\r\n", "\n")
    miss = missing_anchors(s)
    if miss:
        raise SystemExit(f"base_train.py does not match the pinned nanochat; missing: {miss}")
    s = s.replace(TOKENIZER_ANCHOR, TOKENIZER_ANCHOR + TOKENIZER_SHA, 1)
    s = s.replace(LOOP_ANCHOR, PRE_LOOP + LOOP_ANCHOR, 1)
    s = s.replace(LOSS_ANCHOR, LOSS_ANCHOR + LOSS_GUARD, 1)
    s = s.replace(VAL_ANCHOR, VAL_ANCHOR + VAL_GUARD, 1)
    s = s.replace(SAVE_ANCHOR, SAVE_ANCHOR.replace("        save_checkpoint(\n", "") + SAVE_GUARD
                  + "        save_checkpoint(\n", 1)
    i = s.index(SAVE_END, s.index("_ooa_nf = [n for n")) + len(SAVE_END)
    s = s[:i] + PRUNE + s[i:]
    s = s.replace(META_ANCHOR, META_FIX, 1)
    if not compile_ok:
        s = NO_COMPILE_PRELUDE + s.replace(COMPILE_LINE, "pass  # OOA: torch.compile switched off\n", 1)
    return s


STEP_RE = re.compile(r"step (\d+)/(\d+) \(([\d.]+)%\) \| loss: (\S+) \| lrm: \S+ \| dt: ([\d.]+)ms \| "
                     r"tok/sec: ([\d,]+)")
VAL_RE = re.compile(r"Step (\d+) \| Validation bpb: (\S+)")


# ------------------------------------------------------------------------------------------- checkpoints
def checkpoint_steps(d):
    """Steps with model_N.pt and meta_N.json in d, ascending."""
    out = []
    for f in Path(d).glob("model_*.pt") if Path(d).exists() else []:
        m = re.match(r"model_(\d+)\.pt$", f.name)
        if m and (Path(d) / f"meta_{m.group(1)}.json").exists():
            out.append(int(m.group(1)))
    return sorted(out)


def complete_checkpoint(d, step):
    """model + meta + optimizer state of a step are all there (base_train needs the optimizer to resume)."""
    d = Path(d)
    return all((d / f).exists() and (d / f).stat().st_size > 0
               for f in (f"model_{step:06d}.pt", f"meta_{step:06d}.json", f"optim_{step:06d}_rank0.pt"))


def drop_checkpoint(d, step):
    for f in Path(d).glob(f"*_{step:06d}*"):
        if re.match(rf"(?:model|meta|optim)_{step:06d}[._]", f.name):
            f.unlink()


# ------------------------------------------------------------------------------------------- tokenizer
def tokenizer_sha1(tok_dir):
    f = Path(tok_dir) / "tokenizer.pkl"
    return hashlib.sha1(f.read_bytes()).hexdigest() if f.exists() else None


SPECIAL = ["<|bos|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>",
           "<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>"]


def tokenizer_card(tok_dir, render_dirs=(), name="v6", log=print, extra=None):
    """Write tok_dir/tokenizer_card.json: what this tokenizer is (sha1 of tokenizer.pkl, vocab, the chat special
    tokens as the last ids) and what it measures on rendered conversations (held-out val rows when there are, else a
    sample of train rows): characters per token of the messages, characters of a JSONL line per token (what
    sft_setup.file_stats divides by to plan the SFT mixture) and true tokens / ooa_format.approx_tokens (the row-length
    estimate the renderer uses: under 1 means rendered rows fit with room to spare). Needs tiktoken."""
    import pickle
    import statistics
    import sys
    tok_dir = Path(tok_dir)
    with open(tok_dir / "tokenizer.pkl", "rb") as f:
        enc = pickle.load(f)
    st = enc._special_tokens
    n = enc.n_vocab
    ids = [st.get(s) for s in SPECIAL]
    card = {"name": name, "sha1": tokenizer_sha1(tok_dir), "vocab_size": n,
            "specials_ok": None not in ids and ids == list(range(n - len(SPECIAL), n)),
            "made": time.strftime("%Y-%m-%d %H:%M")}
    lines = []
    for d in render_dirs or ():
        d = Path(d)
        fs = [d / "val.jsonl"] if (d / "val.jsonl").exists() else sorted(d.glob("*_train.jsonl"))
        for f in fs:
            with open(f, encoding="utf-8") as fh:
                for i, l in enumerate(fh):
                    if l.strip() and (f.name == "val.jsonl" or i % 20 == 0):
                        lines.append(l)
            if len(lines) > 3000:
                break
    if lines:
        here = Path(__file__).resolve().parent             # training/pretrain (repo) or /content/ooa (Colab)
        sys.path[:0] = [str(here.parent), str(here)]
        try:
            import ooa_format as F
        except ImportError:
            F = None
        finally:
            del sys.path[:2]
        chars = toks = jchars = 0
        ratio = []
        for l in lines[:3000]:
            try:
                row = json.loads(l)
            except ValueError:
                continue
            msgs = [m for m in row.get("messages", []) if isinstance(m, dict)]
            t = sum(len(enc.encode_ordinary(m.get("content") or "")) for m in msgs) + 2 * len(msgs) + 1
            chars += sum(len(m.get("content") or "") for m in msgs)
            jchars += len(l)
            toks += t
            if F is not None:
                ratio.append(t / max(1, F.row_tokens(row)))
        card["measured_rows"] = len(lines[:3000])
        card["chars_per_token"] = round(chars / max(1, toks), 3)
        card["jsonl_chars_per_token"] = round(jchars / max(1, toks), 3)
        if ratio:
            ratio.sort()
            card["true_vs_approx_tokens"] = {"mean": round(statistics.mean(ratio), 3),
                                             "p99": round(ratio[int(0.99 * (len(ratio) - 1))], 3),
                                             "max": round(ratio[-1], 3)}
    card.update(extra or {})
    (tok_dir / "tokenizer_card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    log(f"tokenizer card: {json.dumps(card)}")
    return card


def read_card(tok_dir):
    """The tokenizer card of tok_dir when it describes the tokenizer.pkl next to it (sha1 matches), else None."""
    f = Path(tok_dir) / "tokenizer_card.json"
    try:
        card = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return card if card.get("sha1") and card["sha1"] == tokenizer_sha1(tok_dir) else None


def plan_chars_per_token(tok_dir, default=None):
    """sft_setup.file_stats' chars per token for a model's tokenizer: the card's jsonl_chars_per_token, else default
    (sft_setup.CHARS_PER_TOKEN, measured for the old web tokenizer)."""
    card = read_card(tok_dir) if tok_dir else None
    v = (card or {}).get("jsonl_chars_per_token")
    return float(v) if isinstance(v, (int, float)) and v > 0 else default


# ------------------------------------------------------------------------------------------- the model pack
def write_pack(work, step, pack, info=None, data_card=None, log=print):
    """The finished base model as a pack: pack/tokenizer (+ card), pack/base_checkpoints/d8/{model,meta,optim}_N,
    pack/VERSION, pack/DATA_CARD.md + data_card.json. A pack already there is moved to <pack>_before_<date> (never
    deleted); nothing else is touched (the current base and the installed model stay as they are)."""
    work, pack = Path(work), Path(pack)
    src = work / "base_checkpoints" / MODEL_TAG
    if not complete_checkpoint(src, step):
        raise SystemExit(f"no complete checkpoint of step {step} in {src}")
    if pack.exists():
        old = pack.with_name(pack.name + "_before_" + time.strftime("%Y%m%d_%H%M%S"))
        shutil.move(str(pack), old)
        log(f"the earlier {pack.name} is kept in {old}")
    tmp = pack.with_name(pack.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    d8 = tmp / "base_checkpoints" / MODEL_TAG
    d8.mkdir(parents=True)
    for f in (f"model_{step:06d}.pt", f"meta_{step:06d}.json", f"optim_{step:06d}_rank0.pt"):
        shutil.copy2(src / f, d8 / f)
    shutil.copytree(work / "tokenizer", tmp / "tokenizer")
    meta = json.loads((d8 / f"meta_{step:06d}.json").read_text(encoding="utf-8"))
    sha = tokenizer_sha1(tmp / "tokenizer")
    if meta.get("tokenizer_sha1") not in (None, sha):
        raise SystemExit("the checkpoint was trained with another tokenizer than the one in " + str(work / "tokenizer"))
    meta["tokenizer_sha1"] = sha
    meta["ooa_base"] = dict(info or {}, packed=time.strftime("%Y-%m-%d %H:%M"))
    (d8 / f"meta_{step:06d}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    ver = f"{VERSION_PREFIX}-{time.strftime('%Y%m%d')}-step{step}"
    (tmp / "VERSION").write_text(ver + "\nclean base model: FineWeb-Edu (ODC-By 1.0) + the owner's domain text; see "
                                 "DATA_CARD.md\n", encoding="utf-8")
    dc = Path(data_card) if data_card else None
    for name in ("DATA_CARD.md", "data_card.json"):
        if dc and (dc / name).exists():
            shutil.copy2(dc / name, tmp / name)
    os.replace(tmp, pack)
    log(f"base model pack: {pack} ({ver})")
    return pack
