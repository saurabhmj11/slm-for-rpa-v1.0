"""Train the OOA model on this PC with the Colab v4 recipe (cell 7), in saved chunks that resume after any stop.

    python train_pc.py --mode full  --data DIR   the Colab run: 12000 steps at 4096 from the pretrained BASE model
                                                 (NVIDIA GPU: a few hours; on a CPU it would take weeks)
    ... --base v6                                the same from the clean base model (train_data/base_v6, notebook
                                                 MODE "pretrain"); its own state (local_run/full_v6) and tokenizer
    python train_pc.py --mode topup --data DIR   continue the newest SFT model for a few hundred steps (CPU: a night)
    python train_pc.py --mode refine --data DIR  NVIDIA GPU: continue the finished Colab model (v5) for 1500 steps on
                                                 the newest data (v5, session) and the weak areas, with replay of the
                                                 rest (sft_setup.REFINE_SHARES); saved every 250 steps
    python train_pc.py --mode colab --data DIR   NVIDIA GPU: FINISH the Colab SFT run in progress here (cell 7's
                                                 chunked run, when Colab misbehaves or runs out of hours): data_pc.py
                                                 --mode colab downloads Drive sft_run/state.json, its last saved chunk
                                                 and Drive's sft_render; the remaining chunks train with the run's own
                                                 plan (steps, chunks, learning rates, data mixture, seed, data cursor)
    ... --check                                  only show what would be used
    ... --measure 3                              train 3 steps, print seconds per step, change nothing
    ... --restart                                throw away a run in progress and start over

Same code as Colab: nanochat's scripts/chat_sft.py written fresh from the pinned commit, patched by
training/sft_setup.patch_chat_sft (the 4096 attention-window fix, the data mixture planned for the step budget, rows
longer than the window skipped, "Training data actually read" at the end). --data is the rendered data (sft_render:
the plan-first format, compact answers, gold dropout, beginner rows), prepared by data_pc.py.

Chunks: nanochat saves only at the end of a run, so the run is split into --chunks pieces. Chunk k starts from the
model and optimizer chunk k-1 saved and covers the part [k/K, (k+1)/K] of the run's learning-rate schedule (constant,
then the linear warmdown over the second half, exactly the shape of one long run); the Muon momentum ramp (0.85 to
0.95 over the first 300 steps) also counts the run's steps, not the chunk's (OOA_STEP_OFFSET). Every chunk plans the same data
mixture for the whole run with the run's one seed (a full run seed 1 like Colab, every top-up round a new seed) and
continues reading that list where the chunk before stopped (OOA_DATA_START), so the chunks read what one run reads and
every file is seen the planned number of times. (A run started before this continues as it began: seed + k per chunk.)
With --chunks 1 (default on a GPU) the run is Colab cell 7.
The chunk logic is training/sft_chunks.py, shared with Colab cell 7 (which runs the full mode in SFT_CHUNKS pieces
saved to Drive, so a Colab session that ends loses at most one piece).
The state is kept in RPA/local_run/<mode>/ (state.json, the last two chunks, train.log): full, topup and refine runs
never share a state. The result is the newest chunk; eval_pc.py scores it and installs it only if it is better.

Broken weights (nan / inf) never go further: the start model is checked before anything runs (also --check), and a
broken one is refused with what to use instead (--start-from pc / drive; notebook cell 1 REFINE_FROM), never swapped
silently. The optimizer state saved with the start model is loaded only when it is finite (optimizer_choice). While a
chunk trains, a step whose loss or gradients are not finite is skipped (sft_setup's step guard; float16 keeps its
GradScaler scale at 1 or above) and its output is watched (follow): such steps for NAN_STEPS steps in a row
(chat_sft's guard, sft_setup) or a validation bpb that is not a number stops the chunk; nothing of it is saved, the state notes it, and the
next try of that chunk starts with a fresh optimizer state when the loaded one may be the cause.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RPA_DEFAULT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
import sft_setup as T                                          # noqa: E402  (one source of truth with Colab cell 7)
sys.path.insert(1, str(HERE.parent / "pretrain"))
sys.path.append(str(HERE.parent.parent / "assistant"))
import base_setup as BS                                        # noqa: E402  (the clean base model, v6)
from model_backend import tokenizer_problem                    # noqa: E402  (a model and the tokenizer it needs)
import sft_chunks as SC                                        # noqa: E402  (the chunk logic, shared with Colab cell 7)
from sft_chunks import (NANOCHAT_COMMIT, NAN_STEPS, BASE_LRS, MOMENTUM_OLD, MOMENTUM_NEW, STEP_RE, VAL_RE,  # noqa: E402,F401
                        newest, copy_model, lr_for_chunk, lr_for_span, base_lrs_of, lr_flags, model_health,
                        optimizer_choice, nan_value, pinned_chat_sft, no_compile, data_sig, windows_kill_with_parent,
                        run_chunk, follow, chunk_bounds, chunk_command, prepare_work, chunk_data, data_after_chunk,
                        DATA_ORDER)

TOKENS_PER_STEP = 32768          # Colab: SFT_SEQ 4096 x device batch 2 x 4 micro-batches; every mode keeps it
COLAB_RUN = ("train_data", "colab_run")   # MODE "colab": Drive's SFT run in progress, as data_pc.py downloads it
FULL_STEPS = 12000               # Colab SFT_ITERS 48000 micro-batches = 12000 optimizer steps
FULL_LR = 0.8                    # Colab: nanochat's default --init-lr-frac
TOPUP_LR = 0.3                   # a top-up only adjusts an already trained model
TOPUP_STEPS = 300                # measured on a 4-core CPU at the 4096 window, no torch.compile: 67 s per step at low
                                 # priority with 3 threads (55 s with all 4) -> about 5.6 h; slower PCs up to about 9 h
SMOLTALK_ROWS = 3000             # Colab cell 1 SMOLTALK_ROWS (downloads SmolTalk once); top-ups use none
REFINE_STEPS = 1500              # GPU refine of the finished Colab model: 1/8 of its 12000 steps. The Colab T4 trains
                                 # 12000 steps in about 3.5-4.5 h (1.1-1.35 s per step with torch.compile), so 1500 steps
                                 # take about 30-35 min there; on Windows without torch.compile, or an 8 GB GPU (1 row per
                                 # pass), about 1.5-2.5 s per step -> 40-65 min, plus 1-2 min of loading per chunk
REFINE_CHUNK = 250               # saved every 250 steps (a PC can stop: run again and it continues)
REFINE_LR = 0.35                 # the Colab run peaked at 0.8 and ended at 0 (warmdown); restarting a trained model at
                                 # 0.8 would shake what it learned. A top-up of 300 steps uses 0.3; a refine is 5x longer
                                 # with its own warmdown over the second half, so a little more (0.35, 44% of the peak)
                                 # lets the new rows (v5, session) settle while the replay holds the rest


START_FROM = ("auto", "drive", "pc")     # notebook cell 1 REFINE_FROM (topup / refine)


BASES = ("current", "v6")      # MODE "full": the base model the SFT starts from (notebook cell 1 BASE)


def base_root(rpa, base="current"):
    """The base model pack: "current" the Colab base (train_data/base, from Drive nanochat_base), "v6" the clean
    base (train_data/base_v6: pretrain_pc.py's pack, or Drive nanochat_base_v6 downloaded by data_pc.py)."""
    return Path(rpa) / "train_data" / ("base_v6" if base == "v6" else "base")


def run_name(mode, base="current"):
    """The run's state folder under local_run: a full run on the v6 base has its own (local_run/full_v6), so
    switching BASE never throws a run in progress away."""
    return "full_v6" if mode == "full" and base == "v6" else mode


def pick_tokenizer(rpa, src_root, meta):
    """(tokenizer folder, problem) for a start model: its own pack's tokenizer; the installed model's only for an
    older model whose pack has none, and never one the model was not trained with (tokenizer_problem)."""
    tried = []
    for p in (Path(src_root) / "tokenizer", Path(rpa) / "nanochat_base" / "tokenizer"):
        if (p / "tokenizer.pkl").exists() and (p / "token_bytes.pt").exists():
            why = tokenizer_problem(meta or {}, p)
            if why is None:
                return p, None
            tried.append(why)
        if (meta or {}).get("tokenizer_sha1") and p == Path(src_root) / "tokenizer":
            break                                  # a recorded tokenizer must come with the model
    return None, ("; ".join(tried) if tried else f"no tokenizer (tokenizer.pkl + token_bytes.pt) in "
                  f"{Path(src_root) / 'tokenizer'}")


def base_lrs(rpa, root=None):
    """The learning rates Colab's SFT run used: chat_sft inherits them from the base model's meta (user_config).
    Passed explicitly to every chunk, because a saved SFT checkpoint stores them as None (chat_sft saves its
    user_config before it fills them in) and chat_sft then falls back to its own defaults, which halve the output
    layer's rate (unembedding 0.004 instead of base_train's 0.008). root: the base pack (default train_data/base)."""
    return base_lrs_of(Path(root or Path(rpa) / "train_data" / "base") / "base_checkpoints" / "d8")


def drive_id(root):
    f = Path(root) / ".drive_id"
    return f.read_text(encoding="utf-8").strip() if f.exists() else None


def same_drive_model(a, b):
    """Two Drive model identities ("<path>:<file id>") name the same model: Drive keeps a file's id when it is moved
    (Colab cell 7 moves chatsft_checkpoints to chatsft_checkpoints_before_<date> when a new run starts), so only the
    id counts, not the folder."""
    return bool(a and b) and a.rsplit(":", 1)[-1] == b.rsplit(":", 1)[-1]


def local_origin(rpa):
    """The Drive model the PC's installed model grew from (written into its meta when it was installed), or None."""
    d8 = Path(rpa) / "nanochat_base" / "chatsft_checkpoints" / "d8"
    s = newest(d8)
    if s is None:
        return None
    m = json.loads((d8 / f"meta_{s:06d}.json").read_text(encoding="utf-8"))
    return (m.get("ooa_local") or {}).get("drive_origin")


def topup_start(rpa):
    """The newest SFT model: Drive's newest (train_data/drive_model) unless the PC's installed model already grew
    from that same Drive model by an earlier top-up (then the PC's one is newer); no Drive model: the PC's one."""
    rpa = Path(rpa)
    drive, here = rpa / "train_data" / "drive_model", rpa / "nanochat_base"
    if newest(drive / "chatsft_checkpoints" / "d8") is None:
        return here
    if newest(here / "chatsft_checkpoints" / "d8") is not None and same_drive_model(local_origin(rpa), drive_id(drive)):
        return here
    return drive


def start_label(rpa, root):
    """Where a top-up / refine / RL run starts, in words."""
    rpa, root = Path(rpa), Path(root)
    if root == rpa / "train_data" / "drive_model":
        return "Drive's newest trained model (e.g. the Colab v5 run)"
    if root == rpa / "nanochat_base":
        return "this PC's installed model (it is Drive's newest or already grew from it)"
    return str(root)


def start_root(rpa, start_from="auto"):
    """The model a top-up / refine continues: "auto" the newest (topup_start), "drive" Drive's newest model as
    downloaded (train_data/drive_model), "pc" this PC's installed model (nanochat_base)."""
    rpa = Path(rpa)
    if start_from == "drive":
        return rpa / "train_data" / "drive_model"
    if start_from == "pc":
        return rpa / "nanochat_base"
    return topup_start(rpa)


def model_name(rpa, root):
    rpa, root = Path(rpa), Path(root)
    if root == rpa / "train_data" / "drive_model":
        return "Drive's newest model (the Colab run)"
    if root == rpa / "nanochat_base":
        return "this PC's installed model"
    return f"the model in {root}"


def broken_message(rpa, root, step, bad, mode="refine"):
    """The plain refusal for a start model with nan / inf weights, with the setting to use instead (never a silent
    switch to another model)."""
    rpa, root = Path(rpa), Path(root)
    kinds = sorted({b.rsplit("(", 1)[-1].rstrip(")") for b in bad}) or ["nan"]
    msg = (f"{model_name(rpa, root)} (step {step}) has broken weights ({'/'.join(kinds)}) in: {T.short_list(bad)}; "
           "it cannot be trained or installed.")
    if root == rpa / "train_data" / "drive_model":
        msg += (" The Colab run went unstable: send its cell 7 loss lines (the 'step ... | loss:' and 'Validation "
                "bpb' lines, mainly the first ones that are nan).")
    others = {"pc": rpa / "nanochat_base", "drive": rpa / "train_data" / "drive_model"}
    for key, other in others.items():
        if other == root:
            continue
        d8 = other / "chatsft_checkpoints" / "d8"
        ostep = newest(d8)
        if ostep is None:
            continue
        status, _ = model_health(d8, ostep)
        if status == "ok":
            msg += (f" {model_name(rpa, other)[0].upper() + model_name(rpa, other)[1:]} (step {ostep}) is fine: to "
                    f"{mode} it instead, set REFINE_FROM = \"{key}\" in notebook cell 1 (--start-from {key})"
                    " and run cells 1-5 again.")
            return msg
    msg += (" No other trained model on this PC is usable: train again on Colab (cell 7) from a good model, or MODE "
            "\"full\" on this GPU.")
    return msg


def colab_base(st):
    """The base a Colab run trains on: "v6" when its base folder is nanochat_base_v6, else "current"."""
    return "v6" if str((st.get("base") or {}).get("dir", "")).replace("\\", "/").rstrip("/").endswith("_v6") \
        else "current"


def colab_source(rpa, start=None):
    """The Colab SFT run (cell 7) data_pc.py --mode colab downloaded to train_data/colab_run (or start): -> (folder, its
    state.json, the d8 folder of the model the next chunk starts from, its step). The model is the last chunk Colab
    saved (colab_run/chunks/chunk_<done-1>); a run with no chunk saved yet starts from its base model (train_data/base,
    train_data/base_v6 for a v6 run). ({} , None, None) parts when nothing was downloaded."""
    root = Path(start).expanduser().resolve() if start else Path(rpa).joinpath(*COLAB_RUN)
    try:
        st = json.loads((root / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return root, {}, None, None
    done = int(st.get("done", 0) or 0)
    if done:
        d8 = root / "chunks" / f"chunk_{done - 1:03d}"
    else:
        d8 = base_root(rpa, colab_base(st)) / "base_checkpoints" / ((st.get("base") or {}).get("tag") or "d8")
    return root, st, d8, newest(d8)


def colab_problems(st, data, tok, tps, seq, dbs):
    """Why the downloaded Colab run cannot be continued here (empty: it can)."""
    out = []
    if not st:
        return ["no Colab SFT run here (train_data/colab_run/state.json): run the data cell with MODE \"colab\" "
                "(data_pc.py --mode colab) first"]
    if st.get("finished"):
        out.append("the Colab SFT run is finished (all its chunks are done): nothing to continue; MODE \"install\" "
                   "tests and installs its model")
    if st.get("data_sig") != data_sig(data):
        out.append(f"the training data here ({data}: render {data_sig(data)}) is not the data the Colab run trains on "
                   f"(render {st.get('data_sig')}): run the data cell again with REFRESH_DATA = True so it downloads "
                   "Drive's sft_render, or let Colab finish the run")
    want = (st.get("base") or {}).get("tokenizer_sha1")
    if want and tok and (Path(tok) / "tokenizer.pkl").exists() and SC.file_sha1(Path(tok) / "tokenizer.pkl") != want:
        out.append(f"the tokenizer in {tok} is not the one the Colab run trains with: run the data cell again")
    if tps % (seq * dbs):
        out.append(f"device batch {dbs} x window {seq} does not divide the run's {tps} tokens per step")
    gone = sorted(n for n in (st.get("views") or {}) if not (Path(data) / n).exists())
    if gone:
        out.append(f"files the Colab run trains on are missing in {data}: {T.short_list(gone)} (run the data cell "
                   "again with REFRESH_DATA = True)")
    return out


def write_script(nano, data, views, smoltalk, seed, compile_ok):
    """nanochat/scripts/chat_sft_ooa.py for one chunk (sft_chunks.write_script, the same as Colab cell 7)."""
    return SC.write_script(nano, data, views, smoltalk, seed, compile_ok, HERE.parent)


def device_info():
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return "cuda", torch.cuda.get_device_name(0), p.total_memory / 2**30, torch.cuda.get_device_capability()
    return "cpu", f"{os.cpu_count()} CPU threads", 0.0, None


LOW_RAM_GB = 10                  # notebook cell 2: a CPU PC below this trains at 2048


def default_seq(gpu):
    """The window when --seq is not given (the nightly task): the notebook's rule, 4096 unless a CPU PC has less
    than LOW_RAM_GB of RAM (one 4096-token row per pass needs about 5-6 GB on a CPU)."""
    if gpu:
        return 4096
    try:
        import psutil
        ram = psutil.virtual_memory().total / 2**30
    except Exception:
        return 4096
    return 4096 if ram >= LOW_RAM_GB else 2048


def gpu_batch(mem_gb, seq):
    """Rows per GPU pass: Colab's T4 (15 GB) runs 2 x 4096. Same tokens per step either way."""
    b = 8 if mem_gb >= 60 else 4 if mem_gb >= 30 else 2 if mem_gb >= 14 else 1
    return max(1, min(b, TOKENS_PER_STEP // seq))


def kill_leftover(run_dir):
    """A training process left running by an earlier stop (Windows does not end child processes with their parent)."""
    pid_f = Path(run_dir) / "child.pid"
    if not pid_f.exists():
        return
    try:
        import psutil
        pr = psutil.Process(int(pid_f.read_text(encoding="utf-8")))
        if "chat_sft_ooa" in " ".join(pr.cmdline()):
            print("stopping a training process left running by an earlier stop, pid", pr.pid)
            pr.kill()
    except Exception:
        pass
    pid_f.unlink(missing_ok=True)


def nan_stop(state, state_f, k, why, loaded, log):
    """A chunk stopped because the training went nan: note it in the state and stop with what to do and send."""
    state.setdefault("nan_stops", []).append({"chunk": k, "why": why, "optimizer": "loaded" if loaded else "fresh",
                                              "time": time.strftime("%Y-%m-%d %H:%M")})
    state_f.write_text(json.dumps(state, indent=1), encoding="utf-8")
    fresh_too = any(e["chunk"] == k and e["optimizer"] == "fresh" for e in state["nan_stops"])
    if loaded:
        nxt = ("Run cell 5 again (train_pc.py again): this chunk is retried with a fresh optimizer state, the likely "
               "cause. If it goes nan again, send")
    elif fresh_too and len([e for e in state["nan_stops"] if e["chunk"] == k]) > 1:
        nxt = "It went nan with a fresh optimizer state too, so running it again will not help. Send"
    else:
        nxt = "Send"
    raise SystemExit(
        f"\nSTOPPED: chunk {k + 1} went unstable: {why.rstrip(' .')}. Nothing of this chunk was saved (the chunks "
        f"before it are kept, and the installed model is not touched). {nxt} the lines above (the 'step ... | loss' "
        f"lines and the 'Validation bpb' lines) and the end of {log}.")


def main(argv=None):
    try:
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rpa", default=str(RPA_DEFAULT), help='the "rpa model" folder')
    ap.add_argument("--mode", choices=["full", "topup", "refine", "colab"], required=True)
    ap.add_argument("--data", required=True, help="the rendered training data (sft_render)")
    ap.add_argument("--base", choices=BASES, default="current",
                    help="full: the base model: current (train_data/base, the Colab base) or v6 (train_data/base_v6, "
                         "the clean base; notebook cell 1 BASE)")
    ap.add_argument("--start", default=None, help="topup: a model folder with chatsft_checkpoints/d8 (+ tokenizer) to "
                                                  "continue; full: a folder with base_checkpoints/d8")
    ap.add_argument("--start-from", choices=START_FROM, default="auto",
                    help="topup / refine: auto = the newest trained model; drive = Drive's newest (the Colab run); "
                         "pc = this PC's installed model (notebook cell 1 REFINE_FROM)")
    ap.add_argument("--optimizer", choices=["auto", "load", "fresh"], default="auto",
                    help="the optimizer state saved with the model a chunk starts from: auto = load it when it is "
                         "finite (fresh after a nan try), load, fresh")
    ap.add_argument("--steps", type=int, default=None, help=f"optimizer steps of {TOKENS_PER_STEP:,} tokens (full "
                                                            f"{FULL_STEPS}, topup {TOPUP_STEPS}, refine {REFINE_STEPS})")
    ap.add_argument("--chunks", type=int, default=0, help=f"saved pieces (0: refine steps/{REFINE_CHUNK}; else 1 on a "
                                                          "GPU, steps/50 on a CPU)")
    ap.add_argument("--seq", type=int, default=0, choices=[0, 2048, 4096],
                    help="window (0: 4096 as Colab; 2048 on a CPU PC with less than 10 GB RAM, as notebook cell 2)")
    ap.add_argument("--device-batch", type=int, default=0, help="rows per pass (0: from GPU memory; 1 on a CPU)")
    ap.add_argument("--lr", type=float, default=None, help=f"--init-lr-frac (full {FULL_LR}, topup {TOPUP_LR}, "
                                                           f"refine {REFINE_LR})")
    ap.add_argument("--smoltalk", type=int, default=None, help=f"general chat rows (full {SMOLTALK_ROWS}, full on the "
                                                               "v6 base 0, topup 0)")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0: all but one at low priority)")
    ap.add_argument("--full-speed", action="store_true", help="CPU: all threads, normal priority")
    ap.add_argument("--no-compile", action="store_true", default=(os.name == "nt"),
                    help="run without torch.compile (automatic on Windows)")
    ap.add_argument("--compile", dest="no_compile", action="store_false")
    ap.add_argument("--allow-cpu-full", "--allow-cpu", dest="allow_cpu", action="store_true",
                    help="full or refine on a CPU (days to weeks; only for testing)")
    ap.add_argument("--measure", type=int, default=0, help="train this many steps, print seconds per step, keep nothing")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)

    os.environ.setdefault("PYTHONUTF8", "1")
    rpa = Path(a.rpa).expanduser().resolve()
    data = Path(a.data).expanduser().resolve()
    nano = rpa / "nanochat"
    full, refine, colab = a.mode == "full", a.mode == "refine", a.mode == "colab"
    dev, dev_name, mem, cap = device_info()
    gpu = dev == "cuda"
    seq_given = bool(a.seq)
    a.seq = a.seq or default_seq(gpu)
    a.steps = a.steps or (FULL_STEPS if full else REFINE_STEPS if refine else TOPUP_STEPS)
    tps = TOKENS_PER_STEP
    if colab:                                   # the Colab run's own plan; nothing of it is chosen here
        c_root, c_st, c_d8, c_step = colab_source(rpa, a.start)
        if c_st:
            a.steps, a.chunks, a.seq = int(c_st["steps"]), int(c_st["chunks"]), int(c_st["seq"])
            tps, a.lr, a.smoltalk = int(c_st["tokens_per_step"]), float(c_st["lr"]), int(c_st.get("smoltalk", 0))
            if a.measure:
                a.steps, a.chunks = a.measure, 1
    if a.measure:
        a.steps, a.chunks = a.measure, 1
    a.chunks = a.chunks or (max(1, round(a.steps / REFINE_CHUNK)) if refine else 1 if gpu else max(1, round(a.steps / 50)))
    a.chunks = max(1, min(a.chunks, a.steps))
    a.lr = a.lr if a.lr is not None else (FULL_LR if full else REFINE_LR if refine else TOPUP_LR)
    # SmolTalk (general chat rows) only on the current base: the clean base (v6) stays clean (SmolTalk is largely
    # another model's output, under that model licence's naming / notice terms; training/pretrain/DATA_CARD.md)
    a.smoltalk = a.smoltalk if a.smoltalk is not None else (
        SMOLTALK_ROWS if full and not a.measure and a.base != "v6" else 0)
    dbs = a.device_batch or (gpu_batch(mem, a.seq) if gpu else 1)
    if tps % (a.seq * dbs):
        raise SystemExit(f"--device-batch {dbs} does not divide {tps} tokens per step at window {a.seq}")
    accum = tps // (a.seq * dbs)

    run_dir = rpa / "local_run" / ("measure" if a.measure else run_name(a.mode, a.base))
    state_f = run_dir / "state.json"
    if (a.restart or a.measure) and run_dir.exists() and not a.check:
        shutil.rmtree(run_dir, ignore_errors=True)
    state = json.loads(state_f.read_text(encoding="utf-8")) if state_f.exists() else {}
    if colab and state and not a.measure:
        if a.restart:
            state = {}
        elif not c_st or state.get("colab_started") != c_st.get("started"):
            if c_st:
                print(f"note: the Colab run downloaded now (started {c_st.get('started')}) is not the one this PC "
                      f"continued (started {state.get('colab_started')}): it is continued instead")
            state = {}
        elif state.get("finished"):
            raise SystemExit(f"this Colab run (started {c_st.get('started')}) was already finished on this PC and "
                             f"tested ({state.get('finished')}): nothing to do. --restart (RESTART_TRAINING = True) "
                             "trains its remaining chunks here again.")
        elif int(c_st.get("done", 0) or 0) > int(state.get("done", 0) or 0):
            print(f"note: Colab went further ({c_st.get('done')} chunks) than this PC ({state.get('done')}): this PC "
                  "continues from Colab's newest chunk")
            state = {}
        if not state and not a.check:
            shutil.rmtree(run_dir, ignore_errors=True)
    if state.get("finished"):
        state = {}                                    # the last run is complete (eval_pc.py took it): a new one
    if state and a.mode in ("topup", "refine") and not a.start and a.start_from != "auto" and \
            Path(state["start"]) != start_root(rpa, a.start_from):
        print(f"note: the run in progress started from {model_name(rpa, state['start'])}; --start-from "
              f"{a.start_from} (REFINE_FROM) asks for {model_name(rpa, start_root(rpa, a.start_from))}: a new run "
              "starts from it")
        state = {}
        if not a.check:
            shutil.rmtree(run_dir, ignore_errors=True)
    if state and state.get("done"):                 # a chunk saved by an older version could be broken (nan)
        _last = run_dir / "chunks" / f"chunk_{state['done'] - 1:03d}"
        _st, _bad = model_health(_last, newest(_last))
        if _st == "broken":
            print(f"note: the run in progress is broken: its saved chunk {state['done']} has nan/inf in "
                  f"{T.short_list(_bad, 3)} (saved before the NaN check existed). It is thrown away and the run "
                  "starts over.")
            state = {}
            if not a.check:
                shutil.rmtree(run_dir, ignore_errors=True)

    # where the run starts: the base model (full), the newest SFT model (topup / refine), the Colab run's last chunk
    if colab:
        src_root = c_root
        src_d8, src_step = (Path(state["start_d8"]), state["start_step"]) if state else (c_d8, c_step)
        src_d8 = src_d8 or c_root / "chunks"
    elif state:
        src_root, src_step = Path(state["start"]), state["start_step"]
    else:
        src_root = Path(a.start).expanduser().resolve() if a.start else (
            base_root(rpa, a.base) if full else start_root(rpa, a.start_from))
        src_step = newest(src_root / ("base_checkpoints" if full else "chatsft_checkpoints") / "d8")
    if not colab:
        src_d8 = src_root / ("base_checkpoints" if full else "chatsft_checkpoints") / "d8"
    try:
        src_meta = json.loads((src_d8 / f"meta_{src_step:06d}.json").read_text(encoding="utf-8")) if src_step is not None else {}
    except (OSError, ValueError):
        src_meta = {}
    tok, tok_why = pick_tokenizer(rpa, src_root, src_meta)

    problems = []
    if colab:
        problems += colab_problems(c_st, data, tok, tps, a.seq, dbs)
    if src_step is None and not (colab and not c_st):
        problems.append(f"no model_<N>.pt + meta_<N>.json in {src_d8}"
                        + ((" (the clean base: notebook MODE \"pretrain\" on this PC, or Drive nanochat_base_v6 from "
                            "Colab PROFILE \"v6\")" if a.base == "v6" else
                            " (the pretrained base model from Drive nanochat_base/base_checkpoints/d8)") if full else ""))
    if tok is None:
        problems.append(tok_why)
    for f in (data / "val.jsonl", data / "catalog.jsonl"):
        if not f.exists():
            problems.append(f"missing {f}")
    if not (nano / "scripts" / "chat_sft.py").exists() or not (nano / "nanochat" / "gpt.py").exists():
        problems.append(f"nanochat is missing in {nano}: run notebook cell 2 again (it downloads it; Git is not "
                        f"needed), or: python \"{HERE / 'setup_pc.py'}\" --rpa \"{rpa}\"")
    if not (data / "examples.jsonl").exists():
        problems.append(f"{data} is not rendered data (no examples.jsonl): run the data cell (data_pc.py) first")
    if full and not gpu and not (a.allow_cpu or a.measure):
        problems.append("full training needs an NVIDIA GPU (12000 steps; a CPU would take weeks): use --mode topup")
    if colab and not gpu and not (a.allow_cpu or a.measure):
        problems.append("continuing the Colab run needs an NVIDIA GPU (thousands of steps; a CPU would take weeks)")
    if refine and not gpu and not (a.allow_cpu or a.measure):
        problems.append(f"refine is sized for an NVIDIA GPU ({a.steps} steps; about a day per 1500 steps on a CPU): "
                        "use --mode topup on this PC")
    health = model_health(src_d8, src_step) if src_step is not None else ("unchecked", [])
    if health[0] == "broken":
        problems.append(f"the Colab run's model {src_d8} (step {src_step}) has broken weights (nan/inf) in: "
                        f"{T.short_list(health[1])}: Colab saved no good chunk to continue from" if colab else
                        broken_message(rpa, src_root, src_step, health[1], a.mode) if not full else
                        f"the base model in {src_d8} (step {src_step}) has broken weights (nan/inf) in: "
                        f"{T.short_list(health[1])}; download it again (notebook cell 3 with REFRESH_DATA = True)")
    if gpu and cap and cap < (7, 5) and not a.no_compile:
        problems.append(f"GPU {dev_name} is too old for torch.compile: add --no-compile")
    if state and not seq_given:
        a.seq = state["seq"]                                  # no --seq (the nightly task): the run's own window
    if state and (state.get("steps"), state.get("chunks"), state.get("seq")) != (a.steps, a.chunks, a.seq) \
            and not (a.restart or a.check):
        print(f"note: continuing the run in progress ({state['steps']} steps in {state['chunks']} chunks at "
              f"{state['seq']}); --restart starts a new one with the new settings")
        a.steps, a.chunks, a.seq = state["steps"], state["chunks"], state["seq"]
    if state:                                                 # a run in progress keeps its own settings
        a.lr, dbs = state["lr"], state["device_batch"]
        a.smoltalk = state.get("smoltalk", a.smoltalk)
        tps = state.get("tokens_per_step", tps)
        accum = tps // (a.seq * dbs)
    if state and state.get("data_sig") != data_sig(data) and not colab:
        print("note: the training data changed since this run started; the run continues on the new data")
    try:
        _mt = json.loads((data / "render_report.json").read_text(encoding="utf-8"))["settings"]["max_tokens"]
    except Exception:
        _mt = None
    if _mt and _mt > a.seq - 28:
        print(f"note: the data was rendered for a {_mt + 28}-token window but this run trains at {a.seq}: longer rows "
              "are skipped (the notebook's cell 3 renders it for this window)")

    cpt = BS.plan_chars_per_token(tok) if tok else None          # the tokenizer's own characters per token (v6 card)
    if colab:                                                   # the Colab run's own mixture, never planned again
        views = dict(c_st.get("views") or {})
        stats = T.file_stats(str(data), chars_per_token=cpt or T.CHARS_PER_TOKEN) if data.is_dir() else {}
        bud = T.budget(a.steps, tps) - int(a.smoltalk * 500)
        table = T.mixture_table({n: stats[n] for n in views if n in stats},
                                {n: v for n, v in views.items() if n in stats}, bud) if stats else []
    else:
        views, stats, bud, table = T.plan_for_local(str(data), a.mode, a.steps, tps, smoltalk_rows=a.smoltalk,
                                                    chars_per_token=cpt) \
            if data.exists() and (data / "val.jsonl").exists() else ({}, {}, 0, [])
    done = state.get("done", 0) if state or not colab else int(c_st.get("done", 0) or 0)
    if a.measure:
        done = 0
    bounds = chunk_bounds(a.steps, a.chunks)                    # chunk k trains steps bounds[k]..bounds[k+1]
    chunk = max(bounds[k + 1] - bounds[k] for k in range(a.chunks))
    print("assistant folder :", rpa)
    print("device           :", dev_name, f"({mem:.0f} GB)" if gpu else "", "| torch.compile", "off" if a.no_compile else "on")
    print("mode             :", a.mode, "| from", src_d8, f"step {src_step}")
    if colab and c_st:
        print("Colab run        :", f"started {c_st.get('started')} on Colab, {c_st.get('done', 0)} of "
              f"{c_st.get('chunks')} chunks done there ({c_st.get('base', {}).get('tag')} base "
              f"{colab_base(c_st)}); this PC trains chunks {int(state.get('done', c_st.get('done', 0)) or 0) + 1}-"
              f"{c_st.get('chunks')} with the run's own plan (seed {c_st.get('seed')}, data cursor "
              f"{state.get('data_cursor', c_st.get('data_cursor'))})")
    if not full and not colab:
        print("start model      :", start_label(rpa, src_root) if a.start_from == "auto" or a.start else
              f"{model_name(rpa, src_root)} (--start-from {a.start_from}: notebook cell 1 REFINE_FROM)")
    print("weights          :", {"ok": "all finite (checked)", "broken": "BROKEN (nan/inf), see below"}.get(
        health[0], "not checked: " + "; ".join(health[1])))
    _k = state.get("done", 0)                              # the next chunk: from the start model or the last chunk
    _fd = src_d8 if _k == 0 or (colab and not state) else run_dir / "chunks" / f"chunk_{_k - 1:03d}"
    _fs = src_step if _k == 0 or (colab and not state) else newest(_fd)
    if colab and not state:
        _k = int(c_st.get("done", 0) or 0)
    if _fs is not None and health[0] != "broken":
        print("optimizer        :", optimizer_choice(_fd, _fs, a.optimizer, any(
            e["chunk"] == _k and e["optimizer"] == "loaded" for e in state.get("nan_stops", [])))[1])
    print("training data    :", data, f"(render {data_sig(data)})")
    print(f"plan             : {a.steps} steps x {tps:,} tokens at window {a.seq} (device batch {dbs}, "
          f"{accum} micro-batches per step) in {a.chunks} chunk(s) of up to {chunk}; {done} done | lr {a.lr} | "
          f"SmolTalk rows {a.smoltalk}")
    if full:
        print("base model       :", f"{a.base} ({src_root})", "| tokenizer", tok,
              f"({cpt} characters per token)" if cpt else "")
    print("base rates       :", " ".join(lr_flags(state.get("base_lrs") or (c_st.get("base_lrs") if colab else None)
                                              or base_lrs(rpa, src_root if full else None))),
          "(Colab's, times the lr above)")
    if table:
        print(f"data mixture     : about {bud / 1e6:.1f}M tokens of conversations; views per row:")
        print("\n".join("   " + l for l in table))
    for p_ in problems:
        print("  !", p_)
    if problems:
        raise SystemExit("\nNot ready: fix the lines marked ! above.")
    if a.check:
        print("\nREADY.")
        return 0

    if os.name == "nt":                                     # keep the PC awake while this runs
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except Exception:
            pass
    run_dir.mkdir(parents=True, exist_ok=True)
    kill_leftover(run_dir)
    if os.name != "nt":                                        # a SIGTERM (stopped cell) ends the chunk cleanly too
        import signal
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if colab and not state:                  # the Colab run, continued here: its plan, its last chunk as chunk done-1
        cd = int(c_st.get("done", 0) or 0)
        state = {k_: c_st.get(k_) for k_ in ("steps", "chunks", "seq", "tokens_per_step", "lr", "smoltalk", "seed",
                                              "views", "base_lrs", "data_order", "data_cursor", "data_sig")}
        state.update(mode="colab", base=colab_base(c_st), device_batch=dbs, start=str(c_root), start_d8=str(src_d8),
                     start_step=src_step, data=str(data), done=cd, started=time.strftime("%Y-%m-%d %H:%M"),
                     tokenizer=str(tok), install_step=a.steps, drive_origin=None, colab_started=c_st.get("started"),
                     colab_done=cd, colab_nan_stops=c_st.get("nan_stops", []))
        if cd and not a.measure:
            copy_model(src_d8, src_step, run_dir / "chunks" / f"chunk_{cd - 1:03d}")
        views = state["views"]
        done = cd
        if a.measure:
            state["done"] = done = 0
    if not state:
        state = {"mode": a.mode, "base": a.base if full else None, "steps": a.steps, "chunks": a.chunks, "seq": a.seq, "device_batch": dbs,
                 "start": str(src_root), "start_step": src_step, "lr": a.lr, "smoltalk": a.smoltalk, "data": str(data),
                 "data_sig": data_sig(data), "views": views, "done": 0, "started": time.strftime("%Y-%m-%d %H:%M"),
                 "tokenizer": str(tok), "seed": 1 if full else int(time.time()) % 100000 * 100 + 1,
                 "data_order": DATA_ORDER, "data_cursor": 0,
                 "install_step": a.steps if full else src_step + a.steps,
                 "drive_origin": drive_id(src_root) or (local_origin(rpa) if src_root == rpa / "nanochat_base" else None)}
    elif not colab:
        state["views"] = views                               # same plan unless the data changed
    state.setdefault("base_lrs", base_lrs(rpa, src_root if full else None))   # one set of rates for every chunk
    state_f.write_text(json.dumps(state, indent=1), encoding="utf-8")
    log = run_dir / "train.log"
    threads = a.threads or (os.cpu_count() or 4) if (a.full_speed or gpu) else (a.threads or max(1, (os.cpu_count() or 2) - 1))
    prio = {}
    if not (gpu or a.full_speed):
        prio = {"creationflags": 0x00004000} if os.name == "nt" else {"preexec_fn": lambda: os.nice(10)}
    dtype = ("bfloat16" if cap and cap >= (8, 0) else "float16") if gpu else "float32"
    speeds = []
    for k in range(done, a.chunks):
        steps_k = bounds[k + 1] - bounds[k]
        lr0, wd, final = lr_for_span(bounds[k] / a.steps, bounds[k + 1] / a.steps, a.lr)
        if k == 0:
            from_dir, from_step = src_d8, src_step
        else:
            from_dir = run_dir / "chunks" / f"chunk_{k - 1:03d}"
            from_step = newest(from_dir)
        print(f"\n== chunk {k + 1}/{a.chunks}: {steps_k} steps, lr {lr0:.3f} -> {lr0 * final:.3f} "
              f"({time.strftime('%H:%M')})", flush=True)
        work = run_dir / "work"
        shutil.rmtree(work, ignore_errors=True)
        wb = prepare_work(work / "nanochat_base", from_dir, from_step, tok)
        if colab:
            views = state["views"]
        if (run_dir / "task_data").exists():
            shutil.copytree(run_dir / "task_data", wb / "task_data", dirs_exist_ok=True)
        f_status, f_bad = model_health(from_dir, from_step)
        if f_status == "broken":                     # never train on (or save over) a broken model
            raise SystemExit(broken_message(rpa, src_root, from_step, f_bad, a.mode) if k == 0 else
                             f"chunk {k} of this run (in {from_dir}) has nan/inf in {T.short_list(f_bad, 3)}: run "
                             "again with --restart (notebook: RESTART_TRAINING = True)")
        nan_before = any(e["chunk"] == k and e["optimizer"] == "loaded" for e in state.get("nan_stops", []))
        load_opt, why_opt = optimizer_choice(from_dir, from_step, a.optimizer, nan_before)
        print("   optimizer:", why_opt, flush=True)
        seed_k, data_start = chunk_data(state, k)
        write_script(nano, data, views, a.smoltalk, seed_k, compile_ok=not a.no_compile)
        data_out = work / "data_cursor.json"
        env = dict(os.environ, NANOCHAT_BASE_DIR=str(wb), OOA_TRAINING_DIR=str(HERE.parent), PYTHONUTF8="1",
                   PYTHONIOENCODING="utf-8", NANOCHAT_DTYPE=dtype, OMP_NUM_THREADS=str(1 if gpu else threads),
                   OOA_STEP_OFFSET=str(bounds[k]), OOA_DATA_START=str(data_start), OOA_DATA_OUT=str(data_out))
        cmd = chunk_command(sys.executable, a.seq, dbs, tps, steps_k, (lr0, wd, final), from_step,
                            load_opt, state["base_lrs"], 100 if gpu else -1, None if gpu else "cpu")
        t0 = time.time()
        watch = {}
        rc, sps = run_chunk(cmd, nano, env, log, a.steps, bounds[k], prio, watch)
        if watch.get("nan") or rc == T.NAN_EXIT:
            nan_stop(state, state_f, k, watch.get("nan") or "chat_sft stopped on a loss that is not a number",
                     load_opt, log)
        if rc != 0:
            raise SystemExit(f"chunk {k + 1} stopped with an error (see the end of {log}). Run again to retry it; the "
                             "chunks before it are kept.")
        out_d8 = wb / "chatsft_checkpoints" / "d8"
        cstep = newest(out_d8)
        if cstep is None:
            raise SystemExit(f"chunk {k + 1} produced no model (see {log}); run again to retry it")
        o_status, o_bad = model_health(out_d8, cstep)
        if o_status == "broken":                     # the watchdog missed it: still never saved or advanced
            nan_stop(state, state_f, k, f"the model it saved has nan/inf in {T.short_list(o_bad, 3)}", load_opt, log)
        if sps:
            speeds.append(sps)
        if a.measure:
            print(f"\nMEASURED: {sps:.1f} s per step (window {a.seq}, device batch {dbs}, {threads} threads, "
                  f"compile {'off' if a.no_compile else 'on'}); first step + loading {time.time() - t0:.0f} s in total")
            shutil.rmtree(run_dir, ignore_errors=True)
            return 0
        if not (run_dir / "task_data").exists() and (wb / "task_data").exists():
            shutil.copytree(wb / "task_data", run_dir / "task_data", dirs_exist_ok=True)
        dst = run_dir / "chunks" / f"chunk_{k:03d}"
        shutil.rmtree(dst, ignore_errors=True)
        copy_model(out_d8, cstep, dst)
        old = run_dir / "chunks" / f"chunk_{k - 2:03d}"           # keep the last two chunks (disk space)
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
        data_after_chunk(state, data_out, data_start)
        state["done"] = k + 1
        state.setdefault("seconds_per_step", [])
        if sps:
            state["seconds_per_step"].append(round(sps, 2))
        state_f.write_text(json.dumps(state, indent=1), encoding="utf-8")
        shutil.rmtree(work, ignore_errors=True)
        h = (time.time() - t0) / 3600
        print(f"   chunk {k + 1} saved ({h * 60:.0f} min); about {h * (a.chunks - k - 1):.1f} h to go", flush=True)

    last = run_dir / "chunks" / f"chunk_{a.chunks - 1:03d}"
    fstep = newest(last)
    if fstep is None:
        raise SystemExit("no finished model in " + str(last))
    meta_f = last / f"meta_{fstep:06d}.json"
    m = json.loads(meta_f.read_text(encoding="utf-8"))
    m["ooa_local"] = {"mode": a.mode, "base": state.get("base"), "steps": a.steps, "chunks": a.chunks, "from": str(src_d8), "from_step": src_step,
                      "seq": a.seq, "data_sig": state.get("data_sig"), "time": time.strftime("%Y%m%d_%H%M%S")}
    meta_f.write_text(json.dumps(m, indent=2), encoding="utf-8")
    state["result"] = str(last)
    state_f.write_text(json.dumps(state, indent=1), encoding="utf-8")
    print(f"\nTRAINED: {last / f'model_{fstep:06d}.pt'} (trained window {m.get('trained_window')}). "
          "Next: eval_pc.py scores it against the installed model and installs it only if it is better.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nstopped: run again to continue from the last saved chunk", flush=True)
        sys.exit(130)
