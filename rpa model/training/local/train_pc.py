"""Train the OOA model on this PC with the Colab v4 recipe (cell 7), in saved chunks that resume after any stop.

    python train_pc.py --mode full  --data DIR   the Colab run: 12000 steps at 4096 from the pretrained BASE model
                                                 (NVIDIA GPU: a few hours; on a CPU it would take weeks)
    python train_pc.py --mode topup --data DIR   continue the newest SFT model for a few hundred steps (CPU: a night)
    python train_pc.py --mode refine --data DIR  NVIDIA GPU: continue the finished Colab model (v5) for 1500 steps on
                                                 the newest data (v5, session) and the weak areas, with replay of the
                                                 rest (sft_setup.REFINE_SHARES); saved every 250 steps
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
mixture for the whole run and reads its own share of it with a different shuffle (seed + k; a full run starts at seed
1 like Colab, every top-up round at a new seed), so over the run every file is seen the planned number of times on
average. With --chunks 1 (default on a GPU) the run is Colab cell 7.
The state is kept in RPA/local_run/<mode>/ (state.json, the last two chunks, train.log): full, topup and refine runs
never share a state. The result is the newest chunk; eval_pc.py scores it and installs it only if it is better.

Broken weights (nan / inf) never go further: the start model is checked before anything runs (also --check), and a
broken one is refused with what to use instead (--start-from pc / drive; notebook cell 1 REFINE_FROM), never swapped
silently. The optimizer state saved with the start model is loaded only when it is finite (optimizer_choice). While a
chunk trains, the loss and validation lines are watched (follow): a loss that is not a number for NAN_STEPS steps in
a row, or a validation bpb that is not a number, stops the chunk; nothing of it is saved, the state notes it, and the
next try of that chunk starts with a fresh optimizer state when the loaded one may be the cause.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RPA_DEFAULT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
import sft_setup as T                                          # noqa: E402  (one source of truth with Colab cell 7)

NANOCHAT_COMMIT = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"   # the nanochat version the patch is written against
TOKENS_PER_STEP = 32768          # Colab: SFT_SEQ 4096 x device batch 2 x 4 micro-batches; every mode keeps it
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


NAN_STEPS = T.NAN_STEPS          # loss lines in a row that are not a number -> the chunk is stopped
START_FROM = ("auto", "drive", "pc")     # notebook cell 1 REFINE_FROM (topup / refine)


def newest(d):
    """Highest step N in d with model_N.pt and meta_N.json, or None."""
    best = None
    for f in Path(d).glob("model_*.pt") if Path(d).exists() else []:
        m = re.match(r"model_(\d+)\.pt$", f.name)
        if m and (Path(d) / f"meta_{m.group(1)}.json").exists():
            best = max(best or 0, int(m.group(1)))
    return best


def copy_model(src, step, dst, optimizer=True):
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in Path(src).iterdir():
        if re.match(rf"(model|meta)_{step:06d}\.", f.name) or (optimizer and re.match(rf"optim_{step:06d}_", f.name)):
            shutil.copy2(f, dst / f.name)


def lr_for_chunk(k, chunks, lr):
    """(--init-lr-frac, --warmdown-ratio, --final-lr-frac) for chunk k of chunks, so the chunks together follow one
    run's schedule: lr flat over the first half, then linear down to 0 (nanochat chat_sft defaults: warmdown 0.5,
    final 0). chunks=1 gives exactly those defaults."""
    return lr_for_span(k / chunks, (k + 1) / chunks, lr)


def lr_for_span(g0, g1, lr):
    """The same for a chunk that covers the part [g0, g1] of the run."""
    def f(g):
        return 1.0 if g <= 0.5 else max(0.0, (1.0 - g) / 0.5)
    if g1 <= 0.5:                      # flat: warmdown over the whole chunk from 1 to 1 (warmdown 0 would divide by
        return lr, 1.0, 1.0            # zero in nanochat once progress passes 1.0 by the prefetched batch)
    if g0 >= 0.5:
        return lr * f(g0), 1.0, f(g1) / f(g0)
    return lr, (g1 - 0.5) / (g1 - g0), f(g1)


BASE_LRS = {"embedding_lr": 0.3, "unembedding_lr": 0.008, "matrix_lr": 0.02}   # base_train.py defaults (pinned)


def base_lrs(rpa):
    """The learning rates Colab's SFT run used: chat_sft inherits them from the base model's meta (user_config).
    Passed explicitly to every chunk, because a saved SFT checkpoint stores them as None (chat_sft saves its
    user_config before it fills them in) and chat_sft then falls back to its own defaults, which halve the output
    layer's rate (unembedding 0.004 instead of base_train's 0.008)."""
    out = dict(BASE_LRS)
    d = Path(rpa) / "train_data" / "base" / "base_checkpoints" / "d8"
    step = newest(d)
    if step is not None:
        try:
            uc = json.loads((d / f"meta_{step:06d}.json").read_text(encoding="utf-8")).get("user_config") or {}
            for k in out:
                if isinstance(uc.get(k), (int, float)) and uc[k] > 0:
                    out[k] = float(uc[k])
        except (OSError, ValueError):
            pass
    return out


def lr_flags(lrs):
    return [f"--{k.replace('_', '-')}={v:g}" for k, v in lrs.items()]


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


def model_health(d8, step):
    """Are the weights of model_<step>.pt in d8 all finite? -> ("ok", []), ("broken", [tensor names with nan/inf]) or
    ("unchecked", [why]) when this Python has no torch or the file is not a saved state (nanochat then says so
    itself when it loads it)."""
    if step is None:
        return "unchecked", ["no model"]
    try:
        bad = T.nonfinite_tensors(Path(d8) / f"model_{step:06d}.pt")
    except ImportError:
        return "unchecked", ["torch is not installed in this Python"]
    except Exception as ex:                                     # noqa: BLE001  (not a saved state dict)
        return "unchecked", [f"could not read it ({type(ex).__name__}: {str(ex)[:120]})"]
    return ("broken", bad) if bad else ("ok", [])


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


def optimizer_choice(d8, step, policy="auto", nan_before=False):
    """Load the optimizer state saved with the model a chunk starts from (nanochat --load-optimizer)? -> (load, why).

    The state is loaded when it is there and finite: chat_sft has no learning-rate warmup, and a fresh AdamW's first
    steps move every embedding / scalar by about its full learning rate (its first update is lr x sign of the
    gradient), while the saved second moments damp them; the Muon momentum and AdamW moments are averages over the
    last ~10-20 steps, so their value is that smooth start, nothing more. A float16 run (Colab T4, GradScaler)
    leaves no trace in them: GradScaler unscales the gradients before the optimizer step, so the moments are in real
    units and carry over to bfloat16. Fresh instead: policy "fresh", no saved state, a state that is not finite or
    cannot be read, or (policy "auto") nan_before: an earlier try of this chunk went nan with the state loaded."""
    f = Path(d8) / f"optim_{step:06d}_rank0.pt"
    if policy == "fresh":
        return False, "fresh optimizer state (--optimizer fresh)"
    if not f.exists():
        return False, f"no optimizer state saved with the start model ({f.name}): a fresh one"
    if nan_before and policy == "auto":
        return False, ("the last try of this chunk went nan with the saved optimizer state loaded: this try starts "
                       "with a fresh optimizer state")
    try:
        bad = T.nonfinite_tensors(f)
    except ImportError:
        bad = []
    except Exception as ex:                                     # noqa: BLE001
        return False, f"the saved optimizer state could not be read ({type(ex).__name__}): a fresh one"
    if bad:
        return False, (f"the saved optimizer state has nan/inf in {len(bad)} tensor(s) ({T.short_list(bad, 3)}): "
                       "a fresh optimizer state instead")
    return True, ("the saved optimizer state is finite: loaded (--optimizer load)" if policy == "load" else
                  "the saved optimizer state is finite: loaded, as nanochat does (a fresh one if this chunk goes nan)")


def nan_value(t):
    """A loss / bpb text that is not a finite number ("nan", "inf", "-inf")."""
    try:
        v = float(t)
    except ValueError:
        return True
    return v != v or v in (float("inf"), float("-inf"))


def pinned_chat_sft(nano):
    """scripts/chat_sft.py of the pinned commit: from git when nanochat is a clone (like Colab cell 7), else the file
    as it is, which must be the unpatched pinned version (the zip ships it so)."""
    nano = Path(nano)
    if (nano / ".git").exists():
        try:                                   # Windows without Git: no git program (FileNotFoundError)
            r = subprocess.run(["git", "-C", str(nano), "show", f"{NANOCHAT_COMMIT}:scripts/chat_sft.py"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode == 0 and r.stdout:
                return r.stdout.replace("\r\n", "\n")
        except OSError:
            pass
    s = (nano / "scripts" / "chat_sft.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    if "OOADialogues" in s:
        raise SystemExit(f"{nano / 'scripts' / 'chat_sft.py'} was patched by an older tool: restore it from the pinned "
                         f"nanochat ({NANOCHAT_COMMIT[:7]}) or unzip the assistant again")
    return s


def no_compile(s):
    """Plain Windows has no C++ compiler for torch.compile: run uncompiled (slower, same result). nanochat's optimizer
    is @torch.compile'd too, so torch.compile becomes a no-op before anything imports it."""
    old = "model = torch.compile(model, dynamic=False)"
    if old not in s:
        raise SystemExit("chat_sft.py is not the expected version (torch.compile line)")
    s = s.replace(old, "pass  # torch.compile switched off (--no-compile)", 1)
    return ("import torch as _torch_nc\n"
            "_torch_nc.compile = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)\n") + s


MOMENTUM_OLD = "    muon_momentum = get_muon_momentum(step)\n"
MOMENTUM_NEW = ("    muon_momentum = get_muon_momentum(step + int(os.environ.get(\"OOA_STEP_OFFSET\", \"0\")))"
                "  # the run's step, not the chunk's\n")


def write_script(nano, data, views, smoltalk, seed, compile_ok):
    s = pinned_chat_sft(nano)
    missing = [k for k, a in {"window": T.WINDOW_ANCHOR, "meta": T.META_ANCHOR, "tasks start": T.TASKS_START,
                              "tasks end": T.TASKS_END, "import": T.IMPORT_ANCHOR, "loader": T.LOADER_OLD,
                              "step": T.STEP_ANCHOR, "end": T.END_ANCHOR, "loss line": T.LOSS_LOG_ANCHOR,
                              "validation line": T.VAL_ANCHOR}.items() if a not in s]
    if missing:
        raise SystemExit(f"chat_sft.py does not match the pinned nanochat; missing: {missing}")
    s = T.patch_chat_sft(s, Path(data).as_posix(), views, smoltalk_rows=smoltalk, seed=seed)
    if s.count(MOMENTUM_OLD) != 1:
        raise SystemExit("chat_sft.py is not the expected version (Muon momentum line)")
    s = s.replace(MOMENTUM_OLD, MOMENTUM_NEW, 1)                # chunk k continues the momentum ramp of chunk k-1
    if not compile_ok:
        s = no_compile(s)
    out = Path(nano) / "scripts" / "chat_sft_ooa.py"            # the original chat_sft.py stays untouched
    out.write_text(s, encoding="utf-8")
    shutil.copy2(HERE.parent / "ooa_task.py", Path(nano) / "tasks" / "ooa.py")
    return out


def data_sig(data):
    rep = Path(data) / "render_report.json"
    return hashlib.sha1(rep.read_bytes()).hexdigest()[:12] if rep.exists() else "no-report"


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


def windows_kill_with_parent(p):
    """Windows: put the training process in a job that ends when this process ends, however it ends (a stopped
    notebook cell kills this process with TerminateProcess, which runs no finally block; the nightly task's time
    limit does the same). -> the job handle (keep it open while the child runs), or None."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        k32.CloseHandle.argtypes = (wintypes.HANDLE,)

        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class Extended(ctypes.Structure):
            _fields_ = [("Basic", Basic), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = Extended()
        info.Basic.LimitFlags = 0x2000                         # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):   # Extended info
            k32.CloseHandle(job)
            return None
        h = k32.OpenProcess(0x0001 | 0x0100, False, p.pid)     # PROCESS_TERMINATE | PROCESS_SET_QUOTA
        ok = bool(h) and k32.AssignProcessToJobObject(job, h)
        if h:
            k32.CloseHandle(h)
        if not ok:
            k32.CloseHandle(job)
            return None
        return job                                             # closed by Windows when this process ends
    except Exception:
        return None


def run_chunk(cmd, cwd, env, log, total_steps, step_offset, prio, watch=None):
    """Run one chat_sft chunk; full output to the log, a progress line every 10 steps here. -> (rc, seconds per step).
    watch (a dict) gets "nan": what was seen when follow() stopped the chunk because the loss went nan."""
    dts = []
    watch = {} if watch is None else watch
    with open(log, "a", encoding="utf-8") as lf:
        lf.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        lf.flush()
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", **prio)
        job = windows_kill_with_parent(p)                     # kept open while the chunk runs
        pid_f = Path(log).with_name("child.pid")
        pid_f.write_text(str(p.pid), encoding="utf-8")
        try:
            rc = follow(p, lf, dts, total_steps, step_offset, watch)
        finally:          # a stopped notebook cell must not leave training running (Windows: also the job above)
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=30)
                except Exception:
                    p.kill()
            pid_f.unlink(missing_ok=True)
            if job:
                try:
                    import ctypes
                    ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(job))
                except Exception:
                    pass
    steady = dts[1:] if len(dts) > 1 else dts                     # the first step includes warm-up
    return rc, (sum(steady) / len(steady) if steady else None)


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


STEP_RE = re.compile(r"step (\d+) \(([\d.]+)%\) \| loss: ([-+]?(?:nan|inf|[\d.]+(?:e[-+]?\d+)?))"
                     r".*?dt: ([\d.]+)ms", re.I)
VAL_RE = re.compile(r"Step (\d+) \| Validation bpb: (\S+)")


def follow(p, lf, dts, total_steps, step_offset, watch=None, nan_steps=NAN_STEPS):
    """Copy the chunk's output to the log; a progress line every 10 steps here (and every line once the loss is not a
    number); collect seconds per step. NaN watchdog: a loss that is not a number in nan_steps step lines in a row,
    a validation bpb that is not a number, or chat_sft's own NaN stop (sft_setup NAN_MARK) -> the chunk is ended
    (nothing of it is saved: chat_sft saves only at its last step) and watch["nan"] says what was seen."""
    watch = {} if watch is None else watch
    bad = 0
    for line in p.stdout:
        lf.write(line)
        lf.flush()
        m = STEP_RE.match(line)
        v = VAL_RE.search(line)
        why = None
        if m:
            s, dt = int(m.group(1)), float(m.group(4)) / 1000
            dts.append(dt)
            is_nan = nan_value(m.group(3))
            bad = bad + 1 if is_nan else 0
            if s % 10 == 0 or s <= 3 or is_nan:
                left = (total_steps - step_offset - s) * (sum(dts[-10:]) / len(dts[-10:])) / 3600
                print(f"   step {step_offset + s}/{total_steps} | loss {m.group(3)} | {dt:.1f} s per step"
                      f" | about {left:.1f} h to go ({time.strftime('%H:%M')})", flush=True)
            if bad >= nan_steps:
                why = (f"the loss was not a number ({m.group(3)}) for {bad} steps in a row (from step "
                       f"{step_offset + s - bad + 1})")
        elif v and nan_value(v.group(2)):
            print("   " + line.rstrip(), flush=True)
            why = (f"Validation bpb is {v.group(2)} at step {step_offset + int(v.group(1))}"
                   + (" (before any training step of this chunk: the model it started from gives nan)"
                      if int(v.group(1)) == 0 else ""))
        elif T.NAN_MARK in line:
            print("   " + line.rstrip(), flush=True)
            why = line.split(":", 1)[-1].strip()
        elif any(k in line for k in ("SFT attention window", "OOA training mixture", "Traceback", "Error",
                                     "Training data actually read", "views", "skipped as longer", "Validation bpb",
                                     "optimizer")):
            print("   " + line.rstrip(), flush=True)
        if why:
            watch["nan"] = why
            if p.poll() is None:
                p.terminate()
            try:
                p.wait(timeout=60)
            except Exception:                                      # noqa: BLE001
                p.kill()
            return p.wait()
    return p.wait()


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
        f"\nSTOPPED: chunk {k + 1} went unstable: {why}. Nothing of this chunk was saved (the chunks before it are "
        f"kept, and the installed model is not touched). {nxt} the lines above (the 'step ... | loss' lines and the "
        f"'Validation bpb' lines) and the end of {log}.")


def main(argv=None):
    try:
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rpa", default=str(RPA_DEFAULT), help='the "rpa model" folder')
    ap.add_argument("--mode", choices=["full", "topup", "refine"], required=True)
    ap.add_argument("--data", required=True, help="the rendered training data (sft_render)")
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
    ap.add_argument("--smoltalk", type=int, default=None, help=f"general chat rows (full {SMOLTALK_ROWS}, topup 0)")
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
    full, refine = a.mode == "full", a.mode == "refine"
    dev, dev_name, mem, cap = device_info()
    gpu = dev == "cuda"
    seq_given = bool(a.seq)
    a.seq = a.seq or default_seq(gpu)
    a.steps = a.steps or (FULL_STEPS if full else REFINE_STEPS if refine else TOPUP_STEPS)
    if a.measure:
        a.steps, a.chunks = a.measure, 1
    a.chunks = a.chunks or (max(1, round(a.steps / REFINE_CHUNK)) if refine else 1 if gpu else max(1, round(a.steps / 50)))
    a.chunks = max(1, min(a.chunks, a.steps))
    a.lr = a.lr if a.lr is not None else (FULL_LR if full else REFINE_LR if refine else TOPUP_LR)
    a.smoltalk = a.smoltalk if a.smoltalk is not None else (SMOLTALK_ROWS if full and not a.measure else 0)
    dbs = a.device_batch or (gpu_batch(mem, a.seq) if gpu else 1)
    if TOKENS_PER_STEP % (a.seq * dbs):
        raise SystemExit(f"--device-batch {dbs} does not divide {TOKENS_PER_STEP} tokens per step at window {a.seq}")
    accum = TOKENS_PER_STEP // (a.seq * dbs)

    run_dir = rpa / "local_run" / ("measure" if a.measure else a.mode)
    state_f = run_dir / "state.json"
    if (a.restart or a.measure) and run_dir.exists() and not a.check:
        shutil.rmtree(run_dir, ignore_errors=True)
    state = json.loads(state_f.read_text(encoding="utf-8")) if state_f.exists() else {}
    if state.get("finished"):
        state = {}                                    # the last run is complete (eval_pc.py took it): a new one
    if state and not full and not a.start and a.start_from != "auto" and \
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

    # where the run starts: the base model (full) or the newest SFT model (topup)
    if state:
        src_root, src_step = Path(state["start"]), state["start_step"]
    else:
        src_root = Path(a.start).expanduser().resolve() if a.start else (
            rpa / "train_data" / "base" if full else start_root(rpa, a.start_from))
        src_step = newest(src_root / ("base_checkpoints" if full else "chatsft_checkpoints") / "d8")
    src_d8 = src_root / ("base_checkpoints" if full else "chatsft_checkpoints") / "d8"
    tok = next((p for p in (src_root / "tokenizer", rpa / "nanochat_base" / "tokenizer")
                if (p / "tokenizer.pkl").exists() and (p / "token_bytes.pt").exists()), None)

    problems = []
    if src_step is None:
        problems.append(f"no model_<N>.pt + meta_<N>.json in {src_d8}"
                        + (" (the pretrained base model from Drive nanochat_base/base_checkpoints/d8)" if full else ""))
    if tok is None:
        problems.append(f"no tokenizer (tokenizer.pkl + token_bytes.pt) in {src_root / 'tokenizer'}")
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
    if refine and not gpu and not (a.allow_cpu or a.measure):
        problems.append(f"refine is sized for an NVIDIA GPU ({a.steps} steps; about a day per 1500 steps on a CPU): "
                        "use --mode topup on this PC")
    health = model_health(src_d8, src_step) if src_step is not None else ("unchecked", [])
    if health[0] == "broken":
        problems.append(broken_message(rpa, src_root, src_step, health[1], a.mode) if not full else
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
        accum = TOKENS_PER_STEP // (a.seq * dbs)
    if state and state.get("data_sig") != data_sig(data):
        print("note: the training data changed since this run started; the run continues on the new data")
    try:
        _mt = json.loads((data / "render_report.json").read_text(encoding="utf-8"))["settings"]["max_tokens"]
    except Exception:
        _mt = None
    if _mt and _mt > a.seq - 28:
        print(f"note: the data was rendered for a {_mt + 28}-token window but this run trains at {a.seq}: longer rows "
              "are skipped (the notebook's cell 3 renders it for this window)")

    views, stats, bud, table = T.plan_for_local(str(data), a.mode, a.steps, TOKENS_PER_STEP, smoltalk_rows=a.smoltalk) \
        if data.exists() and (data / "val.jsonl").exists() else ({}, {}, 0, [])
    done = state.get("done", 0)
    bounds = [k * a.steps // a.chunks for k in range(a.chunks + 1)]       # chunk k trains steps bounds[k]..bounds[k+1]
    chunk = max(bounds[k + 1] - bounds[k] for k in range(a.chunks))
    print("assistant folder :", rpa)
    print("device           :", dev_name, f"({mem:.0f} GB)" if gpu else "", "| torch.compile", "off" if a.no_compile else "on")
    print("mode             :", a.mode, "| from", src_d8, f"step {src_step}")
    if not full:
        print("start model      :", start_label(rpa, src_root) if a.start_from == "auto" or a.start else
              f"{model_name(rpa, src_root)} (--start-from {a.start_from}: notebook cell 1 REFINE_FROM)")
    print("weights          :", {"ok": "all finite (checked)", "broken": "BROKEN (nan/inf), see below"}.get(
        health[0], "not checked: " + "; ".join(health[1])))
    _k = state.get("done", 0)                              # the next chunk: from the start model or the last chunk
    _fd = src_d8 if _k == 0 else run_dir / "chunks" / f"chunk_{_k - 1:03d}"
    _fs = src_step if _k == 0 else newest(_fd)
    if _fs is not None and health[0] != "broken":
        print("optimizer        :", optimizer_choice(_fd, _fs, a.optimizer, any(
            e["chunk"] == _k and e["optimizer"] == "loaded" for e in state.get("nan_stops", [])))[1])
    print("training data    :", data, f"(render {data_sig(data)})")
    print(f"plan             : {a.steps} steps x {TOKENS_PER_STEP:,} tokens at window {a.seq} (device batch {dbs}, "
          f"{accum} micro-batches per step) in {a.chunks} chunk(s) of up to {chunk}; {done} done | lr {a.lr} | "
          f"SmolTalk rows {a.smoltalk}")
    print("base rates       :", " ".join(lr_flags(state.get("base_lrs") or base_lrs(rpa))),
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
    if not state:
        state = {"mode": a.mode, "steps": a.steps, "chunks": a.chunks, "seq": a.seq, "device_batch": dbs,
                 "start": str(src_root), "start_step": src_step, "lr": a.lr, "smoltalk": a.smoltalk, "data": str(data),
                 "data_sig": data_sig(data), "views": views, "done": 0, "started": time.strftime("%Y-%m-%d %H:%M"),
                 "tokenizer": str(tok), "seed": 1 if full else int(time.time()) % 100000 * 100 + 1,
                 "install_step": a.steps if full else src_step + a.steps,
                 "drive_origin": drive_id(src_root) or (local_origin(rpa) if src_root == rpa / "nanochat_base" else None)}
    else:
        state["views"] = views                               # same plan unless the data changed
    state.setdefault("base_lrs", base_lrs(rpa))               # one set of rates for every chunk of this run
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
        wb = work / "nanochat_base"
        copy_model(from_dir, from_step, wb / "base_checkpoints" / "d8")
        shutil.copytree(tok, wb / "tokenizer", dirs_exist_ok=True)     # dirs_exist_ok: Windows may leave a
        if (run_dir / "task_data").exists():                             # locked leftover of the last chunk
            shutil.copytree(run_dir / "task_data", wb / "task_data", dirs_exist_ok=True)
        f_status, f_bad = model_health(from_dir, from_step)
        if f_status == "broken":                     # never train on (or save over) a broken model
            raise SystemExit(broken_message(rpa, src_root, from_step, f_bad, a.mode) if k == 0 else
                             f"chunk {k} of this run (in {from_dir}) has nan/inf in {T.short_list(f_bad, 3)}: run "
                             "again with --restart (notebook: RESTART_TRAINING = True)")
        nan_before = any(e["chunk"] == k and e["optimizer"] == "loaded" for e in state.get("nan_stops", []))
        load_opt, why_opt = optimizer_choice(from_dir, from_step, a.optimizer, nan_before)
        print("   optimizer:", why_opt, flush=True)
        write_script(nano, data, views, a.smoltalk, state.get("seed", 1) + k, compile_ok=not a.no_compile)
        env = dict(os.environ, NANOCHAT_BASE_DIR=str(wb), OOA_TRAINING_DIR=str(HERE.parent), PYTHONUTF8="1",
                   PYTHONIOENCODING="utf-8", NANOCHAT_DTYPE=dtype, OMP_NUM_THREADS=str(1 if gpu else threads),
                   OOA_STEP_OFFSET=str(bounds[k]))
        cmd = [sys.executable, "-m", "scripts.chat_sft_ooa", f"--max-seq-len={a.seq}", f"--device-batch-size={dbs}",
               f"--total-batch-size={TOKENS_PER_STEP}", f"--num-iterations={steps_k * accum}",
               f"--eval-every={100 if gpu else -1}", f"--eval-tokens={a.seq * dbs * 8}", "--chatcore-every=-1",
               "--run=dummy", f"--init-lr-frac={lr0:.6f}", f"--warmdown-ratio={wd:.6f}", f"--final-lr-frac={final:.6f}",
               "--model-tag=d8", f"--model-step={from_step}", f"--load-optimizer={int(load_opt)}"] \
            + lr_flags(state["base_lrs"])
        if not gpu:
            cmd.append("--device-type=cpu")
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
    m["ooa_local"] = {"mode": a.mode, "steps": a.steps, "chunks": a.chunks, "from": str(src_d8), "from_step": src_step,
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
