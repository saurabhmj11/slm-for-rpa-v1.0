"""SFT in saved chunks: one long nanochat chat_sft run split into pieces that each start from the model and optimizer
the piece before saved, so a stop (a Colab session that ends, a PC that is switched off) loses at most one piece.

Shared by the PC (training/local/train_pc.py) and Colab cell 7 (colab_sft below); standard library only at import
(torch is needed only to check weights). The pieces follow ONE run's schedule:
  - chunk k of K covers the part [k/K, (k+1)/K] of the run's learning-rate schedule (constant over the first half,
    then the linear warmdown to 0): lr_for_span gives chat_sft's --init-lr-frac / --warmdown-ratio / --final-lr-frac;
  - the Muon momentum ramp counts the run's steps (OOA_STEP_OFFSET, MOMENTUM_NEW);
  - the base learning rates are passed explicitly (a saved SFT checkpoint stores them as None, and chat_sft would fall
    back to its own defaults, which halve the output layer's rate);
  - every chunk plans the same data mixture for the WHOLE run with the run's one seed and reads the next part of that
    one list: chunk k starts reading where chunk k-1 stopped (OOA_DATA_START, the cursor chunk k-1 wrote to
    OOA_DATA_OUT; chunk_data / data_after_chunk), so the chunks read the stream one run reads (up to the ~100
    conversations in the packing buffer at each boundary) and every file, SmolTalk too, is seen as planned. (A run
    started before this kept seed + k per chunk with each chunk reading from the start: continued that way.)
  - a chunk whose loss / validation goes nan is never saved (chat_sft's guard, sft_setup; the watchdog in follow);
    the next try starts with a fresh optimizer state when the loaded one may be the cause (optimizer_choice).

Colab (colab_sft): the chunks and state.json live on Drive (NANOCHAT_BASE_DIR/sft_run), the training itself on the
machine's disk. A new session runs cells 1 -> 2 -> 6 -> 6d -> 7 and cell 7 continues at the next chunk with the
run's own settings; it refuses to mix settings or data that changed (RESTART_SFT = True starts a new run). The finished
model is written to chatsft_checkpoints/<tag> only when every chunk is done and its weights are checked finite.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sft_setup as T                                          # noqa: E402

NANOCHAT_COMMIT = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"   # the nanochat version the patch is written against
NAN_STEPS = T.NAN_STEPS          # loss lines in a row that are not a number -> the chunk is stopped
BASE_LRS = {"embedding_lr": 0.3, "unembedding_lr": 0.008, "matrix_lr": 0.02}   # base_train.py defaults (pinned)
SFT_INIT_LR = 0.8                # nanochat's default --init-lr-frac (the Colab run)


# ------------------------------------------------------------------------------------------- checkpoints
def newest(d):
    """Highest step N in d with model_N.pt and meta_N.json, or None."""
    best = None
    for f in Path(d).glob("model_*.pt") if Path(d).exists() else []:
        m = re.match(r"model_(\d+)\.pt$", f.name)
        if m and (Path(d) / f"meta_{m.group(1)}.json").exists():
            best = max(best or 0, int(m.group(1)))
    return best


def copy_model(src, step, dst, optimizer=True, new_step=None):
    """model / meta (and optim) of step from src into dst; new_step renames them (the run's step on install)."""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in Path(src).iterdir():
        if re.match(rf"(model|meta)_{step:06d}\.", f.name) or (optimizer and re.match(rf"optim_{step:06d}_", f.name)):
            name = f.name if new_step is None else f.name.replace(f"_{step:06d}", f"_{new_step:06d}", 1)
            shutil.copy2(f, dst / name)


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


# ------------------------------------------------------------------------------------------- the schedule
def chunk_bounds(steps, chunks):
    """Chunk k trains the run's steps bounds[k]..bounds[k+1]."""
    return [k * steps // chunks for k in range(chunks + 1)]


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


def base_lrs_of(d8):
    """The learning rates the base model in d8 recorded (its meta's user_config), else base_train's defaults."""
    out = dict(BASE_LRS)
    step = newest(d8)
    if step is not None:
        try:
            uc = json.loads((Path(d8) / f"meta_{step:06d}.json").read_text(encoding="utf-8")).get("user_config") or {}
            for k in out:
                if isinstance(uc.get(k), (int, float)) and uc[k] > 0:
                    out[k] = float(uc[k])
        except (OSError, ValueError):
            pass
    return out


def lr_flags(lrs):
    return [f"--{k.replace('_', '-')}={v:g}" for k, v in lrs.items()]


def chunk_command(python, seq, device_batch, tokens_per_step, steps_k, span, from_step, load_optimizer, base_lrs,
                  eval_every, device_type=None, model_tag="d8"):
    """The chat_sft_ooa command of one chunk (the PC and Colab run the same one)."""
    lr0, wd, final = span
    accum = tokens_per_step // (seq * device_batch)
    cmd = [python, "-m", "scripts.chat_sft_ooa", f"--max-seq-len={seq}", f"--device-batch-size={device_batch}",
           f"--total-batch-size={tokens_per_step}", f"--num-iterations={steps_k * accum}",
           f"--eval-every={eval_every}", f"--eval-tokens={seq * device_batch * 8}", "--chatcore-every=-1",
           "--run=dummy", f"--init-lr-frac={lr0:.6f}", f"--warmdown-ratio={wd:.6f}", f"--final-lr-frac={final:.6f}",
           f"--model-tag={model_tag}", f"--model-step={from_step}", f"--load-optimizer={int(load_optimizer)}"] \
        + lr_flags(base_lrs)
    if device_type:
        cmd.append(f"--device-type={device_type}")
    return cmd


def prepare_work(wb, from_dir, from_step, tokenizer, model_tag="d8"):
    """The folder a chunk runs in (its NANOCHAT_BASE_DIR): the start model as base_checkpoints/<tag> + the tokenizer.
    Anything an earlier chunk left there (its model, its output) is removed first."""
    wb = Path(wb)
    for d in ("base_checkpoints", "chatsft_checkpoints"):
        shutil.rmtree(wb / d, ignore_errors=True)
    copy_model(from_dir, from_step, wb / "base_checkpoints" / model_tag)
    shutil.copytree(tokenizer, wb / "tokenizer", dirs_exist_ok=True)     # dirs_exist_ok: Windows may leave a locked
    return wb                                                              # leftover of the last chunk


# ------------------------------------------------------------------------------------------- the patched script
def pinned_chat_sft(nano):
    """scripts/chat_sft.py of the pinned commit: from git when nanochat is a clone (Colab, a PC with Git), else the file
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


# chunk k continues the run's data stream: the train loader starts at OOA_DATA_START (mod the list's length) and the
# script writes {"start", "fetched", "size"} to OOA_DATA_OUT at the end (rows fetched into the packing buffer)
DATA_ORDER = "one_stream"
CURSOR_OLD = "    cursor = ddp_rank  # Each rank processes different conversations (for fetching)\n"
CURSOR_NEW = (CURSOR_OLD + "    if split == \"train\":  # OOA: continue the run's data stream where the last chunk stopped\n"
              "        cursor = (cursor + int(os.environ.get(\"OOA_DATA_START\", \"0\"))) % dataset_size\n")
FETCH_OLD = "            cursor += ddp_world_size\n"
FETCH_NEW = FETCH_OLD + "            if split == \"train\":\n                _ooa_fetched[0] += ddp_world_size\n"
LOADER_DEF = 'train_loader = sft_data_generator_bos_bestfit("train")\n'
DATA_END = '''if master_process and os.environ.get("OOA_DATA_OUT"):   # OOA: where the next chunk continues reading
    import json as _ooa_json
    with open(os.environ["OOA_DATA_OUT"], "w", encoding="utf-8") as _f:
        _ooa_json.dump({"start": int(os.environ.get("OOA_DATA_START", "0")), "fetched": _ooa_fetched[0],
                        "size": len(train_dataset)}, _f)
'''


def chunk_data(state, k):
    """(seed, data start) of chunk k: the run's one seed and the cursor the chunk before stopped at; a run started
    before the one-stream order: seed + k, each chunk from the start."""
    if state.get("data_order") == DATA_ORDER:
        return int(state.get("seed", 1)), int(state.get("data_cursor", 0) or 0)
    return int(state.get("seed", 1)) + k, 0


def data_after_chunk(state, out_file, start):
    """After a chunk that is saved: state["data_cursor"] = where the next chunk starts reading. -> the record
    {"start", "fetched", "size"} (None, with a warning, when the script wrote none)."""
    try:
        rec = json.loads(Path(out_file).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        if state.get("data_order") == DATA_ORDER:
            print(f"   warning: the chunk wrote no data cursor ({out_file}); the next chunk starts where this one did")
        return None
    if state.get("data_order") == DATA_ORDER:
        state["data_cursor"] = int(start) + int(rec.get("fetched", 0))
        state["data_size"] = rec.get("size")
    return rec


def write_script(nano, data, views, smoltalk, seed, compile_ok, training_dir):
    """nanochat/scripts/chat_sft_ooa.py: the pinned chat_sft.py patched by sft_setup.patch_chat_sft (window fix, the
    mixture, NaN guard) + the run-wide Muon momentum ramp; tasks/ooa.py from training_dir (the original chat_sft.py
    stays untouched)."""
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
    for a, b in ((CURSOR_OLD, CURSOR_NEW), (FETCH_OLD, FETCH_NEW), (LOADER_DEF, "_ooa_fetched = [0]\n" + LOADER_DEF),
                 (T.END_ANCHOR, DATA_END + T.END_ANCHOR)):
        if s.count(a) != 1:
            raise SystemExit("chat_sft.py is not the expected version (data loader: " + a.strip()[:50] + ")")
        s = s.replace(a, b, 1)                                  # chunk k continues the data stream of chunk k-1
    if not compile_ok:
        s = no_compile(s)
    out = Path(nano) / "scripts" / "chat_sft_ooa.py"
    out.write_text(s, encoding="utf-8")
    shutil.copy2(Path(training_dir) / "ooa_task.py", Path(nano) / "tasks" / "ooa.py")
    return out


def data_sig(data):
    """The training data's identity: the sha1 of its render report (sft_render); data that was not rendered: its
    filter report, else its files' names and sizes."""
    data = Path(data)
    rep = data / "render_report.json"
    if rep.exists():
        return hashlib.sha1(rep.read_bytes()).hexdigest()[:12]
    if (data / "filter_report.json").exists():
        return "f" + hashlib.sha1((data / "filter_report.json").read_bytes()).hexdigest()[:11]
    if not data.is_dir():
        return "no-report"
    names = sorted((f.name, f.stat().st_size) for f in data.iterdir() if f.suffix == ".jsonl")
    return "n" + hashlib.sha1(json.dumps(names).encode()).hexdigest()[:11] if names else "no-report"


# ------------------------------------------------------------------------------------------- running a chunk
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


def run_chunk(cmd, cwd, env, log, total_steps, step_offset, prio, watch=None, every=10):
    """Run one chat_sft chunk; full output to the log, a progress line every `every` steps here. -> (rc, seconds per
    step). watch (a dict) gets "nan": what was seen when follow() stopped the chunk because the loss went nan."""
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
            rc = follow(p, lf, dts, total_steps, step_offset, watch, every)
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


STEP_RE = re.compile(r"step (\d+) \(([\d.]+)%\) \| loss: ([-+]?(?:nan|inf|[\d.]+(?:e[-+]?\d+)?))"
                     r".*?dt: ([\d.]+)ms", re.I)
VAL_RE = re.compile(r"Step (\d+) \| Validation bpb: (\S+)")
SKIP_SHOWN = 20                  # skipped-step lines shown in full before only every 50th


def follow(p, lf, dts, total_steps, step_offset, watch=None, every=10):
    """Copy the chunk's output to the log; a progress line every `every` steps here (and the first line whose loss is
    not a number); collect seconds per step. NaN watchdog: a validation bpb that is not a number, or chat_sft's own
    NaN stop (sft_setup NAN_MARK: the step's own loss not a number for NAN_STEPS steps in a row) -> the chunk is ended
    (nothing of it is saved: chat_sft saves only at its last step) and watch["nan"] says what was seen. The loss on
    the step lines is NOT counted: it is chat_sft's running average, which stays nan for good after one nan step
    (a float16 run's GradScaler skips such a step and the model stays fine). The step guard's lines (sft_setup
    SKIP_MARK: a step skipped because a micro-batch loss or the gradients were not finite) are shown: the first
    SKIP_SHOWN, then every 50th."""
    watch = {} if watch is None else watch
    shown_nan = False
    skips = 0
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
            first_nan = is_nan and not shown_nan
            if s % every == 0 or s <= 3 or first_nan:
                left = (total_steps - step_offset - s) * (sum(dts[-10:]) / len(dts[-10:])) / 3600
                print(f"   step {step_offset + s}/{total_steps} | loss {m.group(3)} | {dt:.1f} s per step"
                      f" | about {left:.1f} h to go ({time.strftime('%H:%M')})", flush=True)
            if first_nan:
                shown_nan = True
                print("   (the loss shown is a running average: one step that was not a number keeps it nan; the "
                      f"training stops by itself if {NAN_STEPS} steps in a row are nan)", flush=True)
        elif v and nan_value(v.group(2)):
            print("   " + line.rstrip(), flush=True)
            why = (f"Validation bpb is {v.group(2)} at step {step_offset + int(v.group(1))}"
                   + (" (before any training step of this chunk: the model it started from gives nan)"
                      if int(v.group(1)) == 0 else ""))
        elif line.startswith(T.SKIP_MARK):
            skips += 1
            if skips <= SKIP_SHOWN or skips % 50 == 0:
                print("   " + line.rstrip(), flush=True)
            if skips == SKIP_SHOWN:
                print(f"   (more skipped steps are only in the log, every 50th is shown here; {NAN_STEPS} nan steps in "
                      "a row stop the training)", flush=True)
        elif T.NAN_MARK in line:
            print("   " + line.rstrip(), flush=True)
            why = line.split(":", 1)[-1].strip()
            for tail in ("Stopped; nothing was saved", "the training went unstable"):   # said in nan_stop()
                why = why.rstrip(" :.")
                why = why[:-len(tail)] if why.endswith(tail) else why
            why = why.rstrip(" :.")
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


# ------------------------------------------------------------------------------------------- run state
RUN_KEYS = ("steps", "chunks", "seq", "device_batch", "tokens_per_step", "lr", "smoltalk", "seed", "data", "data_sig",
            "base")
KEY_WORDS = {"steps": "optimizer steps (SFT_ITERS / 4)", "chunks": "SFT_CHUNKS", "seq": "SFT_SEQ",
             "device_batch": "SFT_DEVICE_BATCH", "tokens_per_step": "tokens per step", "lr": "learning rate",
             "smoltalk": "SMOLTALK_ROWS", "seed": "seed", "data": "training data folder",
             "data_sig": "training data (render_report.json: cells 6 / 6d made it again)",
             "base": "base model (BASE, its step or its tokenizer)"}


def read_state(f):
    try:
        return json.loads(Path(f).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_state(f, state):
    """state.json written whole or not at all (a session that ends while it is written keeps the old one)."""
    f = Path(f)
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp, f)


def settings_diff(state, settings):
    """['what: run value -> now', ...] for the run settings that differ."""
    out = []
    for k in RUN_KEYS:
        if state.get(k) != settings.get(k):
            out.append(f"{KEY_WORDS.get(k, k)}: {state.get(k)} -> {settings.get(k)}")
    return out


def plan_run(state, settings, restart=False, chunks_ok=True):
    """What cell 7 does with the run state on Drive -> (action, notes). action:
    "new"       no run, RESTART_SFT, a finished run with other settings, or a run whose saved chunk is missing;
                (RESTART_SFT is used once: a run it started, with the same settings, is continued, never restarted)
    "continue"  a run in progress with exactly these settings: the next chunk;
    "finished"  this run is done and installed: nothing to train (RESTART_SFT trains it again);
    "refuse"    a run in progress with other settings or data: never mixed (RESTART_SFT starts over)."""
    if not state:
        return "new", []
    diff = settings_diff(state, settings)
    if restart and state.get("started_by_restart") and not diff:
        # RESTART_SFT left on: it started THIS run (maybe in an earlier session) and never restarts it again
        note = (f"RESTART_SFT = True is ignored: it already started this run ({state.get('started')}); set it back to "
                "False in cell 1. (To start over once more: delete sft_run/state.json on Drive.)")
        if state.get("finished"):
            return "finished", [note]
        return ("continue", [note]) if chunks_ok else ("new", [note, "the last saved chunk is missing: a new run"])
    if restart:
        return "new", ["RESTART_SFT = True: a new run starts" + (f" (the run in progress had {state.get('done', 0)} of "
                       f"{state.get('chunks')} chunks done; they are thrown away)" if not state.get("finished") else "")]
    if state.get("finished"):
        if diff:
            return "new", ["the last run is finished; these settings differ from it, so a new run starts: "
                           + "; ".join(diff)]
        return "finished", []
    if diff:
        return "refuse", diff
    if not chunks_ok:
        return "new", [f"the run in progress says {state.get('done')} chunk(s) are done, but the last one is not on "
                       "Drive (deleted?): a new run starts"]
    return "continue", []


def refuse_message(diff, state):
    return ("cell 7 stopped: an SFT run is in progress on Drive ({} of {} chunks done) with other settings than now:\n  "
            .format(state.get("done", 0), state.get("chunks")) + "\n  ".join(diff) +
            "\nContinuing would mix two different runs. Either put the settings back as they were (cell 1; for the data:"
            " FORCE_DATA = False so cells 6 / 6d keep the data the run started on) and run cell 7 again, or set "
            "RESTART_SFT = True in cell 1 to throw the run in progress away and start a new one with these settings: "
            "then run cells 1 -> 6 -> 6d -> 7 (with RESTART_SFT on, cells 6 / 6d make the data again when its inputs "
            "changed, e.g. SFT_SEQ, so the new run never trains on data made for the old settings).")


def drive_full_message(what, ex, chunk):
    return (f"cell 7 stopped: {what} failed ({type(ex).__name__}: {ex}). Most likely Google Drive is full. Deleted "
            "files (older chunks, old runs) stay in Drive's TRASH and still count against the space: empty the trash "
            "(drive.google.com -> Trash -> Empty trash), check there is about 4 GB free, then run cell 7 again"
            + (f"; chunk {chunk} is trained again (the chunks before it are kept)." if chunk else
               "; every chunk is done, so it only installs the model."))


def run_in_progress(base_dir):
    """The state of an unfinished SFT run on this base (Colab: NANOCHAT_BASE_DIR/sft_run), or None."""
    st = read_state(Path(base_dir) / "sft_run" / "state.json")
    return st if st and not st.get("finished") else None


def base_tag(base_dir):
    """The base model folder chat_sft takes without --model-tag (nanochat find_largest_model): the dN with the largest
    N that has a model."""
    d = Path(base_dir) / "base_checkpoints"
    tags = [p.name for p in d.iterdir() if re.match(r"d\d+$", p.name) and newest(p) is not None] if d.is_dir() else []
    return max(tags, key=lambda t: int(t[1:])) if tags else None


def aside_name(base_dir):
    """chatsft_checkpoints_before_<date>_<time> (the name the PC's data_pc.py looks for), never an existing folder."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    kept = Path(base_dir) / ("chatsft_checkpoints_before_" + stamp)
    while kept.exists():
        time.sleep(1)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        kept = Path(base_dir) / ("chatsft_checkpoints_before_" + stamp)
    return kept


def file_sha1(f):
    h = hashlib.sha1()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------- Colab cell 7
def colab_sft(base_dir, data, steps, chunks, seq, device_batch, tokens_per_step, lr=SFT_INIT_LR, smoltalk=0,
              chars_per_token=None, seed=1, restart=False, nano="/content/nanochat", training_dir="/content/ooa",
              work="/content/sft_work", eval_every=100, device_type=None, compile_ok=True, stop_after=None,
              every=50, python=sys.executable):
    """SFT in `chunks` saved pieces, resumable across Colab sessions. base_dir: the base model folder on Drive
    (NANOCHAT_BASE_DIR: base_checkpoints/<tag> + tokenizer); the run is kept in base_dir/sft_run (state.json, the last
    two chunks, train.log); the finished model goes to base_dir/chatsft_checkpoints/<tag>. stop_after: train at most
    this many chunks in this call (tests: a session that ends). -> {"status": "finished" | "paused", ...}"""
    base_dir, data, work = Path(base_dir), Path(data), Path(work)
    run_dir = base_dir / "sft_run"
    state_f = run_dir / "state.json"
    tag = base_tag(base_dir)
    if tag is None:
        raise SystemExit(f"no base model in {base_dir / 'base_checkpoints'}: pretrain first (cells 3-4)")
    base_d8 = base_dir / "base_checkpoints" / tag
    base_step = newest(base_d8)
    tok = base_dir / "tokenizer"
    if not (tok / "tokenizer.pkl").exists():
        raise SystemExit(f"no tokenizer in {tok}: the base model must come with its tokenizer (cell 3)")
    if tokens_per_step % (seq * device_batch):
        raise SystemExit(f"SFT_DEVICE_BATCH {device_batch} x SFT_SEQ {seq} does not divide {tokens_per_step} tokens")
    chunks = max(1, min(int(chunks), int(steps)))
    settings = {"steps": int(steps), "chunks": chunks, "seq": int(seq), "device_batch": int(device_batch),
                "tokens_per_step": int(tokens_per_step), "lr": float(lr), "smoltalk": int(smoltalk), "seed": int(seed),
                "data": str(data), "data_sig": data_sig(data),
                "base": {"dir": str(base_dir), "tag": tag, "step": base_step, "tokenizer_sha1": file_sha1(tok / "tokenizer.pkl")}}
    state = read_state(state_f)
    done = int(state.get("done", 0) or 0)
    chunks_ok = done == 0 or newest(run_dir / "chunks" / f"chunk_{done - 1:03d}") is not None
    action, notes = plan_run(state, settings, restart, chunks_ok)
    for n in notes:
        print("note:", n)
    if action == "refuse":
        raise SystemExit(refuse_message(notes, state))
    installed = base_dir / "chatsft_checkpoints" / tag
    if action == "finished":
        print(f"SFT already finished: {state['chunks']} chunks, {state['steps']} steps; the model is in "
              f"{state.get('installed', installed)} (step {state.get('installed_step')}). Nothing to train. "
              "(RESTART_SFT = True in cell 1 trains it again.)")
        return {"status": "finished", "installed": state.get("installed", str(installed)), "state": state,
                "trained_now": 0}

    bounds = chunk_bounds(settings["steps"], chunks)
    if action == "new":
        h, bad = model_health(base_d8, base_step)
        if h == "broken":
            raise SystemExit(f"the base model {base_d8} (step {base_step}) has broken weights (nan/inf) in "
                             f"{T.short_list(bad)}: it cannot be trained. Pretrain again (cells 3-4).")
        views, _stats, budget, lines = T.plan_for_notebook(str(data), settings["steps"], tokens_per_step,
                                                           smoltalk_rows=smoltalk, chars_per_token=chars_per_token)
        shutil.rmtree(run_dir / "chunks", ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        kept = None
        ck = base_dir / "chatsft_checkpoints"
        if ck.exists():        # the previous finished model: moved aside (never deleted) once, when a NEW run starts
            kept = aside_name(base_dir)
            shutil.move(str(ck), str(kept))
            print("previous model kept in", kept)
        state = dict(settings, views=views, base_lrs=base_lrs_of(base_d8), done=0, finished=False,
                     started_by_restart=bool(restart), data_order=DATA_ORDER, data_cursor=0,
                     started=time.strftime("%Y-%m-%d %H:%M"), previous_kept=str(kept) if kept else None,
                     chunk_log=[], nan_stops=[], seconds_per_step=[])
        write_state(state_f, state)
        print(f"NEW SFT RUN: {settings['steps']} steps x {tokens_per_step:,} tokens at window {seq} in {chunks} chunks "
              f"of about {settings['steps'] // chunks} steps (saved to Drive after each) | lr {lr} | SmolTalk rows "
              f"{smoltalk} | base {tag} step {base_step} | data {data} (render {settings['data_sig']})")
        print(f"data mixture planned for the whole run: about {budget / 1e6:.0f}M tokens of conversations; views per row:")
        print("\n".join("   " + l for l in lines))
    else:
        print(f"CONTINUING the SFT run started {state.get('started')}: {done} of {chunks} chunks done "
              f"({bounds[done]} of {settings['steps']} steps); same settings and data (checked). Next: chunk {done + 1}.")
    print("base learning rates:", " ".join(lr_flags(state["base_lrs"])), "(times the lr of each chunk)")

    work.mkdir(parents=True, exist_ok=True)
    wb = work / "nanochat_base"
    log = work / "train.log"
    trained = 0
    k = int(state["done"])
    while k < chunks:
        if stop_after is not None and trained >= stop_after:
            print(f"stopped after {trained} chunk(s) (stop_after): run again to continue at chunk {k + 1}")
            return {"status": "paused", "state": state, "trained_now": trained}
        steps_k = bounds[k + 1] - bounds[k]
        span = lr_for_span(bounds[k] / settings["steps"], bounds[k + 1] / settings["steps"], settings["lr"])
        from_dir = base_d8 if k == 0 else run_dir / "chunks" / f"chunk_{k - 1:03d}"
        from_step = newest(from_dir)
        print(f"\n== chunk {k + 1}/{chunks}: steps {bounds[k] + 1}-{bounds[k + 1]}, lr {span[0]:.3f} -> "
              f"{span[0] * span[2]:.3f} ({time.strftime('%H:%M')})", flush=True)
        f_status, f_bad = model_health(from_dir, from_step)
        if f_status == "broken":                     # never train on (or save over) a broken model
            raise SystemExit(f"the model chunk {k + 1} starts from ({from_dir}, step {from_step}) has nan/inf in "
                             f"{T.short_list(f_bad, 3)}: set RESTART_SFT = True in cell 1 and run cell 7 again")
        nan_before = any(e["chunk"] == k and e["optimizer"] == "loaded" for e in state.get("nan_stops", []))
        load_opt, why_opt = optimizer_choice(from_dir, from_step, "auto", nan_before)
        print("   optimizer:", why_opt, flush=True)
        prepare_work(wb, from_dir, from_step, tok, tag)
        seed_k, data_start = chunk_data(state, k)
        write_script(nano, data, state["views"], settings["smoltalk"], seed_k, compile_ok, training_dir)
        data_out = work / "data_cursor.json"
        data_out.unlink(missing_ok=True)
        env = dict(os.environ, NANOCHAT_BASE_DIR=str(wb), OOA_TRAINING_DIR=str(training_dir), PYTHONUTF8="1",
                   PYTHONIOENCODING="utf-8", OOA_STEP_OFFSET=str(bounds[k]), OOA_DATA_START=str(data_start),
                   OOA_DATA_OUT=str(data_out))
        cmd = chunk_command(python, seq, device_batch, tokens_per_step, steps_k, span, from_step, load_opt,
                            state["base_lrs"], eval_every, device_type, tag)
        t0 = time.time()
        watch = {}
        rc, sps = run_chunk(cmd, nano, env, log, settings["steps"], bounds[k], {}, watch, every)
        out_d8 = wb / "chatsft_checkpoints" / tag
        cstep = newest(out_d8) if rc == 0 and not watch.get("nan") else None
        why = watch.get("nan") or ("chat_sft stopped on a loss that is not a number" if rc == T.NAN_EXIT else None)
        if why is None and cstep is not None:
            o_status, o_bad = model_health(out_d8, cstep)
            if o_status == "broken":                 # the watchdog missed it: still never saved or advanced
                why = f"the model it saved has nan/inf in {T.short_list(o_bad, 3)}"
        if why:
            state.setdefault("nan_stops", []).append({"chunk": k, "why": why, "optimizer": "loaded" if load_opt else
                                                      "fresh", "time": time.strftime("%Y-%m-%d %H:%M")})
            write_state(state_f, state)
            print(f"   chunk {k + 1} went unstable: {why}. Nothing of it was saved.", flush=True)
            if load_opt:
                print("   trying chunk", k + 1, "again with a fresh optimizer state (the likely cause)", flush=True)
                continue
            raise SystemExit(f"STOPPED: chunk {k + 1} went unstable ({why}) with a fresh optimizer state too, so "
                             f"running it again will not help. The chunks before it are kept on Drive and the "
                             f"previous model is untouched. To finish this run on a PC with an NVIDIA GPU instead "
                             f"(bfloat16 there does not overflow like the T4's float16): the PC notebook with "
                             f"MODE = \"colab\" continues it from chunk {k + 1} (train_pc.py --mode colab). Send "
                             f"the 'step ... | loss', 'OOA SKIP' and 'Validation bpb' lines above and the end of {log}.")
        if rc != 0 or cstep is None:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:] if log.exists() else []
            print("\n".join("   | " + l for l in tail))
            raise SystemExit(f"chunk {k + 1} stopped with an error (exit code {rc}; the end of {log} is above). Run "
                             "cell 7 again to retry it; the chunks before it are kept on Drive.")
        # save to Drive: a .part folder first, renamed when complete; then state.json; then the old chunk goes
        dst = run_dir / "chunks" / f"chunk_{k:03d}"
        part = dst.with_name(dst.name + ".part")
        shutil.rmtree(part, ignore_errors=True)
        try:
            copy_model(out_d8, cstep, part)
            shutil.rmtree(dst, ignore_errors=True)
            os.replace(part, dst)
        except OSError as ex:                       # Drive full (or gone): a clear stop instead of a traceback
            shutil.rmtree(part, ignore_errors=True)
            raise SystemExit(drive_full_message(f"saving chunk {k + 1} to Drive", ex, k + 1))
        rec = data_after_chunk(state, data_out, data_start)
        state["done"] = k + 1
        state.setdefault("chunk_log", []).append({"chunk": k, "steps": [bounds[k], bounds[k + 1]],
                                                  "lr": [round(x, 6) for x in span], "optimizer": "loaded" if load_opt
                                                  else "fresh", "seed": seed_k,
                                                  "data": [data_start, data_start + rec["fetched"], rec["size"]]
                                                  if rec else None,
                                                  "minutes": round((time.time() - t0) / 60, 1)})
        if sps:
            state.setdefault("seconds_per_step", []).append(round(sps, 2))
        write_state(state_f, state)
        try:
            shutil.copy2(log, run_dir / "train.log")
        except OSError:
            pass
        old = run_dir / "chunks" / f"chunk_{k - 2:03d}"           # keep the last two chunks (Drive space)
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
        trained += 1
        mins = (time.time() - t0) / 60
        left = mins * (chunks - k - 1) / 60
        print(f"   chunk {k + 1}/{chunks} saved to Drive ({mins:.0f} min). {bounds[k + 1]} of {settings['steps']} "
              f"steps done; about {left:.1f} h to go." + (f" If the session ends now: new session, run cells 1 -> 2 "
                                                          f"-> 6 -> 6d -> 7; it continues at chunk {k + 2}."
                                                          if k + 1 < chunks else ""), flush=True)
        k += 1

    # every chunk done: the last one, checked finite, becomes the trained model (renamed to the run's step)
    last = run_dir / "chunks" / f"chunk_{chunks - 1:03d}"
    fstep = newest(last)
    status, bad = model_health(last, fstep)
    if status != "ok":
        raise SystemExit(f"the last chunk ({last}, step {fstep}) is not a usable model ({status}: "
                         f"{T.short_list(bad, 3)}); it is not installed. RESTART_SFT = True trains again.")
    ck = base_dir / "chatsft_checkpoints"
    if ck.exists() and any(ck.iterdir()):
        kept = aside_name(base_dir)
        shutil.move(str(ck), str(kept))
        print("a model that appeared in chatsft_checkpoints during the run is kept in", kept)
    part = base_dir / "chatsft_checkpoints.part"
    shutil.rmtree(part, ignore_errors=True)
    try:
        copy_model(last, fstep, part / tag, new_step=settings["steps"])
    except OSError as ex:
        shutil.rmtree(part, ignore_errors=True)
        raise SystemExit(drive_full_message("installing the finished model on Drive", ex, None))
    meta_f = part / tag / f"meta_{settings['steps']:06d}.json"
    m = json.loads(meta_f.read_text(encoding="utf-8"))
    m["step"] = settings["steps"]
    m["ooa_colab"] = {"steps": settings["steps"], "chunks": chunks, "seq": seq, "device_batch": device_batch,
                      "lr": settings["lr"], "seed": settings["seed"], "data": str(data), "data_sig": settings["data_sig"],
                      "base": settings["base"], "base_lrs": state["base_lrs"], "started": state.get("started"),
                      "finished": time.strftime("%Y-%m-%d %H:%M"), "nan_stops": len(state.get("nan_stops", []))}
    meta_f.write_text(json.dumps(m, indent=2), encoding="utf-8")
    shutil.rmtree(ck, ignore_errors=True)
    os.replace(part, ck)
    state.update(finished=True, installed=str(ck / tag), installed_step=settings["steps"],
                 finished_at=time.strftime("%Y-%m-%d %H:%M"))
    write_state(state_f, state)
    print(f"\nSFT FINISHED: all {chunks} chunks; the model (weights checked finite) is in {ck / tag} as step "
          f"{settings['steps']} (trained window {m.get('trained_window')}).")
    return {"status": "finished", "installed": str(ck / tag), "state": state, "trained_now": trained}


# ------------------------------------------------------------------------------------------- Colab cells 6 / 6d
CELL6_CODE = ("build_catalog.py", "build_sft_data.py", "gen_flows.py", "data_filter.py", "ooa_reward.py", "repair.py",
              "ooa_format.py", "safety.py")
CELL6D_CODE = ("render_dataset.py", "ooa_format.py", "retrieval.py", "repair.py")
CELL6_OUT = ("sft/catalog.jsonl", "sft/val.jsonl", "sft/eval.jsonl", "sft/eval_synth.jsonl", "sft/eval_pattern.jsonl",
             "sft/synth_train.jsonl", "sft/synth_pattern.jsonl", "sft/train_catalog.jsonl")
CELL6D_OUT = ("sft_render/render_report.json", "sft_render/examples.jsonl", "sft_render/val.jsonl",
              "sft_render/catalog.jsonl", "sft_render/train_catalog.jsonl")
STAMP6 = "sft/cell6_inputs.json"
STAMP6D = "sft_render/cell6d_inputs.json"


def code_sig(folder, names):
    """sha1 over the named code files in folder (what makes the data)."""
    h = hashlib.sha1()
    for n in sorted(names):
        p = Path(folder) / n
        h.update(n.encode() + b"\0" + (p.read_bytes() if p.exists() else b"(missing)") + b"\0")
    return h.hexdigest()[:12]


def missing_outputs(drive_dir, outputs):
    return [o for o in outputs if not (Path(drive_dir) / o).exists() or (Path(drive_dir) / o).stat().st_size == 0]


def data_skip(drive_dir, stamp, want, outputs, force=False, base_dir=None, restart=False):
    """May cell 6 / 6d skip its heavy work? -> (skip, why). Skipped when its outputs are on Drive and either were made
    from these inputs (the stamp it wrote last time equals want, compared without the platform commit, which is
    pinned to the stamp's) or an SFT run is in progress on them (base_dir/sft_run: rebuilding would change the data
    under it). FORCE_DATA (force) always rebuilds. RESTART_SFT (restart): cell 7 starts a new run (or, left on, its
    plan follows the settings), so the run in progress does not keep the data: data whose inputs changed is made
    again for the new run."""
    if force:
        return False, "FORCE_DATA = True: made again"
    miss = missing_outputs(drive_dir, outputs)
    if miss:
        return False, "not on Drive yet: " + ", ".join(miss[:4])
    old = read_state(Path(drive_dir) / stamp)
    if old and {k: v for k, v in old.items() if k not in ("platform", "time")} == \
            {k: v for k, v in want.items() if k not in ("platform", "time")}:
        return True, f"Drive already has it for the same inputs (made {old.get('time', '?')})"
    run = run_in_progress(base_dir) if base_dir and not restart else None
    if run:
        what = [k for k in sorted(set(old) | set(want)) if k not in ("platform", "time") and old.get(k) != want.get(k)]
        return True, (f"an SFT run is in progress on this data ({run.get('done', 0)} of {run.get('chunks')} chunks "
                      f"done), so it is kept for cell 7 to continue" + (f" (differs now in: {', '.join(what)})" if
                                                                        old and what else "") +
                      "; FORCE_DATA = True makes it again (cell 7 then needs RESTART_SFT = True)")
    return False, (("RESTART_SFT = True: the new run gets data made for these settings; " if restart and old else "")
                   + "its inputs changed: " + ", ".join(k for k in sorted(set(old) | set(want))
                                                     if k not in ("platform", "time") and old.get(k) != want.get(k))
                   if old else "no record of the inputs it was made from (made by an older notebook)")


def write_stamp(drive_dir, stamp, want):
    f = Path(drive_dir) / stamp
    f.parent.mkdir(parents=True, exist_ok=True)
    write_state(f, dict(want, time=time.strftime("%Y-%m-%d %H:%M")))


def clear_stamp(drive_dir, stamp):
    """Before the work starts again: a half-made output is never taken for a finished one."""
    try:
        (Path(drive_dir) / stamp).unlink()
    except OSError:
        pass
