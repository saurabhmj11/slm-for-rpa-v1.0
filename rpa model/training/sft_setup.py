"""How the Colab notebook (cell 7) sets up nanochat's SFT for our data: the per-file data mixture for the real step
budget, and the patch of scripts/chat_sft.py (pinned nanochat commit 92d63d4). Standard library only, so it is tested
locally (test_sft_setup.py) and copied into the notebook like the other training modules.

1. plan_mixture: how many times each training file is seen (views per row), worked out from the SFT budget
   (optimizer steps x tokens per step x packing efficiency) instead of "epochs x repeats". With the old list the
   4000-step run used 5% of a 1.4M-conversation list: long rows were seen 4.6 times, generated flows 0.3 times (73%
   never) and catalog questions about as rarely. Here every file gets a wanted number of views (WANT), long and
   business rows at most LONG_CAP views, catalog questions at least CATALOG_MIN, and the rest is scaled to fill the
   budget.
2. patch_chat_sft: (a) the SFT window. chat_sft builds the model from the BASE checkpoint's config, so the attention
   window ("L" = sequence_len, gpt.py _compute_window_sizes) stays the pretraining length (1024 on the t4 profile)
   while the saved SFT config says --max-seq-len (4096): the served model then looks 4096 tokens back, which it never
   trained for (loss past 2048 was 100-400x higher, JSON broke near token 2060). The patch sets the config's
   sequence_len to --max-seq-len right after loading and recomputes the window sizes (and the rotary table when it is
   too short), prints the windows, and saves "trained_window" in the checkpoint meta for the serving check
   (assistant/model_backend.window_check). (b) our data mixture, (c) rows longer than the window are skipped, not cut,
   (d) per-file consumption is counted and printed at the end, (e) a step guard: a step whose micro-batch loss or
   unscaled gradients are not finite is skipped (the model unchanged, an "OOA SKIP" line), and float16's GradScaler
   scale is floored at SCALE_FLOOR (a collapsed scale made the unscale write nan into every weight unnoticed), (f) a
   NaN guard: the run stops (exit code NAN_EXIT, before anything is saved) after NAN_STEPS such steps in a row,
   SKIP_STOP skipped steps in a row, or a validation bpb that is not a number, saying whether the weights themselves
   are finite, so a run that went unstable never saves a broken model over a good one.
3. nonfinite_tensors / check_model_dir: which tensors of a saved model or optimizer state hold nan / inf (torch is
   imported only there: Colab cells 7-8 and the PC scripts check a model before they train, test or install it).
"""
import json
import os

# wanted views per row (before scaling to the budget). Long / business rows are long and few: at most LONG_CAP.
WANT = {
    "train_flows.jsonl": 5,          # the platform's own flows: few and real
    "train_catalog.jsonl": 3,        # catalog questions: short; each action gets 3+ views (CATALOG_MIN)
    "synth_train.jsonl": 1,          # 30k generated flows
    "synth_pattern.jsonl": 1.5,      # long generated flows following real-flow patterns
    "natural_train.jsonl": 3, "core_train.jsonl": 3, "chains_train.jsonl": 3, "combos_train.jsonl": 3,
    "notask_train.jsonl": 3, "tools_train.jsonl": 3,
    "scenarios_train.jsonl": 3, "long_train.jsonl": 5.5, "business2_train.jsonl": 5.5,
    "targeted_train.jsonl": 3, "fix_train.jsonl": 2, "clarify_train.jsonl": 3,
    "self_improved.jsonl": 2, "hard_fixed.jsonl": 2, "feedback_train.jsonl": 3,
    "beginner_train.jsonl": 4,       # beginner wording (gen_beginner.py): restate, then plan; friendly questions
    "v5_train.jsonl": 6,             # v5 hand-written: business words, app / action choice, Hinglish & voice, asking
    "session_train.jsonl": 5,        # sign in / open once, later steps use the session, sign out / close at the end
}
LONG_FILES = ("long_train.jsonl", "business2_train.jsonl", "scenarios_train.jsonl", "synth_pattern.jsonl")
LONG_CAP = 3.0
# the business processes the model fails on get more practice than the other long files (compact answers made
# them about 45% shorter, so this costs fewer tokens than before); the 4000-step run had seen them ~4.6 times
PROCESS_FILES = ("long_train.jsonl", "business2_train.jsonl")
PROCESS_CAP = 5.5
CATALOG_MIN = 3.0
MAX_VIEWS = 8.0
DEFAULT_WANT = 2.0
PACKING = 0.9                        # share of each 4096-token row that holds conversation (best-fit packing)
MARGIN = 1.1                         # the list is a little longer than the budget: chat_sft stops early when its
                                     # list runs out, so the step count, not the list, must end training
CHARS_PER_TOKEN = 2.8                # rendered rows (measured 2.7-3.0 with the 16k tokenizer)
SKIP = ("catalog.jsonl", "examples.jsonl", "val.jsonl", "render_report.json", "rl_train.jsonl", "rl_val.jsonl")


def file_stats(folder, names=None, chars_per_token=CHARS_PER_TOKEN):
    """{file name: (rows, mean tokens per row)} for the training files in folder: the files named in WANT and any other
    *_train.jsonl (a new data file gets DEFAULT_WANT views). Tokens are estimated from characters."""
    out = {}
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".jsonl") or name in SKIP or name.startswith(("val", "eval")):
            continue
        if name not in WANT and not name.endswith("_train.jsonl"):
            continue                       # train.jsonl (the union of the split files), logs, queues
        if names is not None and name not in names:
            continue
        rows = chars = 0
        with open(os.path.join(folder, name), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows += 1
                    chars += len(line)
        if rows:
            out[name] = (rows, chars / rows / chars_per_token)
    return out


def plan_mixture(stats, budget_tokens, want=None, long_cap=LONG_CAP, catalog_min=CATALOG_MIN, max_views=MAX_VIEWS):
    """Views per row for each file so the whole list fills budget_tokens (the tokens the SFT steps will read).
    stats: {name: (rows, mean tokens)}. Files get want[name] views (DEFAULT_WANT when not listed), times one common
    scale; long files never more than long_cap, the catalog never fewer than catalog_min, no file more than
    max_views. Returns {name: views} (floats)."""
    want = dict(WANT, **(want or {}))
    fixed, views = {}, {n: float(want.get(n, DEFAULT_WANT)) for n in stats}

    def cap(n):
        if n in PROCESS_FILES:
            return max(long_cap, PROCESS_CAP)
        return long_cap if n in LONG_FILES else max_views

    def floor(n):
        return catalog_min if n == "train_catalog.jsonl" else 0.0

    def tokens(vs):
        return sum(vs[n] * stats[n][0] * stats[n][1] for n in vs)

    for _ in range(20):                                  # scale the free files; pin the ones that hit a bound
        free = [n for n in views if n not in fixed]
        if not free:
            break
        rest = budget_tokens - sum(fixed[n] * stats[n][0] * stats[n][1] for n in fixed)
        base = sum(float(want.get(n, DEFAULT_WANT)) * stats[n][0] * stats[n][1] for n in free)
        s = rest / base if base > 0 else 1.0
        changed = False
        for n in free:
            v = float(want.get(n, DEFAULT_WANT)) * s
            if v > cap(n):
                fixed[n], changed = cap(n), True
            elif v < floor(n):
                fixed[n], changed = floor(n), True
            else:
                views[n] = v
        if not changed:
            break
    views.update(fixed)
    # every file at its cap and still short of the budget: the list must not run out before the last step (chat_sft
    # stops when it does), so the files that are not long ones go over max_views
    short = budget_tokens - tokens(views)
    more = sum(views[n] * stats[n][0] * stats[n][1] for n in views if n not in LONG_FILES)
    if short > 0.01 * budget_tokens and more > 0:
        s = 1 + short / more
        views = {n: (v * s if n not in LONG_FILES else v) for n, v in views.items()}
    return {n: round(max(v, 0.01), 3) for n, v in views.items()}


def mixture_table(stats, views, budget_tokens):
    """Lines of text: per file rows, tokens per row, views per row, share of the training tokens."""
    total = sum(views[n] * stats[n][0] * stats[n][1] for n in views) or 1
    lines = [f"{'file':26s} {'rows':>7s} {'tok/row':>7s} {'views':>6s} {'%tokens':>7s}"]
    for n in sorted(views, key=lambda x: -views[x] * stats[x][0] * stats[x][1]):
        lines.append(f"{n:26s} {stats[n][0]:7d} {stats[n][1]:7.0f} {views[n]:6.2f} "
                     f"{100 * views[n] * stats[n][0] * stats[n][1] / total:7.1f}")
    lines.append(f"list tokens {total / 1e6:.1f}M for a budget of {budget_tokens / 1e6:.1f}M "
                 f"({100 * min(1.0, budget_tokens / total):.0f}% of the list fits)")
    return lines


def budget(steps, tokens_per_step, packing=PACKING, margin=MARGIN):
    """Tokens of conversation the SFT steps read (steps x tokens per step x packing), times margin."""
    return int(steps * tokens_per_step * packing * margin)


# ---------------------------------------------------------------------------------------------- chat_sft.py patch
WINDOW_ANCHOR = "orig_model = model\nmodel = torch.compile(model, dynamic=False)\n"
WINDOW_FIX = '''# OOA: train with the attention window the checkpoint is served with. The model is built from the BASE config, whose
# sequence_len (and so the "L" window, gpt.py _compute_window_sizes) is the pretraining length; the SFT checkpoint
# saves --max-seq-len, so without this the served window would be one the model never trained with.
_base_seq = model.config.sequence_len
if _base_seq != args.max_seq_len:
    model.config.sequence_len = args.max_seq_len
    model.window_sizes = model._compute_window_sizes(model.config)
    if model.rotary_seq_len < args.max_seq_len * 10:
        _hd = model.config.n_embd // model.config.n_head
        model.rotary_seq_len = args.max_seq_len * 10
        model.cos, model.sin = model._precompute_rotary_embeddings(model.rotary_seq_len, _hd, device=model.cos.device)
assert model.window_sizes[-1][0] == args.max_seq_len, (model.window_sizes, args.max_seq_len)
print0(f"SFT attention window: base config sequence_len {_base_seq} -> {model.config.sequence_len}; "
       f"window per layer {[w for w, _ in model.window_sizes]}; rotary table {model.rotary_seq_len}")
'''
META_ANCHOR = '                "user_config": user_config, # inputs to the training script\n'
META_FIX = ('                "user_config": user_config, # inputs to the training script\n'
            '                "trained_window": orig_model.window_sizes[-1][0], # OOA: the window SFT trained with\n'
            '                "compute_dtype": str(COMPUTE_DTYPE).replace("torch.", ""), # OOA: bfloat16 / float16\n'
            '                "tokenizer_sha1": _ooa_tok_sha1(), # OOA: the model pack must carry this tokenizer\n')
TOK_SHA = '''
def _ooa_tok_sha1():   # OOA: the tokenizer this model trains with (model_backend.tokenizer_problem checks it)
    import hashlib as _h
    from nanochat.common import get_base_dir as _gbd
    with open(_os_ooa.path.join(_gbd(), "tokenizer", "tokenizer.pkl"), "rb") as _f:
        return _h.sha1(_f.read()).hexdigest()'''
TASKS_START = "train_tasks = ["
TASKS_END = "# total: 24K"
IMPORT_ANCHOR = "from tasks.smoltalk import SmolTalk"
LOADER_OLD = "            ids, mask = tokenizer.render_conversation(conversation)\n            conv_buffer.append((ids, mask))"
LOADER_NEW = ("            ids, mask = tokenizer.render_conversation(conversation, max_tokens=10**9)\n"
              "            if split == \"train\":\n"
              "                _seen[train_dataset.index_map[cursor][0]] += 1\n"
              "            if len(ids) <= row_capacity:\n"
              "                conv_buffer.append((ids, mask))\n"
              "            else:\n"
              "                skipped_long[0] += 1")
STEP_ANCHOR = "last_step = False # we will toggle this"
END_ANCHOR = 'print0(f"Total training time:'
END_REPORT = '''print0(f"Conversations skipped as longer than the window (not cut): {skipped_long[0]}")
print0("Training data actually read (rows fetched / rows in the file = views per row):")
for _i, (_name, _rows) in enumerate(_task_files):
    print0(f"  {_name:28s} {_seen[_i]:8d} / {_rows:6d} = {_seen[_i] / max(1, _rows):5.2f} views")
'''


NAN_STEPS = 5                        # steps in a row whose loss / gradients are not a number -> stop
SKIP_STOP = 50                       # steps in a row skipped for any reason (float16 overflow too) -> stop
SCALE_FLOOR = 1.0                    # float16: the GradScaler's scale is never let below this (see the step guard)
NAN_EXIT = 86                        # chat_sft's exit code then (train_pc.py and Colab cell 7 recognise it)
NAN_MARK = "OOA NAN STOP"            # the line it prints first
SKIP_MARK = "OOA SKIP"               # a training step that was skipped (nothing in the model changed)
LOSS_LOG_ANCHOR = '    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f}'
VAL_ANCHOR = '        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")\n'

# --- the step guard (float16 with GradScaler, and bfloat16 / float32) -----------------------------------------------
# Why (the Colab T4 float16 incident: chunk 4 nan from step ~3447; the v5 run that ended with ALL weights nan):
#  - GradScaler checks the SCALED gradients for inf/nan, then multiplies them by 1/scale; the result is never checked.
#    Every skipped step halves the scale (it doubles only after 2000 clean steps), so frequent float16 overflows push
#    it down. At scale <= 2^-128, 1/scale is inf in float32 and the unscale turns zero gradients into nan WITHOUT
#    flagging the step: the optimizer then writes nan into every weight. Already at scale <= 2^-88 the few gradient
#    elements that did not underflow come back huge (6e-8 / 2^-110 ~ 1e26): AdamW's grad^2 overflows to inf, and its
#    state and weights go nan on the next step. -> the scale is floored at SCALE_FLOOR, and the gradients are checked
#    AFTER the unscale (their total norm): a step whose gradients are not finite is skipped whatever the scaler said.
#  - chat_sft looked at the LAST micro-batch's loss only: an overflow in micro-batches 1-3 skipped a step silently and
#    never counted for the NaN guard, and the printed loss (a running average) stayed nan for good after one nan step.
#    -> every micro-batch's loss is checked; a step with one that is not a number is skipped and printed (SKIP_MARK:
#    which micro-batch, the gradient norm, the scale), and the running average leaves it out.
#  - no gradient clipping: what broke the weights is the scale collapse above, not large real gradients (Muon
#    normalises its updates anyway); clipping would change the training against the chunks already trained.
# A skipped step changes nothing in the model or the optimizer state (no optimizer step, the gradients are dropped);
# with float16 the scale still drops (scaler.update with the step marked as overflowed), never below the floor.
MICRO_ANCHOR = ("    for micro_step in range(grad_accum_steps):\n        loss = model(x, y)\n"
                "        train_loss = loss.detach() # for logging\n")
MICRO_NEW = ("    _ooa_losses = []   # OOA: every micro-batch's loss (chat_sft looked at the last one only)\n"
             "    for micro_step in range(grad_accum_steps):\n        loss = model(x, y)\n"
             "        if (step + 1, micro_step + 1) in _ooa_inject:   # OOA test hook (env OOA_TEST_NAN): a nan loss\n"
             "            loss = loss * float(\"nan\")\n"
             "        _ooa_losses.append(loss.detach().float())\n"
             "        train_loss = loss.detach() # for logging\n")
OPT_ANCHOR = """    if scaler is not None:
        scaler.unscale_(optimizer)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
"""
OPT_NEW = """    _ooa_found = 0.0
    if scaler is not None:
        scaler.unscale_(optimizer)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        if scaler.is_enabled():
            _ooa_found = float(sum(v.item() for v in scaler._found_inf_per_device(optimizer).values()))
    # OOA step guard (sft_setup): the micro-batch losses and the UNSCALED gradients must be finite, else no step
    _ooa_g = [p.grad.detach() for p in orig_model.parameters() if p.grad is not None]
    _ooa_gn = (torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float()) for g in _ooa_g]))
               if _ooa_g else torch.zeros((), device=device))
    _ooa_v = torch.cat([torch.stack(_ooa_losses), _ooa_gn.reshape(1).float()])
    _ooa_flags = torch.stack([(~torch.isfinite(_ooa_v[:-1])).any(), ~torch.isfinite(_ooa_v[-1])]).float()
    if is_ddp_initialized():
        dist.all_reduce(_ooa_flags, op=dist.ReduceOp.MAX)   # every rank skips the same steps
    _ooa_vals = _ooa_v.tolist()
    _ooa_loss_bad, _ooa_grad_bad = bool(_ooa_flags[0].item()), bool(_ooa_flags[1].item())
    _ooa_gnorm = _ooa_vals[-1]
    # counts for the NaN stop: a loss that is not a number, or gradients that are not finite although the scaler saw
    # no overflow; a float16 overflow the scaler caught (finite losses) is its normal working: only skipped
    _ooa_bad_step = _ooa_loss_bad or (_ooa_grad_bad and _ooa_found == 0)
    if not _ooa_loss_bad:
        _ooa_step_loss[0] = sum(_ooa_vals[:-1]) / len(_ooa_vals[:-1])   # the step's loss: mean of its micro-batches
    if _ooa_loss_bad or _ooa_grad_bad or _ooa_found > 0:
        _ooa_skipped[0] += 1
        _ooa_skip_run[0] += 1
        _ooa_scale_was = scaler.get_scale() if scaler is not None and scaler.is_enabled() else None
        if _ooa_scale_was is not None:
            for v in scaler._found_inf_per_device(optimizer).values():
                v.fill_(1.0)                                 # marked as overflowed: update() lowers the scale
            scaler.update()
            if scaler.get_scale() < SCALE_FLOOR:
                scaler.update(new_scale=SCALE_FLOOR)         # never below the floor: 1/scale stays sane
        _ooa_mb = [i + 1 for i, v in enumerate(_ooa_vals[:-1]) if not _ooa_math.isfinite(v)]
        print0("SKIP_MARK step " + f"{step + 1:05d}: "
               + (f"loss not a number in micro-batch {_ooa_mb} of {grad_accum_steps}; " if _ooa_mb else "")
               + f"gradient norm {_ooa_gnorm:.4g}"
               + (f"; float16 overflow caught by GradScaler" if _ooa_found > 0 and not _ooa_mb else "")
               + (f"; scale {_ooa_scale_was:g} -> {scaler.get_scale():g}" if _ooa_scale_was is not None else "")
               + f": step skipped, the model is unchanged ({_ooa_skipped[0]} skipped in this run, "
               f"{_ooa_skip_run[0]} in a row)")
    else:
        _ooa_skip_run[0] = 0
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    if scaler is not None and scaler.is_enabled() and scaler.get_scale() < SCALE_FLOOR:
        scaler.update(new_scale=SCALE_FLOOR)   # never below the floor: 1/scale stays sane
    model.zero_grad(set_to_none=True)
""".replace("SKIP_MARK", SKIP_MARK).replace("SCALE_FLOOR", repr(SCALE_FLOOR))
EMA_OLD = ("    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() "
           "# EMA the training loss\n")
EMA_NEW = ("    if not _ooa_loss_bad:   # OOA: a skipped step's nan never enters the running average (mean of the "
           "micro-batches)\n"
           "        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * _ooa_step_loss[0]\n")
LOSS_TAIL_OLD = '| epoch: {current_epoch} | total time: {total_training_time/60:.2f}m")'
LOSS_TAIL_NEW = ('| epoch: {current_epoch} | total time: {total_training_time/60:.2f}m | gnorm: {_ooa_gnorm:.3g}'
                 ' | scale: {scaler.get_scale() if scaler is not None and scaler.is_enabled() else 1:g}'
                 ' | skipped: {_ooa_skipped[0]}")')
STEP_SETUP = ('skipped_long = [0]\n_ooa_bad_loss = [0]\n_ooa_skipped = [0]\n_ooa_skip_run = [0]\n'
              '_ooa_step_loss = [0.0]\n_ooa_loss_bad = False\n_ooa_gnorm = 0.0\n'
              '_ooa_inject = {tuple(int(v) for v in _p.split(":")) for _p in os.environ.get("OOA_TEST_NAN", "")'
              '.split(",") if _p}\n')
WEIGHT_NOTE = '''
def _ooa_weight_note(m):   # OOA: are the weights themselves broken, or only this computation?
    import torch as _t
    _bad = [n for n, p in m.named_parameters() if not bool(_t.isfinite(p.detach()).all())]
    if _bad:
        return f"{len(_bad)} weight tensor(s) hold nan/inf: " + ", ".join(_bad[:6])
    _big = max(((float(p.detach().abs().max()), n) for n, p in m.named_parameters()), default=(0.0, "-"))
    return (f"the weights are all finite (largest |w| {_big[0]:.4g} in {_big[1]}): the {COMPUTE_DTYPE} computation "
            "overflowed, the model itself is not nan")'''
LOSS_GUARD = f'''    # OOA: a step whose loss or gradients are not a number was skipped above (the model is unchanged); {NAN_STEPS} such
    # steps in a row (or {SKIP_STOP} skipped steps in a row for any reason) mean the run went unstable: stop before
    # anything is saved (a checkpoint is written only at the last step), so a broken model never replaces a good one.
    _ooa_bad_loss[0] = _ooa_bad_loss[0] + 1 if _ooa_bad_step else 0
    if _ooa_bad_loss[0] >= {NAN_STEPS} or _ooa_skip_run[0] >= {SKIP_STOP}:
        print0(f"{NAN_MARK}: " + (f"the training loss / gradients were not a number (nan/inf) for {NAN_STEPS} steps in "
               f"a row (step {{step}})" if _ooa_bad_loss[0] >= {NAN_STEPS} else f"{SKIP_STOP} steps in a row were "
               f"skipped (step {{step}}: float16 overflow the GradScaler could not get under)")
               + ": the training went unstable; " + _ooa_weight_note(orig_model) + ". Stopped; nothing was saved.")
        raise SystemExit({NAN_EXIT})
'''
VAL_GUARD = f'''        if not _ooa_math.isfinite(val_bpb):   # OOA: the model gives nan / inf: stop before anything is saved
            print0(f"{NAN_MARK}: Validation bpb is not a number ({{val_bpb}}) at step {{step}}"
                   + (": the start model is broken before any training step" if step == 0 else
                      ": the training went unstable") + "; " + _ooa_weight_note(orig_model)
                   + ". Stopped; nothing was saved.")
            raise SystemExit({NAN_EXIT})
'''


def tasks_block(sft_dir, views, smoltalk_rows=0, seed=1):
    """The code that replaces chat_sft's train / val task lists: one OOADialogues per file with its views."""
    lines = ["import os as _os",
             "_task_files, train_tasks = [], []"]
    for n, v in sorted(views.items()):
        p = f"{sft_dir}/{n}"
        lines.append(f"if _os.path.exists({p!r}):")
        lines.append(f"    _t = OOADialogues({p!r}, views={v!r}, seed={seed})")
        lines.append(f"    train_tasks.append(_t); _task_files.append(({n!r}, _t.num_rows))")
    if smoltalk_rows > 0:
        lines.append(f"train_tasks.append(SmolTalk(split=\"train\", stop={int(smoltalk_rows)})); "
                     f"_task_files.append((\"smoltalk\", {int(smoltalk_rows)}))")
    lines += ["_seen = [0] * len(train_tasks)",
              "train_dataset = TaskMixture(train_tasks)",
              'print0(f"OOA training mixture: {len(train_dataset):,} rows from {len(train_tasks)} files")',
              f"val_dataset = TaskMixture([OOADialogues({sft_dir + '/val.jsonl'!r})])"]
    return "\n".join(lines)


def patch_chat_sft(s, sft_dir, views, smoltalk_rows=0, seed=1, window_fix=True):
    """chat_sft.py text (pinned nanochat) -> the patched text. Raises AssertionError when an anchor is missing
    (nanochat changed upstream): the notebook must stop rather than train without the window fix.
    window_fix=False leaves the window as nanochat has it (only for the comparison test)."""
    for a in (WINDOW_ANCHOR, META_ANCHOR, TASKS_START, TASKS_END, IMPORT_ANCHOR, LOADER_OLD, STEP_ANCHOR, END_ANCHOR,
              LOSS_LOG_ANCHOR, VAL_ANCHOR):
        assert s.count(a) >= 1, "chat_sft.py changed upstream, cannot patch: " + a[:60]
    for a in (MICRO_ANCHOR, OPT_ANCHOR, EMA_OLD, LOSS_TAIL_OLD):          # the step guard: each exactly once
        assert s.count(a) == 1, "chat_sft.py changed upstream, cannot patch: " + a.strip()[:60]
    s = s.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + "\nfrom tasks.ooa import OOADialogues\nimport math as _ooa_math"
                  + ("\nimport os as _os_ooa" + TOK_SHA if window_fix else "") + WEIGHT_NOTE, 1)
    a = s.index(TASKS_START)
    b = s.index("\n", s.index(TASKS_END))
    s = s[:a] + tasks_block(sft_dir, views, smoltalk_rows, seed) + s[b:]
    if window_fix:
        s = s.replace(WINDOW_ANCHOR, WINDOW_FIX + WINDOW_ANCHOR, 1)
        s = s.replace(META_ANCHOR, META_FIX, 1)
    s = s.replace(LOADER_OLD, LOADER_NEW, 1)
    s = s.replace(STEP_ANCHOR, STEP_SETUP + STEP_ANCHOR, 1)
    for a, b in ((MICRO_ANCHOR, MICRO_NEW), (OPT_ANCHOR, OPT_NEW), (EMA_OLD, EMA_NEW), (LOSS_TAIL_OLD, LOSS_TAIL_NEW)):
        s = s.replace(a, b, 1)
    s = s.replace(END_ANCHOR, END_REPORT + END_ANCHOR, 1)
    i = s.index("\n", s.index(LOSS_LOG_ANCHOR)) + 1                  # after the whole "step ... | loss:" line
    s = s[:i] + LOSS_GUARD + s[i:]
    s = s.replace(VAL_ANCHOR, VAL_ANCHOR + VAL_GUARD, 1)
    return s


# ---------------------------------------------------------------------------------------------- broken weights
def nonfinite_tensors(path):
    """Names of the floating-point tensors with nan / inf in a saved model_<N>.pt or optim_<N>_rank0.pt (nested
    optimizer state is walked: "state.3.exp_avg"), each with "(nan)" or "(inf)". [] = all finite. Loaded on the CPU
    with weights_only=True. Raises what torch.load raises for a file that is not a saved state."""
    import torch
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    bad = []

    def walk(name, v):
        if torch.is_tensor(v):
            if v.is_floating_point() and not bool(torch.isfinite(v).all()):
                bad.append(f"{name} ({'nan' if bool(torch.isnan(v).any()) else 'inf'})")
        elif isinstance(v, dict):
            for k, x in v.items():
                walk(f"{name}.{k}" if name else str(k).removeprefix("_orig_mod."), x)
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                walk(f"{name}[{i}]", x)
    walk("", sd)
    return bad


def newest_step(d8):
    """Highest N with model_N.pt and meta_N.json in d8, or None."""
    import re
    best = None
    if os.path.isdir(d8):
        for f in os.listdir(d8):
            m = re.match(r"model_(\d+)\.pt$", f)
            if m and os.path.exists(os.path.join(d8, f"meta_{m.group(1)}.json")):
                best = max(best or 0, int(m.group(1)))
    return best


def check_model_dir(d8):
    """(step, names of broken tensors) of the newest model in a chatsft_checkpoints/d8 folder; (None, None) when
    there is no model there."""
    step = newest_step(str(d8))
    if step is None:
        return None, None
    return step, nonfinite_tensors(os.path.join(str(d8), f"model_{step:06d}.pt"))


def short_list(names, n=6):
    """At most n names, then "and K more"."""
    return ", ".join(names[:n]) + (f" and {len(names) - n} more" if len(names) > n else "")


def plan_for_notebook(sft_dir, steps, tokens_per_step, smoltalk_rows=0, smoltalk_tokens=500, want=None,
                      chars_per_token=None):
    """(views, stats, budget, table lines) for cell 7. chars_per_token: characters of a JSONL line per token of the
    base model's tokenizer (its tokenizer_card.json, base_setup.plan_chars_per_token); None = CHARS_PER_TOKEN, the
    old web-only tokenizer's."""
    stats = file_stats(sft_dir, chars_per_token=chars_per_token or CHARS_PER_TOKEN)
    b = budget(steps, tokens_per_step)
    if smoltalk_rows:
        b -= int(smoltalk_rows * smoltalk_tokens)
    views = plan_mixture(stats, b, want)
    return views, stats, b, mixture_table(stats, views, b)



# ------------------------------------------------------------------ the local PC notebook (training/local/train_pc.py)
# The neutral-record files of Drive sft/ that Colab cell 6d renders into sft_render (cell 1 NEW_FILES; the local
# notebook renders the same files when Drive has no sft_render). test_train_pc.py checks they match cell 1.
RENDER_EXTRA = ("long_train.jsonl", "business2_train.jsonl", "targeted_train.jsonl", "fix_train.jsonl",
                "clarify_train.jsonl", "beginner_train.jsonl", "v5_train.jsonl", "session_train.jsonl")
# A top-up continues the newest SFT model for a few hundred steps on a CPU (about 1/25 of the Colab budget). Spread
# like the full mix, every file would get about 0.1-0.3 views and the round would teach little; so a top-up spends
# most of its tokens where the model is weakest and the data is newest: beginner wording (restate, then plan; ask
# when a detail only the user knows is missing) and the long business processes, then the clarify / targeted / fix
# files. TOPUP_REPLAY of the tokens replay all the other files in the full mix's proportions, so nothing learned is
# forgotten. Shares of files that are missing go to the others in proportion.
TOPUP_SHARES = {"beginner_train.jsonl": 0.35, "long_train.jsonl": 0.20, "business2_train.jsonl": 0.20,
                "clarify_train.jsonl": 0.06, "targeted_train.jsonl": 0.04, "fix_train.jsonl": 0.03,
                "feedback_train.jsonl": 0.02}
TOPUP_REPLAY = 0.10
# A refine (notebook MODE "refine", NVIDIA GPU) continues the finished Colab model (v5: 12000 steps) for REFINE_STEPS
# (train_pc.py: 1500, about 1/8 of the Colab budget). Its tokens go first to what is newest and was seen least: the v5
# hand-written rows and the session rows (made after the v5 run, so the v5 model never saw them), then the weak areas
# (beginner wording, the long business processes, clarify / targeted / fix, the user's own feedback). REFINE_REPLAY of
# the tokens replay every other file in the full mix's proportions so nothing learned is forgotten (the shares of
# missing files go to the others in proportion, so with no feedback the replay is about 18.6%).
REFINE_SHARES = {"v5_train.jsonl": 0.14, "session_train.jsonl": 0.08, "beginner_train.jsonl": 0.20,
                 "long_train.jsonl": 0.13, "business2_train.jsonl": 0.08, "clarify_train.jsonl": 0.05,
                 "targeted_train.jsonl": 0.06, "fix_train.jsonl": 0.05, "feedback_train.jsonl": 0.03}
REFINE_REPLAY = 0.18


def plan_topup(stats, budget_tokens, shares=None, replay=TOPUP_REPLAY):
    """Views per row for a top-up round: the files in shares get that share of budget_tokens, every other file shares
    replay in proportion to WANT x its tokens (the full mix's proportions). Then plan_mixture applies the usual caps
    (long files, MAX_VIEWS) and scales to the budget; no catalog floor (a top-up is too short for it)."""
    shares = dict(TOPUP_SHARES if shares is None else shares)
    focus = {n: s for n, s in shares.items() if n in stats}
    others = [n for n in stats if n not in focus]
    total = sum(focus.values()) + (replay if others else 0.0)
    if total <= 0:
        return plan_mixture(stats, budget_tokens, catalog_min=0.0)
    want = {}
    for n, s in focus.items():
        want[n] = s / total * budget_tokens / max(1.0, stats[n][0] * stats[n][1])
    if others:
        weight = {n: float(WANT.get(n, DEFAULT_WANT)) * stats[n][0] * stats[n][1] for n in others}
        wsum = sum(weight.values()) or 1.0
        for n in others:
            share = replay / total * weight[n] / wsum
            want[n] = share * budget_tokens / max(1.0, stats[n][0] * stats[n][1])
    return plan_mixture(stats, budget_tokens, want, catalog_min=0.0)


def plan_for_local(sft_dir, mode, steps, tokens_per_step, smoltalk_rows=0, smoltalk_tokens=500, chars_per_token=None):
    """(views, stats, budget, table lines) for the local notebook: "full" plans exactly like Colab cell 7
    (plan_for_notebook); "topup" plans a top-up round (plan_topup with TOPUP_SHARES); "refine" a GPU refine of the
    finished Colab model (plan_topup with REFINE_SHARES and REFINE_REPLAY)."""
    if mode == "full":
        return plan_for_notebook(sft_dir, steps, tokens_per_step, smoltalk_rows, smoltalk_tokens,
                                 chars_per_token=chars_per_token)
    stats = file_stats(sft_dir, chars_per_token=chars_per_token or CHARS_PER_TOKEN)
    b = budget(steps, tokens_per_step)
    if smoltalk_rows:
        b -= int(smoltalk_rows * smoltalk_tokens)
    if mode == "refine":
        views = plan_topup(stats, b, REFINE_SHARES, REFINE_REPLAY)
    else:
        views = plan_topup(stats, b)
    return views, stats, b, mixture_table(stats, views, b)


if __name__ == "__main__":
    import sys
    d, steps, tps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    v, st, b, lines = plan_for_notebook(d, steps, tps)
    print("\n".join(lines))
    print(json.dumps(v))
