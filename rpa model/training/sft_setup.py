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
   (d) per-file consumption is counted and printed at the end.
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
            '                "trained_window": orig_model.window_sizes[-1][0], # OOA: the window SFT trained with\n')
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
    for a in (WINDOW_ANCHOR, META_ANCHOR, TASKS_START, TASKS_END, IMPORT_ANCHOR, LOADER_OLD, STEP_ANCHOR, END_ANCHOR):
        assert s.count(a) >= 1, "chat_sft.py changed upstream, cannot patch: " + a[:60]
    s = s.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + "\nfrom tasks.ooa import OOADialogues", 1)
    a = s.index(TASKS_START)
    b = s.index("\n", s.index(TASKS_END))
    s = s[:a] + tasks_block(sft_dir, views, smoltalk_rows, seed) + s[b:]
    if window_fix:
        s = s.replace(WINDOW_ANCHOR, WINDOW_FIX + WINDOW_ANCHOR, 1)
        s = s.replace(META_ANCHOR, META_FIX, 1)
    s = s.replace(LOADER_OLD, LOADER_NEW, 1)
    s = s.replace(STEP_ANCHOR, "skipped_long = [0]\n" + STEP_ANCHOR, 1)
    s = s.replace(END_ANCHOR, END_REPORT + END_ANCHOR, 1)
    return s


def plan_for_notebook(sft_dir, steps, tokens_per_step, smoltalk_rows=0, smoltalk_tokens=500, want=None):
    """(views, stats, budget, table lines) for cell 7."""
    stats = file_stats(sft_dir)
    b = budget(steps, tokens_per_step)
    if smoltalk_rows:
        b -= int(smoltalk_rows * smoltalk_tokens)
    views = plan_mixture(stats, b, want)
    return views, stats, b, mixture_table(stats, views, b)



# ------------------------------------------------------------------ the local PC notebook (training/local/train_pc.py)
# The neutral-record files of Drive sft/ that Colab cell 6d renders into sft_render (cell 1 NEW_FILES; the local
# notebook renders the same files when Drive has no sft_render). test_train_pc.py checks they match cell 1.
RENDER_EXTRA = ("long_train.jsonl", "business2_train.jsonl", "targeted_train.jsonl", "fix_train.jsonl",
                "clarify_train.jsonl", "beginner_train.jsonl")
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


def plan_for_local(sft_dir, mode, steps, tokens_per_step, smoltalk_rows=0, smoltalk_tokens=500):
    """(views, stats, budget, table lines) for the local notebook: "full" plans exactly like Colab cell 7
    (plan_for_notebook); "topup" plans a top-up round (plan_topup)."""
    if mode == "full":
        return plan_for_notebook(sft_dir, steps, tokens_per_step, smoltalk_rows, smoltalk_tokens)
    stats = file_stats(sft_dir)
    b = budget(steps, tokens_per_step)
    if smoltalk_rows:
        b -= int(smoltalk_rows * smoltalk_tokens)
    views = plan_topup(stats, b)
    return views, stats, b, mixture_table(stats, views, b)


if __name__ == "__main__":
    import sys
    d, steps, tps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    v, st, b, lines = plan_for_notebook(d, steps, tps)
    print("\n".join(lines))
    print(json.dumps(v))
