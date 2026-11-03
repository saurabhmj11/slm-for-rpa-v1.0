"""
evaluate.py — 1000-Case Accuracy Evaluation for Praman Brain Mini Layer 1

Generates 1000 test trajectories (seed=123, different from training seed=42),
runs model inference on the prompt portion only, then scores accuracy on
5 dimensions per output:

  1. Route accuracy    — did it pick the right agent?
  2. Ticker accuracy   — did it copy the right ticker into the tool call?
  3. Event accuracy    — did it identify the right event type?
  4. Observation acc.  — did it produce the right sentiment/impact/signal?
  5. Full accuracy     — all 4 correct simultaneously

Prints a live progress bar every 100 cases and a final report table.

Usage:
    python evaluate.py
    python evaluate.py --n 500      # faster run with 500 cases
    python evaluate.py --verbose    # print each wrong prediction
"""
import argparse
import json
import re
import random
import torch
from collections import defaultdict

from model import GPT
from tokenizer import AgentTokenizer
from data_gen import gen_sentiment_example, gen_impact_example, gen_signal_example

# ── Config ────────────────────────────────────────────────────────────────
N_TEST      = 1000
MAX_NEW     = 60       # max tokens to generate per prompt
TEMPERATURE = 0.2      # low temperature for reliable structured generation
TOP_K       = 5
NGRAM_N     = 0        # disabled by default: n-gram ban breaks repeating JSON tokens like '": "'

tok = AgentTokenizer()
model = None
cfg = None
end_id = tok.end_id


def load_model(ckpt_path: str):
    global model, cfg
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ── Helpers ───────────────────────────────────────────────────────────────

def parse_output(text: str) -> dict:
    """
    Extract route and first observation token from model output or ground-truth.
    Ticker/event are no longer in the JSON (model is too small to copy them).
    Returns a dict with keys: route, observation.
    """
    result = {"route": "", "observation": ""}

    # Route: first token after <route>
    if "<route>" in text:
        after = text.split("<route>")[1].strip()
        splits = after.split()
        if splits:
            result["route"] = splits[0]

    # Observation: first meaningful token after <observation>
    if "<observation>" in text:
        obs_raw = text.split("<observation>")[1]
        obs_raw = obs_raw.split("<action>")[0].strip()
        tokens  = obs_raw.split()
        result["observation"] = tokens[0] if tokens else ""

    return result


@torch.no_grad()
def generate(prompt: str, temperature: float = TEMPERATURE, top_k: int = TOP_K, greedy: bool = False) -> str:
    """Run model inference on prompt string, return decoded full output."""
    generated = tok.encode(prompt)

    for _ in range(MAX_NEW):
        window   = generated[-cfg.block_size:]
        idx_cond = torch.tensor([window], dtype=torch.long)
        logits, _ = model(idx_cond)
        logits     = logits[:, -1, :]

        if greedy or temperature <= 1e-4:
            next_id = torch.argmax(logits, dim=-1).item()
        else:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs   = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_id)
        if next_id == end_id:
            break

    return tok.decode(generated)


# ── Build 1000 test cases ─────────────────────────────────────────────────

def build_test_cases(n: int) -> list:
    """
    Generate n test cases balanced across the 3 skills.
    Uses seed=123 (different from training seed=42) to ensure
    test cases are not seen during training.
    """
    random.seed(123)
    skills = [
        ("sentiment", gen_sentiment_example),
        ("impact",    gen_impact_example),
        ("signal",    gen_signal_example),
    ]
    cases = []
    n_each = n // 3
    for skill_name, gen_fn in skills:
        for _ in range(n_each):
            cases.append((skill_name, gen_fn()))
    # Fill remainder
    for i in range(n - len(cases)):
        skill_name, gen_fn = skills[i % 3]
        cases.append((skill_name, gen_fn()))
    random.shuffle(cases)
    return cases


# ── Evaluation loop ───────────────────────────────────────────────────────

def run_eval(n: int, verbose: bool, greedy: bool = False):
    print(f"\nPraman Brain Mini — Layer 1 Evaluation")
    print(f"{'='*60}")
    print(f"Model     : {model.num_params():,} params")
    print(f"Test cases: {n}  (seed=123, separate from training seed=42)")
    print(f"Decoding  : {'Greedy' if greedy else f'Temp={TEMPERATURE}, TopK={TOP_K}'}")
    print(f"{'='*60}\n")

    cases   = build_test_cases(n)
    metrics = defaultdict(lambda: defaultdict(int))
    total   = defaultdict(int)
    errors  = []

    for i, (skill, full_traj) in enumerate(cases):
        if "<|im_start|> assistant" in full_traj:
            prompt_text = full_traj.split("<|im_start|> assistant")[0] + "<|im_start|> assistant\n"
        else:
            prompt_text = full_traj.split("<thought>")[0]
        prompt = prompt_text

        expected  = parse_output(full_traj)
        raw_gen = generate(prompt, greedy=greedy)
        predicted = parse_output(raw_gen)

        # Print first 2 cases so user sees what is actually being generated
        if i < 2:
            print(f"--- Sample {i+1} ({skill}) ---")
            print(f"  Prompt:    {repr(prompt[-80:])}")
            print(f"  Generated: {repr(raw_gen[len(prompt):len(prompt)+120])}")
            print(f"  Expected:  {expected}")
            print(f"  Predicted: {predicted}")
            print()

        total[skill] += 1
        total["all"]  += 1

        route_ok = predicted["route"] == expected["route"]
        obs_ok   = predicted["observation"] == expected["observation"]
        full_ok  = route_ok and obs_ok

        for grp in (skill, "all"):
            if route_ok: metrics[grp]["route"] += 1
            if obs_ok:   metrics[grp]["obs"]   += 1
            if full_ok:  metrics[grp]["full"]  += 1

        if verbose and not full_ok:
            errors.append({
                "i": i + 1, "skill": skill,
                "prompt":    prompt_text,
                "expected":  expected,
                "predicted": predicted,
                "raw":       raw_gen,
            })

        # Live progress every 100 cases
        if (i + 1) % 100 == 0:
            n_done = i + 1
            r = metrics["all"]["route"] / n_done * 100
            o = metrics["all"]["obs"]   / n_done * 100
            f = metrics["all"]["full"]  / n_done * 100
            print(f"  [{n_done:4d}/{n}]  route={r:5.1f}%  obs={o:5.1f}%  full={f:5.1f}%")

    # ── Final report ──────────────────────────────────────────────────────
    print(f"\n{'='*54}")
    print(f"  FINAL RESULTS — {n} Test Cases (seed=123)")
    print(f"{'='*54}")
    print(f"\n{'Skill':<14} {'N':>5} {'Route':>7} {'Obs':>7} {'Full':>7}")
    print(f"{'-'*54}")

    for key in ["sentiment", "impact", "signal", "all"]:
        nn = total[key]
        if nn == 0:
            continue
        label = "OVERALL" if key == "all" else key.capitalize()
        r = metrics[key]["route"] / nn * 100
        o = metrics[key]["obs"]   / nn * 100
        f = metrics[key]["full"]  / nn * 100
        sep = "\n" + "-"*54 if key == "all" else ""
        print(f"{sep}\n{label:<14} {nn:>5}  {r:>6.1f}%  {o:>6.1f}%  {f:>6.1f}%")

    print(f"\n{'='*54}")

    # Verdict
    overall_full  = metrics["all"]["full"]  / total["all"] * 100
    overall_route = metrics["all"]["route"] / total["all"] * 100
    overall_obs   = metrics["all"]["obs"]   / total["all"] * 100
    print(f"\n  Routing accuracy : {overall_route:.1f}%  "
          f"({'PASS' if overall_route >= 90 else 'NEEDS RETRAINING'}  target: 90%+)")
    print(f"  Obs accuracy     : {overall_obs:.1f}%  "
          f"({'PASS' if overall_obs >= 80 else 'NEEDS RETRAINING'}  target: 80%+)")
    print(f"  Full accuracy    : {overall_full:.1f}%  "
          f"({'PASS' if overall_full >= 80 else 'NEEDS RETRAINING'}  target: 80%+)")

    # Verbose error dump
    if verbose and errors:
        print(f"\n  --- Wrong predictions ({len(errors)} cases) ---")
        for err in errors[:20]:   # cap at 20 to avoid flooding terminal
            print(f"\n  [{err['i']:4d}] skill={err['skill']}")
            print(f"         prompt    : {err['prompt']}")
            print(f"         expected  : {err['expected']}")
            print(f"         predicted : {err['predicted']}")
            print(f"         RAW       : {repr(err['raw'])}")
        if len(errors) > 20:
            print(f"\n  ... and {len(errors)-20} more (use --verbose to see all)")

    print()


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Praman Brain Mini — 1000-Case Evaluation")
    parser.add_argument("--ckpt",    type=str,  default="dpo_model_v2.pt",
                        help="checkpoint path (e.g. dpo_model_v2.pt or best_model_v2.pt)")
    parser.add_argument("--n",       type=int,  default=N_TEST,
                        help=f"number of test cases (default: {N_TEST})")
    parser.add_argument("--greedy",  action="store_true",
                        help="use greedy decoding instead of temperature sampling")
    parser.add_argument("--verbose", action="store_true",
                        help="print wrong predictions")
    args = parser.parse_args()

    load_model(args.ckpt)
    run_eval(args.n, args.verbose, greedy=args.greedy)
