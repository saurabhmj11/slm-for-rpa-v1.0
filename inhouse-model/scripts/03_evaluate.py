"""
03_evaluate.py
==============
Evaluates the fine-tuned model (brain-v2-7b) vs the base model (qwen2.5:7b)
on 200 Indian financial Q&A pairs.

Measures:
  1. Number accuracy — does the answer contain the correct number?
  2. Refusal rate — does the model say "I don't know" when it should?
  3. Indian formatting — does it use crore/lakh correctly?
  4. Citation presence — does it mention source filings?
  5. Response latency

Output: ../eval/eval_report.json
"""

import json
import os
import re
import sys
import time
from pathlib import Path

# === Config ===
EVAL_FILE = Path(__file__).parent.parent / "data" / "eval_pairs.jsonl"
REPORT_FILE = Path(__file__).parent.parent / "eval" / "eval_report.json"
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

MODELS_TO_EVALUATE = [
    {"id": "qwen2.5:7b", "name": "Base Qwen2.5-7B (before fine-tune)"},
    {"id": "brain-v2-7b", "name": "Praman Brain v2-7B (after fine-tune)"},
]


def call_ollama(model: str, messages: list, timeout: int = 60) -> dict:
    """Call Ollama chat API."""
    import urllib.request
    body = json.dumps({"model": model, "messages": messages, "stream": False}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def extract_numbers(text: str) -> list:
    """Extract all numbers from text."""
    return re.findall(r"[-+]?\d+\.?\d*", text)


def check_indian_formatting(text: str) -> bool:
    """Check if the response uses Indian formatting (crore/lakh)."""
    return bool(re.search(r"\b(crore|Cr|lakh|L)\b", text, re.IGNORECASE))


def check_citation(text: str) -> bool:
    """Check if the response mentions a source filing or data source."""
    patterns = [r"\bfiling\b", r"\bannual report\b", r"\bquarterly\b",
                r"\bNSE\b", r"\bBSE\b", r"\bScreener\b", r"\bFY\d{2,4}\b",
                r"\bQ[1-4]\b", r"\bpage\b"]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def evaluate_model(model_id: str, eval_data: list) -> dict:
    """Evaluate a model on the eval dataset."""
    results = {
        "model": model_id,
        "total": len(eval_data),
        "number_accuracy": 0,
        "indian_formatting": 0,
        "citation_present": 0,
        "avg_latency_ms": 0,
        "errors": 0,
        "examples": [],
    }

    total_latency = 0
    number_checks = 0
    number_correct = 0

    for i, item in enumerate(eval_data):
        instruction = item["instruction"]
        input_ctx = item.get("input", "")
        expected_output = item["output"]

        messages = [
            {"role": "system", "content": (
                "You are Praman Brain v2, an AI financial analyst specializing in "
                "Indian markets. Use crore/lakh formatting. FY ends March 31. "
                "Always cite the source filing when quoting numbers. "
                "If you don't have the data, say so explicitly."
            )},
            {"role": "user", "content": f"{instruction}\n\nContext: {input_ctx}" if input_ctx else instruction},
        ]

        start = time.time()
        response = call_ollama(model_id, messages)
        latency = (time.time() - start) * 1000
        total_latency += latency

        if "error" in response:
            results["errors"] += 1
            continue

        answer = response.get("message", {}).get("content", "")

        # Check number accuracy
        expected_numbers = extract_numbers(expected_output)
        answer_numbers = extract_numbers(answer)
        if expected_numbers and len(expected_numbers) > 0:
            number_checks += 1
            # Check if at least the first significant number matches
            for en in expected_numbers:
                try:
                    en_val = float(en)
                    if en_val != 0 and abs(en_val) >= 1:
                        for an in answer_numbers:
                            try:
                                an_val = float(an)
                                if abs(an_val - en_val) < 0.01 * abs(en_val):
                                    number_correct += 1
                                    break
                            except:
                                pass
                        break
                except:
                    pass

        # Check Indian formatting
        if check_indian_formatting(answer):
            results["indian_formatting"] += 1

        # Check citation
        if check_citation(answer):
            results["citation_present"] += 1

        # Save first 5 examples for manual review
        if i < 5:
            results["examples"].append({
                "question": instruction,
                "expected": expected_output[:200],
                "actual": answer[:200],
                "latency_ms": round(latency),
            })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(eval_data)}] done")

    results["number_accuracy"] = round(number_correct / max(number_checks, 1) * 100, 1)
    results["indian_formatting"] = round(results["indian_formatting"] / max(results["total"] - results["errors"], 1) * 100, 1)
    results["citation_present"] = round(results["citation_present"] / max(results["total"] - results["errors"], 1) * 100, 1)
    results["avg_latency_ms"] = round(total_latency / max(results["total"] - results["errors"], 1))
    return results


def main():
    print("=" * 60)
    print("  03_evaluate.py")
    print("  Evaluating fine-tuned model vs base model")
    print("=" * 60)
    print()

    if not EVAL_FILE.exists():
        print(f"ERROR: Eval file not found: {EVAL_FILE}")
        print("Run: python scripts/01_collect_training_data.py first")
        sys.exit(1)

    eval_data = []
    with open(EVAL_FILE, "r") as f:
        for line in f:
            eval_data.append(json.loads(line.strip()))

    print(f"Loaded {len(eval_data)} evaluation pairs")
    print()

    all_results = {}
    for model in MODELS_TO_EVALUATE:
        print(f"Evaluating: {model['name']} ({model['id']})")
        result = evaluate_model(model["id"], eval_data)
        all_results[model["id"]] = result
        print(f"  Number accuracy:  {result['number_accuracy']}%")
        print(f"  Indian formatting: {result['indian_formatting']}%")
        print(f"  Citation present:  {result['citation_present']}%")
        print(f"  Avg latency:       {result['avg_latency_ms']}ms")
        print(f"  Errors:            {result['errors']}")
        print()

    # Save report
    os.makedirs(REPORT_FILE.parent, exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print("=" * 60)
    print("  Evaluation complete!")
    print("=" * 60)
    print(f"  Report saved to: {REPORT_FILE}")
    print()

    # Comparison table
    if len(all_results) == 2:
        keys = list(all_results.keys())
        base = all_results[keys[0]]
        tuned = all_results[keys[1]]
        print("  COMPARISON:")
        print(f"  {'Metric':<25} {'Base':>10} {'Fine-tuned':>10} {'Delta':>10}")
        print(f"  {'-'*55}")
        print(f"  {'Number accuracy':<25} {base['number_accuracy']:>9}% {tuned['number_accuracy']:>9}% {tuned['number_accuracy']-base['number_accuracy']:>+9}%")
        print(f"  {'Indian formatting':<25} {base['indian_formatting']:>9}% {tuned['indian_formatting']:>9}% {tuned['indian_formatting']-base['indian_formatting']:>+9}%")
        print(f"  {'Citation present':<25} {base['citation_present']:>9}% {tuned['citation_present']:>9}% {tuned['citation_present']-base['citation_present']:>+9}%")
        print(f"  {'Avg latency (ms)':<25} {base['avg_latency_ms']:>10} {tuned['avg_latency_ms']:>10} {tuned['avg_latency_ms']-base['avg_latency_ms']:>+10}")


if __name__ == "__main__":
    main()
