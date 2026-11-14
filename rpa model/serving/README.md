# OOA assistant server (router + verifier + client policy)

One OpenAI-compatible endpoint. A request is tried on our own model first when it looks simple; every answer is repaired
(typo-level fixes only) and verified against the platform's rules; if that fails and the client's policy allows it, the
request escalates to Claude (with PII masked and the verifier's complaints attached); if nothing verifies, it returns an
honest error, never an unchecked flow.

```bash
pip install jsonschema
python ooa_server.py --catalog ../data/catalog/catalog.jsonl --platform PLATFORM_REPO --policy policy.example.json --port 8080
python ooa_server.py ... --local nanochat      # load the trained model (needs nanochat + torch)
python test_server.py PLATFORM_REPO            # end-to-end tests with fake models
python test_layers.py PLATFORM_REPO            # the model layers below, with scripted models and a stub endpoint
```
Register it in the platform as an `openai_compatible` model connection pointing at `http://HOST:8080/v1`.
For cloud escalation set `ANTHROPIC_API_KEY` (production: route through the platform's AI Gateway instead), or set
`escalate_url` (+ `escalate_model`, key in the variable named by `escalate_key_env`) for any OpenAI-compatible endpoint.

## Layers of the local model (Router)
The model format is defined in `training/ooa_format.py`; training data is rendered into it by `training/render_dataset.py`.
1. Retrieval: the request + 12 BM25 actions (plus loop / if and the actions of the similar examples), up to 2 similar
   training requests from `examples.jsonl` with their action outline, and the variables Studio sends.
2. Plan first: the model writes a numbered plan (one action id per line), then the flow. The catalog lock
   (`assistant/constrained.py`) applies only inside the flow JSON.
3. Clarify: a request too vague to automate gets `Question: ...` back; Studio shows it as notes, with no steps.
4. Verify: repair, the platform checker, and plan coverage (every action the plan names must be in the flow).
5. Self-correction: the next attempt is given its previous flow and the problems.
6. Decomposition: a plan over 6 lines (after one failed attempt), or any plan after every attempt failed, is built in
   parts of 2-4 top-level items, each with the variables made so far; the merged flow must pass the checker.
7. Escalation (only when allowed): after the local attempts fail, or at once when the plan is over 12 lines.
8. Safety: `training/safety.py` warnings for destructive steps go into notes, `warnings` and the `ooa` field.
`prompt_format` "auto" uses the request alone (models trained before this format) unless the model pack has
`examples.jsonl` next to its catalog.

## Client policy (policy.json), default = local-only, stores nothing
| Switch | Meaning |
|---|---|
| allow_escalation | may a request leave for the cloud model (default false) |
| mask_pii | mask emails, phones, card-like numbers, IPs before anything leaves (default true) |
| store_logs | keep masked request/result logs and feedback (default false) |
| allow_training_on_data | stored rows are flagged `training_ok`; only flagged rows may be used to train (default false) |
| local_retries, max_clauses_local | how hard the local model tries; longer requests go straight to the cloud |
| local_first | our model tries every request first, whatever its length (the PC assistant sets this) |
| escalate_url, escalate_model, escalate_key_env | escalation to an OpenAI-compatible endpoint instead of the Anthropic API |
| escalate_plan_lines | a failed local flow with a longer plan goes to escalation at once (default 12) |
| prompt_format, context_k, decompose_lines, part_attempts | the layers above (auto, 12, 6, 2) |

## Not done yet
Browser export of the model; per-client adapter training; true constrained decoding (nanochat's generator has no logits
hook, so near-miss names are repaired afterwards instead); request-complexity scoring is a simple clause count; the local
backend (`backends.py`) is untested here (needs GPU/nanochat); the Claude call is untested against the live API.
Masking is regex-based and will miss things: in production rely on the AI Gateway's PII guard as well.
