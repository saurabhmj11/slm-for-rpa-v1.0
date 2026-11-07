# OOA assistant server (router + verifier + client policy)

One OpenAI-compatible endpoint. A request is tried on our own model first when it looks simple; every answer is repaired
(typo-level fixes only) and verified against the platform's rules; if that fails and the client's policy allows it, the
request escalates to Claude (with PII masked and the verifier's complaints attached); if nothing verifies, it returns an
honest error, never an unchecked flow.

```bash
pip install jsonschema
python ooa_server.py --catalog ../data/catalog/catalog.jsonl --platform PLATFORM_REPO --policy policy.example.json --port 8080
python ooa_server.py ... --local nanochat      # load the trained model (needs nanochat + torch)
python test_server.py PLATFORM_REPO            # 8 end-to-end tests with fake models
```
Register it in the platform as an `openai_compatible` model connection pointing at `http://HOST:8080/v1`.
For cloud escalation set `ANTHROPIC_API_KEY` (production: route through the platform's AI Gateway instead).

## Client policy (policy.json), default = local-only, stores nothing
| Switch | Meaning |
|---|---|
| allow_escalation | may a request leave for the cloud model (default false) |
| mask_pii | mask emails, phones, card-like numbers, IPs before anything leaves (default true) |
| store_logs | keep masked request/result logs and feedback (default false) |
| allow_training_on_data | stored rows are flagged `training_ok`; only flagged rows may be used to train (default false) |
| local_retries, max_clauses_local | how hard the local model tries; longer requests go straight to the cloud |

## Not done yet
Browser export of the model; per-client adapter training; true constrained decoding (nanochat's generator has no logits
hook, so near-miss names are repaired afterwards instead); request-complexity scoring is a simple clause count; the local
backend (`backends.py`) is untested here (needs GPU/nanochat); the Claude call is untested against the live API.
Masking is regex-based and will miss things: in production rely on the AI Gateway's PII guard as well.
