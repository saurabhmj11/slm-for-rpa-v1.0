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
   training requests from `examples.jsonl` with their action outline, and the variables Studio sends. A request of
   2+ clauses (cut on , ; : . then / and / if / for each / otherwise ...) also gets per clause 2 BM25 hits and the 3
   actions most used by the 20 most similar training requests, plus 4 voted for the whole request (at most 40 actions).
   Training renders the same context (`ooa_format.context_actions`). The context budget is what the model was trained
   with (`render_report.json` beside `examples.jsonl`; 2,600 characters without it; render_dataset uses 4,800 for a
   4,096-token window), capped by the backend's `info["sequence_len"]`; trimming drops the per-clause extras first.
   Recall of the needed actions on eval/plain_requests sets 1 / 2 / 3: 0.88 / 0.89 / 0.56 before, 0.98 / 0.92 / 0.70
   now with 2,600 characters, 0.98 / 0.92 / 0.75 with 4,800. A model trained with 2,600 is worse with more: the first
   4,096-token model's plans named 39% of the needed set-3 actions with 4,800 characters, 49% with 2,600.
2. Plan first: the model writes a numbered plan (one action id per line), then the flow. The catalog lock
   (`assistant/constrained.py`) applies only inside the flow JSON.
3. Clarify: a request too vague to automate gets `Question: ...` back; Studio shows it as notes, with no steps.
4. Verify: repair, the platform checker, and plan coverage (every action the plan names must be in the flow).
   Compact answers (models trained with `render_dataset.py --answer-format compact`, the default from this round)
   have no variables list, no step ids and no descriptions; `ooa_format.expand_flow` rebuilds them first (variables
   from what the steps make and read, Credential for password / token / secret names, types from the catalog; ids;
   descriptions from the plan; name from the request). A full flow (older models) is checked as written. Every
   compact rendering of the training flows expands to a flow that passes the checker (all 26,105 source flows and
   25,963 rendered rows; the full flows pass 99.8-99.9%).
   A self-correction prompt shows the previous compact answer expanded, as the training rows do.
5. Self-correction: the next attempt is given its previous flow and the problems. An answer the backend stopped as a
   loop (`assistant/model_backend.py`: RepetitionGuard for exact repeats up to 200 tokens long, AnswerWatch for a flow
   that outgrows its plan or writes one "id" 3 times) is a failed attempt; the next one starts afresh.
6. Decomposition (parts first): when the plan is over 6 lines (over 4 for a request of 5+ clauses) the first answer
   stops right after its plan and the flow is built in parts at once (`ooa_format.plan_parts`: up to 4 items / 5 plan
   lines each; a long loop, try or if is cut: the part with its line also writes its first children and its
   "On error" / "Otherwise" lines, its later children are built as parts that go inside it, also several levels
   deep, such as a long try inside a loop), each a short answer
   (1,500 tokens) that starts with the part's own plan lines (given as the answer's start, so the model writes only
   the flow), with the variables made so far; the merged flow must pass the checker. If the parts fail, the
   remaining attempts are whole answers. A backend without `on_plan` gets the old order (one whole attempt first).
7. Escalation (only when allowed): after the local attempts fail, or at once when the plan is over 12 lines.
8. Safety: `training/safety.py` warnings for destructive steps go into notes, `warnings` and the `ooa` field.
9. What the model understood: a model trained on beginner data (`training/gen_beginner.py`) may open its answer with
   "Clear request: <the request restated in tool words>" before the plan. The verified answer's Studio notes then
   start with "I understood: ..." and the result (and the `ooa` field) carries it as `understood`; the Studio shape
   (`variables`, `steps`, `notes`, `warnings`) is unchanged.
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
| context_chars | context budget in characters (0: what the model was trained with) |
| parts_first, long_clauses, long_plan_lines, part_lines | parts at once for long plans (true, 5, 4, 5) |
| max_tokens, part_max_tokens | answer budget for a whole flow / one part (2500, 1500; the backend keeps it inside its window) |

## Not done yet
Browser export of the model; per-client adapter training; true constrained decoding (nanochat's generator has no logits
hook, so near-miss names are repaired afterwards instead); request-complexity scoring is a simple clause count; the local
backend (`backends.py`) is untested here (needs GPU/nanochat); the Claude call is untested against the live API.
Masking is regex-based and will miss things: in production rely on the AI Gateway's PII guard as well.
