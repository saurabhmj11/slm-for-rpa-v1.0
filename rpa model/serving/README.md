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

## Modes and the agent team

A client picks how hard the assistant works on a request. The modes are `Policy` presets (`MODES` in
`ooa_server.py`, the only place they are defined) on top of the client's policy (escalation, masking and logs stay):

| Mode | Agents | Limits |
| --- | --- | --- |
| `fast` | planner + one builder (the same answer) + checker + reporter | 1 attempt, no parts, no self-correction |
| `smart` | planner, parallel builders for a long plan, checker, fixer, reporter | 2 retries; failed parts rebuilt for 2 rounds; no new attempt after 100 s |
| `ultra` | understander, planner, best-of-4 answers, parallel builders (2 samples per part), checker, fixer, critic, reporter | 3 retries, 3 fixer rounds, parts from plans of 5+ lines; no new attempt after 120 s, a running generation cut at 150 s (Studio waits 180 s) |
| `auto` (default) | picks one per request (`choose_mode`) | |

Best-of-N keeps the first verified sample; a sample's question ends the request only when no sample of its batch
verified (and never on a fixer retry once a plan exists), so `ultra` never answers with a question where `fast` would
have built the flow. The caller's deadline (the PC app sets it when the request arrives, so queue time counts) also
caps the cut of a running generation: never more than `HARD_GRACE` (30 s) after it. That cut needs a backend with
`generate_batch(deadline=)` (`NanochatLocal`); other backends only stop starting new attempts.

Auto's rule (`choose_mode`, from `ooa_format.request_features`): **ultra** for a long process (5+ clauses or more
than 60 words); for a vague request that names no file, folder, app or data, **ultra** when the model was trained to
restate requests (rewrite rows), else **smart**; **fast** for 2 clauses or fewer, 25 words or fewer, no loop words
(each, every, repeat ...) and no condition words (if, when, otherwise, only ...); **smart** otherwise.

How a client picks: the model name (`GET /v1/models`: `ooa-assistant` runs the server's default mode;
`ooa-flow-assistant` is auto; `ooa-flow-assistant-fast`, `-smart`, `-ultra`), a `"mode"` field in the body, or the
`X-OOA-Mode` header (field first, then header, then name); the server default is `--mode` (auto; `none` keeps the
settings from before modes). The reply's `ooa` field says `mode`, `mode_asked`, `mode_reason`, `agents` (each with its
seconds) and `report`; Studio's notes end with the report ("Mode: Smart (picked by Auto: a loop), 12.3 s. What it
does: ... Assumed: you give folder when it runs ... Agents: ...").

The team is one model playing roles with different prompts (`Router._team`):
- **understander**: the samples but one start with "Clear request: " (only for a model trained with rewrite rows);
  the free sample may ask a question instead, which ends the request.
- **planner**: writes the plan; a plan long enough to build in parts stops at "Flow:", else it writes the flow
  itself (the single builder).
- **builders**: every part of the plan (`plan_parts`) in ONE batched generation (`NanochatLocal.generate_batch` ->
  `constrained.generate_batch`: nanochat's Engine batches samples of one prompt only, so prompts are prefilled one
  by one, their caches joined, and each decode step runs all rows with per-row positions). A part inside a loop / try
  / if waits for the part that writes it (`part_dependencies`). Backends without `generate_batch` build the parts one
  after another (`_decompose`, as before).
- **assembler**: links a variable a part read under its own name to what an earlier part made
  (`link_part_variables`), then merges (`merge_flows`).
- **checker**: repair + verify for each answer, part and the merged flow. **fixer**: rebuilds only the failed parts
  (batched), each with its previous flow and problems; whole answers get self-correction retries.
- **step builder** (smart, ultra; `serving/step_builder.py`, `Policy.step_builder`: "long" by default, "off",
  "always"): for a long or multi-app request (`step_builder.wants`: 3+ steps with 25+ words or a loop / if, or 2+
  steps in 2+ named apps) the request itself is cut into steps by rules (sentences, ";", then / after that / also /
  finally, ", <verb>", " and <verb>"; "for each ..." opens a loop, "if ..." an if, "otherwise ..." its else, a step
  after "then" that sums up ("how many", "the list") closes the loop). Each step is asked as a short plain request
  of its own ("Read the total for the invoice PDF": pronouns and "each one" made concrete) with retrieval for that
  step and the variables known for it; ALL steps go into ONE batched generation, best-of-3 at temperatures
  0.2 / 0.6 / 0.9, each answer checked; the best checked piece of a step is the one that agrees with the other
  samples, uses the step's best catalog matches and the action its verb points at, and adds nothing it did not ask
  for (no extra loop or if, no app nobody named, few actions). Loops are built by code (the newest collection
  before them, a table through `to_records`, then `core/loop`); failed steps are asked once more (one batch) with
  their problems and the variables the steps before them made; a step whose piece lacks what its words clearly ask
  for ("email the team ..." without an email) is asked again in other words. A step whose words clearly point at
  actions ("keep the ones whose ..." -> a filter, "more than 7 days ago" -> now / parse / difference, "collect the
  links", "working days", a free slot in Outlook; `STEP_HINTS`) gets one more, guided sample in the same batch: its
  answer starts with those actions as its plan and the model writes the flow (kept only when it passes the checker).
  A story ("First I open ..., then I ...") drops its "I" / "we" and the sentences about the bot ("Can the bot do
  this?"); auto leaves a story in fewer than 2 named apps to the single answer. The pieces are joined into their
  containers, their variables linked, then repaired and checked as a whole. It runs BEFORE the single answer: a
  step-built flow that leaves no clause of the request uncovered is kept at once; otherwise the single answer runs
  in the time left and the step-built flow is kept unless the single answer covers more of the request's clauses
  (`step_builder.prefer`). The reply's `ooa` field has `step_builder` (steps, built, failed, verified, kept,
  seconds, model_calls). The step path has its own time budget (`Policy.step_time`, 60 s: its generations are cut
  then, and its second wave starts only in the first 60% of it), so the single answer always has the rest of the
  mode's time. Fast mode never runs it.
- **critic** (ultra): a request clause that names something to do and that no step covers (`uncovered_clauses`) is
  built once as a new part and added; if that fails the check, the flow stays and the report says what to check.
- **reporter**: the summary above. With `store_logs`, requests.jsonl rows carry the mode and the agents' times.

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
| step_builder, step_best_of, step_time | the step builder in smart / ultra: "long" (default), "off" or "always"; samples per step (3); its time budget in seconds (60; 0: only the mode's limits) |

## Platform contract (model connection "OneOffice in-house model")

The platform registers this server as an `openai_compatible` model connection with base URL
`http://127.0.0.1:<port>/v1` and model **`ooa-flow-1`** (the stable id: it runs the server's default mode and is
echoed back; the model version is in `ooa.version`). `ooa-assistant` and the `ooa-flow-assistant[-fast|-smart|-ultra]`
names keep working.

- `POST /v1/chat/completions`: the normal OpenAI reply (the flow, question or error as text in
  `choices[0].message.content`) plus, next to it, `ooa.result` (contract `ooa-flow-result/1`):
  `{status: "flow"|"question"|"no_flow", flow: <JSON object or null>, plan: [...], question, verdict: {verified,
  problems, repairs, attempts, warnings, reason}, mode, report, seconds}`. Probes ("Answer with the single word OK")
  have no `result`.
- Optional bearer token: set `OOA_API_KEY` (assistant/app.py also reads `OOA_ASSISTANT_TOKEN` and `--token` first);
  `/v1/*` then needs `Authorization: Bearer <key>`; `/health` (and app.py's `/ready`) stay open.
- Limits: request body 4 MB (413), one generation at a time with a bounded queue (429 + Retry-After), a time budget
  per request (app.py `--time-budget`, default 120 s). No streaming; no CORS (callers are servers, not browsers).
- Use assistant/app.py as the endpoint (queue, readiness, version route); this bare server has the token and body
  limit but no queue.

## Not done yet
Browser export of the model; per-client adapter training; true constrained decoding (nanochat's generator has no logits
hook, so near-miss names are repaired afterwards instead); request-complexity scoring is a simple clause count; the local
backend (`backends.py`) is untested here (needs GPU/nanochat); the Claude call is untested against the live API.
Masking is regex-based and will miss things: in production rely on the AI Gateway's PII guard as well.
