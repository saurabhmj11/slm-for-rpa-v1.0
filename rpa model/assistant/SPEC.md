# OOA Assistant (local web app): build spec

A local web app (Python standard library HTTP server + one HTML page, no CDN, works offline) that runs OUR model on the
user's PC (CPU is enough) and offers four things:

1. **Build**: plain request -> model -> repair -> verifier -> flow, with every step explained in plain English.
2. **Learn**: search the 1,351-action catalog, read an action's card (what it does, inputs, outputs), lessons per toolkit,
   and "ask the model" (catalog Q&A) next to the catalog's ground truth.
3. **Practice**: quizzes generated from the catalog ("which action does X?", "which inputs does Y require?").
4. **Run**: run a flow on this PC with the platform's Python reference runner, only for the actions it supports, inside a
   sandbox folder, with risky actions blocked unless the user confirms.
5. **Self-learning**: the user accepts / edits / rejects each built flow. Feedback is stored locally; "Export training
   data" turns it into verified training rows (training/ingest_feedback.py rules) for the next Colab training run.
   The model does NOT change live; it improves at the next training run. Say so in the UI.

Honesty rules (the UI must follow them): never show an unverified flow as if it were fine; show the verifier's problems
when it fails; say what the model is good at today (step-by-step requests, catalog questions) and what not (everyday
wording is weak, long business flows fail).

## Files (all new, under `rpa model/assistant/`)
| File | Owner | Purpose |
|---|---|---|
| `knowledge.py` | knowledge agent | catalog, explanations, search, lessons, quizzes (no model, no torch) |
| `runner_bridge.py` | runner agent | can-run check, sandboxed run through the reference runner in a subprocess |
| `model_backend.py` | model agent | nanochat backend with a repetition guard; a fake backend for tests |
| `app.py` | app agent | HTTP server, routes below, wiring, feedback store, export |
| `static/index.html` | UI agent | the single page (tabs: Build, Learn, Practice, Run, Progress) |
| `test_assistant.py` | each agent adds its own `test_*` functions in its own file `tests/test_<module>.py`; app agent writes `tests/test_app.py` | plain `python tests/test_x.py` runnable, no pytest needed |
| `README.md`, `start_assistant.bat`, `start_assistant.sh` | app agent | setup on Windows (no GPU) and Linux |

Reuse, do not copy: `../training/ooa_reward.py` (verifier: `configure(catalog, schema)`, `score(text, gold=None)`,
`problems(text)`, `actions_of(flow)`, `walk(nodes)`), `../training/repair.py` (`repair(flow, cat) -> (flow, notes)`),
`../training/retrieval.py` (`ActionIndex(entries).search(q, k)`, `candidate_line(e)`), `../serving/ooa_server.py`
(`Router`, `Policy`, `extract_json`), `../training/ingest_feedback.py` (CLI: FEEDBACK OUT_DIR CATALOG SCHEMA).
Add `sys.path` entries relative to `Path(__file__)`; never hard-code `/home/user` or the scratchpad.

## Catalog entry shape (catalog.jsonl, one per line)
`{"id": "ooa.csv/read", "name": "Read to table", "toolkit": "CSV/TXT", "toolkit_id": "ooa.csv", "kind": "action",
  "description": "...", "notes": "Time limit 60 s; ...", "inputs": [{"name","type","required","description",...}],
  "outputs": [{"name","type","description"}]}`. Control-flow ids are `core/if`, `core/loop`, `core/try`, ... (toolkit_id `core`).
Flow JSON: `{"kind","id","name","description","variables":[{"name","type","direction","required","default"}],
"steps":[{"id","action","description","args":{...},"out":{output_name: variable},"children":[...],"else":[...],
"elseIf":[{"children":[...]}],"catch":[{"codes":[],"children":[...]}],"finally":[...]}]}`. Args reference variables as
`"$name"` or expressions with `${...}`.

## Module contracts (exact signatures; other agents code against these)

### knowledge.py
```python
class Knowledge:
    def __init__(self, catalog_path: str): ...
    entries: list[dict]; by_id: dict[str, dict]
    def toolkits(self) -> list[dict]          # [{"id","name","count"}], sorted by name
    def action_card(self, action_id) -> dict | None   # {"id","name","toolkit","description","notes",
                                                      #  "required":[{"name","type","description"}],"optional":[...],"outputs":[...]}
    def search(self, query, k=10) -> list[dict]       # cards (short: id,name,toolkit,description) via retrieval.ActionIndex
    def lesson(self, toolkit_id) -> dict      # {"toolkit","intro","actions":[short cards]}; intro is a 1-2 sentence plain summary built from the cards
    def explain_flow(self, flow: dict) -> dict
        # {"summary": str, "inputs":[{"name","type","description"}], "outputs":[...],
        #  "steps":[{"id","depth","action","title","what","args":[{"name","value","meaning"}],"outs":[{"name","variable"}],
        #            "known": bool, "block": None | "if"|"else"|"elseif"|"loop"|"try"|"catch"|"finally"}]}
        # Plain English, e.g. "Step 3 (inside the loop): CSV/TXT -> Read to table: Read a CSV file into a table.
        # Reads from: $path. Saves the table in: table." Unknown action -> known False and "what" says it is not in the catalog.
    def quiz(self, rng=None, kind=None) -> dict
        # kinds: "which_action" (description -> pick the action among 4, distractors from the same toolkit when possible),
        #        "required_inputs" (action -> pick its required-input set among 4), "which_toolkit".
        # {"id": opaque str encoding answer-free state, "kind","question","choices":[str x4]}  (answer NOT included)
    def check_quiz(self, quiz_id, choice_index) -> dict  # {"correct": bool, "answer_index": int, "explanation": str}
        # quiz_id must let the server check without storing state (e.g. base64 json of kind/action/choices seed, plus an HMAC
        # with a per-process random key so the browser cannot forge "correct").
```

### runner_bridge.py
Uses the platform's Python reference runner (`reference/ooa`, package `oneofficeautomation-reference`, may be absent).
```python
RISKY = {...}   # action ids blocked unless the user confirms: ooa.file/delete, ooa.file/write_text when it would overwrite?
                # (decide: keep it simple and explicit), ooa.rest/*, ooa.genai/*, core/run_bot, ooa.agents/*, anything not file/excel/datatable/json/string/core
class RunnerBridge:
    def __init__(self, platform_repo: str | None, sandbox_dir: str): ...
    available: bool                    # reference runner importable (from platform_repo/reference or installed)
    def supported_actions(self) -> set[str]
    def can_run(self, flow: dict) -> dict
        # {"ok": bool, "unsupported":[action ids the local runner lacks], "risky":[ids], "problems":[reference validate_flow messages],
        #  "inputs":[{"name","type","required","default","description"}]}   # flow input variables the user must fill
    def run(self, flow: dict, inputs: dict, allow_risky: bool = False, timeout_s: int = 60) -> dict
        # refuses (ok False, "refused": reason) if not can_run, or risky without allow_risky.
        # Runs in a SUBPROCESS (sys.executable -I), cwd = sandbox_dir, killed after timeout_s.
        # Path guard: any literal string input or arg that looks like a path must resolve inside sandbox_dir (reject absolute
        # paths outside it and '..' escapes); relative paths are resolved against sandbox_dir. Document that this is a guard,
        # not a security boundary (the flow runs as the user).
        # -> {"ok","outputs":{...},"events":[{"type","step","message"} ...] (at most 200),"error": str|None,"seconds","files":[sandbox files after run]}
```
The reference runner API (read it first: `reference/ooa/platform.py`, `reference/ooa/runner/interpreter.py`,
`reference/ooa/runner/validate.py`, `reference/ooa/cli.py` cmd_run): `Platform(home: Path, sink=ListSink(), work_dir=Path)`,
`p.registry.all_manifests()`, `validate_flow(flow, p.registry, raise_on_error=False)`, and how `ooa run` executes a flow.
Note: the catalog lists MORE optional inputs than the reference implements (e.g. `ooa.excel/open` `header`), so
`validate_flow` problems must be shown to the user as "cannot run locally: ..." not hidden.

### model_backend.py
```python
class NanochatLocal:
    def __init__(self, model_dir: str, nanochat_dir: str, source="sft", device=None): ...  # sets NANOCHAT_BASE_DIR=model_dir,
                                                    # NANOCHAT_DTYPE=float32 on CPU, imports nanochat from nanochat_dir
    def generate(self, request: str, attempt: int = 0, max_tokens=2500) -> str
        # same prompt format as training: [bos, <|user_start|>, request, <|user_end|>, <|assistant_start|>], temperature
        # 0.2 + 0.3*attempt, top_k 50, stops at <|assistant_end|> or bos.
        # REPETITION GUARD: stop early when the output is looping (e.g. the last 200 tokens are one n-gram (n<=20) repeated
        # >= 6 times, or the same 30-token window appeared 3 times). Loops are the main failure today and cost ~20 s each.
    def ask(self, question: str) -> str   # catalog Q&A, max 300 tokens
    info: dict   # {"device","params_M","step","source"}
class FakeModel:   # for tests: generate() returns a canned valid flow for known prompts, garbage otherwise
```
Check: `generate` must be compatible with `serving/ooa_server.Router` (it calls `local.generate(request, attempt)`).

### app.py
`python app.py --model-dir DIR --nanochat DIR --platform DIR [--catalog FILE] [--port 8765] [--data-dir DIR] [--fake-model]`
Defaults: catalog = `MODEL_DIR/../sft/catalog.jsonl` if it exists else `../data/catalog/catalog.jsonl`; data dir
`./assistant_data` (feedback.jsonl, sandbox/, exports/). Binds 127.0.0.1 only. Loads the model once at start.
Routes (JSON in/out; errors -> {"error": str} with 400/404/500):
- `GET /` -> static/index.html; `GET /static/<file>` (no path traversal)
- `GET /api/status` -> model info, catalog size, runner available + supported count, feedback counts, honest "good at / weak at" text
- `POST /api/build {"request"}` -> Router (local only, policy: no escalation, retries 1) -> `{"verified","flow","problems","repairs",
  "seconds","explanation": Knowledge.explain_flow(flow) if verified, "can_run": RunnerBridge.can_run(flow) if verified}`
- `POST /api/explain {"flow"}`; `POST /api/ask {"question"}` -> `{"model_answer", "catalog_matches": search(q,5)}`
- `GET /api/search?q=&k=`; `GET /api/action?id=`; `GET /api/toolkits`; `GET /api/lesson?toolkit=`
- `GET /api/quiz?kind=`; `POST /api/quiz/check {"id","choice"}`
- `POST /api/can_run {"flow"}`; `POST /api/run {"flow","inputs","allow_risky"}`
- `POST /api/feedback {"request","flow","verdict":"accept"|"edit"|"reject","edited_flow"?,"note"?}` -> append to
  data/feedback.jsonl in ingest_feedback's format (`prompt`, `flow`, `accepted`, `edited_flow`) plus time/verdict/note
- `GET /api/progress` -> counts by verdict, last 20 feedback rows (prompt, verdict, time)
- `POST /api/export` -> run ingest_feedback (as a subprocess) into data/exports/<timestamp>/ -> `{"usable","dropped","rejected","path"}`
  with instructions: upload `feedback_train.jsonl` to Drive `ooa-nanochat/sft/`; the notebook's training cell mixes it in.
Thread safety: one lock around model calls (one generation at a time).

### static/index.html
One self-contained file (inline CSS + vanilla JS, no external URLs). Tabs: **Build** (textarea, Build button, spinner with
elapsed seconds, result: verified badge or problems list, step-by-step explanation as an indented list, collapsible raw
JSON, buttons Accept / Edit (opens the JSON in a textarea, then Save edit) / Reject, and "Try running it" when can_run.ok),
**Learn** (search box with results, action card view, toolkit list -> lesson, "Ask the model" box showing model answer next
to catalog matches with a note that the catalog is the source of truth), **Practice** (quiz with 4 choices, feedback,
running score kept in localStorage wrapped in try/catch), **Run** (paste or reuse the built flow, input form generated
from can_run.inputs, risky warning + confirm checkbox, Run button, outputs/events/files), **Progress** (feedback counts,
recent feedback, Export training data button and the instructions it returns). Example request chips that the model is
good at (step-by-step wording). Light and dark (prefers-color-scheme). Works at 360 px wide. Accessible labels.

## Test environment (cloud session that builds this; not the user's PC)
- Python with torch CPU + nanochat deps: `/tmp/claude-0/-home-user-slm-for-rpa-v1-0/747ef98a-1702-5aa7-8c30-1ec154c87d91/scratchpad/venv/bin/python`
  (has torch, rustbpe, tiktoken, filelock, numpy, jsonschema, and the platform reference installed editable).
- Trained model: `.../scratchpad/dl/nanochat_base` (chatsft_checkpoints/d8/model_002000.pt, tokenizer/); catalog `.../scratchpad/dl/sft/catalog.jsonl` (1,351 entries)
- nanochat code (commit 92d63d4): `.../scratchpad/nanochat`
- Platform repo: `/home/user/oneofficeautomation-private-limited/oneofficeautomation-platform` (schema at
  `contracts/flow-schema/flow.v1.json`, reference runner at `reference/`)
- The real model takes ~1-20 s per request on this 4-core CPU; one process at a time (do not run model tests in parallel).
- Do not commit or push. Do not edit files outside `rpa model/assistant/` (other agents own other files) unless the
  integration agent must fix a real bug, and then say so in its report.
