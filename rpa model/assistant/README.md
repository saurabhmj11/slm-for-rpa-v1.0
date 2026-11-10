# OOA Assistant (runs on your own PC)

A small web page that runs **our own** flow-writing model on your computer. No GPU and no internet are needed once it is
set up, and nothing you type leaves your PC (the server only listens on `127.0.0.1`, your own machine).

It does five things:

| Tab | What it does |
|---|---|
| **Build** | You type a request. The model writes a flow, the app repairs small mistakes and the platform's own rules check it. You see every step explained in plain English. A flow that fails the check is **never** shown as if it were fine: you see the problems instead. Then you **Accept**, **Edit** or **Reject** it. |
| **Learn** | Search the 1,351 actions of the catalog, open an action's card (what it does, required and optional inputs, outputs), read a short lesson per toolkit, and "Ask the model" a question. The model's answer is shown next to the catalog's matching actions: **the catalog is the source of truth**, the model can be wrong. |
| **Practice** | Quizzes made from the catalog: "which action does this?", "which inputs does this action require?", "which toolkit has this action?". Your score is kept in the browser. |
| **Run** | Runs a flow on this PC with the platform's Python reference runner, inside a sandbox folder (`assistant_data/sandbox`). Only actions the local runner supports can run. Risky actions (deleting files, overwriting, network, AI calls, running other bots) are blocked unless you tick the confirm box. The sandbox is a guard against mistakes, **not** a security boundary: the flow runs with your own permissions. |
| **Progress** | How many flows you accepted, edited and rejected, your recent feedback, and the **Export training data** button. |

## Setup on a Windows PC (no GPU)

You need about 2 GB of free disk space. Put everything in one folder, for example `C:\ooa`.

0. **Get this repo** (`slm-for-rpa-v1.0`, with `git clone` or as a zip) into e.g. `C:\ooa\slm-for-rpa-v1.0`. Run
   everything from its `rpa model\assistant` folder and keep the folder next to its siblings `rpa model\serving` and
   `rpa model\training`: the app imports the checker, repair and router code from them. Copying only the `assistant`
   folder does not work.
1. **Install Python 3.12 or 3.13** from <https://www.python.org/downloads/windows/>. On the first installer screen tick
   **"Add python.exe to PATH"**. Check it in a new Command Prompt: `python --version`.
2. **Install the Python packages** (in Command Prompt):
   ```
   python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   python -m pip install rustbpe tiktoken filelock jsonschema
   ```
   The first line installs the CPU-only build of PyTorch (smaller; no graphics card needed).
3. **Get the nanochat code** at the exact version the model was trained with. The `ooa_assistant.zip` bundle already
   has it in `rpa model\nanochat`. From a git checkout instead (needs Git: <https://git-scm.com/download/win>):
   ```
   cd "rpa model"
   git clone https://github.com/karpathy/nanochat.git
   cd nanochat
   git checkout 92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
   ```
4. **Download the trained model** from the Google Drive folder **`ooa-nanochat`**. You only need these (not the whole folder):
   - `nanochat_base/tokenizer/` (the whole folder)
   - `nanochat_base/chatsft_checkpoints/d8/model_002000.pt` and `meta_002000.json`
   - `sft/catalog.jsonl` (optional; `data\catalog\catalog.jsonl` is used when it is missing)

   Put them inside `rpa model`, next to the `assistant` folder:
   ```
   rpa model\nanochat_base\tokenizer\...
   rpa model\nanochat_base\chatsft_checkpoints\d8\model_002000.pt
   rpa model\nanochat_base\chatsft_checkpoints\d8\meta_002000.json
   rpa model\sft\catalog.jsonl
   ```
5. **The platform's flow rules.** The app checks every flow against `contracts\flow-schema\flow.v1.json` from the
   platform repo (`oneofficeautomation-platform`). The zip bundle carries the few files needed in `rpa model\platform`
   (`contracts\`, `docs\llms.txt` and the reference runner `reference\`). From a git checkout, point `PLATFORM_DIR` at
   your copy of the platform repo instead.
   *Optional, for the Run tab:* `python -m pip install -e "rpa model\platform\reference"`
   (without it everything works except Run, and the page says so).
6. **Start it**: double-click `start_assistant.bat` (in this folder). With the layout above no editing is needed; if
   you keep things elsewhere, open it with Notepad and change the paths at the top (`MODEL_DIR`, `NANOCHAT_DIR`,
   `PLATFORM_DIR`, `CATALOG`). Your browser opens <http://127.0.0.1:8765/>. Loading the model takes up to a minute;
   refresh the page when the window says "OOA Assistant is running". Close the window to stop it.

To try the page without the model, set `FAKE_MODEL=1` in the .bat (canned answers only).
On Linux/macOS use `start_assistant.sh` (edit the paths at the top or set them as environment variables).

Command line, if you prefer:
```
python app.py --model-dir ..\nanochat_base --nanochat ..\nanochat --platform ..\platform [--catalog FILE] [--port 8765] [--data-dir DIR] [--fake-model]
```

## What the model is good and weak at today

The model is small (it runs on a laptop CPU) and young. Be honest with it and it will be honest with you:

- **Good at:** step-by-step requests worded like the catalog's action descriptions, naming the inputs and outputs, e.g.
  *"Read a text file, then parse JSON text into a Dictionary or List. The flow takes path as input. It returns value."*
  (the example buttons on the Build tab were all checked with the real model). Also questions about the catalog
  ("which action reads a CSV file?"). Even short free wording such as "List the files in a folder." can loop and fail.
- **Weak at:** everyday wording (*"sort out my invoices"*): rewrite it as plain steps. Long business flows with many
  steps usually fail the check. Its answers to catalog questions can be wrong: trust the catalog cards shown next to them.
- A request takes about 1 to 20 seconds on a 4-core CPU. One request is processed at a time.
- Every flow is checked by the platform's rules (schema, real actions, correct inputs, declared variables). If no attempt
  passes, you get the list of problems, never an unchecked flow.

## How your feedback becomes training data ("self-learning")

The model does **not** change while you use it. It improves at the **next training run**, from your feedback:

1. In **Build**, press **Accept** (the flow is right), **Edit** (fix the JSON, then Save edit) or **Reject**.
   Each click adds one line to `assistant_data\feedback.jsonl` on your PC (your request, the flow, your verdict, the time,
   and an optional note).
2. In **Progress**, press **Export training data**. The app runs `training/ingest_feedback.py` and writes a new folder
   `assistant_data\exports\<date-time>\` with:
   - `feedback_train.jsonl`: accepted flows and your edited flows (an edit wins over the original), **only if they pass
     the platform's rules**. Flows that fail are dropped, never trained on.
   - `rejected_prompts.jsonl`: requests you rejected without an edit, a list of hard cases worth writing good answers for.
3. Upload `feedback_train.jsonl` to Google Drive, folder `ooa-nanochat/sft/`. The Colab training notebook's training cell
   mixes it in.
4. After training, download the new `chatsft_checkpoints/d8` files into `rpa model\nanochat_base\` and restart the assistant.

## Files

| File | What it is |
|---|---|
| `app.py` | The local web server and all `/api/...` routes, the feedback store and the export |
| `knowledge.py` | Catalog search, action cards, lessons, flow explanations, quizzes |
| `model_backend.py` | Loads the model on CPU and stops it early when it starts repeating itself |
| `runner_bridge.py` | Sandboxed local runs through the platform's reference runner |
| `static/index.html` | The page |
| `assistant_data/` | Created on first start: `feedback.jsonl`, `sandbox/`, `exports/` |

Tests (no model needed): `python tests/test_app.py` (and the other `tests/test_*.py`). They need the platform repo; set
`OOA_PLATFORM` to its folder if it is not next to this repo. Some tests are skipped (they print `SKIP name`) unless
these are set:

| Variable | What it enables |
|---|---|
| `OOA_PLATFORM` | the platform repo folder (example flows, reference runner) |
| `OOA_CATALOG` | the full `catalog.jsonl` (default: the repo copy under `rpa model/data/catalog/`) |
| `OOA_SCHEMA` | the flow schema (`contracts/flow-schema/flow.v1.json`), for the checker tests |
| `OOA_MODEL_DIR` | the model folder (`nanochat_base`), for the real-model tests |
| `OOA_NANOCHAT_DIR` | the nanochat code folder, for the real-model tests |

`tests/test_ui.py` needs Playwright (`pip install playwright` and `playwright install chromium`); without it it is skipped.

**Security notes.** The server only listens on 127.0.0.1. It also refuses requests whose `Host` is not
`127.0.0.1:PORT` / `localhost:PORT`, requests from other web sites (`Origin`), and any POST without
`Content-Type: application/json` and the header `X-OOA-Assistant: 1` (the page sends it), so a web site you visit cannot
make it run flows. During a run every file path an action opens must be inside the sandbox folder, even paths built
at run time. Build requests are limited to 500 characters and 6 steps: longer ones are not sent to the model.
