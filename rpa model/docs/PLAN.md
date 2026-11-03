# OneOfficeAutomation Conversational AI: Step-by-Step Plan

Goal: an in-house conversational assistant that understands the OneOfficeAutomation (OOA) platform. It
builds and edits OOA Flow JSON, operates bots, explains failures, and remembers context across a conversation.

## Assumptions (change if wrong)
- Scope: **Studio co-pilot first** (chat to Flow JSON), then Control Room operations.
- Base model: **Qwen2.5-7B-Instruct**, fine-tuned with QLoRA. Fallback: 3B if GPU memory is tight.
- Serving: vLLM (constrained decoding) or Ollama. Platform integration: the existing `openai_compatible` provider.
- The tiny from-scratch models in this folder are not used for the assistant. A small router can come later.
- Source repo: `OneOfficeAutomation-Platform`. The assistant is verified by that platform's own validator.

## Success criteria (what "perfect" means, measurable)
| Metric | Target for v1 |
|---|---|
| Output passes `ooa validate` | >= 98% |
| Correct action chosen (vs. gold flow) | >= 90% |
| Flow runs and passes in sandbox on the test set | >= 80% |
| Asks a clarifying question when a required input is missing | >= 90% |
| Recalls a fact stated 10 turns earlier | >= 90% |
| Sensitive actions proposed without approval flag | 0% |

## Phase 0: Foundations (week 1)
1. Python 3.11/3.12 venv (the current Python 3.14 + torch is blocked by Application Control). Confirm GPU access
   (local NVIDIA 16GB+ or a rented A100).
2. Put the project under git. Ignore large data and checkpoints.
3. Clone the platform repo as a pinned dependency. Install the Python reference (`cd reference && pip install -e ".[test]"`)
   and run its tests, so `ooa validate` and `ooa manifest` work locally.
4. Decide the data policy: may docs and flows go to a cloud LLM for synthetic data? If not, generate with a local model.

## Phase 1: Knowledge extraction (week 1-2)
1. Run `ooa manifest` and parse `docs/reference/toolkits/*.md` into a single **action catalog** (id, description,
   typed inputs and outputs, risk level). About 1,100 actions.
2. Parse the 42 use-case flows, the refund example, the templates and the conformance cases into a gold set.
3. Chunk the architecture and functional docs for retrieval.
4. Build the **retrieval index**: embeddings of action docs and doc chunks (bge-m3 or similar), plus keyword search.

## Phase 2: Evaluation first (week 2)
1. Write 150-300 evaluation tasks in four groups:
   - **Create:** request to flow (varied wording, typos, vague requests).
   - **Edit:** a multi-turn change to an existing flow ("add a retry", "also email the result").
   - **Explain and fix:** a flow plus an error, to an explanation and a corrected flow.
   - **Operate:** run, schedule, check status, and approval handling.
2. Include **memory probes** at distances of 1, 5, 10 and 30 turns, in the style of `context_bench.py`.
3. Automatic scoring: schema validity, action match, run in sandbox, and a human review of a sample.
4. Freeze the test set. Never train on it.

## Phase 3: Baseline with no training (week 2-3)
1. Off-the-shelf Qwen2.5-7B-Instruct plus the `llms.txt` rules as the system prompt.
2. Retrieval of the top-k actions and doc chunks per turn, and the Flow JSON schema as a constrained-decoding grammar.
3. A **validate-and-repair loop**: generate, run `ooa validate`, feed errors back, and retry up to 3 times.
4. Measure against Phase 2. This is the number training must beat. It may already be good enough for some groups.

## Phase 4: Context and memory layer (week 3-4)
1. **Conversation state:** the current flow draft, the user's goal, answers collected so far.
2. **Rolling summary** of older turns.
3. **Long-term memory:** user and team preferences, and past flows, in a vector store.
4. **Prompt assembly:** system rules, retrieved actions, draft flow, summary, then the recent turns, within a token budget.
5. Re-run the memory probes.

## Phase 5: Dataset (week 3-5)
1. Seeds: the gold flows, the docs, and (if available) real customer flows. Real flows are worth more than synthetic ones.
2. Generate multi-turn dialogues with a strong LLM from the action catalog. Cover happy paths, missing parameters,
   corrections, errors, approvals and out-of-scope requests.
3. **Filter hard:** keep only examples whose flows pass `ooa validate`, and where possible a sandbox run.
4. Deduplicate, split by workflow family (so test flows are never near-copies of training flows), and review
   a sample by hand.
5. Target 5,000-20,000 dialogues. Quality matters more than count.

## Phase 6: Fine-tune (week 5-6)
1. QLoRA on the base model: rank 64, learning rate 2e-4, loss on assistant tokens only.
2. Train on retrieval-augmented prompts (the same format used at inference), so the model learns to use the
   retrieved actions.
3. Compare against the Phase 3 baseline on the frozen test set. Keep the fine-tune only if it wins.
4. Optional: preference tuning (DPO) using validator pass/fail as the signal.

## Phase 7: Integration (week 6-8)
1. Expose the model as an OpenAI-compatible endpoint and register it as a One Model Connection.
2. Add a chat panel to Studio. The assistant proposes a flow as a diff, and the user accepts it.
3. Add the Control Room tools (run, schedule, status, approvals) as typed tools. Sensitive tools require human approval,
   enforced in code, not by the model.
4. Log every model call through the platform's AI governance (PII masking, budgets, audit).

## Phase 8: Pilot and improve (week 8+)
1. Internal pilot with 5-10 users. Capture thumbs up and down and the edits users make to generated flows.
2. Turn corrected flows into new validated training data and retrain monthly.
3. Track the success metrics on a dashboard. Add new toolkits by re-running Phase 1, with no retraining needed for catalog updates.

## Risks and mitigations
| Risk | Mitigation |
|---|---|
| Model invents actions or parameters | Retrieval, schema-constrained decoding, validator loop |
| Overfitting to 42 example flows | Split by workflow family, add real data, large synthetic set |
| Unsafe actions | Approval and permission checks in code, not in the model |
| Sensitive data leakage | Platform PII masking, on-premises generation if required |
| Catalog changes break the model | Retrieval-based design, regenerate the catalog on each platform release |
| Scope creep | Ship the Studio co-pilot first, then Control Room operations |

## Open questions that change the plan
1. Cloud LLM allowed for generating data, or fully on-premises?
2. GPU: local card (VRAM?) or rented?
3. Do real customer flows exist that can be used?
4. Team size and deadline.
