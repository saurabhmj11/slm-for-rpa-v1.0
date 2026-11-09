# OOA Assistant: what it is, how it learns, what is proven

*Plain-English guide for colleagues and clients. Status as of 2026-11-09. Honest numbers are in section 6.*

## 1. The problem
OneOfficeAutomation flows are JSON documents (1,300+ possible actions, loops, conditions, error handling). Writing one by hand is slow.
We want to say what we need in plain words and get a correct flow, **without** sending client data to a model we do not control.

## 2. The idea in one picture
```
 user request
      |
  [ router ]  simple?  ->  OUR OWN small model  -> repair -> VERIFY -> flow            (cheap, local, private)
      |                                                          | fails
      +--- complex / failed, and the client allows it -> CLAUDE -> repair -> VERIFY -> flow
                                                                 | still fails
                                                      honest error (never an unchecked flow)
```
The **verifier** is the platform's own rules: valid JSON, matches the Flow schema, every action and input really exists, every
variable declared. It is cheap, exact, and does not need a person or another AI to judge.

## 3. The pieces (all in this repository)
| Piece | What it does |
|---|---|
| Catalog | Reads the platform docs: every action, its inputs and outputs (1,300+ entries) |
| Generator | Builds thousands of valid training flows (1-4 steps, loops, if, try/catch) with no AI involved |
| Our model | A small transformer **trained from scratch by us** (nanochat recipe, MIT licence). No Qwen, no one else's weights |
| Verifier + reward | Scores any answer 0 to 1 using the platform's rules |
| Server + router | One OpenAI-compatible endpoint; plugs into the platform's AI Gateway as a normal model connection |
| Client policy | Switches each client controls: allow cloud help, mask PII, store logs, allow training on their data (default: local only, store nothing) |

## 4. How it improves by itself (three loops, one scoreboard)
All three are judged by the same verifier, so "better" always means *passes the platform's rules more often*.
1. **Data loop (self-improvement rounds):** the model attempts fresh requests; verified wins become new training data; failures are
   stored with the correct flow and retried. Training restarts from the base model each time, never only on its own output.
2. **Policy loop (reinforcement):** the model writes 8 answers per request; answers the verifier scores above average are reinforced.
3. **Recipe loop (autoresearch, new):** inspired by Andrej Karpathy's `autoresearch` (MIT). Each experiment changes **one** thing in
   the training recipe (data mix or learning settings), trains for the **same fixed 12M tokens**, and is scored on a **frozen** test
   (40 generated + 4 real flows, hash-checked). It is kept only if it beats the best by more than a noise margin. It runs with a
   scripted search (`loop.py`) or with an AI agent following `autoresearch/program.md`.

Safeguards that make the numbers trustworthy: the scoreboard and referee code are read-only for experiments; the frozen test is
hash-checked; training rows that overlap it are removed automatically; the budget cannot be changed; every result is logged.
(The runner refuses a changed budget or unknown settings: tested.)

## 5. Real use adds a fourth loop
Accepted and edited flows from Studio (when the client's policy allows) become training data, **only if they also pass the verifier**.
This is the long-term source of real examples; the endpoint to receive them exists, the Studio side does not yet.

## 6. What is proven and what is not
**Tested here (no GPU needed):** verifier and reward (12 checks), flow generator (30,000 flows with full schema checks in about a minute),
router and server (8 end-to-end tests with fake models: routing, repair, PII masking, policy switches, honest errors), the
autoresearch search logic (keeps real gains, discards noise and crashes, resumes), and the runner's guards. All 48 real example flows
pass the schema and catalog checks.

**Measured on a real run (free Colab T4):** a small 8-layer model pretrained for 300M tokens, then fine-tuned. On 20 fresh generated
requests per round it solved **5-10%** exactly, and answers used real action names only about **22%** of the time and correct input
names about **3%** of the time. That is an early, small, weak result, **not** a working product.

**Important finding (2026-11-07):** real flows average **18 steps and about 11 different actions across about 4 toolkits**; our generated
training flows have 1-4 steps. A search for the right actions from a one-sentence description finds only about **10%** of what real flows use.
So scores on generated flows do **not** predict performance on real flows. This is why the design keeps our small model for **simple**
requests and escalates complex ones to Claude (when allowed), always behind the verifier.

**Not done / not tested:** the bigger fine-tuning run is pending a result; reinforcement learning has not been run; the autoresearch
loop has not been run on a GPU (each experiment is estimated at about 10 minutes on a T4, not measured); the Claude call is untried
against the live API; the local-model backend needs a GPU; browser version, per-client adapters, true constrained decoding are not built.

## 7. Questions clients ask
- **Is the model open-source or Chinese?** The model weights are ours, trained from scratch. The training code is open-source (MIT, US author).
  Pre-training text and a small chat dataset are public datasets: their licences must be listed before sales use (not yet verified).
- **Does our data leave?** Not by default. Cloud help is off unless the client's policy turns it on, with PII masking before anything leaves.
  Masking is pattern-based and will miss things; in production the platform's AI Gateway PII guard should also be used.
- **Can we train it on our data?** Planned: per-client adapters kept in the client's environment. Logged data is flagged for training only if allowed.
- **How do you know it is right?** Every flow is checked against the platform's own rules before anyone sees it. That proves it is *valid*,
  not that it does the *right job*; users still review the flow before it runs.

## 8. Demo script (10 minutes)
1. Show the catalog and a real flow next to a generated one (explain why real flows are harder).
2. Run the server tests: `python serving/test_server.py PLATFORM_REPO` (routing, masking, policy, honest error).
3. Show the policy file and flip `allow_escalation` to show local-only versus cloud-assisted behaviour.
4. Show `results.tsv` from the autoresearch loop (keep / discard / crash rows) and the frozen-test hash check.
5. Close with section 6: what is proven, what is not.

## Credits
nanochat and autoresearch by Andrej Karpathy (MIT licence). This repository's code was written for OOA; ideas, not code, were taken from autoresearch.
