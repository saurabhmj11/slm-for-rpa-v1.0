# Plug the OOA assistant into the platform (no platform code change)

The platform already supports "One Model Connections" with an `openai_compatible` provider (local or hosted servers with the OpenAI
interface), and Studio's assistant ("Build from a description") calls one by name. Our server speaks exactly that.

## What Studio sends and expects (read from `StudioServer.assistantBuild` and `LlmClient`)
- It calls `POST {endpoint}/chat/completions` with a system prompt (its instructions plus the live action catalog) and a user message
  `Existing variables: [...]\nRequest: <what the user typed>`, temperature 0, max_tokens 4000.
- It reads `choices[0].message.content`, finds the first `{`..last `}`, and expects `{"variables": [...], "steps": [...], "notes": "..."}`;
  it then validates the steps itself and shows problems. It reads `usage.prompt_tokens` / `completion_tokens` for budgets.
- Our server returns exactly that shape, verified first (schema, real actions, inputs, declared variables). If nothing verifies it returns
  empty steps with the reason in `notes`.

## Steps
1. Start the server where the platform can reach it:
   `python serving/ooa_server.py --catalog data/catalog/catalog.jsonl --platform PLATFORM_REPO --policy serving/policy.example.json --port 8080`
   add `--local nanochat` for our trained model, and set `allow_escalation: true` + `ANTHROPIC_API_KEY` to let hard requests go to Claude.
2. In the platform: AI > One Model Connections > New: Provider `openai_compatible`, Model `ooa-assistant`, Endpoint `http://HOST:8080/v1`,
   Key: leave empty (our server needs none; the platform only requires a key for `openai`). Optionally restrict roles and set budgets.
   If the platform refuses plain http for a non-local host, put the server behind https.
3. In Studio: Build from a description, pick that connection.
4. Check: `python serving/test_server.py PLATFORM_REPO` (11 tests, including Studio's request and answer shape).

## Not yet verified against a live platform
These tests use fake models and a simulated Studio request. Nobody has registered the connection in a running Command Center yet; the
exact UI labels above come from the docs (docs/functional/09-ai-features.md) and may differ slightly.
