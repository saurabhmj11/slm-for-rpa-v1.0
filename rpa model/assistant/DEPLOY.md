# Running the OOA Assistant as a server for One Agent Studio

This runs the flow-writing model in a container (CPU only) so Studio's **Assistant** on other machines can use it as an
OpenAI-compatible model connection. On a single PC you do not need any of this: see [README.md](README.md).

What the server offers:

| Route | Auth | What it does |
|---|---|---|
| `POST /v1/chat/completions` | bearer token | OpenAI-style request in, flow JSON out (Studio's `{variables, steps, notes}` for Studio's Assistant) |
| `GET /v1/models` | bearer token | lists the one model id, `ooa-assistant` |
| `GET /api/assistant/v1/admin/version` | bearer token | `{version, contract, model_step, catalog_actions, catalog_sha256, schema_sha256}` |
| `GET /health` | none | `200 {"ok": true}` while the process is up (also while the model loads) |
| `GET /ready` | none | `200` once the model, the action catalog and the flow schema are loaded; `503` with `missing` before |

Other behaviour worth knowing:

- One generation at a time. Up to 4 more requests wait (`OOA_MAX_WAITING`); one more gets `429` with `Retry-After`.
- Each request has a time budget of 120 seconds (`OOA_TIME_BUDGET`): after it, no retry attempt starts. One attempt
  takes from a few seconds to about a minute on a CPU.
- The connection check messages ("Reply with the single word OK.", "Answer with the single word OK.") are answered
  with `OK` at once, without the model.
- Requests with `tools` are refused with `400`: the model writes flows only.
- The browser page of the assistant (Build / Learn / Run tabs, `/api/*`) is off while a token is set, which is always
  the case in the container. It has no login of its own, so it is not served to anyone, not even the machine itself.

## 1. Build the image

Build from the `rpa model` folder of this repo. The platform repo is a second build context: only its flow schema
(`contracts/`) and `docs/llms.txt` are copied in, so use the platform version that Studio runs.

```
cd "rpa model"
docker build -f assistant/Dockerfile \
  --build-context platform=/path/to/oneofficeautomation-platform \
  -t ooa-assistant:2026-11-14 .
```

Needs Docker 23 or newer (BuildKit, for `--build-context`). The image pins the nanochat code to commit `92d63d4`, the
version the model was trained with (build argument `NANOCHAT_COMMIT`), and installs the CPU-only build of PyTorch.
Tag images with a date or a version number, not only `latest`, so you can roll back.

## 2. Put the model files on the host

The image has no model in it. Keep each model version in its own folder under one models folder, which the container
mounts read-only at `/models`:

```
/srv/ooa-models/
  2026-11-14/                       <- one model pack
    VERSION                         <- one line, e.g. 2026-11-14 (shown as "version" by the version route)
    nanochat_base/tokenizer/        <- tokenizer.pkl, token_bytes.pt
    nanochat_base/chatsft_checkpoints/<depth>/model_<step>.pt and meta_<step>.json
    sft/catalog.jsonl               <- optional: the action list the model was trained with
  2026-10-02/                       <- the previous pack, kept for roll-back
```

The files come from the training run's output (the `ooa-nanochat` Drive folder: `nanochat_base/tokenizer/`,
`nanochat_base/chatsft_checkpoints/...` and `sft/catalog.jsonl`). Without `sft/catalog.jsonl` the catalog built into
the image (`data/catalog/catalog.jsonl`) is used.

## 3. Run it

Make a long random token (at least 16 characters) and keep it in your secret store:

```
openssl rand -hex 32
```

```
docker run -d --name ooa-assistant --restart unless-stopped \
  -p 8765:8765 \
  -e OOA_ASSISTANT_TOKEN=<the token> \
  -e OOA_MODEL_PACK=2026-11-14 \
  -v /srv/ooa-models:/models:ro \
  -v ooa-assistant-data:/data \
  ooa-assistant:2026-11-14
```

- `OOA_ASSISTANT_TOKEN` is required: the server listens on all interfaces inside the container and refuses to start
  without a token. Prefer `--env-file` or your orchestrator's secrets over typing it on the command line.
- `OOA_MODEL_PACK` names the folder under `/models` to load (default `current`).
- `/data` keeps feedback from the browser page; it can be an empty volume.
- Optional: `OOA_MAX_WAITING` (default 4), `OOA_TIME_BUDGET` (seconds, default 120).

Check it:

```
curl http://<host>:8765/health
curl http://<host>:8765/ready                       # 503 until the model has loaded (about a minute)
curl -H "Authorization: Bearer <the token>" http://<host>:8765/api/assistant/v1/admin/version
curl -H "Authorization: Bearer <the token>" -H "Content-Type: application/json" \
  -d '{"model":"ooa-assistant","messages":[{"role":"user","content":"Reply with the single word OK."}]}' \
  http://<host>:8765/v1/chat/completions
```

The container's health check calls `/health`. Use `/ready` for a load balancer or a readiness probe.

Put it on a private network or behind a TLS proxy: the token travels in a header, so plain HTTP is only safe on a
network you trust.

Always set the token, also when the server listens on 127.0.0.1 and a proxy on the same machine forwards to it. A
proxy connects from 127.0.0.1, so without a token every client of the proxy can use the model routes. With a token
the model routes need it and the browser page is off. The page also refuses any request that carries `Forwarded`,
`X-Forwarded-For`, `X-Forwarded-Host` or `X-Real-IP`.

## 4. Point Studio at it

On the machine where the platform runs, set an environment variable that holds the token (for example
`OOA_ASSISTANT_TOKEN`) for the platform process, or save the token as a credential. Then in Studio: sun icon >
**One Model Connections...** > **+ Connection** (the "Other compatible service" preset):

| Field | Value |
|---|---|
| Name | `ooa` |
| Provider | OpenAI-compatible (`openai_compatible`) |
| Model | `ooa-assistant` |
| Endpoint (base URL) | `http://<host>:8765/v1` |
| API key | the environment variable holding the token (`api_key_env`), or the credential |
| Use it by default when an agent names no connection | **untick it** (it starts ticked when the workspace has no default connection yet) |

Do not make it the default connection. Every GenAI step that has no connection of its own is sent to the default
connection, and this model only writes flows; it cannot answer general prompts.

**Test** should answer `OK`. Then **Assistant > Build from a description**, pick `ooa` as the Model Connection,
describe the job and **Generate steps**. In `model-connections.json` of the workspace the entry looks like:

```json
{"name": "ooa", "provider": "openai_compatible", "model": "ooa-assistant",
 "base_url": "http://<host>:8765/v1", "api_key_env": "OOA_ASSISTANT_TOKEN",
 "description": "Flow-writing model (OOA Assistant)"}
```

## 5. Upgrade

1. Copy the new model pack into its own folder, e.g. `/srv/ooa-models/2026-12-01/` (with its `VERSION`). Do not
   overwrite or delete the folder the running container uses.
2. If the code or the platform version changed, build a new image with a new tag (step 1).
3. Start the new version and check `/ready` and the version route before sending traffic to it:
   ```
   docker rm -f ooa-assistant
   docker run -d --name ooa-assistant ... -e OOA_MODEL_PACK=2026-12-01 ... ooa-assistant:2026-12-01
   ```
4. In Studio, **Test** the `ooa` connection and build one flow you know.

The version route shows `catalog_sha256` and `schema_sha256`: if the platform was upgraded, rebuild the image against
the new platform so the schema matches what Studio checks.

## 6. Roll back

Start the previous image tag with the previous model folder, which you kept:

```
docker rm -f ooa-assistant
docker run -d --name ooa-assistant ... -e OOA_MODEL_PACK=2026-10-02 ... ooa-assistant:2026-10-02
```

Keep at least the previous model folder and image tag until the new version has run for a while.

## Running without Docker

The same server runs straight from the repo:

```
OOA_ASSISTANT_TOKEN=<the token> python assistant/app.py --host 0.0.0.0 \
  --model-dir /srv/ooa-models/2026-11-14/nanochat_base --nanochat /path/to/nanochat --platform /path/to/platform
```

`--allowed-hosts NAME ...` adds Host header names the browser page accepts when no token is set (for example another
name for this machine). The model routes do not check the Host header when a token is set: the token protects them.
With a token the browser page is off; to use it, start a second copy on this machine without `--host` and `--token`.
