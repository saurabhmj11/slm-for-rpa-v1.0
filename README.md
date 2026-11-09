# OOA Assistant (slm-for-rpa)

An in-house assistant that writes OneOfficeAutomation Flow JSON from plain-language requests, built around a small model we
train ourselves, a verifier that checks every answer against the platform's rules, a router that can escalate hard requests to
Claude (only if the client's policy allows), and loops that let the system improve itself.

**Start here:** [`rpa model/docs/EXPLAINER.md`](rpa%20model/docs/EXPLAINER.md) (what it is, how it learns, what is proven).
Plan and notes: [`rpa model/docs/TRACK_B_NANOCHAT.md`](rpa%20model/docs/TRACK_B_NANOCHAT.md).

## Layout (`rpa model/`)
| Folder | What |
|---|---|
| `data/catalog/` | builds `catalog.jsonl` from the platform's toolkit docs |
| `training/` | flow generator, SFT data, verifier/reward, RL task, self-improvement, retrieval, repair, Colab and Kaggle notebooks |
| `training/autoresearch/` | frozen scoreboard, fixed-budget experiment runner, scripted search loop, agent instructions (`program.md`) |
| `serving/` | OpenAI-compatible server, router, client policy, feedback endpoint |
| `eval/` | flow checker (schema + catalog) |

## Run the tests (no GPU, needs `jsonschema`; platform repo cloned somewhere)
```bash
python "rpa model/training/test_reward.py" PLATFORM_REPO
python "rpa model/serving/test_server.py" PLATFORM_REPO
python "rpa model/training/autoresearch/test_loop.py"
```
Train: open `rpa model/training/ooa_nanochat_colab.ipynb` (Colab) or `ooa_nanochat_kaggle.ipynb` (Kaggle) and follow the cells.

## Status
Early. Pipeline and safeguards are tested; the trained model is small and weak, and real flows (about 18 steps) are much harder than
the generated training flows. See the explainer for honest numbers. Not production software.

Built on ideas and code from [karpathy/nanochat](https://github.com/karpathy/nanochat) (MIT); the recipe loop follows
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) (MIT).
