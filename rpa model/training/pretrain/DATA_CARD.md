# Data card: the clean base model's pretraining data

What the new base model is pretrained on, where it comes from, under which licence, and what is left out. The
builder is `pretrain_data.py`; every build also writes its own `data_card.json` and `DATA_CARD.md` next to the shards
with the exact characters, shares, repeats, file hashes and clean-up counts of that build.

Licences were checked on 2026-11-25 against the dataset cards at the pinned revisions below. This is an engineering
record, not legal advice.

## Sources

| source | share of characters | licence | version |
|---|---|---|---|
| FineWeb-Edu, `sample/10BT` (`HuggingFaceFW/fineweb-edu`) | 85% | ODC-By 1.0; use also subject to the Common Crawl Terms of Use | dataset revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` (2025-07-11); files `000_00000.parquet` .. `013_00000.parquet`, sha256 per file recorded in each build's card |
| rendered OOA training conversations (sft_render: request, retrieval context, plan, flow JSON) | 15% x 0.55 | the owner's own data, generated from the platform and catalog | sha256 of the files in each build's card |
| OneOfficeAutomation platform code / JSON (`contracts`, `examples`, `templates`, `reference`, `platform`, README, SECURITY, module-map) | 15% x 0.20 | the owner's own code | sha256 of the files in each build's card |
| OneOfficeAutomation platform docs (`docs/`) | 15% x 0.15 | the owner's own docs | sha256 of the files in each build's card |
| OOA action catalog (`data/catalog/catalog.jsonl`: records + an id / name index) | 15% x 0.10 | the owner's own (built from the platform) | sha256 in each build's card |

Licence evidence:

- FineWeb-Edu card, "Licensing Information": "released under the Open Data Commons Attribution License (ODC-By) v1.0
  ... The use of this dataset is also subject to CommonCrawl's Terms of Use."
  https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu (revision above; card front matter `license: odc-by`),
  https://opendatacommons.org/licenses/by/1-0/, https://commoncrawl.org/terms-of-use.
  ODC-By allows commercial use; it asks for attribution of the database. Attribution line for the model's
  documentation: *"Pretrained in part on FineWeb-Edu (Hugging Face, ODC-By 1.0)."*
- No other third-party text is in the base data. A permissive code / JSON corpus was considered and left out: the
  platform's own code and the rendered flows already cover JSON, and a code corpus would bring per-file licences to
  track.

## Clean-up (domain sources)

In this order, counts per build in its card:

1. Files whose names look like key material are never read (`.env*`, `id_rsa*`, `*.pem`, `*.key`, `*.p12`,
   `*.pfx`, `*.jks`, `*.keystore`, `*.crt`, `credentials*`, `secrets*`, ...); build / dependency folders are skipped;
   files over 512 KB are skipped.
2. A document holding a hard secret is dropped whole: private-key blocks, AWS / GitHub / Slack / OpenAI-style / Google
   API key formats, Slack webhooks, storage AccountKey=, JWTs. (Sample build: 5 platform files, all test fixtures
   or a PEM header constant, and 1 doc page.)
3. `password` / `passwd` / `pwd` / `secret` / `api_key` / `access_token` / `auth_token` / `client_secret` literals are
   replaced by `<redacted>`; flow variables such as `"$password$"` are kept.
4. Rendered rows whose request is too close to any `eval/*.jsonl` prompt are dropped, using the same check the SFT
   data uses (`eval/beginner_score.too_close`: Jaccard over 0.6 or content-word containment of 0.7 or more). Any
   domain document that quotes an eval prompt verbatim (whitespace / case normalised) is dropped too.
5. Held-out rendered rows (`val*.jsonl`, `eval*.jsonl`, `rl_val*.jsonl`) go only to the validation shard; a training
   row asking the same request as a held-out row is dropped. `rl_train*.jsonl` (prompts without answers),
   `catalog.jsonl` and `examples.jsonl` of a render folder are not used (the catalog has its own source).
6. Exact duplicates (whitespace-normalised) are dropped across all domain sources.

FineWeb-Edu is used as published (it is already deduplicated and filtered by its authors).

## Split and format

- nanochat parquet shards, one column `text`, row groups of 1,024 documents, zstd. Train shards
  `shard_00000.parquet` ...; validation `shard_99999.parquet` (the last file, nanochat's val split).
- Validation = the last 2 row groups of FineWeb-Edu file `000_00000.parquet` (never used for training) + held-out
  rendered rows at the same 15% share.
- Domain pieces are at most 8,000 characters (cut at line ends), below `tok_train`'s 10,000-character document cap.
- Deterministic and resumable: the domain stream does not depend on the requested size; a larger rerun keeps the
  finished shards (sha256-checked) and appends. Other settings or changed domain sources are refused.

## Not included

- **ClimbMix** (`nvidia/Nemotron-ClimbMix`, CC BY-NC 4.0, non-commercial), the data of the current base model.
  nanochat downloads it from the mirror `karpathy/climbmix-400b-shuffle`, whose card says `license: mit`; a mirror
  cannot relicense the original's non-commercial terms, so it is treated as CC BY-NC 4.0.
- **SmolTalk** or any other third-party chat / instruction data. (The base data has none.)
- Any eval prompt or request too close to one; held-out val rows; key material.

## Later stages: SFT data (recorded here, not part of the base data)

The SFT recipe (`make_colab_notebook.py`, `SMOLTALK_ROWS = 3000`) mixes 3,000 rows of
`HuggingFaceTB/smol-smoltalk` (nanochat `tasks/smoltalk.py`) into the OOA rows. Its card says `license: apache-2.0`
(revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`, https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk).
But it is a subset of SmolTalk, whose core component Smol-Magpie-Ultra was "generated using the Magpie pipeline with
Llama-3.1-405B-Instruct" (https://huggingface.co/datasets/HuggingFaceTB/smoltalk). The Llama 3.1 Community License
says: "If you use the Llama Materials or any outputs or results of the Llama Materials to create, train, fine tune, or
otherwise improve an AI model, which is distributed or made available, you shall also include 'Llama' at the
beginning of any such AI model name" and asks for "Built with Llama" when distributing
(https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE, section 1.b.i). SmolTalk also folds
in OpenHermes-2.5 and other sets with their own terms. For a commercial model that is distributed, set
`SMOLTALK_ROWS = 0` (or have the terms reviewed) so the clean base is not followed by an encumbered SFT.

## Tokenizer

The new base model gets its own tokenizer, trained with nanochat `scripts/tok_train.py` (rustbpe, nanochat's split
pattern and its 9 chat special tokens, unchanged) on the first characters of the built train shards:

    NANOCHAT_BASE_DIR=<base> python -m scripts.tok_train --vocab-size 16384 --max-chars 2000000000

(`<base>/base_data_climbmix` = the built shards: the folder name is nanochat's.) The tokenizer is part of the model
pack; a model trained with it must never be served with another tokenizer. `tok_compare.py` measures tokenizers on
held-out text. Sample build (313M characters, 15% domain), held-out text:

| tokenizer | vocab | request | context | plan | flow JSON | whole conversation | eval prompts | FineWeb-Edu val | flow tokens | action id tokens (mean / max) |
|---|---|---|---|---|---|---|---|---|---|---|
| current (web text only) | 16,384 | 3.94 | 2.71 | 2.82 | 2.14 | 2.73 | 3.79 | 4.34 | baseline | 11.4 / 24 |
| FineWeb-Edu only | 16,384 | 3.92 | 2.68 | 2.78 | 1.72 | 2.64 | 3.79 | 4.38 | +24.7% | 11.4 / 24 |
| **web + 15% domain** | **16,384** | **4.22** | **3.81** | **3.69** | **4.17** | **3.85** | **4.17** | **4.33** | **-48.8%** | **5.5 / 13** |
| web + 15% domain | 24,576 | 4.28 | 3.87 | 3.78 | 4.30 | 3.93 | 4.32 | 4.53 | -50.3% | 5.1 / 12 |
| web + 15% domain | 32,768 | 4.32 | 3.91 | 3.82 | 4.36 | 3.96 | 4.40 | 4.65 | -51.0% | 4.9 / 12 |

(Characters per token; higher is better. Special tokens: same names, order and ids at the end of the vocabulary in
every row.) Recommended: **16,384**. Going to 24,576 saves only 3% of flow tokens and 2% of conversation tokens but
adds 25.2M parameters at d8 (wte + lm_head + 4 value-embedding tables: 6 x 512 per vocabulary entry; 75.5M -> 100.7M)
and 12% more FLOPs per token (lm_head); at d10 +36.7M (7 x 640 per entry; 122.6M -> 159.3M). The rare extra tokens would also get few
training updates in a few hundred million tokens of pretraining.

## Size

Measured on the sample: 0.41 GB of shards per 1B characters; the 16,384 domain tokenizer gives 4.18 characters per
token on the training mix, so **about 1.7 GB of shards per 1B training tokens** (4.2B characters). nanochat's d8
default horizon (12 tokens per scaling parameter) is 403M tokens = about 0.7 GB. Building is network-bound: about 6 s
per 100M characters here, plus about 90 s to prepare the domain sources.

The domain stream holds about 503M characters (each source at its maximum passes: render 3, code 4, docs 4, catalog
6), enough for a 15% share up to about 3.3B training characters (about 0.8B tokens). Beyond that the share falls
(the builder warns). `--repeat-scale 2` doubles every source's passes (render 6, code 8, docs 8, catalog 12: about 1B
characters, a 15% share up to about 6.7B characters); the v6 profiles (`base_setup.py`, 1.51B tokens, about 6.8B
characters) use it. It is a build setting: shards built with another value are refused, older manifests count as 1.
