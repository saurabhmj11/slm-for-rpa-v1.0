# Stage 2: BPE tokenizer on the OOA corpus

Date: 2026-11-03. Script: `tokenizer/train_tokenizer.py`. Trained tokenizers: `tokenizer/out/ooa_bpe_{512,1024,2048,4096}.json`.

## Setup
- Corpus copied to `data/raw/`: 48 flow JSON files (+ schema) and 109 doc files (toolkit reference, architecture, functional docs).
- Held out 10 flows for measuring compression. Tokenizers trained on 38 flows (minified) + all docs = 2.16 MB.
- Algorithm: byte-level BPE with the GPT-4 split regex (same as minbpe's RegexTokenizer), with merges run over
  unique chunks weighted by frequency so training takes seconds.

## Results (10 held-out flows)
| vocab | train time | tokens, minified | bytes/token | tokens, pretty-printed |
|---|---|---|---|---|
| bytes only | - | 134,159 | 1.00 | 288,142 |
| 512 | 9 s | 61,300 | 2.19 | 174,097 |
| 1024 | 24 s | 48,313 | 2.78 | 118,811 |
| 2048 | 54 s | 41,059 | 3.27 | 95,725 |
| 4096 | 110 s | 38,030 | 3.53 | 82,908 |

## What it shows
1. **Minifying JSON is the biggest single win.** Pretty-printed flows are 2.15x the bytes of minified ones, and cost
   2.2x the tokens at vocab 4096 (82,908 vs 38,030). Train and run the model on minified JSON.
2. **Returns diminish.** Going 512 to 1024 saves 21% of tokens, 2048 to 4096 saves only 7%. A bigger vocab also makes
   the embedding table larger and rare tokens poorly trained.
3. **Context length is the real constraint.** At vocab 4096, held-out minified flows are 617 to 14,338 tokens, median
   about 2,000. Only 5 of 10 fit in a 2048-token window, and that is before adding the request, retrieved action docs
   and chat history. The assistant cannot generate whole large flows in one pass with a 2048 window.
4. **The vocabulary is biased toward docs.** Docs are about 70% of the training text, so merges like
   ` ONEOFFICEAUTOMATION`, ` Bedrock`, `ControlRoom` are learned, while flow-specific strings like `","`, `":"`
   and `/flow` are learned but flow ids and action names get less weight.

## Caveats
- The held-out set is not clean: many flows have a `-capture` twin (for example `W1-...` and `W1-...-capture`). A
  held-out flow can have a near-copy in training, which flatters the compression numbers.
- 48 flows is a tiny corpus. Real compression on new flows will be a bit worse.

## Decisions for the project
- Always minify flow JSON for training and inference.
- Candidate vocab: 4096 for a small model (compression plateau), pending a check on model size and embedding cost.
- Reweight the tokenizer corpus toward flows (for example repeat flows 3x), and re-measure.
- Fix the context problem before training: use 4096-token context if affordable, or generate flows step by step
  (the plan's edit-based approach) rather than whole flows. Consider shorter step encodings (compact DSL that compiles to Flow JSON).

## Exercises
1. Read `minbpe-master/minbpe-master/minbpe/regex.py` and compare with `train_bpe` here.
2. Re-run with flows repeated 3x in the training text and compare the table.
3. Remove the `-capture` flows from the held-out comparison and see how the numbers change.
4. Compare with a stock tokenizer (tiktoken cl100k_base) on the same flows. Needs a small vocab download, not done here.
