# New raw training files shipped with the code

Neutral records (`{"kind": "flow" | "rewrite" | "clarify" | ...}`) of `training/sft_setup.RENDER_EXTRA` files that were
made after the last Colab render, so the PC notebook can train on them even when they are not on Google Drive yet:

- `session_train.jsonl` (733 records: 683 flows + 50 friendly questions): sign in / open once, later steps reuse the
  session, sign out / close at the end.
- `decomposition_train.jsonl` (1273 records: 961 long flows + 312 step-builder parts): 7-16-step business processes
  (finance / AP / HR / IT / sales / ops) across 2-3 apps with sessions, loops, ifs, try / catch and the glue steps
  (table -> to_records -> loop, now -> parse -> difference -> if, rest/request -> json/to_table, OCR -> extract ->
  rename, AI inside email loops, the right ticketing app), each with a clear request and the gold flow; plus single
  parts of those flows asked the way the server's parts builder asks them ("With the current <item>: ...", the
  variables the earlier steps made given as known variables) -> a 1-3 action fragment.
- `wording_train.jsonl` (1360 records: 1113 flows + 247 questions): messy wording -> clear request + flow: Indian
  English ("kindly do the needful", "today itself", "one excel", "prepone"), story style, voice-style rambling,
  Hinglish, typos, very short requests; and the same styles with a detail only the user knows left out (which folder,
  who gets it, which rule, which system) -> one question.

Every flow passed repair.session_fix (no change) and has no ooa_reward problems against data/catalog/catalog.jsonl and
the platform's flow.v1 schema; every flow request names the details its flow hard-codes (addresses, channels, sheets,
columns, thresholds, input paths); no request is too close (eval/beginner_score.too_close) to any eval/*.jsonl prompt.
Records carry the thinking fields (understand, facts, missing, steps) for render_dataset.py --think.

`training/local/data_pc.py` uses a file from here only when Drive's `sft/` does not have it (Drive's upload wins), and
only for the modes that continue a trained model (refine, topup; also install / rl for their context): it renders the
file alone with the Colab render's settings, catalog and example index (`render_extras`). The full run (MODE "full")
stays exactly the Colab data. Upload the files to Drive `ooa-nanochat/sft/` too, so the next Colab run trains on them
(Colab cell 1 NEW_FILES lists them; cell 6d says MISSING for one that is not uploaded).
