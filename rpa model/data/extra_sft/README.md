# New raw training files shipped with the code

Neutral records (`{"kind": "flow" | "rewrite" | "clarify" | ...}`) of `training/sft_setup.RENDER_EXTRA` files that were
made after the last Colab render, so the PC notebook can train on them even when they are not on Google Drive yet:

- `session_train.jsonl` (733 records: 683 flows + 50 friendly questions): sign in / open once, later steps reuse the
  session, sign out / close at the end.

`training/local/data_pc.py` uses a file from here only when Drive's `sft/` does not have it (Drive's upload wins), and
only for the modes that continue a trained model (refine, topup; also install / rl for their context): it renders the
file alone with the Colab render's settings, catalog and example index (`render_extras`). The full run (MODE "full")
stays exactly the Colab data. Upload the file to Drive `ooa-nanochat/sft/` too, so the next Colab run trains on it.
