# New raw training files shipped with the code

Neutral records (`{"kind": "flow" | "rewrite" | "clarify" | ...}`) of `training/sft_setup.RENDER_EXTRA` files that were
made after the last Colab render, so the PC notebook can train on them even when they are not on Google Drive yet:

- `session_train.jsonl` (730 records: 680 flows + 50 friendly questions): sign in / open once, later steps reuse the
  session, sign out / close at the end.
- `decomposition_train.jsonl` (1269 records: 961 long flows + 308 step-builder parts): 7-16-step business processes
  (finance / AP / HR / IT / sales / ops) across 2-3 apps with sessions, loops, ifs, try / catch and the glue steps
  (table -> to_records -> loop, now -> parse -> difference -> if, rest/request -> json/to_table, OCR -> extract ->
  rename, AI inside email loops, the right ticketing app), each with a clear request and the gold flow; plus single
  parts of those flows asked the way the server's parts builder asks them ("With the current <item>: ...", the
  variables the earlier steps made given as known variables) -> a 1-3 action fragment.
- `wording_train.jsonl` (1353 records: 1106 flows + 247 questions): messy wording -> clear request + flow: Indian
  English ("kindly do the needful", "today itself", "one excel", "prepone"), story style, voice-style rambling,
  Hinglish, typos, very short requests; and the same styles with a detail only the user knows left out (which folder,
  who gets it, which rule, which system) -> one question.

Every flow passed repair.session_fix (no change) and has no ooa_reward problems against data/catalog/catalog.jsonl and
the platform's flow.v1 schema; no request is too close (eval/beginner_score.too_close) to any eval/*.jsonl prompt.

Grounding (`training/grounding_check.py`, fixed by `training/fix_grounding.py`, tested by `training/test_grounding.py`):
a flow hard-codes only details its request (for a step-builder part: the step text and the known variables) gives.
Anything else -- a saved connection (`$jira_connection`, as the platform's Connections), an address, a SharePoint site,
an AD distinguished name (`$leavers_ou`, `$all_staff_group_dn`, also inside an LDAP filter), a team / channel / group /
sheet / table / procedure / warehouse / mailbox / project / printer name, a database type, a currency, a link, a path
or the folder of a path (also an unzip archive, a knowledge index, a sharing-link item) -- is an input variable with a
description and no default. A new file the flow writes keeps its name (in a folder the request names, or in a folder
variable: `${download_folder}\fx_rates.zip`), and every step that reads it uses the same text; a step never reads a
file the request does not name or an earlier step did not write. A file type filter the request does not ask for is
dropped ("all files" -> no extension filter), subjects / bodies name no made-up company or link, and template text
("2 copy/copies", "hour(s)") is written out. The texts (clear, understand, facts, steps lines, flow / step / variable
descriptions) only repeat the request and say "the download folder", "the policy PDF", "the printer" ... for a
variable (never "the given folder"); a step-builder part's understand is its request. A value built from a row inside
text is written `${row.Name}` (`D:\out\${row.Name}.txt`; a bare `$row.Name` inside text is sent literally). Clarify
records carry "missing". Records the fix cannot make faithful (a made-up Salesforce field, a guessed reply field, two
made-up folders that would read the same) are dropped: 14 of 3366. After a regenerate, run
`python training/fix_grounding.py data/extra_sft/*.jsonl` and `python training/grounding_check.py data/extra_sft/*.jsonl`
(exit 0 = nothing made up).
Records carry the thinking fields (understand, facts, missing, steps) for render_dataset.py --think.

`training/local/data_pc.py` uses a file from here only when Drive's `sft/` does not have it (Drive's upload wins), and
only for the modes that continue a trained model (refine, topup; also install / rl for their context): it renders the
file alone with the Colab render's settings, catalog and example index (`render_extras`). The full run (MODE "full")
stays exactly the Colab data. Upload the files to Drive `ooa-nanochat/sft/` too, so the next Colab run trains on them
(Colab cell 1 NEW_FILES lists them; cell 6d says MISSING for one that is not uploaded).
