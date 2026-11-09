"""THE ONE FILE AN EXPERIMENT MAY CHANGE: the SFT recipe (what the model is trained on, and how).
Everything else (frozen eval, verifier, catalog, token budget, runner) is read-only. Keep it simple: all else equal,
fewer rows / fewer knobs changed is better. `token_budget` is fixed by constants.py and must not be changed."""
from constants import FIXED_TOKEN_BUDGET

RECIPE = {
    "token_budget": FIXED_TOKEN_BUDGET,   # do not change
    # ---- data mix ----
    "synth_train": 8000,                  # generated flows (rows)
    "flow_repeat": 15,                    # each real gold flow is repeated this many times
    "smoltalk_rows": 3000,                # general chat rows so the model still talks (0 = none)
    "include_catalog": True,              # catalog Q&A rows ("which action ...", "what inputs ...")
    "step_weights": [25, 35, 25, 15],     # share of generated flows with 1, 2, 3, 4 steps
    "wrap_weights": [55, 17, 14, 14],     # share with no container / loop / if / try
    # ---- optimisation ----
    "seq_len": 2048,
    "device_batch": 4,
    "grad_accum": 4,
    "init_lr_frac": 0.8,                  # starting learning rate as a fraction of the pretrained base rate
    "warmdown_ratio": 0.5,
    "warmup_ratio": 0.0,
    "final_lr_frac": 0.0,
}
