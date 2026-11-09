"""FROZEN settings shared by every experiment. Changing these makes old results incomparable. Agents must not edit this file."""
FIXED_TOKEN_BUDGET = 12_000_000     # SFT training tokens per experiment (fixed token budget, not time: portable across GPUs)
KEEP_EPSILON = 0.003                # an experiment must beat the best score by more than this to be kept (noise guard)
EVAL_SAMPLES = 4                    # answers sampled per frozen-eval request
EVAL_TEMPERATURE = 0.6
EVAL_MAX_NEW_TOKENS = 1024
