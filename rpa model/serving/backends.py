"""Local model backend for the server. NanochatBackend loads our trained nanochat checkpoint (same calls as
training/ooa_self_improve.py). NOT tested in the build environment (needs nanochat + torch); tested path uses fakes."""
import sys


def tokenizer_problem(meta):
    """model_backend.tokenizer_problem for the model pack nanochat loads ($NANOCHAT_BASE_DIR/tokenizer): a model
    must be served with the tokenizer it was trained with (the clean base v6 has its own)."""
    import os
    import sys
    from pathlib import Path
    from nanochat.common import get_base_dir
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant"))
    try:
        from model_backend import tokenizer_problem as check
    except ImportError:                                     # a serving folder without the assistant: no check
        return None
    finally:
        sys.path.pop(0)
    return check(meta, os.path.join(get_base_dir(), "tokenizer"))


class NanochatBackend:
    def __init__(self, source="sft", max_new_tokens=1024, base_temp=0.2):
        import torch
        from nanochat.common import compute_init, autodetect_device_type
        from nanochat.checkpoint_manager import load_model
        from nanochat.engine import Engine
        _, _, _, _, self.device = compute_init(autodetect_device_type())
        self.model, self.tok, meta = load_model(source, self.device, phase="eval")
        bad = tokenizer_problem(meta)
        if bad:
            raise RuntimeError("model pack: " + bad)
        self.engine = Engine(self.model, self.tok)
        self.max_new_tokens, self.base_temp = max_new_tokens, base_temp

    def generate(self, request, attempt=0):
        conv = {"messages": [{"role": "user", "content": request}, {"role": "assistant", "content": ""}]}
        tokens = self.tok.render_for_completion(conv)
        seqs, _ = self.engine.generate_batch(tokens, num_samples=1, max_tokens=self.max_new_tokens,
                                             temperature=self.base_temp + 0.3 * attempt, top_k=50, seed=1000 + attempt)
        return self.tok.decode(seqs[0][len(tokens):])
