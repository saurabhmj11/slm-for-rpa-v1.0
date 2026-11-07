"""Local model backend for the server. NanochatBackend loads our trained nanochat checkpoint (same calls as
training/ooa_self_improve.py). NOT tested in the build environment (needs nanochat + torch); tested path uses fakes."""
import sys


class NanochatBackend:
    def __init__(self, source="sft", max_new_tokens=1024, base_temp=0.2):
        import torch
        from nanochat.common import compute_init, autodetect_device_type
        from nanochat.checkpoint_manager import load_model
        from nanochat.engine import Engine
        _, _, _, _, self.device = compute_init(autodetect_device_type())
        self.model, self.tok, _ = load_model(source, self.device, phase="eval")
        self.engine = Engine(self.model, self.tok)
        self.max_new_tokens, self.base_temp = max_new_tokens, base_temp

    def generate(self, request, attempt=0):
        conv = {"messages": [{"role": "user", "content": request}, {"role": "assistant", "content": ""}]}
        tokens = self.tok.render_for_completion(conv)
        seqs, _ = self.engine.generate_batch(tokens, num_samples=1, max_tokens=self.max_new_tokens,
                                             temperature=self.base_temp + 0.3 * attempt, top_k=50, seed=1000 + attempt)
        return self.tok.decode(seqs[0][len(tokens):])
