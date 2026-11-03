"""
Logistics SLM — Multi-turn tokenizer + shared encoding utilities.

Run once to train the BPE tokenizer on the multi-turn conversations:
    $env:PYTHONPATH="..;..\\minbpe-master\\minbpe-master"; py tokenizer_ctx.py

Also provides `CtxTokenizer`, a cached fast encoder that returns a per-token
loss mask (train on assistant turns only, skip tool <observation> payloads,
which come from the environment and are unpredictable).
"""
import json
import os
import random
import regex as re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "minbpe-master", "minbpe-master")))
from minbpe import RegexTokenizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PREFIX = os.path.join(HERE, "models", "ctx_bpe")
SPECIAL_TOKENS_LIST = [
    "<|im_start|>", "<|im_end|>",
    "<thought>", "<route>", "<tool_call>", "<observation>", "<action>",
    "<end>", "<pad>",
]
SPECIAL_SPLIT = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS_LIST) + ")")


class CtxTokenizer:
    def __init__(self, path=TOKENIZER_PREFIX + ".model"):
        self.tok = RegexTokenizer()
        self.tok.load(path)
        self.special = self.tok.special_tokens
        self.pad_id = self.special["<pad>"]
        self.end_id = self.special["<end>"]
        self.obs_id = self.special["<observation>"]
        self.vocab_size = max(max(self.tok.vocab), max(self.special.values())) + 1
        self._cache = {}

    def _ordinary(self, text):
        out = []
        pattern = self.tok.compiled_pattern if hasattr(self.tok, 'compiled_pattern') else re.compile(self.tok.pattern)
        for chunk in re.findall(pattern, text):
            ids = self._cache.get(chunk)
            if ids is None:
                ids = self._cache[chunk] = self.tok._encode_chunk(chunk.encode("utf-8"))
            out.extend(ids)
        return out

    def encode(self, text):
        return self.encode_with_mask(text)[0]

    def encode_with_mask(self, text):
        """Returns (ids, mask). mask[i]=1 if token i should be *predicted* (assistant content)."""
        ids, mask = [], []
        in_asst = in_obs = False
        expect_role = False
        for part in SPECIAL_SPLIT.split(text):
            if not part:
                continue
            if part in self.special:
                if part == "<|im_start|>":
                    expect_role, in_asst, in_obs = True, False, False
                elif part == "<observation>":
                    in_obs = True
                elif part == "<action>":
                    in_obs = False
                ids.append(self.special[part])
                mask.append(1 if in_asst and part != "<|im_start|>" else 0)
                if part == "<end>":
                    in_asst = False
                continue
            p_ids = self._ordinary(part)
            if expect_role:
                expect_role = False
                in_asst = part.startswith(" assistant")
                mask.extend([0] * len(p_ids))      # the role header itself is given, not predicted
            else:
                m = 1 if in_asst and not in_obs else 0
                mask.extend([m] * len(p_ids))
            ids.extend(p_ids)
        return ids, mask

    def decode(self, ids):
        return self.tok.decode(ids)


if __name__ == "__main__":
    with open(os.path.join(HERE, "conv_train.jsonl"), encoding="utf-8") as f:
        convs = [json.loads(line)["text"] for line in f]
    random.seed(42)
    sample = random.sample(convs, 400)
    text = SPECIAL_SPLIT.sub(" ", "\n".join(sample))[:800_000]
    print(f"training tokenizer on {len(text):,} chars...")
    tok = RegexTokenizer()
    vocab_size = 1000
    tok.train(text, vocab_size, verbose=False)
    tok.register_special_tokens({t: vocab_size + i for i, t in enumerate(SPECIAL_TOKENS_LIST)})
    os.makedirs(os.path.dirname(TOKENIZER_PREFIX), exist_ok=True)
    tok.save(TOKENIZER_PREFIX)

    ct = CtxTokenizer()
    ex = convs[0]
    ids, mask = ct.encode_with_mask(ex)
    assert ct.decode(ids) == ex, "round-trip failed"
    lens = [len(ct.encode(c)) for c in convs[:2000]]
    n_turns = sum(c.count("<end>") for c in convs[:2000])
    print(f"vocab {ct.vocab_size} | avg conv {sum(lens)/len(lens):.0f} tokens | max {max(lens)} | "
          f"{sum(lens)/n_turns:.1f} tokens/turn | trained-on fraction {sum(mask)/len(mask):.2f}")
    print(f"saved -> {TOKENIZER_PREFIX}.model")
