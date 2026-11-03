"""
Praman Brain Mini -- Domain Tokenizer

Uses a small word-level vocabulary covering only the tokens that appear
in Layer-1 training data (~130 tokens). This keeps the embedding table
tiny (~4K params for n_embd=32) so the transformer weights actually
matter and the model can learn.

The old tiktoken (100K vocab) used 3.2M params for embeddings alone,
leaving only ~21K params for the actual transformer -- not enough to learn.
"""
import re

# ---------------------------------------------------------------------------
# Vocabulary list.  The index in this list == token ID.
# ---------------------------------------------------------------------------
_RAW_VOCAB = [
    # special
    "<pad>", "<unk>", "<end>",
    # ChatML / agent tags
    "<|im_start|>", "<|im_end|>",
    "<thought>", "<route>", "<tool_call>", "<observation>", "<action>",
    # role words
    "user", "assistant",
    # question words / template glue
    "what", "is", "the", "sentiment", "of", "news", "?",
    "market", "impact",
    "will", "incline", "or", "decline", "due", "to",
    "wants", "know", "for", "with", "faces",
    "stock", "direction", "signal", "from",
    "a", "an", "in", "and", "based", "on", "be", "likely",
    # tickers
    "reliance", "tcs", "hdfc", "infosys", "sbi", "bharti",
    "itc", "larsen", "bajaj", "adani", "kotak", "wipro",
    "maruti", "sunpharma", "ongc",
    # events
    "quarterly_profit", "quarterly_loss", "revenue_beat", "revenue_miss",
    "contract_win", "contract_loss", "merger", "acquisition",
    "regulatory_approval", "regulatory_fine", "dividend", "ceo_change",
    "credit_upgrade", "credit_downgrade", "global_rally", "global_selloff",
    "crude_fall", "crude_rise", "rate_cut", "rate_hike",
    # agents
    "sentiment_agent", "impact_agent", "signal_agent",
    # tools (also appear as words in generated JSON)
    "analyze_sentiment", "assess_impact", "get_signal",
    # JSON punctuation
    "{", "}", "\"", ":", ",",
    # JSON key words
    "tool", "args",
    # observation: sentiment
    "bullish", "bearish", "neutral",
    "strong", "moderate", "mild",
    # observation: impact
    "positive", "negative", "mixed",
    "high", "medium", "low",
    # observation: signal
    "sideways",
    # extra action-phrase words
    "analysis", "strongly", "significantly",
    "\n",
]

# Deduplicate while preserving order
_seen: set = set()
_VOCAB: list = []
for _t in _RAW_VOCAB:
    if _t not in _seen:
        _seen.add(_t)
        _VOCAB.append(_t)


class AgentTokenizer:
    """
    Tiny word-level tokenizer with a fixed domain vocabulary (~130 tokens).

    Encoding:
      1. The text is split using a regex that keeps tags atomic.
      2. Known words/tags -> their integer ID.
      3. Unknown words    -> per-character fallback IDs.

    Decoding:
      Token strings are joined with spaces, then whitespace around
      newlines and "?" is cleaned up.
    """

    def __init__(self):
        # Build stoi / itos from the fixed vocab
        self.stoi: dict = {tok: i for i, tok in enumerate(_VOCAB)}
        self.itos: dict = {i: tok for i, tok in enumerate(_VOCAB)}

        # Add every printable ASCII character as a fallback so we never
        # silently drop characters.
        _next = len(_VOCAB)
        for ch in (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789 \t_-.!'@#$%^&*()[]<>/\\"
        ):
            if ch not in self.stoi:
                self.stoi[ch] = _next
                self.itos[_next] = ch
                _next += 1

        self.vocab_size: int = _next

    # ------------------------------------------------------------------

    def encode(self, text: str) -> list:
        ids = []
        for tok in _tokenise(text):
            if tok in self.stoi:
                ids.append(self.stoi[tok])
            else:
                for ch in tok:
                    ids.append(self.stoi.get(ch, self.stoi["<unk>"]))
        return ids

    def decode(self, ids: list) -> str:
        parts = [self.itos.get(i, "<unk>") for i in ids]
        text = " ".join(parts)
        # Clean up spacing
        text = text.replace(" \n ", "\n").replace(" \n", "\n").replace("\n ", "\n")
        text = text.replace(" ?", "?")
        return text

    @property
    def end_id(self) -> int:
        return self.stoi["<end>"]

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TAG_PAT = re.compile(r'(<\|[^|>]+\|>|<[^>]+>|\n|[^\s]+)')

def _tokenise(text: str) -> list:
    """Split text preserving angle-bracket tags and newlines as atoms."""
    return _TAG_PAT.findall(text)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tok = AgentTokenizer()
    print(f"Vocabulary size: {tok.vocab_size}")
    print(f"  end_id = {tok.end_id}  pad_id = {tok.pad_id}")

    sample = (
        "<|im_start|> user\n"
        "what is the sentiment of reliance quarterly_profit news ?<|im_end|>\n"
        "<|im_start|> assistant\n"
        "<thought> user wants news sentiment analysis "
        "<route> sentiment_agent "
        '<tool_call> {"tool": "analyze_sentiment"} '
        "<observation> bullish strong "
        "<action> the news for reliance is bullish with strong positive sentiment"
        "<end>"
    )
    ids = tok.encode(sample)
    print(f"Token count: {len(ids)}")
    print("IDs:", ids)
    print("Decoded:", tok.decode(ids))

