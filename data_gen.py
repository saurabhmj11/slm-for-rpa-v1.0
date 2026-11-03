"""
Praman Brain Mini — Layer 1: News Intelligence Data Generator.

Generates synthetic agent trajectories for 3 news skills:

  Skill A — News Sentiment  (sentiment_agent)
    "what is the sentiment of <ticker> <event> news ?"
    → routes to sentiment_agent, emits bullish/bearish/neutral + strength

  Skill B — Market Impact   (impact_agent)
    "what is the market impact of <ticker> <event> news ?"
    → routes to impact_agent, emits positive/negative/mixed + high/medium/low

  Skill C — Stock Signal    (signal_agent)
    "will <ticker> incline or decline due to <event> news ?"
    → routes to signal_agent, emits incline/decline/sideways

All trajectories use the same DSL proved by the original 100K model:
  <user> ... <thought> ... <route> AGENT <tool_call> {json}
  <observation> ... <action> ... <end>

Event polarity is rule-based (not learned from real data). The model learns:
  quarterly_profit → bullish → incline
  regulatory_fine  → bearish → decline
  …and so on for all 20 event types.
"""
import json
import random

TICKERS = [
    "reliance", "tcs", "hdfc", "infosys", "sbi", "bharti", "itc", "larsen", 
    "bajaj", "adani", "kotak", "wipro", "maruti", "sunpharma", "ongc"
]

EVENTS = [
    "quarterly_profit", "quarterly_loss", "revenue_beat", "revenue_miss", 
    "contract_win", "contract_loss", "merger", "acquisition", 
    "regulatory_approval", "regulatory_fine", "dividend", "ceo_change", 
    "credit_upgrade", "credit_downgrade", "global_rally", "global_selloff", 
    "crude_fall", "crude_rise", "rate_cut", "rate_hike"
]

random.seed(42)

# ── Event classification ──────────────────────────────────────────────────
# Each event is deterministically positive, negative, or neutral.
# This gives the model clean, consistent signal to learn from.

POSITIVE_EVENTS = {
    "quarterly_profit", "revenue_beat", "contract_win",
    "merger", "acquisition", "regulatory_approval",
    "dividend", "credit_upgrade", "global_rally",
    "crude_fall", "rate_cut",
}

NEGATIVE_EVENTS = {
    "quarterly_loss", "revenue_miss", "contract_loss",
    "regulatory_fine", "credit_downgrade",
    "global_selloff", "crude_rise", "rate_hike",
}

NEUTRAL_EVENTS = {"ceo_change"}


def _sentiment(event: str) -> str:
    if event in POSITIVE_EVENTS:
        return "bullish"
    if event in NEGATIVE_EVENTS:
        return "bearish"
    return "neutral"


def _signal(event: str) -> str:
    if event in POSITIVE_EVENTS:
        return "incline"
    if event in NEGATIVE_EVENTS:
        return "decline"
    return "sideways"


def _impact_dir(event: str) -> str:
    if event in POSITIVE_EVENTS:
        return "positive"
    if event in NEGATIVE_EVENTS:
        return "negative"
    return "mixed"


# ── ChatML formatting helper ────────────────────────────────────────────────

def _format_chatml(user_msg: str, assistant_msg: str) -> str:
    return (
        f"<|im_start|> user\n{user_msg}<|im_end|>\n"
        f"<|im_start|> assistant\n{assistant_msg}<end>"
    )

# ── Skill A: News Sentiment ───────────────────────────────────────────────

def gen_sentiment_example() -> str:
    ticker    = random.choice(TICKERS)
    event     = random.choice(EVENTS)
    sentiment = _sentiment(event)
    strength  = random.choice(["strong", "moderate", "mild"])
    sent_qual = (
        "positive" if sentiment == "bullish" else
        "negative" if sentiment == "bearish" else
        "mixed"
    )
    user_msg = f"what is the sentiment of {ticker} {event} news ?"
    asst_msg = (
        f"<thought> user wants news sentiment analysis "
        f"<route> sentiment_agent "
        f"<tool_call> analyze_sentiment "
        f"<observation> {sentiment} {strength} "
        f"<action> the news for {ticker} is {sentiment} "
        f"with {strength} {sent_qual} sentiment"
    )
    return _format_chatml(user_msg, asst_msg)


# ── Skill B: Market Impact ────────────────────────────────────────────────

def gen_impact_example() -> str:
    ticker     = random.choice(TICKERS)
    event      = random.choice(EVENTS)
    impact_dir = _impact_dir(event)
    impact_lvl = random.choice(["high", "medium", "low"])
    user_msg = f"what is the market impact of {ticker} {event} news ?"
    asst_msg = (
        f"<thought> user wants to know market impact of news "
        f"<route> impact_agent "
        f"<tool_call> assess_impact "
        f"<observation> {impact_dir} {impact_lvl} "
        f"<action> {ticker} faces {impact_lvl} {impact_dir} impact due to {event}"
    )
    return _format_chatml(user_msg, asst_msg)


# ── Skill C: Stock Signal ─────────────────────────────────────────────────

def gen_signal_example() -> str:
    ticker    = random.choice(TICKERS)
    event     = random.choice(EVENTS)
    signal    = _signal(event)
    user_msg = f"will {ticker} incline or decline due to {event} news ?"
    asst_msg = (
        f"<thought> user wants stock direction signal from news "
        f"<route> signal_agent "
        f"<tool_call> get_signal "
        f"<observation> {signal} likely "
        f"<action> {ticker} is likely to {signal} based on {event} news"
    )
    return _format_chatml(user_msg, asst_msg)


GENERATORS = [gen_sentiment_example, gen_impact_example, gen_signal_example]


def generate_dataset(n_examples: int):
    return [random.choice(GENERATORS)() for _ in range(n_examples)]


if __name__ == "__main__":
    examples = generate_dataset(9)  # 3 per skill
    for ex in examples:
        print(ex)
        print("---")
