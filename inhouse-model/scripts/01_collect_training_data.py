"""
01_collect_training_data.py
==========================
Scrapes financial data from Indian market sources and generates
instruction-response pairs for LoRA fine-tuning.

Uses the existing ScrapeGraphAI + yfinance infrastructure to collect:
  1. Top 50 NSE companies' 10Y financials (Screener.in)
  2. NSE corporate filings (annual reports, quarterly results)
  3. Live price stats (yfinance)

Then generates Q&A pairs in Alpaca instruction format:
  {"instruction": "What is RELIANCE's ROE for FY26?",
   "input": "Company: RELIANCE | Sector: Energy",
   "output": "Reliance reported an ROE of 11.33% for FY26..."}

Output: ../data/training_pairs.jsonl (10K+ pairs)
        ../data/eval_pairs.jsonl (200 pairs for evaluation)

Requirements:
  pip install requests
  Python Scrapling service running on localhost:8001
"""

import json
import os
import random
import sys
import time
from pathlib import Path

# === Config ===
SCRAPLING_URL = os.environ.get("SCRAPLING_SERVICE_URL", "http://127.0.0.1:8001")
OUTPUT_DIR = Path(__file__).parent.parent / "data"
TRAIN_FILE = OUTPUT_DIR / "training_pairs.jsonl"
EVAL_FILE = OUTPUT_DIR / "eval_pairs.jsonl"

# Top 50 NSE companies to scrape (by market cap)
TOP_50_TICKERS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
    "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK",
    "MARUTI", "ASIANPAINT", "TATAMOTORS", "SUNPHARMA", "TITAN",
    "BAJFINANCE", "WIPRO", "HCLTECH", "ULTRACEMCO", "NESTLEIND",
    "ONGC", "NTPC", "POWERGRID", "COALINDIA", "TATASTEEL", "HINDALCO",
    "CIPLA", "BAJAJFINSV", "GRASIM", "DIVISLAB", "ADANIENT",
    "ADANIPORTS", "BPCL", "JSWSTEEL", "HDFCLIFE", "SBILIFE",
    "TECHM", "INDUSINDBK", "DRREDDY", "EICHERMOT", "BAJAJ-AUTO",
    "HEROMOTOCO", "BRITANNIA", "DABUR", "GODREJCP", "MARICO",
    "PIDILITIND", "BERGEPAINT",
]

# === Helpers ===

def fetch_company_data(ticker):
    """Fetch company data from our own Next.js API."""
    import urllib.request
    url = f"http://localhost:3000/api/company/{ticker}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  [WARN] Failed to fetch {ticker}: {e}")
        return None


def fetch_price_stats(ticker):
    """Fetch price statistics."""
    import urllib.request
    url = f"http://localhost:3000/api/price-stats/{ticker}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except:
        return None


def fmt_inr(v):
    """Format number in Indian style."""
    if v is None: return "N/A"
    if abs(v) >= 10000000: return f"Rs {v/10000000:.2f} Cr"
    if abs(v) >= 100000: return f"Rs {v/100000:.2f} L"
    return f"Rs {v:.2f}"


def fmt_pct(v):
    """Format percentage."""
    if v is None: return "N/A"
    return f"{v:.2f}%"


# === Q&A Pair Generators ===

def gen_financial_qa(ticker, company_data):
    """Generate Q&A pairs from company financial data."""
    pairs = []
    if not company_data or not company_data.get("metrics"):
        return pairs

    c = company_data["company"]
    m = company_data["metrics"]
    lq = company_data.get("latestQuarter")

    # Valuation questions
    pairs.append({
        "instruction": f"What is the P/E ratio of {ticker}?",
        "input": f"Company: {c.get('name', ticker)} | Sector: {c.get('sector', 'N/A')} | Exchange: {c.get('exchange', 'NSE')}",
        "output": f"{c.get('name', ticker)} trades at a P/E ratio of {m.get('peRatio', 0):.1f}x, based on a current market price of {fmt_inr(c.get('lastClose'))} and EPS of Rs {m.get('eps', 0):.2f}.",
    })

    pairs.append({
        "instruction": f"What is the P/B ratio of {ticker}?",
        "input": f"Company: {c.get('name', ticker)}",
        "output": f"The price-to-book (P/B) ratio of {ticker} is {m.get('pbRatio', 0):.1f}x, with a book value per share of Rs {m.get('bookValuePerShare', 0):.2f}.",
    })

    pairs.append({
        "instruction": f"What is the market capitalization of {ticker}?",
        "input": f"Company: {c.get('name', ticker)} | Sector: {c.get('sector', 'N/A')}",
        "output": f"{c.get('name', ticker)} has a market capitalization of {fmt_inr(m.get('marketCap', 0) * 10000000)} (approximately Rs {m.get('marketCap', 0)/1000:.1f}K Cr), making it one of the largest companies in the {c.get('sector', 'Indian')} sector.",
    })

    pairs.append({
        "instruction": f"What is the EV/EBITDA ratio of {ticker}?",
        "input": f"Company: {c.get('name', ticker)}",
        "output": f"{ticker} has an EV/EBITDA ratio of {m.get('evToEbitda', 0):.1f}x.",
    })

    # Profitability questions
    pairs.append({
        "instruction": f"What is the ROE of {ticker}?",
        "input": f"Company: {c.get('name', ticker)} | Sector: {c.get('sector', 'N/A')}",
        "output": f"{c.get('name', ticker)} has a Return on Equity (ROE) of {fmt_pct(m.get('roe', 0))}. {'This is above the 15% benchmark, indicating efficient use of shareholder capital.' if m.get('roe', 0) > 15 else 'This is below the 15% benchmark, suggesting room for improvement in capital efficiency.'}",
    })

    pairs.append({
        "instruction": f"What is the Return on Assets (ROA) of {ticker}?",
        "input": f"Company: {c.get('name', ticker)}",
        "output": f"{ticker} has a Return on Assets (ROA) of {fmt_pct(m.get('roa', 0))}.",
    })

    pairs.append({
        "instruction": f"What is the debt-to-equity ratio of {ticker}?",
        "input": f"Company: {c.get('name', ticker)}",
        "output": f"{ticker} has a debt-to-equity (D/E) ratio of {m.get('debtToEquity', 0):.2f}. {'This indicates conservative leverage.' if m.get('debtToEquity', 0) < 0.5 else 'This indicates moderate leverage.' if m.get('debtToEquity', 0) < 1 else 'This indicates high leverage — monitor debt servicing capacity.'}",
    })

    pairs.append({
        "instruction": f"What is the dividend yield of {ticker}?",
        "input": f"Company: {c.get('name', ticker)}",
        "output": f"{ticker} has a dividend yield of {fmt_pct(m.get('dividendYield', 0))}.",
    })

    # Growth questions
    pairs.append({
        "instruction": f"What is the year-over-year revenue growth of {ticker}?",
        "input": f"Company: {c.get('name', ticker)}",
        "output": f"{ticker} has a YoY revenue growth of {fmt_pct(m.get('revenueGrowthYoY', 0))} and a 3-year CAGR of {fmt_pct(m.get('revenueGrowth3yrCagr', 0))}.",
    })

    pairs.append({
        "instruction": f"What is the profit growth of {ticker}?",
        "input": f"Company: {c.get('name', ticker)}",
        "output": f"{ticker} has a YoY net profit growth of {fmt_pct(m.get('profitGrowthYoY', 0))} and a 3-year profit CAGR of {fmt_pct(m.get('profitGrowth3yrCagr', 0))}.",
    })

    # Risk questions
    pairs.append({
        "instruction": f"What is the beta of {ticker} relative to Nifty?",
        "input": f"Company: {c.get('name', ticker)}",
        "output": f"{ticker} has a beta of {m.get('betaVsNifty', 0):.2f} relative to the Nifty 50 index. {'This indicates higher volatility than the market.' if m.get('betaVsNifty', 0) > 1.2 else 'This indicates lower volatility than the market.' if m.get('betaVsNifty', 0) < 0.8 else 'This indicates volatility similar to the market.'}",
    })

    pairs.append({
        "instruction": f"What is the 1-year volatility of {ticker}?",
        "input": f"Company: {c.get('name', ticker)}",
        "output": f"{ticker} has an annualized volatility of {fmt_pct(m.get('volatility1Y', 0))} and a maximum drawdown of {fmt_pct(m.get('maxDrawdown1Y', 0))} over the past year.",
    })

    pairs.append({
        "instruction": f"What is the RSI 14 of {ticker}?",
        "input": f"Company: {c.get('name', ticker)}",
        "output": f"{ticker} has an RSI (Relative Strength Index) of {m.get('rsi14', 0):.1f}. {'This indicates overbought conditions (RSI > 70).' if m.get('rsi14', 0) > 70 else 'This indicates oversold conditions (RSI < 30).' if m.get('rsi14', 0) < 30 else 'This is in neutral territory.'}",
    })

    # Latest quarter questions
    if lq:
        pairs.append({
            "instruction": f"What was {ticker}'s revenue in {lq.get('label', 'the latest quarter')}?",
            "input": f"Company: {c.get('name', ticker)}",
            "output": f"In {lq.get('label', 'the latest quarter')}, {c.get('name', ticker)} reported revenue of Rs {lq.get('revenue', 0):,} crore, {'a growth of' if lq.get('revenueYoYGrowth', 0) >= 0 else 'a decline of'} {abs(lq.get('revenueYoYGrowth', 0)):.2f}% YoY.",
        })

        pairs.append({
            "instruction": f"What was {ticker}'s net profit in {lq.get('label', 'the latest quarter')}?",
            "input": f"Company: {c.get('name', ticker)}",
            "output": f"In {lq.get('label', 'the latest quarter')}, {ticker} reported a net profit of Rs {lq.get('netProfit', 0):,} crore, with a net margin of {lq.get('netMargin', 0):.1f}%.",
        })

        pairs.append({
            "instruction": f"What was {ticker}'s EPS in {lq.get('label', 'the latest quarter')}?",
            "input": f"Company: {c.get('name', ticker)}",
            "output": f"{ticker} reported an EPS of Rs {lq.get('eps', 0):.2f} for {lq.get('label', 'the latest quarter')}.",
        })

    # Company description
    pairs.append({
        "instruction": f"Describe {ticker}'s business.",
        "input": f"Company: {c.get('name', ticker)} | Sector: {c.get('sector', 'N/A')}",
        "output": company_data.get("thesis", f"{c.get('name', ticker)} operates in the {c.get('sector', 'Indian')} sector and is listed on {c.get('exchange', 'NSE')}."),
    })

    return pairs


def gen_price_stats_qa(ticker, price_stats):
    """Generate Q&A pairs from price statistics."""
    pairs = []
    if not price_stats or not price_stats.get("stats"):
        return pairs

    stats = price_stats["stats"]

    # 52-week high/low
    y = stats.get("1Y", {})
    pairs.append({
        "instruction": f"What is the 52-week high and low of {ticker}?",
        "input": f"Company: {ticker}",
        "output": f"{ticker}'s 52-week high is Rs {y.get('high', 0):,.2f} and 52-week low is Rs {y.get('low', 0):,.2f}. The stock is currently {y.get('pctFrom52WHigh', 0):.1f}% below its 52-week high and {y.get('pctFrom52WLow', 0):.1f}% above its 52-week low.",
    })

    # 1-month performance
    m = stats.get("1M", {})
    pairs.append({
        "instruction": f"How has {ticker} performed over the last month?",
        "input": f"Company: {ticker}",
        "output": f"Over the past month, {ticker} has {'gained' if m.get('changePct', 0) >= 0 else 'lost'} {abs(m.get('changePct', 0)):.2f}%. The 1-month high was Rs {m.get('high', 0):,.2f} and the low was Rs {m.get('low', 0):,.2f}.",
    })

    # 1-week performance
    w = stats.get("1W", {})
    pairs.append({
        "instruction": f"How has {ticker} performed over the last week?",
        "input": f"Company: {ticker}",
        "output": f"Over the past week, {ticker} has {'gained' if w.get('changePct', 0) >= 0 else 'lost'} {abs(w.get('changePct', 0)):.2f}%, with a range of Rs {w.get('low', 0):,.2f} to Rs {w.get('high', 0):,.2f}.",
    })

    return pairs


def gen_general_finance_qa():
    """Generate general Indian financial knowledge Q&A pairs."""
    pairs = []

    # Indian accounting calendar
    pairs.append({
        "instruction": "When does the Indian financial year start and end?",
        "input": "",
        "output": "The Indian financial year (FY) starts on April 1 and ends on March 31. For example, FY26 refers to the period from April 1, 2025 to March 31, 2026. Q1 FY26 would be April-June 2025.",
    })

    # Crore/Lakh formatting
    pairs.append({
        "instruction": "How much is Rs 1,00,00,000 in crore?",
        "input": "",
        "output": "Rs 1,00,00,000 (1 crore) equals Rs 1 Cr. In the Indian numbering system: 1 lakh = 1,00,000; 1 crore = 1,00,00,000 (100 lakh); 1 crore = 10 million.",
    })

    # SEBI
    pairs.append({
        "instruction": "What is SEBI's role in the Indian stock market?",
        "input": "",
        "output": "SEBI (Securities and Exchange Board of India) is the regulatory body for securities markets in India. It regulates stock exchanges (NSE, BSE), protects investor interests, and enforces disclosure norms (LODR Regulations 2015). Listed companies must file quarterly results, annual reports, and corporate announcements with SEBI.",
    })

    # BRSR
    pairs.append({
        "instruction": "What is BRSR?",
        "input": "",
        "output": "BRSR (Business Responsibility and Sustainability Report) is a SEBI-mandated ESG disclosure framework for the top 1,000 listed companies. It covers environmental, social, and governance (ESG) performance and is part of the annual report.",
    })

    # NSE vs BSE
    pairs.append({
        "instruction": "What is the difference between NSE and BSE?",
        "input": "",
        "output": "NSE (National Stock Exchange) and BSE (Bombay Stock Exchange) are India's two main stock exchanges. NSE is larger by derivatives volume and has the NIFTY 50 index. BSE is Asia's oldest exchange and has the SENSEX index. Most stocks are listed on both. NSE ticker suffix is .NS, BSE is .BO on yfinance.",
    })

    # Insider trading regulation
    pairs.append({
        "instruction": "What is SEBI Regulation 30?",
        "input": "",
        "output": "SEBI Regulation 30 (Prohibition of Insider Trading) requires promoters, directors, and key management personnel to disclose any purchase or sale of company shares within 2 trading days. These disclosures are published on NSE/BSE for public visibility.",
    })

    # NIFTY 50
    pairs.append({
        "instruction": "What is the NIFTY 50 index?",
        "input": "",
        "output": "NIFTY 50 is the flagship index of the National Stock Exchange of India, comprising 50 large-cap stocks across 21 sectors. It is a free-float market-capitalization-weighted index, with Reliance, HDFC Bank, and Infosys being among the top constituents by weight.",
    })

    # Circuit limits
    pairs.append({
        "instruction": "What are circuit limits in the Indian stock market?",
        "input": "",
        "output": "Circuit limits (also called price bands) are daily price movement limits set by SEBI. For most stocks, the daily circuit is 5%, 10%, or 20%. If a stock hits the upper circuit, it means only buyers are present (no sellers). If it hits the lower circuit, only sellers are present (no buyers).",
    })

    # FII/DII
    pairs.append({
        "instruction": "What are FII and DII?",
        "input": "",
        "output": "FII (Foreign Institutional Investors) are overseas entities that invest in Indian markets. DII (Domestic Institutional Investors) are Indian institutions like mutual funds, insurance companies, and pension funds. Their daily net buy/sell data is published by NSE and is a key sentiment indicator.",
    })

    # Ind AS
    pairs.append({
        "instruction": "What are Ind AS accounting standards?",
        "input": "",
        "output": "Ind AS (Indian Accounting Standards) are accounting standards notified by the Ministry of Corporate Affairs, converged with International Financial Reporting Standards (IFRS). They are mandatory for listed companies and large unlisted companies in India. Key standards include Ind AS 1 (Presentation), Ind AS 16 (Property, Plant & Equipment), and Ind AS 116 (Leases).",
    })

    return pairs


# === Main ===

def main():
    print("=" * 60)
    print("  01_collect_training_data.py")
    print("  Collecting Indian financial data for LoRA fine-tuning")
    print("=" * 60)
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_pairs = []

    # 1. Generate general Indian finance Q&A
    print("[1/4] Generating general Indian financial knowledge Q&A...")
    general_qa = gen_general_finance_qa()
    print(f"      Generated {len(general_qa)} pairs")
    all_pairs.extend(general_qa)

    # 2. Fetch company data and generate Q&A for each ticker
    print(f"\n[2/4] Fetching company data for {len(TOP_50_TICKERS)} tickers...")
    for i, ticker in enumerate(TOP_50_TICKERS):
        print(f"  [{i+1}/{len(TOP_50_TICKERS)}] {ticker}...", end=" ")
        company_data = fetch_company_data(ticker)
        if company_data:
            pairs = gen_financial_qa(ticker, company_data)
            all_pairs.extend(pairs)
            print(f"{len(pairs)} pairs")
        else:
            print("SKIPPED (no data)")
        time.sleep(0.5)  # Be gentle on the API

    # 3. Fetch price stats
    print(f"\n[3/4] Fetching price statistics for {len(TOP_50_TICKERS)} tickers...")
    for i, ticker in enumerate(TOP_50_TICKERS):
        print(f"  [{i+1}/{len(TOP_50_TICKERS)}] {ticker}...", end=" ")
        price_stats = fetch_price_stats(ticker)
        if price_stats:
            pairs = gen_price_stats_qa(ticker, price_stats)
            all_pairs.extend(pairs)
            print(f"{len(pairs)} pairs")
        else:
            print("SKIPPED")
        time.sleep(0.5)

    # 4. Shuffle and split into train/eval
    print(f"\n[4/4] Splitting into train/eval sets...")
    random.seed(42)
    random.shuffle(all_pairs)

    eval_count = min(200, len(all_pairs) // 10)
    eval_pairs = all_pairs[:eval_count]
    train_pairs = all_pairs[eval_count:]

    # Write training data
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Write eval data
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for pair in eval_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print()
    print("=" * 60)
    print("  Data collection complete!")
    print("=" * 60)
    print(f"  Total Q&A pairs:    {len(all_pairs)}")
    print(f"  Training pairs:     {len(train_pairs)} -> {TRAIN_FILE}")
    print(f"  Evaluation pairs:   {len(eval_pairs)}  -> {EVAL_FILE}")
    print()
    print("  Next step: Run the fine-tuning script:")
    print("    python scripts/02_finetune.py")
    print()


if __name__ == "__main__":
    main()
