"""Lexicon-based macro news sentiment (DEMO).

The bundled ``data/raw/news_headlines.csv`` is a small, hand-written toy sample
(not real wire headlines). It only covers a few weeks, so the sentiment score is
reported as a standalone demo and is **not** used in the historical backtest.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

POSITIVE = {
    "beat", "beats", "boost", "boosts", "expand", "expands", "expansion", "gain", "gains",
    "growth", "improve", "improves", "improved", "rally", "rallies", "rebound", "rebounds",
    "record", "recovery", "resilient", "robust", "strong", "stronger", "surge", "surges",
    "upgrade", "upgrades", "optimism", "accelerates", "hiring", "eases", "easing", "cools",
    "cooling", "moderates", "stabilise", "stabilize", "stabilises", "stabilizes", "upbeat",
    "tame", "soft-landing", "rises", "jump", "jumps", "outperform",
}
NEGATIVE = {
    "contraction", "contracts", "crisis", "cut", "cuts", "decline", "declines", "default",
    "defaults", "downgrade", "downgrades", "fall", "falls", "fear", "fears", "layoffs",
    "loss", "losses", "miss", "misses", "plunge", "plunges", "recession", "risk", "selloff",
    "sell-off", "slump", "slumps", "slowdown", "slows", "stress", "tumble", "tumbles", "weak",
    "weaker", "weakens", "worse", "worst", "volatility", "shock", "hot", "sticky",
    "accelerating", "spike", "spikes", "widen", "widens", "turmoil", "warns", "jobless",
    "shutdown", "tariffs", "drops", "sinks",
}
NEGATORS = {"not", "no", "never", "without", "fails", "fail", "less"}
# Context rules: inflation going up is bad news for markets; inflation cooling is good.
PHRASES = {
    "inflation cools": 1, "inflation eases": 1, "prices ease": 1, "rate cut": 1, "rate cuts": 1,
    "inflation accelerates": -1, "inflation rises": -1, "inflation jumps": -1, "prices surge": -1,
    "unemployment rises": -1, "jobless claims rise": -1, "jobless claims fall": 1,
    "yields spike": -1, "spreads widen": -1, "spreads tighten": 1,
}

_TOKEN = re.compile(r"[a-z][a-z\-']*")


def score_headline(text: str) -> float:
    """Net tone in [-1, 1]: (pos - neg) / (pos + neg), phrase rules first, simple negation."""
    t = text.lower()
    pos = neg = 0
    for phrase, sign in PHRASES.items():
        if phrase in t:
            pos += sign > 0
            neg += sign < 0
            t = t.replace(phrase, " ")
    toks = _TOKEN.findall(t)
    for i, w in enumerate(toks):
        s = 1 if w in POSITIVE else -1 if w in NEGATIVE else 0
        if s and any(x in NEGATORS for x in toks[max(0, i - 3):i]):
            s = -s
        pos += s > 0
        neg += s < 0
    return 0.0 if pos + neg == 0 else (pos - neg) / (pos + neg)


def classify(score: float, band: float = 0.2) -> str:
    return "positive" if score > band else "negative" if score < -band else "neutral"


def load_headlines(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["score"] = df["headline"].map(score_headline)
    df["predicted"] = df["score"].map(classify)
    return df


def evaluate(df: pd.DataFrame) -> dict:
    acc = float((df["predicted"] == df["label"]).mean())
    cm = pd.crosstab(df["label"], df["predicted"]).reindex(
        index=["negative", "neutral", "positive"], columns=["negative", "neutral", "positive"], fill_value=0)
    return {"n": len(df), "accuracy": acc, "confusion": cm}


def weekly_index(df: pd.DataFrame) -> pd.Series:
    """Average tone per week — the kind of series one would feed a model with real data."""
    return df.set_index("date")["score"].resample("W-FRI").mean().dropna().rename("sentiment")
