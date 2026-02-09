from typing import Dict, Optional
import re

import numpy as np
import pandas as pd


def build_macro_features(macro_df: pd.DataFrame, lags: int = 2) -> pd.DataFrame:
    df = macro_df.copy().sort_index()
    cols = df.columns

    for col in cols:
        for lag in range(1, lags + 1):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        df[f"{col}_chg"] = df[col] - df[col].shift(1)
        df[f"{col}_roll_std"] = df[col].rolling(4, min_periods=3).std()

    return df.dropna()


def build_sentiment_index(
    text_df: pd.DataFrame,
    positive_words: Optional[Dict[str, int]] = None,
    negative_words: Optional[Dict[str, int]] = None,
    freq: str = "Q",
) -> pd.DataFrame:
    if positive_words is None:
        positive_words = {
            "strong": 1,
            "growth": 1,
            "robust": 1,
            "optimism": 1,
            "record": 1,
            "confidence": 1,
            "positive": 1,
            "rally": 1,
            "gain": 1,
        }
    if negative_words is None:
        negative_words = {
            "recession": -1,
            "slowdown": -1,
            "tension": -1,
            "fear": -1,
            "volatility": -1,
            "panic": -1,
            "crisis": -1,
            "worry": -1,
        }

    df = text_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    def score(headline: str) -> int:
        tokens = re.findall(r"[a-zA-Z']+", str(headline).lower())
        s = 0
        for t in tokens:
            if t in positive_words:
                s += positive_words[t]
            if t in negative_words:
                s += negative_words[t]
        return s

    df["sent_score"] = df["headline"].apply(score)
    daily = df.groupby("date")["sent_score"].sum().to_frame("sentiment_raw")

    rule = {"M": "M", "Q": "Q"}[freq]
    sent_ts = daily.resample(rule).sum()
    sent_ts["sentiment_norm"] = (
        (sent_ts["sentiment_raw"] - sent_ts["sentiment_raw"].mean())
        / (sent_ts["sentiment_raw"].std() or 1)
    )
    return sent_ts


def combine_macro_and_sentiment(
    macro_feat: pd.DataFrame, sentiment_ts: pd.DataFrame
) -> pd.DataFrame:
    combined = macro_feat.join(sentiment_ts[["sentiment_norm"]], how="inner")
    return combined.dropna()
