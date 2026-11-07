from pathlib import Path

from macro_ai_copilot import sentiment

CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "news_headlines.csv"


def test_scores_have_expected_sign():
    assert sentiment.score_headline("Payrolls beat expectations as hiring surges") > 0
    assert sentiment.score_headline("Recession fears grow as stocks plunge") < 0
    assert sentiment.score_headline("Inflation cools in October") > 0
    assert sentiment.score_headline("Inflation accelerates again") < 0
    assert sentiment.score_headline("Central bank publishes minutes") == 0


def test_negation_flips_sign():
    assert sentiment.score_headline("Economy is not weak") > 0


def test_demo_csv_loads_and_scores():
    df = sentiment.load_headlines(CSV)
    assert len(df) >= 35
    assert set(df["label"]) <= {"positive", "neutral", "negative"}
    res = sentiment.evaluate(df)
    assert 0 <= res["accuracy"] <= 1
    assert len(sentiment.weekly_index(df)) > 0
