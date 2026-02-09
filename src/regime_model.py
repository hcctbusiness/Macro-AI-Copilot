from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def label_macro_regimes(
    macro_df: pd.DataFrame,
    growth_col: str = "gdp_growth",
    unemp_col: str = "unemp",
) -> pd.Series:
    g = macro_df[growth_col]
    u = macro_df[unemp_col]

    labels: List[str] = []
    for gi, ui in zip(g, u):
        if gi > 2.0 and ui < 6.0:
            labels.append("expansion")
        elif gi < 0.5 and ui > 6.5:
            labels.append("recession")
        elif gi > 0.5 and gi <= 2.0 and ui >= 6.0:
            labels.append("recovery")
        else:
            labels.append("slowdown")

    return pd.Series(labels, index=macro_df.index, name="regime")


@dataclass
class RegimeModel:
    model: Any
    feature_cols: list


def train_regime_classifier(
    features: pd.DataFrame,
    regimes: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[RegimeModel, Dict[str, Any]]:
    df = features.join(regimes, how="inner").dropna()
    X = df[features.columns]
    y = df["regime"]

    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))

    metrics = {
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": sorted(y.unique()),
    }

    regime_model = RegimeModel(model=clf, feature_cols=list(features.columns))
    return regime_model, metrics


def predict_regimes(regime_model: RegimeModel, features: pd.DataFrame) -> pd.Series:
    X = features[regime_model.feature_cols]
    preds = regime_model.model.predict(X)
    return pd.Series(preds, index=features.index, name="regime_pred")
