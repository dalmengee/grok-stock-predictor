"""주가 예측 모델 모듈."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import get_latest_features, prepare_features


@dataclass
class PredictionResult:
    """예측 결과."""

    ticker: str
    current_price: float
    predicted_price: float
    forecast_days: int
    predicted_return: float
    direction: str
    model_name: str
    metrics: dict[str, float]
    feature_importance: pd.DataFrame
    historical: pd.DataFrame


MODELS = {
    "random_forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
    ),
}


def _build_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "random_forest",
    n_splits: int = 5,
) -> dict[str, float]:
    """시계열 교차 검증으로 모델을 평가합니다."""
    model = MODELS.get(model_name, MODELS["random_forest"])
    pipeline = _build_pipeline(model)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    maes, rmses, r2s = [], [], []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        maes.append(mean_absolute_error(y_test, preds))
        rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
        r2s.append(r2_score(y_test, preds))

    return {
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "r2": float(np.mean(r2s)),
    }


def quick_predict(
    df: pd.DataFrame,
    forecast_days: int = 5,
    model_name: str = "random_forest",
) -> tuple[float, float, float]:
    """교차 검증 없이 빠르게 예측합니다. (스크리너용)"""
    X, y, feature_cols = prepare_features(df, forecast_days=forecast_days)
    if len(X) < 30:
        raise ValueError("학습 데이터가 부족합니다.")

    model = MODELS.get(model_name, MODELS["random_forest"])
    pipeline = _build_pipeline(model)
    pipeline.fit(X, y)

    latest_features = get_latest_features(df, feature_cols)
    predicted_price = float(pipeline.predict(latest_features)[0])
    current_price = float(df["close"].iloc[-1])
    predicted_return = (predicted_price / current_price) - 1
    return current_price, predicted_price, predicted_return


def train_and_predict(
    df: pd.DataFrame,
    ticker: str,
    forecast_days: int = 5,
    model_name: str = "random_forest",
) -> PredictionResult:
    """모델을 학습하고 미래 주가를 예측합니다."""
    X, y, feature_cols = prepare_features(df, forecast_days=forecast_days)
    metrics = evaluate_model(X, y, model_name=model_name)

    model = MODELS.get(model_name, MODELS["random_forest"])
    pipeline = _build_pipeline(model)
    pipeline.fit(X, y)

    latest_features = get_latest_features(df, feature_cols)
    predicted_price = float(pipeline.predict(latest_features)[0])
    current_price = float(df["close"].iloc[-1])
    predicted_return = (predicted_price / current_price) - 1

    if predicted_return > 0.01:
        direction = "상승"
    elif predicted_return < -0.01:
        direction = "하락"
    else:
        direction = "보합"

    model_step = pipeline.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        importance = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model_step.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
    else:
        importance = pd.DataFrame({"feature": feature_cols, "importance": [0.0] * len(feature_cols)})

    return PredictionResult(
        ticker=ticker,
        current_price=current_price,
        predicted_price=predicted_price,
        forecast_days=forecast_days,
        predicted_return=predicted_return,
        direction=direction,
        model_name=model_name,
        metrics=metrics,
        feature_importance=importance,
        historical=df,
    )