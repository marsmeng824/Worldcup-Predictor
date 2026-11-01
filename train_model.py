from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)


BASE_FEATURES = [
    "home_tier",
    "away_tier",
    "tier_diff",
    "home_elo",
    "away_elo",
    "elo_diff",
    "neutral",
]
FORM_WINDOWS = [5, 10]
FORM_FEATURES = [
    feature
    for window in FORM_WINDOWS
    for feature in [
        f"home_last{window}_matches",
        f"home_last{window}_goals_for",
        f"home_last{window}_goals_against",
        f"home_last{window}_goal_diff",
        f"away_last{window}_matches",
        f"away_last{window}_goals_for",
        f"away_last{window}_goals_against",
        f"away_last{window}_goal_diff",
        f"last{window}_goal_diff_delta",
    ]
]
FEATURES = BASE_FEATURES + FORM_FEATURES

TARGET = "label"
SCORE_TARGETS = ["home_score", "away_score"]
LABEL_ORDER = ["home_win", "draw", "away_win"]


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {"date", TARGET, *BASE_FEATURES, *SCORE_TARGETS}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad_rows = int(df["date"].isna().sum())
        raise ValueError(f"Found {bad_rows} rows with invalid dates.")

    df["neutral"] = df["neutral"].astype(bool).astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    missing_form_features = [feature for feature in FORM_FEATURES if feature not in df.columns]
    if missing_form_features:
        df = add_recent_form_features(df)
    return df


def get_recent_form(history: list[tuple[int, int]], window: int) -> dict[str, int]:
    recent_matches = history[-window:]
    goals_for = sum(match[0] for match in recent_matches)
    goals_against = sum(match[1] for match in recent_matches)
    return {
        "matches": len(recent_matches),
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_diff": goals_for - goals_against,
    }


def add_recent_form_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    team_history: dict[str, list[tuple[int, int]]] = defaultdict(list)
    feature_rows = []

    for _, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        feature_row = {}

        for window in FORM_WINDOWS:
            home_form = get_recent_form(team_history[home_team], window)
            away_form = get_recent_form(team_history[away_team], window)

            for stat_name, stat_value in home_form.items():
                feature_row[f"home_last{window}_{stat_name}"] = stat_value
            for stat_name, stat_value in away_form.items():
                feature_row[f"away_last{window}_{stat_name}"] = stat_value

            feature_row[f"last{window}_goal_diff_delta"] = (
                home_form["goal_diff"] - away_form["goal_diff"]
            )

        feature_rows.append(feature_row)

        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        team_history[home_team].append((home_score, away_score))
        team_history[away_team].append((away_score, home_score))

    form_df = pd.DataFrame(feature_rows, index=df.index)
    return pd.concat([df, form_df], axis=1)


def labels_from_scores(home_scores: pd.Series, away_scores: pd.Series) -> pd.Series:
    labels = np.select(
        [
            home_scores > away_scores,
            home_scores < away_scores,
        ],
        [
            "home_win",
            "away_win",
        ],
        default="draw",
    )
    return pd.Series(labels, index=home_scores.index)


def split_by_time(df: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.5 <= train_fraction < 1:
        raise ValueError("train_fraction must be between 0.5 and 1.0.")

    split_index = int(len(df) * train_fraction)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced an empty dataset.")
    return train_df, test_df


def train_model(train_df: pd.DataFrame) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_df[FEATURES], train_df[TARGET])
    return model


def train_score_regressor(train_df: pd.DataFrame) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_df[FEATURES], train_df[SCORE_TARGETS])
    return model


def evaluate_model(
    model: RandomForestClassifier,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_test = test_df[TARGET]
    y_pred = model.predict(test_df[FEATURES])
    y_prob = model.predict_proba(test_df[FEATURES])

    metrics = [
        ("train_rows", len(train_df)),
        ("test_rows", len(test_df)),
        ("train_start_date", train_df["date"].min().date().isoformat()),
        ("train_end_date", train_df["date"].max().date().isoformat()),
        ("test_start_date", test_df["date"].min().date().isoformat()),
        ("test_end_date", test_df["date"].max().date().isoformat()),
        ("accuracy", accuracy_score(y_test, y_pred)),
        ("balanced_accuracy", balanced_accuracy_score(y_test, y_pred)),
        ("log_loss", log_loss(y_test, y_prob, labels=model.classes_)),
    ]
    metrics_df = pd.DataFrame(metrics, columns=["metric", "value"])

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=[label for label in LABEL_ORDER if label in model.classes_],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose().reset_index(names="label")

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    confusion_df = pd.DataFrame(
        cm,
        index=[f"actual_{label}" for label in model.classes_],
        columns=[f"predicted_{label}" for label in model.classes_],
    )

    return metrics_df, report_df, confusion_df


def build_predictions(model: RandomForestClassifier, test_df: pd.DataFrame) -> pd.DataFrame:
    predictions = test_df.copy()
    predictions["predicted_label"] = model.predict(test_df[FEATURES])

    probabilities = model.predict_proba(test_df[FEATURES])
    for class_name, class_prob in zip(model.classes_, probabilities.T):
        predictions[f"prob_{class_name}"] = class_prob

    return predictions


def evaluate_score_regressor(
    model: RandomForestRegressor,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_predictions = model.predict(test_df[FEATURES])
    predicted_home_score = pd.Series(score_predictions[:, 0], index=test_df.index).clip(lower=0)
    predicted_away_score = pd.Series(score_predictions[:, 1], index=test_df.index).clip(lower=0)

    predicted_home_goals = predicted_home_score.round().astype(int)
    predicted_away_goals = predicted_away_score.round().astype(int)
    predicted_labels = labels_from_scores(predicted_home_goals, predicted_away_goals)

    y_test = test_df[TARGET]
    actual_scores = test_df[SCORE_TARGETS]

    home_rmse = mean_squared_error(
        actual_scores["home_score"],
        predicted_home_score,
    ) ** 0.5
    away_rmse = mean_squared_error(
        actual_scores["away_score"],
        predicted_away_score,
    ) ** 0.5

    metrics = [
        ("train_rows", len(train_df)),
        ("test_rows", len(test_df)),
        ("train_start_date", train_df["date"].min().date().isoformat()),
        ("train_end_date", train_df["date"].max().date().isoformat()),
        ("test_start_date", test_df["date"].min().date().isoformat()),
        ("test_end_date", test_df["date"].max().date().isoformat()),
        ("home_score_mae", mean_absolute_error(actual_scores["home_score"], predicted_home_score)),
        ("away_score_mae", mean_absolute_error(actual_scores["away_score"], predicted_away_score)),
        ("home_score_rmse", home_rmse),
        ("away_score_rmse", away_rmse),
        ("derived_label_accuracy", accuracy_score(y_test, predicted_labels)),
        ("derived_label_balanced_accuracy", balanced_accuracy_score(y_test, predicted_labels)),
    ]
    metrics_df = pd.DataFrame(metrics, columns=["metric", "value"])

    report_dict = classification_report(
        y_test,
        predicted_labels,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose().reset_index(names="label")

    cm = confusion_matrix(y_test, predicted_labels, labels=LABEL_ORDER)
    confusion_df = pd.DataFrame(
        cm,
        index=[f"actual_{label}" for label in LABEL_ORDER],
        columns=[f"predicted_{label}" for label in LABEL_ORDER],
    )

    predictions_df = test_df.copy()
    predictions_df["predicted_home_score"] = predicted_home_score
    predictions_df["predicted_away_score"] = predicted_away_score
    predictions_df["predicted_home_goals"] = predicted_home_goals
    predictions_df["predicted_away_goals"] = predicted_away_goals
    predictions_df["predicted_label_from_goals"] = predicted_labels

    return metrics_df, report_df, confusion_df, predictions_df


def save_outputs(
    model: RandomForestClassifier,
    score_regressor: RandomForestRegressor,
    output_dir: Path,
    model_dir: Path,
    metrics_df: pd.DataFrame,
    report_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    regression_metrics_df: pd.DataFrame,
    regression_report_df: pd.DataFrame,
    regression_confusion_df: pd.DataFrame,
    regression_predictions_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)
    report_df.to_csv(output_dir / "model_classification_report.csv", index=False)
    confusion_df.to_csv(output_dir / "model_confusion_matrix.csv")
    predictions_df.to_csv(output_dir / "model_test_predictions.csv", index=False)

    importance_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(output_dir / "model_feature_importance.csv", index=False)

    regression_importance_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": score_regressor.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    regression_importance_df.to_csv(
        output_dir / "regression_feature_importance.csv",
        index=False,
    )
    regression_metrics_df.to_csv(output_dir / "regression_metrics.csv", index=False)
    regression_report_df.to_csv(
        output_dir / "regression_classification_report.csv",
        index=False,
    )
    regression_confusion_df.to_csv(output_dir / "regression_confusion_matrix.csv")
    regression_predictions_df.to_csv(
        output_dir / "regression_test_predictions.csv",
        index=False,
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "classes": list(model.classes_),
        },
        model_dir / "worldcup_match_model.joblib",
    )
    joblib.dump(
        {
            "model": score_regressor,
            "features": FEATURES,
            "score_targets": SCORE_TARGETS,
        },
        model_dir / "worldcup_score_regressor.joblib",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a World Cup match outcome model.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "analysis_data.csv",
        help="Path to the training dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for metrics and prediction CSV outputs.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory for the saved model.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Chronological fraction used for training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input)
    train_df, test_df = split_by_time(df, args.train_fraction)

    model = train_model(train_df)
    metrics_df, report_df, confusion_df = evaluate_model(model, train_df, test_df)
    predictions_df = build_predictions(model, test_df)

    score_regressor = train_score_regressor(train_df)
    (
        regression_metrics_df,
        regression_report_df,
        regression_confusion_df,
        regression_predictions_df,
    ) = evaluate_score_regressor(score_regressor, train_df, test_df)

    save_outputs(
        model,
        score_regressor,
        args.output_dir,
        args.model_dir,
        metrics_df,
        report_df,
        confusion_df,
        predictions_df,
        regression_metrics_df,
        regression_report_df,
        regression_confusion_df,
        regression_predictions_df,
    )

    print("Classification model training complete.")
    print(metrics_df.to_string(index=False))
    print("\nFeature importance:")
    print(
        pd.DataFrame(
            {"feature": FEATURES, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .to_string(index=False)
    )
    print("\nRegression score model training complete.")
    print(regression_metrics_df.to_string(index=False))
    print("\nRegression feature importance:")
    print(
        pd.DataFrame(
            {"feature": FEATURES, "importance": score_regressor.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
