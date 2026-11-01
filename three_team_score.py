from __future__ import annotations

from itertools import combinations
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor


DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SUBJECT_TEAMS = [
    "Brazil",
    "Japan",
    "Netherlands",
    "France",
    "England",
    "United States",
    "Germany",
    "Spain",
    "Colombia",
    "Croatia",
    "Portugal",
    "Panama",
    "Ghana",
    "Paraguay",
    "Canada",
    "Belgium",
    "China",
    "South Korea",
    "Czech Republic",
    "India",
    "Morocco",
    "Norway",
    "Australia",
    "Italy",
    "Poland",
    "Denmark",
    "Ukraine",
    "Russia",
]
ROLLING_WINDOW = 20
DRAW_THRESHOLD = 0.03

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
RAW_RESULTS_PATH = DATA_DIR / "international_results.csv"


FEATURE_COLUMNS = [
    "team_a_ppm_20",
    "team_a_win_rate_20",
    "team_a_draw_rate_20",
    "team_a_attack_20",
    "team_a_defense_20",
    "team_a_goal_diff_20",
    "team_b_ppm_20",
    "team_b_win_rate_20",
    "team_b_draw_rate_20",
    "team_b_attack_20",
    "team_b_defense_20",
    "team_b_goal_diff_20",
    "ppm_diff",
    "attack_vs_defense_diff",
    "defense_vs_attack_diff",
    "goal_diff_profile_diff",
    "draw_pressure",
    "home_advantage",
    "neutral",
    "is_friendly",
    "match_importance",
]


def download_results() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if RAW_RESULTS_PATH.exists():
        return

    request = Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        RAW_RESULTS_PATH.write_bytes(response.read())


def tournament_weight(tournament: str) -> float:
    tournament_lower = str(tournament).lower()
    if "fifa world cup" in tournament_lower:
        return 2.0
    if any(
        cup in tournament_lower
        for cup in [
            "uefa euro",
            "copa américa",
            "copa america",
            "afc asian cup",
            "africa cup of nations",
            "concacaf gold cup",
            "ofc nations cup",
        ]
    ):
        return 1.6
    if "qualification" in tournament_lower or "qualifying" in tournament_lower:
        return 1.2
    if tournament_lower == "friendly":
        return 0.6
    return 1.0


def result_label(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "team_a_win"
    if home_score < away_score:
        return "team_b_win"
    return "draw"


def points_for(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def load_results() -> pd.DataFrame:
    download_results()
    results = pd.read_csv(RAW_RESULTS_PATH, parse_dates=["date"])
    results["match_id"] = results.index
    return results


def build_long_results(results: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "match_id": results.index,
            "date": results["date"],
            "team": results["home_team"],
            "opponent": results["away_team"],
            "goals_for": results["home_score"],
            "goals_against": results["away_score"],
        }
    )
    away = pd.DataFrame(
        {
            "match_id": results.index,
            "date": results["date"],
            "team": results["away_team"],
            "opponent": results["home_team"],
            "goals_for": results["away_score"],
            "goals_against": results["home_score"],
        }
    )

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    long["goal_diff"] = long["goals_for"] - long["goals_against"]
    long["points"] = [
        points_for(gf, ga) for gf, ga in zip(long["goals_for"], long["goals_against"])
    ]
    long["win"] = (long["goal_diff"] > 0).astype(int)
    long["draw"] = (long["goal_diff"] == 0).astype(int)

    rolling_specs = {
        "ppm_20": "points",
        "win_rate_20": "win",
        "draw_rate_20": "draw",
        "attack_20": "goals_for",
        "goals_against_20": "goals_against",
        "goal_diff_20": "goal_diff",
    }

    for output_col, source_col in rolling_specs.items():
        long[output_col] = (
            long.groupby("team")[source_col]
            .transform(lambda s: s.shift().rolling(ROLLING_WINDOW, min_periods=5).mean())
        )

    long["defense_20"] = -long["goals_against_20"]
    return long


def latest_profile(long_results: pd.DataFrame, team: str) -> pd.Series:
    rows = long_results[long_results["team"] == team].sort_values("date")
    if rows.empty:
        raise ValueError(f"No historical matches found for {team}")

    recent = rows.tail(ROLLING_WINDOW)
    return pd.Series(
        {
            "ppm_20": recent["points"].mean(),
            "win_rate_20": recent["win"].mean(),
            "draw_rate_20": recent["draw"].mean(),
            "attack_20": recent["goals_for"].mean(),
            "defense_20": -recent["goals_against"].mean(),
            "goal_diff_20": recent["goal_diff"].mean(),
        }
    )


def add_matchup_features(dataset: pd.DataFrame) -> pd.DataFrame:
    output = dataset.copy()
    output["ppm_diff"] = output["team_a_ppm_20"] - output["team_b_ppm_20"]
    output["attack_vs_defense_diff"] = output["team_a_attack_20"] + output["team_b_defense_20"]
    output["defense_vs_attack_diff"] = output["team_a_defense_20"] + output["team_b_attack_20"]
    output["goal_diff_profile_diff"] = (
        output["team_a_goal_diff_20"] - output["team_b_goal_diff_20"]
    )
    output["draw_pressure"] = (output["team_a_draw_rate_20"] + output["team_b_draw_rate_20"]) / 2
    return output


def build_classification_dataset(results: pd.DataFrame, long_results: pd.DataFrame) -> pd.DataFrame:
    profile_cols = [
        "match_id",
        "team",
        "ppm_20",
        "win_rate_20",
        "draw_rate_20",
        "attack_20",
        "defense_20",
        "goal_diff_20",
    ]
    profiles = long_results[profile_cols]

    dataset = results.copy()
    dataset["team_a"] = dataset["home_team"]
    dataset["team_b"] = dataset["away_team"]
    dataset["home_advantage"] = np.where(dataset["neutral"], 0, 1)
    dataset["neutral"] = dataset["neutral"].astype(int)
    dataset["is_friendly"] = (dataset["tournament"] == "Friendly").astype(int)
    dataset["match_importance"] = dataset["tournament"].map(tournament_weight)
    dataset["sample_weight"] = dataset["match_importance"]
    dataset["label"] = [
        result_label(home, away)
        for home, away in zip(dataset["home_score"], dataset["away_score"])
    ]

    team_a_profiles = profiles.rename(
        columns={
            "team": "team_a",
            "ppm_20": "team_a_ppm_20",
            "win_rate_20": "team_a_win_rate_20",
            "draw_rate_20": "team_a_draw_rate_20",
            "attack_20": "team_a_attack_20",
            "defense_20": "team_a_defense_20",
            "goal_diff_20": "team_a_goal_diff_20",
        }
    )
    team_b_profiles = profiles.rename(
        columns={
            "team": "team_b",
            "ppm_20": "team_b_ppm_20",
            "win_rate_20": "team_b_win_rate_20",
            "draw_rate_20": "team_b_draw_rate_20",
            "attack_20": "team_b_attack_20",
            "defense_20": "team_b_defense_20",
            "goal_diff_20": "team_b_goal_diff_20",
        }
    )

    dataset = dataset.merge(team_a_profiles, on=["match_id", "team_a"], how="left")
    dataset = dataset.merge(team_b_profiles, on=["match_id", "team_b"], how="left")
    dataset = add_matchup_features(dataset)
    dataset = dataset.dropna(subset=FEATURE_COLUMNS + ["label", "home_score", "away_score"])
    return dataset.sort_values("date").reset_index(drop=True)


def train_classifier(dataset: pd.DataFrame):
    split_index = int(len(dataset) * 0.8)
    train_df = dataset.iloc[:split_index]
    val_df = dataset.iloc[split_index:]

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        train_df[FEATURE_COLUMNS],
        train_df["label"],
        sample_weight=train_df["sample_weight"],
    )

    val_prob = model.predict_proba(val_df[FEATURE_COLUMNS])
    val_pred = model.predict(val_df[FEATURE_COLUMNS])
    metrics = {
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "validation_accuracy": round(accuracy_score(val_df["label"], val_pred), 3),
        "validation_log_loss": round(log_loss(val_df["label"], val_prob, labels=model.classes_), 3),
    }
    return model, metrics


def train_score_regressor(dataset: pd.DataFrame):
    split_index = int(len(dataset) * 0.8)
    train_df = dataset.iloc[:split_index]
    val_df = dataset.iloc[split_index:]

    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )
    )
    model.fit(
        train_df[FEATURE_COLUMNS],
        train_df[["home_score", "away_score"]],
        sample_weight=train_df["sample_weight"],
    )

    val_pred = model.predict(val_df[FEATURE_COLUMNS])
    metrics = {
        "score_validation_mae_team_a": round(
            mean_absolute_error(val_df["home_score"], val_pred[:, 0]),
            3,
        ),
        "score_validation_mae_team_b": round(
            mean_absolute_error(val_df["away_score"], val_pred[:, 1]),
            3,
        ),
    }
    return model, metrics


def build_prediction_rows(long_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for team_a, team_b in combinations(SUBJECT_TEAMS, 2):
        profile_a = latest_profile(long_results, team_a)
        profile_b = latest_profile(long_results, team_b)
        rows.append(
            {
                "team_a": team_a,
                "team_b": team_b,
                "team_a_ppm_20": profile_a["ppm_20"],
                "team_a_win_rate_20": profile_a["win_rate_20"],
                "team_a_draw_rate_20": profile_a["draw_rate_20"],
                "team_a_attack_20": profile_a["attack_20"],
                "team_a_defense_20": profile_a["defense_20"],
                "team_a_goal_diff_20": profile_a["goal_diff_20"],
                "team_b_ppm_20": profile_b["ppm_20"],
                "team_b_win_rate_20": profile_b["win_rate_20"],
                "team_b_draw_rate_20": profile_b["draw_rate_20"],
                "team_b_attack_20": profile_b["attack_20"],
                "team_b_defense_20": profile_b["defense_20"],
                "team_b_goal_diff_20": profile_b["goal_diff_20"],
                "home_advantage": 0,
                "neutral": 1,
                "is_friendly": 0,
                "match_importance": 2.0,
            }
        )

    return add_matchup_features(pd.DataFrame(rows))


def prediction_from_prob(row: pd.Series) -> str:
    probs = {
        "team_a_win": row["team_a_win_prob"],
        "draw": row["draw_prob"],
        "team_b_win": row["team_b_win_prob"],
    }
    top_label = max(probs, key=probs.get)
    sorted_probs = sorted(probs.values(), reverse=True)
    if top_label == "draw" or sorted_probs[0] - sorted_probs[1] < DRAW_THRESHOLD:
        return "Draw tendency"
    if top_label == "team_a_win":
        return row["team_a"]
    if top_label == "team_b_win":
        return row["team_b"]
    return "Draw tendency"


def main() -> None:
    results = load_results()
    long_results = build_long_results(results)
    dataset = build_classification_dataset(results, long_results)
    model, metrics = train_classifier(dataset)
    score_model, score_metrics = train_score_regressor(dataset)
    metrics.update(score_metrics)

    predictions = build_prediction_rows(long_results)
    probs = model.predict_proba(predictions[FEATURE_COLUMNS])
    expected_scores = score_model.predict(predictions[FEATURE_COLUMNS])
    class_to_index = {label: index for index, label in enumerate(model.classes_)}
    predictions["team_a_win_prob"] = probs[:, class_to_index["team_a_win"]]
    predictions["draw_prob"] = probs[:, class_to_index["draw"]]
    predictions["team_b_win_prob"] = probs[:, class_to_index["team_b_win"]]
    predictions["team_a_expected_goals"] = np.clip(expected_scores[:, 0], 0, 6)
    predictions["team_b_expected_goals"] = np.clip(expected_scores[:, 1], 0, 6)
    predictions["predicted_winner"] = predictions.apply(prediction_from_prob, axis=1)
    predictions["score_gap"] = (
        predictions[["team_a_win_prob", "draw_prob", "team_b_win_prob"]]
        .apply(lambda row: row.sort_values(ascending=False).iloc[0] - row.sort_values(ascending=False).iloc[1], axis=1)
        .round(3)
    )
    predictions["draw_threshold"] = DRAW_THRESHOLD

    display_predictions = predictions[
        [
            "team_a",
            "team_b",
            "team_a_win_prob",
            "draw_prob",
            "team_b_win_prob",
            "team_a_expected_goals",
            "team_b_expected_goals",
            "predicted_winner",
            "score_gap",
            "draw_threshold",
        ]
    ].copy()
    for col in ["team_a_win_prob", "draw_prob", "team_b_win_prob"]:
        display_predictions[col] = display_predictions[col].round(3)
    for col in ["team_a_expected_goals", "team_b_expected_goals"]:
        display_predictions[col] = display_predictions[col].round(2)

    dataset.to_csv(DATA_DIR / "three_team_classification_dataset.csv", index=False)
    display_predictions.to_csv(DATA_DIR / "three_team_pairwise_decisions.csv", index=False)
    display_predictions.to_csv(DATA_DIR / "three_team_classification_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(DATA_DIR / "three_team_model_metrics.csv", index=False)

    print("Saved:")
    print(DATA_DIR / "three_team_classification_dataset.csv")
    print(DATA_DIR / "three_team_classification_predictions.csv")
    print(DATA_DIR / "three_team_pairwise_decisions.csv")
    print(DATA_DIR / "three_team_model_metrics.csv")
    print()
    print("Model metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print()
    print("Pairwise decisions:")
    print(display_predictions.to_string(index=False))


if __name__ == "__main__":
    main()
