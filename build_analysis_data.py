from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


BASE_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
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

TEAM_ALIASES = {
    "Czechia": "Czech Republic",
    "Curacao": "Cura\u00e7ao",
}

TIER_OVERRIDES = {
    "Turkey": 3,
    "China": 1,
    "India": 1,
}


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def normalize_team_name(team: str) -> str:
    return TEAM_ALIASES.get(team, team)


def strength_to_tier(strength: float) -> int:
    if strength >= 95:
        return 5
    if strength >= 88:
        return 4
    if strength >= 80:
        return 3
    if strength >= 60:
        return 2
    return 1


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


def load_raw_matches(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_columns = sorted(set(BASE_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required match columns: {missing_columns}")

    df = df[BASE_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"] = df["neutral"].map(parse_bool)

    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.drop_duplicates(subset=BASE_COLUMNS)
    return df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def load_target_teams_and_tiers(path: Path) -> tuple[set[str], dict[str, int]]:
    teams_df = pd.read_csv(path)
    if "team" not in teams_df.columns:
        raise ValueError("Team file must contain a 'team' column.")

    teams_df = teams_df.copy()
    teams_df["team"] = teams_df["team"].map(normalize_team_name)
    target_teams = set(teams_df["team"])

    tier_map: dict[str, int] = {}
    for _, row in teams_df.iterrows():
        team = row["team"]
        if "strength" in teams_df.columns and pd.notna(row.get("strength")):
            tier_map[team] = strength_to_tier(float(row["strength"]))
        else:
            tier_map[team] = 1

    tier_map.update(TIER_OVERRIDES)
    return target_teams, tier_map


def add_dynamic_features(
    df: pd.DataFrame,
    tier_map: dict[str, int],
    k_factor: float,
    home_advantage: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    elo: defaultdict[str, float] = defaultdict(lambda: 1500.0)
    team_history: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    feature_rows = []

    for _, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_elo = elo[home_team]
        away_elo = elo[away_team]
        home_bonus = 0 if row["neutral"] else home_advantage

        feature_row = {
            "label": labels_from_scores(
                pd.Series([row["home_score"]]),
                pd.Series([row["away_score"]]),
            ).iloc[0],
            "goal_diff": row["home_score"] - row["away_score"],
            "home_tier": tier_map.get(home_team, 0),
            "away_tier": tier_map.get(away_team, 0),
            "tier_diff": tier_map.get(home_team, 0) - tier_map.get(away_team, 0),
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": (home_elo + home_bonus) - away_elo,
        }

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

        if row["home_score"] > row["away_score"]:
            home_result = 1
        elif row["home_score"] == row["away_score"]:
            home_result = 0.5
        else:
            home_result = 0

        expected_home = 1 / (1 + 10 ** ((away_elo - (home_elo + home_bonus)) / 400))
        expected_away = 1 - expected_home
        elo[home_team] = home_elo + k_factor * (home_result - expected_home)
        elo[away_team] = away_elo + k_factor * ((1 - home_result) - expected_away)

        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        team_history[home_team].append((home_score, away_score))
        team_history[away_team].append((away_score, home_score))

    features_df = pd.DataFrame(feature_rows, index=df.index)
    enriched_df = pd.concat([df, features_df], axis=1)

    snapshot_rows = []
    teams = sorted(set(df["home_team"]).union(df["away_team"]))
    for team in teams:
        snapshot_row = {
            "team": team,
            "tier": tier_map.get(team, 0),
            "elo": elo[team],
            "matches_in_history": len(team_history[team]),
        }
        for window in FORM_WINDOWS:
            form = get_recent_form(team_history[team], window)
            for stat_name, stat_value in form.items():
                snapshot_row[f"last{window}_{stat_name}"] = stat_value
        snapshot_rows.append(snapshot_row)

    snapshot_df = pd.DataFrame(snapshot_rows)
    return enriched_df, snapshot_df


def build_analysis_data(
    matches_path: Path,
    teams_path: Path,
    output_path: Path,
    snapshot_path: Path,
    history_start_date: str,
    analysis_start_date: str,
    k_factor: float,
    home_advantage: float,
) -> pd.DataFrame:
    matches_df = load_raw_matches(matches_path)
    target_teams, tier_map = load_target_teams_and_tiers(teams_path)

    history_start = pd.Timestamp(history_start_date)
    analysis_start = pd.Timestamp(analysis_start_date)
    history_df = matches_df[matches_df["date"] >= history_start].copy()

    enriched_df, snapshot_df = add_dynamic_features(
        history_df,
        tier_map=tier_map,
        k_factor=k_factor,
        home_advantage=home_advantage,
    )

    selected_mask = (
        enriched_df["home_team"].isin(target_teams)
        | enriched_df["away_team"].isin(target_teams)
    )
    analysis_df = enriched_df[
        selected_mask & (enriched_df["date"] >= analysis_start)
    ].copy()

    ordered_columns = [
        *BASE_COLUMNS,
        "label",
        "goal_diff",
        "home_tier",
        "away_tier",
        "tier_diff",
        "home_elo",
        "away_elo",
        "elo_diff",
        *FORM_FEATURES,
    ]
    analysis_df = analysis_df[ordered_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_df.to_csv(output_path, index=False)
    snapshot_df.to_csv(snapshot_path, index=False)
    return analysis_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build World Cup analysis dataset.")
    parser.add_argument(
        "--matches",
        type=Path,
        default=Path("data") / "international_results.csv",
        help="Raw historical match results CSV.",
    )
    parser.add_argument(
        "--teams",
        type=Path,
        default=Path("worldcup_2026_teams.csv"),
        help="Target teams CSV with team and strength columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "analysis_data.csv",
        help="Output analysis dataset CSV.",
    )
    parser.add_argument(
        "--snapshot-output",
        type=Path,
        default=Path("data") / "team_current_features.csv",
        help="Current team Elo and recent-form snapshot CSV.",
    )
    parser.add_argument("--history-start-date", default="2014-01-01")
    parser.add_argument("--analysis-start-date", default="2014-01-01")
    parser.add_argument("--k-factor", type=float, default=30)
    parser.add_argument("--home-advantage", type=float, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_df = build_analysis_data(
        matches_path=args.matches,
        teams_path=args.teams,
        output_path=args.output,
        snapshot_path=args.snapshot_output,
        history_start_date=args.history_start_date,
        analysis_start_date=args.analysis_start_date,
        k_factor=args.k_factor,
        home_advantage=args.home_advantage,
    )
    print(f"Saved {len(analysis_df)} rows to {args.output}")
    print(f"Date range: {analysis_df['date'].min()} to {analysis_df['date'].max()}")


if __name__ == "__main__":
    main()
