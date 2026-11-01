import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import pandas as pd
import joblib

APP_DIR=Path(__file__).parent
DRAW_GOAL_THRESHOLD = 0.35


@st.cache_resource
def load_models():
    classifier_bundle = joblib.load(APP_DIR / "models" / "worldcup_match_model.joblib")
    regressor_bundle = joblib.load(APP_DIR / "models" / "worldcup_score_regressor.joblib")
    return classifier_bundle, regressor_bundle


@st.cache_data
def load_team_features():
    return pd.read_csv(APP_DIR / "data" / "team_current_features.csv").set_index("team")


def build_match_features(team_a, team_b, team_features, neutral=True):
    a = team_features.loc[team_a]
    b = team_features.loc[team_b]
    home_bonus = 0 if neutral else 50

    row = {
        "home_tier": a["tier"],
        "away_tier": b["tier"],
        "tier_diff": a["tier"] - b["tier"],
        "home_elo": a["elo"],
        "away_elo": b["elo"],
        "elo_diff": (a["elo"] + home_bonus) - b["elo"],
        "neutral": int(neutral),
    }

    for window in [5, 10]:
        row[f"home_last{window}_matches"] = a[f"last{window}_matches"]
        row[f"home_last{window}_goals_for"] = a[f"last{window}_goals_for"]
        row[f"home_last{window}_goals_against"] = a[f"last{window}_goals_against"]
        row[f"home_last{window}_goal_diff"] = a[f"last{window}_goal_diff"]

        row[f"away_last{window}_matches"] = b[f"last{window}_matches"]
        row[f"away_last{window}_goals_for"] = b[f"last{window}_goals_for"]
        row[f"away_last{window}_goals_against"] = b[f"last{window}_goals_against"]
        row[f"away_last{window}_goal_diff"] = b[f"last{window}_goal_diff"]

        row[f"last{window}_goal_diff_delta"] = (
            a[f"last{window}_goal_diff"] - b[f"last{window}_goal_diff"]
        )

    return pd.DataFrame([row])


def result_from_expected_goals(team_a_goals, team_b_goals):
    if abs(team_a_goals - team_b_goals) <= DRAW_GOAL_THRESHOLD:
        return "draw"
    if team_a_goals > team_b_goals:
        return "home_win"
    return "away_win"

st.markdown(
    """
    <h1 style="
        text-align:center;
        font-size:56px;
        font-weight:800;
        color:purple;
        margin-bottom:30px;
    ">
        WORLD CUP Prediction Dashboard
    </h1>
    """,
    unsafe_allow_html=True,
)
st.divider()

teams = [
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
    "Argentina",
    "Switzerland",
    "Turkey",
    "Italy",
    "Poland",
    "Denmark",
    "Ukraine",
    "Russia",
]

flag_urls = {
    "Brazil": "https://flagcdn.com/w320/br.png",
    "Japan": "https://flagcdn.com/w320/jp.png",
    "Netherlands": "https://flagcdn.com/w320/nl.png",
    "France": "https://flagcdn.com/w320/fr.png",
    "England": "https://flagcdn.com/w320/gb-eng.png",
    "United States": "https://flagcdn.com/w320/us.png",
    "Germany": "https://flagcdn.com/w320/de.png",
    "Spain": "https://flagcdn.com/w320/es.png",
    "Colombia": "https://flagcdn.com/w320/co.png",
    "Croatia": "https://flagcdn.com/w320/hr.png",
    "Portugal": "https://flagcdn.com/w320/pt.png",
    "Panama": "https://flagcdn.com/w320/pa.png",
    "Ghana": "https://flagcdn.com/w320/gh.png",
    "Paraguay": "https://flagcdn.com/w320/py.png",
    "Canada": "https://flagcdn.com/w320/ca.png",
    "Belgium": "https://flagcdn.com/w320/be.png",
    "China": "https://flagcdn.com/w320/cn.png",
    "South Korea": "https://flagcdn.com/w320/kr.png",
    "Czech Republic": "https://flagcdn.com/w320/cz.png",
    "India": "https://flagcdn.com/w320/in.png",
    "Morocco": "https://flagcdn.com/w320/ma.png",
    "Norway": "https://flagcdn.com/w320/no.png",
    "Australia": "https://flagcdn.com/w320/au.png",
    "Argentina": "https://flagcdn.com/w320/ar.png",
    "Switzerland": "https://flagcdn.com/w320/ch.png",
    "Turkey": "https://flagcdn.com/w320/tr.png",
    "Italy": "https://flagcdn.com/w320/it.png",
    "Poland": "https://flagcdn.com/w320/pl.png",
    "Denmark": "https://flagcdn.com/w320/dk.png",
    "Ukraine": "https://flagcdn.com/w320/ua.png",
    "Russia": "https://flagcdn.com/w320/ru.png",
}

with st.sidebar:
    team_a = st.selectbox("Select First National Team", teams)
    team_b_options = [team for team in teams if team != team_a]
    team_b = st.selectbox("Select Second National Team", team_b_options)

components.html(
    f"""
<section style="
    display:flex;
    align-items:center;
    justify-content:center;
    gap:64px;
    background:#0b1020;
    padding:30px 28px;
    border-radius:16px;
    border:3px solid #e11d48;
    font-family:Arial, sans-serif;
    margin-bottom:40px;
">
    <div style="text-align:center; min-width:220px;">
        <img
            src="{flag_urls[team_a]}"
            alt="{team_a} flag"
            style="width:160px; border-radius:8px; display:block; margin:0 auto;"
        >
        <div style="font-size:32px; font-weight:800; margin-top:12px; color:purple;">
            {team_a}
        </div>
    </div>

    <div style="font-size:46px; font-weight:900; color:purple; transform: translateY(-30px);">
        VS
    </div>

    <div style="text-align:center; min-width:220px;">
        <img
            src="{flag_urls[team_b]}"
            alt="{team_b} flag"
            style="width:160px; border-radius:8px; display:block; margin:0 auto;"
        >
        <div style="font-size:32px; font-weight:800; margin-top:12px; color:purple;">
            {team_b}
        </div>
    </div>
</section>
""",
    height=250,
)

st.divider()

st.markdown("""
<style>          
div[data-testid="stButton"] > button {
    background-color: #111827 !important;
    color: #e11d48 !important;
    border: 3px solid #e11d48 !important;
    border-radius: 18px !important;

    width: 280px !important;
    height: 90px !important;

    padding: 20px 40px !important;
}

div[data-testid="stButton"] > button p {
    font-size: 24px !important;
    font-weight: 800 !important;
    color: purple !important;
    margin: 0 !important;
}

div[data-testid="stButton"] > button:hover {
    background-color: #1e293b !important;
    border: 3px solid #e11d48 !important;
}

div[data-testid="stButton"] > button:hover p {
    color: #f43f5e !important;
}
</style>
""", unsafe_allow_html=True)

left, middle, right = st.columns([1.2, 1, 1.8])

with middle:
    show_result = st.button("Show Result", use_container_width=True)


if show_result:
    try:
        classifier_bundle, regressor_bundle = load_models()
        team_features = load_team_features()

        classifier = classifier_bundle["model"]
        regressor = regressor_bundle["model"]
        classifier_features = classifier_bundle["features"]
        regressor_features = regressor_bundle["features"]

        X = build_match_features(team_a, team_b, team_features, neutral=True)
        X_classifier = X.reindex(columns=classifier_features, fill_value=0)
        X_regressor = X.reindex(columns=regressor_features, fill_value=0)

        probabilities = classifier.predict_proba(X_classifier)[0]
        prob_map = dict(zip(classifier.classes_, probabilities))

        team_a_win_prob = prob_map.get("home_win", 0)
        draw_prob = prob_map.get("draw", 0)
        team_b_win_prob = prob_map.get("away_win", 0)

        score_prediction = regressor.predict(X_regressor)[0]
        team_a_expected_goals = max(0, score_prediction[0])
        team_b_expected_goals = max(0, score_prediction[1])

        predicted_label = result_from_expected_goals(
            team_a_expected_goals,
            team_b_expected_goals,
        )
        if predicted_label == "home_win":
            winner = team_a
        elif predicted_label == "away_win":
            winner = team_b
        else:
            winner = "Draw"
    except FileNotFoundError as error:
        st.error(f"Missing model or feature file: {error.filename}")
        st.stop()
    except KeyError as error:
        st.error(f"Missing team or feature in team_current_features.csv: {error}")
        st.stop()

    st.divider()

    components.html(
        f"""
        <div style="
            background:#0b1020;
            border:3px solid #e11d48;
            border-radius:16px;
            color:white;
            padding:22px 28px;
            text-align:center;
            font-family:Arial, sans-serif;
            margin-bottom:10px;
        ">
            <div style="font-size:24px; font-weight:700; color:purple;">
                Expected Score
            </div>
            <div style="font-size:30px; font-weight:600; margin-top:8px; color:#e11d48;">
                {team_a} {team_a_expected_goals:.2f} - {team_b_expected_goals:.2f} {team_b}
            </div>
        </div>
        """,
        height=150,
    )

    st.divider()

    components.html(
        f"""
    <section style="
        display:grid;
        grid-template-columns:repeat(3,1fr);
        gap:18px;
        font-family:Arial, sans-serif;
        margin-top:10px
        margin-bottom:10px;
    ">
        <div style="
            background:#0f172a;
            border:3px solid #e11d48;
            border-radius:16px;
            padding:24px 12px;
            text-align:center;
        ">
            <div style="
                font-size:20px;
                font-weight:700;
                color:#94a3b8;
                margin-bottom:12px;
            ">
                {team_a} Win
            </div>
            <div style="
                font-size:44px;
                font-weight:900;
                color:#e11d48;
            ">
                {team_a_win_prob:.1%}
            </div>
        </div>

        <div style="
            background:#0b1020;
            border:3px solid #e11d48;
            border-radius:16px;
            padding:24px 12px;
            text-align:center;
        ">
            <div style="
                font-size:20px;
                font-weight:700;
                color:purple;
                margin-bottom:12px;
            ">
                Draw
            </div>
            <div style="
                font-size:44px;
                font-weight:900;
                color:#94a3b8;
            ">
                {draw_prob:.1%}
            </div>
        </div>

        <div style="
            background:#0b1020;
            border:3px solid #e11d48;
            border-radius:16px;
            padding:24px 12px;
            text-align:center;
        ">
            <div style="
                font-size:20px;
                font-weight:700;
                color:#94a3b8;
                margin-bottom:12px;
            ">
                {team_b} Win
            </div>
            <div style="
                font-size:44px;
                font-weight:900;
                color:#e11d48;
            ">
                {team_b_win_prob:.1%}
            </div>
        </div>
    </section>
    """,
    height=160
    )

    st.divider()

    if winner == "Draw":
        components.html(
            f"""
            <div style="
                background:#111827;
                border:2px solid #e11d48;
                color:#e11d48;
                padding:24px;
                border-radius:14px;
                font-size:34px;
                font-weight:900;
                text-align:center;
                font-family:Arial, sans-serif;
            ">
                Draw
            </div>
            """,
            height=120,
        )
    else:
        components.html(
            f"""
            <div style="
                display:flex;
                align-items:center;
                justify-content:center;
                gap:24px;
                background:#111827;
                border:3px solid #e11d48;
                color:#e11d48;
                padding:24px;
                border-radius:14px;
                font-size:34px;
                font-weight:900;
                font-family:Arial, sans-serif;
            ">
                <img src="{flag_urls[winner]}" style="width:100px; border-radius:6px;">
                <span>{winner} Win</span>
            </div>
            """,
            height=130,
        )
