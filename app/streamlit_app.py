import sys
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.predict import load_artifact, predict_match
from src.data_preprocessing import get_h2h_records

st.set_page_config(
    page_title='Matchday Pro | FIFA World Cup AI Predictor',
    page_icon='⚽',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom High-End Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
* { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: #080e1a; color: #f1f5f9; }
[data-testid="stSidebar"] { background: #0b1528; border-right: 1px solid #1e293b; }
.hero-header {
    background: linear-gradient(135deg, #0f2744 0%, #0d3840 50%, #0b1c2d 100%);
    border: 1px solid #1e3a5f;
    padding: 2.2rem;
    border-radius: 1.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
.eyebrow {
    color: #4ade80 !important;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff !important;
    margin: 0;
    letter-spacing: -0.02em;
}
.subtitle {
    color: #94a3b8 !important;
    font-size: 1rem;
    margin-top: 0.5rem;
}
.card {
    background: #0f1c32;
    border: 1px solid #1e3250;
    border-radius: 1rem;
    padding: 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.card-winner {
    background: linear-gradient(135deg, #132d4b 0%, #103a35 100%);
    border: 1px solid #22c55e;
    border-radius: 1rem;
    padding: 1.5rem;
    margin-bottom: 1rem;
    text-align: center;
}
.prob-label {
    font-weight: 600;
    font-size: 0.95rem;
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.35rem;
}
.prob-bar {
    height: 0.85rem;
    background: #1e293b;
    border-radius: 1rem;
    overflow: hidden;
    margin-bottom: 1rem;
}
.prob-fill-win { height: 100%; background: linear-gradient(90deg, #10b981, #34d399); border-radius: 1rem; }
.prob-fill-draw { height: 100%; background: linear-gradient(90deg, #64748b, #94a3b8); border-radius: 1rem; }
.prob-fill-loss { height: 100%; background: linear-gradient(90deg, #3b82f6, #60a5fa); border-radius: 1rem; }
.stat-pill {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 0.5rem;
    font-size: 0.8rem;
    font-weight: 600;
    background: #1e293b;
    color: #cbd5e1;
    margin-right: 0.4rem;
}
div.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: #ffffff;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 0.75rem;
    padding: 0.75rem 1.5rem;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    transition: all 0.2s ease;
}
div.stButton button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(16, 185, 129, 0.45);
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner='Initializing match models, FIFA ratings & shootout intelligence…')
def get_model():
    return load_artifact()

# App Header
st.markdown("""
<div class="hero-header">
    <p class="eyebrow">Matchday Pro · Global Football Intelligence</p>
    <h1 class="main-title">FIFA World Cup AI Match Predictor</h1>
    <p class="subtitle">Ensemble Machine Learning · Historical FIFA World Rankings · Dynamic Chronological Elo · Penalty Shootout Modeling</p>
</div>
""", unsafe_allow_html=True)

try:
    artifact = get_model()
except Exception as exc:
    st.error(f"Error loading prediction engine: {exc}")
    st.stop()

# Sidebar: Model Intelligence & Held-Out Validation
with st.sidebar:
    st.markdown("### 🏆 Engine Performance")
    m = artifact['metrics']
    col_a, col_b = st.columns(2)
    col_a.metric('Holdout Accuracy', f"{m['accuracy']:.1%}")
    col_b.metric('Log Loss', f"{m['log_loss']:.3f}")

    st.caption(f"📊 Evaluated on **{m['matches']:,} matches** in 2022–2025 out-of-time holdout.")
    st.divider()

    st.markdown("### 📚 Data Sources")
    st.markdown("""
    - **49,200+ Scored Matches** (1872–present)
    - **Official FIFA World Rankings** (1992–2024)
    - **675 Penalty Shootouts** (Knockout tiebreaks)
    - **47,600+ Goalscorers** (Attacking metrics)
    """)
    st.caption(f"Latest calibrated fixture date: `{artifact['last_result']}`")
    st.divider()
    st.caption("Engine: Calibrated Logistic Regression + Histogram Gradient Boosting Soft Voting Ensemble with Bivariate Poisson xG.")

teams = artifact['teams']
team_stats = artifact['team_stats']

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Match Simulator",
    "⚔️ Head-to-Head Records",
    "📊 Team Intelligence Radar",
    "🏆 World Cup 2026 Arena"
])

with tab1:
    st.subheader("Configure Fixture")

    with st.form("match_form"):
        col1, col2 = st.columns(2)
        default_home = teams.index('Brazil') if 'Brazil' in teams else 0
        default_away = teams.index('France') if 'France' in teams else 1

        home_team = col1.selectbox("Home / Team 1", teams, index=default_home)
        away_team = col2.selectbox("Away / Team 2", teams, index=default_away)

        # Quick stat chips
        h_s = team_stats.get(home_team, {})
        a_s = team_stats.get(away_team, {})

        c1, c2 = st.columns(2)
        c1.markdown(f"""
        <span class="stat-pill">FIFA #{h_s.get('fifa_rank', 'N/A')}</span>
        <span class="stat-pill">Elo: {h_s.get('elo', 1500)}</span>
        <span class="stat-pill">{h_s.get('confederation', '')}</span>
        """, unsafe_allow_html=True)

        c2.markdown(f"""
        <span class="stat-pill">FIFA #{a_s.get('fifa_rank', 'N/A')}</span>
        <span class="stat-pill">Elo: {a_s.get('elo', 1500)}</span>
        <span class="stat-pill">{a_s.get('confederation', '')}</span>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        col_s1, col_s2, col_s3 = st.columns(3)

        tournament = col_s1.selectbox(
            "Tournament Category",
            ['FIFA World Cup', 'UEFA Euro', 'Copa América', 'FIFA World Cup qualification', 'UEFA Nations League', 'Friendly'],
            index=0
        )
        neutral = col_s2.checkbox("Neutral Venue", value=True, help="World Cup tournament matches are played on neutral territory.")
        is_knockout = col_s3.checkbox("Knockout Match", value=True, help="Simulate extra time and penalty shootouts if regulation ends in a draw.")

        submitted = st.form_submit_button("Predict Outcome & Simulate Match", type="primary", use_container_width=True)

    if submitted:
        if home_team == away_team:
            st.error("Please select two distinct teams.")
        else:
            res = predict_match(home_team, away_team, neutral=neutral, tournament=tournament, is_knockout=is_knockout, artifact=artifact)
            probs = res['probabilities']

            st.markdown(f"## {home_team} vs {away_team}")

            # Top Winner Highlight Card
            win_pct = max(probs.values())
            win_label = res['winner']
            favored_name = home_team if win_label == 'Home win' else away_team if win_label == 'Away win' else 'Draw'

            st.markdown(f"""
            <div class="card-winner">
                <p class="eyebrow">Most Likely Regulation Outcome</p>
                <h2 style="font-size: 2rem; margin: 0.3rem 0; color: #ffffff;">{favored_name} ({win_label})</h2>
                <p style="color: #4ade80; font-weight: 700; font-size: 1.2rem; margin: 0;">{win_pct:.1%} Estimated Probability</p>
            </div>
            """, unsafe_allow_html=True)

            # Probabilities Grid
            col_p1, col_p2, col_p3 = st.columns(3)

            col_p1.markdown(f"""
            <div class="card">
                <div class="prob-label"><span>{home_team} Win</span><span>{probs['Home win']:.1%}</span></div>
                <div class="prob-bar"><div class="prob-fill-win" style="width: {probs['Home win']*100:.1f}%;"></div></div>
                <span class="stat-pill">Expected Goals: {res['expected_goals'][home_team]}</span>
            </div>
            """, unsafe_allow_html=True)

            col_p2.markdown(f"""
            <div class="card">
                <div class="prob-label"><span>Draw</span><span>{probs['Draw']:.1%}</span></div>
                <div class="prob-bar"><div class="prob-fill-draw" style="width: {probs['Draw']*100:.1f}%;"></div></div>
                <span class="stat-pill">Full Time 90 min</span>
            </div>
            """, unsafe_allow_html=True)

            col_p3.markdown(f"""
            <div class="card">
                <div class="prob-label"><span>{away_team} Win</span><span>{probs['Away win']:.1%}</span></div>
                <div class="prob-bar"><div class="prob-fill-loss" style="width: {probs['Away win']*100:.1f}%;"></div></div>
                <span class="stat-pill">Expected Goals: {res['expected_goals'][away_team]}</span>
            </div>
            """, unsafe_allow_html=True)

            # Knockout Tiebreak Section
            if is_knockout and res['knockout_advance']:
                adv = res['knockout_advance']
                st.markdown("### 🏆 Knockout Tiebreak & Advancement")
                k1, k2 = st.columns(2)
                h_adv = adv[home_team]
                a_adv = adv[away_team]

                k1.markdown(f"""
                <div class="card" style="border-left: 4px solid #10b981;">
                    <h3 style="margin-top:0;">{home_team} Advance: {h_adv:.1%}</h3>
                    <p style="color: #94a3b8; font-size: 0.9rem;">
                        Shootout Win Rate: <strong>{h_s.get('shootout_win_rate', 50)}%</strong> ({h_s.get('shootout_wins', 0)}/{h_s.get('shootout_total', 0)} won)
                    </p>
                </div>
                """, unsafe_allow_html=True)

                k2.markdown(f"""
                <div class="card" style="border-left: 4px solid #3b82f6;">
                    <h3 style="margin-top:0;">{away_team} Advance: {a_adv:.1%}</h3>
                    <p style="color: #94a3b8; font-size: 0.9rem;">
                        Shootout Win Rate: <strong>{a_s.get('shootout_win_rate', 50)}%</strong> ({a_s.get('shootout_wins', 0)}/{a_s.get('shootout_total', 0)} won)
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Scoreline & Goal Market Analytics
            st.markdown("### 🎯 Expected Goals & Scoreline Probabilities")
            s_col1, s_col2 = st.columns([1.5, 1])

            with s_col1:
                st.markdown("<p style='color: #94a3b8; font-size: 0.9rem;'>Top 5 Most Probable Exact Scorelines (Bivariate Poisson Distribution):</p>", unsafe_allow_html=True)
                score_cards = []
                for score, p in res['top_scorelines']:
                    st.markdown(f"""
                    <div style="background: #111e33; padding: 0.6rem 1rem; border-radius: 0.6rem; margin-bottom: 0.4rem; display: flex; justify-content: space-between; border: 1px solid #1e293b;">
                        <span style="font-weight: 700; color: #f1f5f9;">{score}</span>
                        <span style="color: #4ade80; font-weight: 600;">{p:.1%}</span>
                    </div>
                    """, unsafe_allow_html=True)

            with s_col2:
                st.markdown("<p style='color: #94a3b8; font-size: 0.9rem;'>Match Metrics & Betting Indicators:</p>", unsafe_allow_html=True)
                st.metric("Over 2.5 Goals Probability", f"{res['over_2_5_prob']:.1%}")
                st.metric("Both Teams to Score (BTTS)", f"{res['btts_prob']:.1%}")

with tab2:
    st.subheader("All-Time Head-to-Head Encounter Records")
    col_t1, col_t2 = st.columns(2)
    t1 = col_t1.selectbox("First Team", teams, index=teams.index('Brazil') if 'Brazil' in teams else 0, key="h2h_t1")
    t2 = col_t2.selectbox("Second Team", teams, index=teams.index('Argentina') if 'Argentina' in teams else 1, key="h2h_t2")

    if t1 == t2:
        st.warning("Please choose two different nations.")
    else:
        h2h_data = get_h2h_records(t1, t2)
        total_m = h2h_data['total']

        if total_m == 0:
            st.info(f"No official historical matches on record between {t1} and {t2}.")
        else:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Matches", total_m)
            m2.metric(f"{t1} Wins", h2h_data['a_wins'])
            m3.metric("Draws", h2h_data['draws'])
            m4.metric(f"{t2} Wins", h2h_data['b_wins'])
            m5.metric("Goal Tally", f"{h2h_data['a_goals']} - {h2h_data['b_goals']}")

            st.markdown("#### Recent Encounters")
            df_history = pd.DataFrame(h2h_data['matches'])
            st.dataframe(df_history, use_container_width=True)

with tab3:
    st.subheader("Comparative Team Intelligence")
    comp_col1, comp_col2 = st.columns(2)
    sel_a = comp_col1.selectbox("Select Nation A", teams, index=teams.index('England') if 'England' in teams else 0, key="radar_a")
    sel_b = comp_col2.selectbox("Select Nation B", teams, index=teams.index('Germany') if 'Germany' in teams else 1, key="radar_b")

    sa = team_stats.get(sel_a, {})
    sb = team_stats.get(sel_b, {})

    comp_table = [
        {"Metric": "Official FIFA World Rank", sel_a: f"#{sa.get('fifa_rank', 'N/A')}", sel_b: f"#{sb.get('fifa_rank', 'N/A')}"},
        {"Metric": "Official FIFA Points", sel_a: str(sa.get('fifa_points', 'N/A')), sel_b: str(sb.get('fifa_points', 'N/A'))},
        {"Metric": "Chronological Elo Rating", sel_a: str(sa.get('elo', 1500)), sel_b: str(sb.get('elo', 1500))},
        {"Metric": "Confederation", sel_a: str(sa.get('confederation', 'Other')), sel_b: str(sb.get('confederation', 'Other'))},
        {"Metric": "Recent Form (PPG)", sel_a: f"{sa.get('form_ppg', 0.0)} / 3.0", sel_b: f"{sb.get('form_ppg', 0.0)} / 3.0"},
        {"Metric": "Recent Goals Scored / Match", sel_a: str(sa.get('form_gf', 0.0)), sel_b: str(sb.get('form_gf', 0.0))},
        {"Metric": "Recent Goals Conceded / Match", sel_a: str(sa.get('form_ga', 0.0)), sel_b: str(sb.get('form_ga', 0.0))},
        {"Metric": "Clean Sheet % (Last 10)", sel_a: f"{sa.get('clean_sheet_pct', 0.0)}%", sel_b: f"{sb.get('clean_sheet_pct', 0.0)}%"},
        {"Metric": "Penalty Shootout Record", sel_a: f"{sa.get('shootout_wins', 0)}/{sa.get('shootout_total', 0)} ({sa.get('shootout_win_rate', 50)}%)", sel_b: f"{sb.get('shootout_wins', 0)}/{sb.get('shootout_total', 0)} ({sb.get('shootout_win_rate', 50)}%)"}
    ]

    st.table(pd.DataFrame(comp_table).set_index("Metric"))

with tab4:
    st.subheader("FIFA World Cup 2026 Simulation Arena")
    st.markdown("""
    Explore high-profile simulated clashes under 2026 World Cup tournament conditions (neutral venue, maximum tournament importance weight `60`, and knockout tiebreak modeling).
    """)

    preset_matches = [
        ('Argentina', 'France'),
        ('Brazil', 'England'),
        ('Spain', 'Germany'),
        ('Portugal', 'Netherlands'),
        ('United States', 'Mexico'),
        ('Japan', 'South Korea')
    ]

    selected_preset = st.selectbox(
        "Choose a Featured Clash",
        [f"{p[0]} vs {p[1]}" for p in preset_matches]
    )

    p_home, p_away = selected_preset.split(" vs ")

    if st.button("Run World Cup Simulation", type="primary"):
        sim_res = predict_match(p_home, p_away, neutral=True, tournament='FIFA World Cup', is_knockout=True, artifact=artifact)
        p_probs = sim_res['probabilities']
        p_adv = sim_res['knockout_advance']

        st.markdown(f"### 🏆 Simulated Outcome: {p_home} vs {p_away}")

        sim1, sim2 = st.columns(2)
        sim1.metric(f"{p_home} Win (90 min)", f"{p_probs['Home win']:.1%}", f"xG: {sim_res['expected_goals'][p_home]}")
        sim2.metric(f"{p_away} Win (90 min)", f"{p_probs['Away win']:.1%}", f"xG: {sim_res['expected_goals'][p_away]}")

        st.info(f"**Overall Knockout Advance Probability:** {p_home} **{p_adv[p_home]:.1%}** — {p_away} **{p_adv[p_away]:.1%}**")

        st.markdown("#### Top Predicted Scorelines:")
        for sc, prob_val in sim_res['top_scorelines'][:3]:
            st.write(f"- **{sc}** ({prob_val:.1%})")
