import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from .train_model import MODEL_PATH, train
from .data_preprocessing import get_h2h_records

def load_artifact():
    """Loads pre-trained model artifact or initiates training if absent."""
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else train()

def predict_match(home: str, away: str, neutral: bool = True, tournament: str = 'FIFA World Cup', is_knockout: bool = False, artifact: dict = None):
    """Predicts 3-way regulation outcome, expected goals, exact scorelines, and knockout advancement."""
    artifact = artifact if artifact is not None else load_artifact()

    if home == away:
        raise ValueError('Home and Away teams must be distinct.')
    if home not in artifact['teams'] or away not in artifact['teams']:
        raise ValueError(f"One or both teams not recognized. Available teams: {len(artifact['teams'])}")

    ratings = artifact['ratings']
    fifa = artifact['fifa']
    shootouts = artifact['shootouts']
    model = artifact['model']
    now = pd.to_datetime(artifact['last_result'])

    feats = ratings.features(home, away, neutral, fifa, now, tournament)
    probs = model.predict_proba([feats])[0]

    outcome_probs = {
        'Home win': float(probs[0]),
        'Draw': float(probs[1]),
        'Away win': float(probs[2])
    }

    # Expected Goals (xG / Lambda) estimation via calibrated Poisson
    elo_diff = (ratings.elo[home] - ratings.elo[away] + (0.0 if neutral else 65.0)) / 400.0
    home_form = ratings._team_form(home)
    away_form = ratings._team_form(away)

    form_offset = 0.10 * (home_form[1] - away_form[2]) # GF_home - GA_away
    lambda_home = float(max(0.20, min(4.50, 1.35 * np.exp(0.35 * elo_diff + form_offset))))
    lambda_away = float(max(0.20, min(4.50, 1.35 * np.exp(-0.35 * elo_diff - form_offset))))

    # Exact scoreline probability grid (0 to 5 goals each)
    score_probs = {}
    over_25 = 0.0
    btts = 0.0

    for h_goals in range(6):
        p_h = poisson.pmf(h_goals, lambda_home)
        for a_goals in range(6):
            p_a = poisson.pmf(a_goals, lambda_away)
            prob_score = p_h * p_a
            score_probs[f"{h_goals} - {a_goals}"] = float(prob_score)
            if h_goals + a_goals > 2.5:
                over_25 += prob_score
            if h_goals > 0 and a_goals > 0:
                btts += prob_score

    # Sort top 5 exact scorelines
    top_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:5]

    # Knockout stage advance probability (Extra time & Penalty Shootout simulation)
    knockout_advance = None
    if is_knockout:
        h_sh = shootouts.stats(home)['win_rate']
        a_sh = shootouts.stats(away)['win_rate']

        # Shootout probability blends historical shootout rate with Elo quality
        elo_prob = 1.0 / (1.0 + 10.0 ** ((ratings.elo[away] - ratings.elo[home]) / 400.0))
        h_adv_given_draw = 0.60 * ((h_sh) / (h_sh + a_sh)) + 0.40 * elo_prob

        h_total_adv = outcome_probs['Home win'] + outcome_probs['Draw'] * h_adv_given_draw
        a_total_adv = 1.0 - h_total_adv

        knockout_advance = {
            home: float(h_total_adv),
            away: float(a_total_adv),
            'draw_shootout_split': {home: float(h_adv_given_draw), away: float(1.0 - h_adv_given_draw)}
        }

    return {
        'probabilities': outcome_probs,
        'winner': max(outcome_probs, key=outcome_probs.get),
        'expected_goals': {home: round(lambda_home, 2), away: round(lambda_away, 2)},
        'over_2_5_prob': float(over_25),
        'btts_prob': float(btts),
        'top_scorelines': top_scores,
        'knockout_advance': knockout_advance,
        'h2h': get_h2h_records(home, away)
    }

# Backward compatibility alias
def predict(home: str, away: str, neutral: bool = True, artifact: dict = None):
    res = predict_match(home, away, neutral=neutral, artifact=artifact)
    return res['probabilities']
