from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, log_loss
from .data_preprocessing import load_matches, load_fifa, load_shootouts, build_features

MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'model.joblib'

def build_ensemble():
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    hgb = HistGradientBoostingClassifier(random_state=42, max_iter=200, min_samples_leaf=30)
    return VotingClassifier(
        estimators=[('lr', lr), ('hgb', hgb)],
        voting='soft',
        weights=[1.0, 1.0]
    )

def train():
    print("Loading matches, FIFA rankings, and shootout data...")
    matches = load_matches()
    fifa = load_fifa()
    shootouts = load_shootouts()

    print("Building chronological feature matrix...")
    features, labels, dates, ratings = build_features(matches, fifa)
    x, y = np.asarray(features), np.asarray(labels)

    train_mask = dates < '2022-01-01'
    test_mask = (dates >= '2022-01-01') & (dates < '2026-01-01')

    print(f"Validating on holdout (Train: {train_mask.sum()}, Test: {test_mask.sum()})...")
    val_model = build_ensemble()
    val_model.fit(x[train_mask], y[train_mask])

    probs = val_model.predict_proba(x[test_mask])
    preds = val_model.predict(x[test_mask])

    acc = float(accuracy_score(y[test_mask], preds))
    ll = float(log_loss(y[test_mask], probs, labels=[0, 1, 2]))

    print(f"Holdout Validation -> Accuracy: {acc:.2%}, Log loss: {ll:.4f}")

    print("Refitting ensemble on all scored historical matches...")
    full_model = build_ensemble()
    full_model.fit(x, y)

    teams = sorted(set(matches.home_team) | set(matches.away_team))
    latest_date = matches.date.max()

    # Precompute team intelligence stats for instant UI lookup
    team_stats = {}
    for team in teams:
        pts, rnk = fifa.get_fifa(team, latest_date)
        conf = fifa.get_confederation(team)
        sh_stats = shootouts.stats(team)
        elo_val = float(ratings.elo[team])
        ppg, gf, ga, cs = ratings._team_form(team)

        team_stats[team] = {
            'elo': round(elo_val, 1),
            'fifa_rank': int(rnk),
            'fifa_points': round(pts, 1),
            'confederation': conf,
            'form_ppg': round(float(ppg), 2),
            'form_gf': round(float(gf), 2),
            'form_ga': round(float(ga), 2),
            'clean_sheet_pct': round(float(cs * 100), 1),
            'shootout_wins': sh_stats['wins'],
            'shootout_total': sh_stats['total'],
            'shootout_win_rate': round(sh_stats['win_rate'] * 100, 1)
        }

    artifact = {
        'model': full_model,
        'ratings': ratings,
        'fifa': fifa,
        'shootouts': shootouts,
        'metrics': {
            'accuracy': acc,
            'log_loss': ll,
            'matches': int(test_mask.sum())
        },
        'teams': teams,
        'team_stats': team_stats,
        'global_avg_goals': 1.35,
        'last_result': str(latest_date.date())
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"Artifact successfully cached to {MODEL_PATH}")
    return artifact

if __name__ == '__main__':
    train()
