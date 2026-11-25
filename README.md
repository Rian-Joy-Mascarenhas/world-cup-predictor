# Matchday — World Cup Match Predictor

An interactive Streamlit web application that estimates regulation-time outcome probabilities (**Home win**, **Draw**, **Away win**) for international football matches using chronological Elo ratings and recent team form.

## Overview

- **Leak-free chronological features:** Team Elo ratings and rolling 10-match form statistics are computed strictly from matches played prior to each fixture date; matches played on the same calendar date cannot influence one another's pre-match features.
- **Calibrated three-way probabilities:** A standardized multinomial logistic regression model outputs probabilities for all three full-time outcomes rather than just a binary winner.
- **Self-contained startup:** On first launch, the app trains the model from `data/matches.csv`, logs holdout validation metrics, and caches the artifact at `models/model.joblib` for fast subsequent runs.
- **Interactive dashboard:** Select any two teams from 323 international sides, toggle neutral-venue status, and inspect predicted outcome probabilities alongside held-out evaluation metrics in the sidebar.

## Evaluation Summary

Model validation uses a temporal split rather than random shuffling so metrics reflect out-of-time generalization:

| Metric | Value |
| --- | --- |
| Training window (validation) | `1990-01-01` to `2021-12-31` (27,687 scored matches) |
| Holdout window | `2022-01-01` to `2025-12-31` (4,249 scored matches) |
| Holdout accuracy | **60.3%** (`0.6032`) |
| Holdout multiclass log loss | **0.881** (`0.8806`) |
| Published refit dataset | All scored matches from 1990 through `2026-03-31` (32,101 matches, 323 teams) |

> **Note:** Holdout metrics cover all scored international fixtures in 2022–2025 (friendlies, qualifiers, continental tournaments, and World Cup matches), not World Cup finals exclusively.

## Getting Started

### Prerequisites

- Python 3.12 or 3.13
- `pip`

### Installation

1. Clone the repository and enter the project directory:
   ```bash
   git clone https://github.com/Rian-Joy-Mascarenhas/world-cup-predictor.git
   cd world-cup-predictor
   ```

2. Create and activate a virtual environment:

   **Windows (PowerShell):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   **macOS / Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Start the Streamlit dashboard:
```bash
streamlit run app/streamlit_app.py
```

On the first run, the app automatically trains the model and saves `models/model.joblib` (ignored by Git).

To retrain the cached artifact manually after updating `data/matches.csv`:
```bash
python -m src.train_model
```

## Modeling Approach

1. **Data Filtering (`src/data_preprocessing.py`):**
   Loads `data/matches.csv` (49,287 historical rows dating back to 1872), drops unplayed/unscored fixtures, and retains scored international matches from `1990` onward (32,101 matches).
2. **Feature Engineering (`src/feature_engineering.py`):**
   Generates five pre-match features for every fixture:
   - **Elo difference:** `home_elo - away_elo` (teams initialized at `1500.0`, `K = 24`, with a `+65` Elo home-field offset applied when `neutral` is `False`).
   - **Rolling 10-match points-per-game difference:** Win = `1.0`, Draw = `0.5`, Loss = `0.0` (defaults to `0.5` prior to a team's first recorded match).
   - **Rolling 10-match goals scored difference** (defaults to `1.2`).
   - **Rolling 10-match goals conceded difference** (defaults to `1.2`).
   - **Home-venue indicator:** `1` if non-neutral venue, `0` if neutral.
3. **Training & Calibration (`src/train_model.py`):**
   Trains a `scikit-learn` pipeline (`StandardScaler` + `LogisticRegression(max_iter=500)`), computes 2022–2025 holdout accuracy and multiclass log loss, and refits the pipeline on all scored matches (`1990–2026-03-31`) for live inference.

## Project Structure

```text
world-cup-predictor/
├── app/
│   └── streamlit_app.py        # Streamlit UI and prediction dashboard
├── data/
│   ├── matches.csv             # Historical international match results and fixtures
│   └── archive/                # Supplementary datasets (former_names, goalscorers, shootouts)
├── src/
│   ├── data_preprocessing.py   # Match loading and chronological feature construction
│   ├── feature_engineering.py  # Elo rating system and 10-match rolling form tracker
│   ├── train_model.py          # Holdout evaluation, full refit, and joblib serialization
│   └── predict.py              # Artifact loading and 3-way probability inference
├── models/                     # Auto-generated locally (gitignored)
│   └── model.joblib            # Cached trained pipeline, ratings state, and metrics
├── requirements.txt
└── README.md
```

## Limitations & Scope

- **Regulation time only:** Predictions estimate 90-minute match outcomes (`Home win`, `Draw`, `Away win`) and do not model extra time or penalty shootouts.
- **Team-level history only:** Player lineups, injuries, tactical changes, travel fatigue, and FIFA rankings are not included.
- **Name continuity:** Teams that changed official names over time are not currently unified using `data/archive/former_names.csv`.
- **Baseline benchmark:** Changes to ratings, hyper-parameters, or features should be validated on an untouched future time window before claiming accuracy improvements.
- **Educational use:** Intended for data science experimentation and match analysis, not wagering or financial decisions.

## Author

**Rian Mascarenhas** — [GitHub](https://github.com/Rian-Joy-Mascarenhas)
