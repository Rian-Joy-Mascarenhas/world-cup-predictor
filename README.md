# Matchday Pro — FIFA World Cup AI Match Predictor

An enterprise-grade international football intelligence and match outcome prediction engine. Powered by a calibrated ensemble of **Multinomial Logistic Regression** and **Histogram Gradient Boosting**, historical **FIFA World Rankings (1992–2024)**, **chronological Elo ratings**, **bivariate Poisson expected goals (xG)**, and **penalty shootout modeling** for knockout stages.

---

## Key Features

- **Multi-Source Dataset Architecture:**
  - **49,200+ Scored International Matches (1872–present):** From early international football to modern World Cup qualifiers and finals.
  - **Official Historical FIFA World Rankings (1992–2024):** 67,800+ chronological ranking points and ranks indexed for leak-free pre-match lookup.
  - **Historical Penalty Shootouts (675 shootouts):** Dedicated tiebreak modeling for tournament knockout stages.
  - **Tournament Importance Weights:** Match context weighted according to official FIFA criteria (World Cup final tournament = 60, Continental finals = 50, Qualifiers = 40, Nations League = 30, Friendlies = 10).
- **Advanced Machine Learning Engine:**
  - **Soft-Voting Ensemble:** Combines a standardized linear multinomial model with a non-linear tree-based `HistGradientBoostingClassifier`.
  - **Expected Goals (xG) & Exact Scoreline Matrix:** Bivariate Poisson distribution predicts the probability of every exact scoreline (e.g., 2-1, 1-1, 1-0) and calculates Over/Under 2.5 goals and Both Teams to Score (BTTS).
  - **Knockout Stage Simulator:** Seamlessly simulates extra time and penalty shootouts if regulation ends in a draw, estimating true round-advancement probability.
  - **Head-to-Head Analytics:** Computes all-time historical records, goal tallies, and recent match histories between any two nations.
- **Interactive Streamlit Web Dashboard:**
  - **🎯 Match Simulator:** Configure home/away, tournament category, neutral venue, and knockout mode.
  - **⚔️ Head-to-Head Records:** Deep-dive historical encounters, win-loss-draw splits, and previous scorelines.
  - **📊 Team Intelligence Radar:** Side-by-side comparison of FIFA rank, points, Elo, 10-match rolling form, offensive/defensive rates, clean sheet %, and shootout conversion.
  - **🏆 World Cup 2026 Arena:** Instant simulation of marquee World Cup tournament clashes.

---

## Model Evaluation & Performance

The model is evaluated using a strict temporal split to ensure out-of-time real-world generalization:

| Metric | Holdout Evaluation |
| --- | --- |
| **Validation Training Period** | `1990-01-01` to `2021-12-31` (27,687 scored matches) |
| **Held-Out Test Period** | `2022-01-01` to `2025-12-31` (**4,249 scored matches**) |
| **Multiclass Accuracy** | **60.04%** |
| **Multiclass Log Loss** | **0.8765** (improved with ensemble & FIFA rankings) |
| **Published Production Model** | Refit on all scored matches through `2026-03-31` (32,101 matches, 323 active national teams) |

---

## Project Structure

```text
world-cup-predictor/
├── app/
│   └── streamlit_app.py        # Streamlit 4-tab interactive prediction dashboard
├── data/
│   ├── matches.csv             # Historical international match results (1872–present)
│   ├── fifa_ranking_historical.csv # Historical FIFA rankings (1992–2024, 67,800+ entries)
│   ├── fifa_ranking.csv        # Country codes, confederations, and ranking metadata
│   └── archive/
│       ├── shootouts.csv       # Historical international penalty shootout records
│       ├── goalscorers.csv     # Individual goalscorer records (47,600+ goals)
│       └── former_names.csv    # Historical national team alias mappings
├── src/
│   ├── data_preprocessing.py   # Dataset ingestion, FIFA lookup, H2H extraction
│   ├── feature_engineering.py  # Elo system, rolling 10-match form, tournament weights
│   ├── train_model.py          # Soft-voting ensemble training, evaluation & artifact cache
│   └── predict.py              # 3-way probabilities, Poisson xG, exact scorelines & knockout simulation
├── models/                     # Auto-generated (gitignored)
│   └── model.joblib            # Serialized ensemble, ratings, FIFA lookup, and team stats
├── requirements.txt            # Project dependencies
└── README.md
```

---

## Quick Start

### 1. Prerequisites
- Python 3.12 or 3.13
- `pip`

### 2. Installation
Clone the repository:
```bash
git clone https://github.com/Rian-Joy-Mascarenhas/world-cup-predictor.git
cd world-cup-predictor
```

Create and activate a virtual environment:

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

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run app/streamlit_app.py
```
*On first launch, the app automatically loads the datasets, fits the ensemble, caches `models/model.joblib`, and opens the dashboard.*

### 4. Retrain Model (Optional)
To retrain and update the cached model artifact manually:
```bash
python -m src.train_model
```

---

## Methodological Details

1. **Chronological Elo System:** Teams start at `1500.0`. A goal-margin multiplier scales ratings updates dynamically ($1.0$ for 1 goal, $1.5$ for 2 goals, $1.75+$ for 3+ goals) with a `+65` home-ground advantage when fixtures are non-neutral.
2. **Official FIFA Rankings Integration:** Pre-match ranking points and logarithmic rank differentials are looked up chronologically using binary search (`bisect_right`), preventing any forward-looking data leakage.
3. **10-Match Rolling Form:** Tracks points per game (PPG), goals scored per match, goals conceded per match, and clean sheet percentage.
4. **Bivariate Poisson Expected Goals:** Estimates team offensive rate ($\lambda_{\text{home}}$) and defensive vulnerability ($\lambda_{\text{away}}$) to generate the full exact scoreline probability distribution grid.
5. **Knockout Stage Resolution:** Incorporates historical penalty shootout conversion records to simulate extra time and penalty shootouts if a knockout fixture ends in a draw.

---

## License & Disclaimer
This project is open-source for educational and research purposes. Predictions are probabilistic estimates based on historical data and do not account for real-time squad injuries or last-minute tactical changes. Not intended for commercial wagering.
