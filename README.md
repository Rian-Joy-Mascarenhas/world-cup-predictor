# Matchday: World Cup Match Predictor

An interactive Streamlit dashboard for international football match outcome probabilities.

## Run

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The first run trains the model and caches an ignored `models/model.joblib` artifact. To retrain after changing `data/matches.csv`, remove that file and run `python -m src.train_model`.

## Method

Historical scored international matches from 1990 onward generate pre-match Elo differences, recent ten-match form, goals scored and conceded, and venue status. Ratings update after each date, so simultaneous fixtures cannot influence one another's features. A scaled multinomial logistic regression estimates home win, draw and away win probabilities at full time.

Training before 2022 is evaluated on 2022–2025 matches using accuracy and multiclass log loss. The published model is then refit on all scored matches. The dashboard reports the holdout metrics and most recent scored result date. Fixtures with missing scores are excluded.

This evaluates international matches generally, rather than claiming World Cup-only accuracy. Predictions omit squads, injuries, extra time and penalties. Historical team-name changes may split a country's record. Ratings are a simple baseline; performance gains need comparison on an untouched later period before being claimed.
