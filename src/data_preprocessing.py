from pathlib import Path
import pandas as pd
from .feature_engineering import Ratings

DATA = Path(__file__).resolve().parents[1] / 'data' / 'matches.csv'

def load_matches():
    df = pd.read_csv(DATA, parse_dates=['date']).dropna(subset=['date','home_score','away_score'])
    df = df[df.date.dt.year >= 1990].copy()
    df['neutral'] = df.neutral.astype(str).str.lower().eq('true')
    return df.sort_values('date',kind='stable')

def build_features(matches):
    ratings = Ratings()
    x,y,dates = [],[],[]
    for date,group in matches.groupby('date',sort=True):
        updates = []
        for row in group.itertuples(index=False):
            x.append(ratings.features(row.home_team,row.away_team,row.neutral))
            y.append(0 if row.home_score > row.away_score else 2 if row.home_score < row.away_score else 1)
            dates.append(date)
            updates.append((row.home_team,row.away_team,row.home_score,row.away_score,row.neutral))
        for update in updates:
            ratings.update(*update)
    return x,y,pd.DatetimeIndex(dates),ratings
