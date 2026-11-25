from pathlib import Path
import pandas as pd
from .feature_engineering import FIFARankings, ShootoutTracker, Ratings

BASE_DIR = Path(__file__).resolve().parents[1]
MATCHES_DATA = BASE_DIR / 'data' / 'matches.csv'
FIFA_HISTORICAL = BASE_DIR / 'data' / 'fifa_ranking_historical.csv'
FIFA_METADATA = BASE_DIR / 'data' / 'fifa_ranking.csv'
SHOOTOUTS_DATA = BASE_DIR / 'data' / 'archive' / 'shootouts.csv'

def load_matches():
    """Loads scored international matches from 1990 onward."""
    df = pd.read_csv(MATCHES_DATA, parse_dates=['date']).dropna(subset=['date', 'home_score', 'away_score'])
    df = df[df.date.dt.year >= 1990].copy()
    df['neutral'] = df.neutral.astype(str).str.lower().eq('true')
    return df.sort_values('date', kind='stable')

def load_all_matches():
    """Loads all historical match records from 1872 to present for H2H analytics."""
    df = pd.read_csv(MATCHES_DATA, parse_dates=['date'])
    df['neutral'] = df.neutral.astype(str).str.lower().eq('true')
    return df.sort_values('date', ascending=False)

def load_fifa():
    """Loads and indexes official historical FIFA rankings."""
    return FIFARankings(FIFA_HISTORICAL, FIFA_METADATA)

def load_shootouts():
    """Loads historical international penalty shootout records."""
    return ShootoutTracker(SHOOTOUTS_DATA)

def build_features(matches: pd.DataFrame, fifa_rankings: FIFARankings):
    """Chronologically builds features and updates ratings without future leakage."""
    ratings = Ratings()
    x, y, dates = [], [], []

    for date, group in matches.groupby('date', sort=True):
        updates = []
        for row in group.itertuples(index=False):
            feats = ratings.features(
                row.home_team,
                row.away_team,
                row.neutral,
                fifa_rankings,
                date,
                getattr(row, 'tournament', 'FIFA World Cup')
            )
            outcome = 0 if row.home_score > row.away_score else 2 if row.home_score < row.away_score else 1
            x.append(feats)
            y.append(outcome)
            dates.append(date)
            updates.append((row.home_team, row.away_team, row.home_score, row.away_score, row.neutral))

        for update in updates:
            ratings.update(*update)

    return x, y, pd.DatetimeIndex(dates), ratings

def get_h2h_records(team_a: str, team_b: str):
    """Retrieves all historical head-to-head encounters between two national teams."""
    df = load_all_matches().dropna(subset=['home_score', 'away_score'])
    cond = ((df['home_team'] == team_a) & (df['away_team'] == team_b)) | \
           ((df['home_team'] == team_b) & (df['away_team'] == team_a))
    h2h_df = df[cond].copy()

    total = len(h2h_df)
    if total == 0:
        return {
            'total': 0, 'a_wins': 0, 'draws': 0, 'b_wins': 0,
            'a_goals': 0, 'b_goals': 0, 'matches': []
        }

    a_wins = 0
    b_wins = 0
    draws = 0
    a_goals = 0
    b_goals = 0
    match_list = []

    for row in h2h_df.itertuples(index=False):
        if row.home_team == team_a:
            hg, ag = int(row.home_score), int(row.away_score)
            a_goals += hg
            b_goals += ag
            if hg > ag: a_wins += 1
            elif hg < ag: b_wins += 1
            else: draws += 1
        else:
            hg, ag = int(row.away_score), int(row.home_score) # team_a is away
            a_goals += hg
            b_goals += ag
            if hg > ag: a_wins += 1
            elif hg < ag: b_wins += 1
            else: draws += 1

        match_list.append({
            'date': str(row.date.date()),
            'home': row.home_team,
            'away': row.away_team,
            'score': f"{int(row.home_score)} - {int(row.away_score)}",
            'tournament': row.tournament,
            'neutral': row.neutral
        })

    return {
        'total': total,
        'a_wins': a_wins,
        'draws': draws,
        'b_wins': b_wins,
        'a_goals': a_goals,
        'b_goals': b_goals,
        'matches': match_list[:15] # Most recent 15 meetings
    }
