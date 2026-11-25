import bisect
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd

TEAM_ALIASES = {
    'South Korea': 'Korea Republic',
    'United States': 'USA',
    'Ivory Coast': "Côte d'Ivoire",
    'Iran': 'IR Iran',
    'DR Congo': 'Congo DR',
    'North Korea': 'Korea DPR',
    'Cape Verde': 'Cabo Verde',
    'Saint Vincent and the Grenadines': 'St. Vincent and the Grenadines',
    'Saint Lucia': 'St. Lucia',
    'Saint Kitts and Nevis': 'St. Kitts and Nevis',
    'Kyrgyzstan': 'Kyrgyz Republic',
    'Taiwan': 'Chinese Taipei',
    'Czech Republic': 'Czechia',
    'Turkey': 'Türkiye',
    'Eswatini': 'Swaziland',
    'North Macedonia': 'FYR Macedonia',
    'Brunei': 'Brunei Darussalam',
    'Curacao': 'Curaçao',
    'Syria': 'Syrian Arab Republic'
}

TOURNAMENT_WEIGHTS = {
    'FIFA World Cup': 60,
    'FIFA World Cup qualification': 40,
    'UEFA Euro': 50,
    'UEFA Euro qualification': 40,
    'Copa América': 50,
    'African Cup of Nations': 50,
    'AFC Asian Cup': 50,
    'CONCACAF Gold Cup': 50,
    'UEFA Nations League': 30,
    'Friendly': 10
}

class FIFARankings:
    """Historical FIFA rankings with chronological binary search lookup."""
    def __init__(self, historical_path: Path, metadata_path: Path = None):
        self.team_data = {}
        self.confederations = {}
        self._load(historical_path, metadata_path)

    def _load(self, historical_path: Path, metadata_path: Path):
        df = pd.read_csv(historical_path, parse_dates=['date']).dropna(subset=['team', 'date', 'total_points'])
        df = df.sort_values(['team', 'date'], kind='stable')
        df['rank'] = df.groupby('date')['total_points'].rank(ascending=False, method='min')

        for team, group in df.groupby('team'):
            dates = group['date'].values
            points = group['total_points'].values
            ranks = group['rank'].values
            self.team_data[team] = (dates, points, ranks)

        if metadata_path and metadata_path.exists():
            meta = pd.read_csv(metadata_path)
            if 'country_full' in meta.columns and 'confederation' in meta.columns:
                self.confederations = meta.dropna(subset=['country_full', 'confederation']).drop_duplicates('country_full').set_index('country_full')['confederation'].to_dict()

    def get_fifa(self, team: str, date: pd.Timestamp):
        t = TEAM_ALIASES.get(team, team)
        if t not in self.team_data:
            return 1200.0, 100.0
        dates, pts, rnks = self.team_data[t]
        target = np.datetime64(date)
        idx = bisect.bisect_right(dates, target) - 1
        if idx < 0:
            return 1200.0, 100.0
        return float(pts[idx]), float(rnks[idx])

    def get_confederation(self, team: str) -> str:
        t = TEAM_ALIASES.get(team, team)
        return self.confederations.get(t, self.confederations.get(team, 'Other'))


def default_elo():
    return 1500.0

def default_form():
    return deque(maxlen=10)

def default_h2h():
    return [0, 0, 0]

class ShootoutTracker:
    """Historical penalty shootouts analyzer for knockout tiebreaks."""
    def __init__(self, shootouts_path: Path = None):
        self.wins = defaultdict(int)
        self.total = defaultdict(int)
        if shootouts_path and shootouts_path.exists():
            df = pd.read_csv(shootouts_path)
            for row in df.itertuples(index=False):
                h, a, w = row.home_team, row.away_team, row.winner
                self.total[h] += 1
                self.total[a] += 1
                if pd.notna(w):
                    self.wins[w] += 1

    def stats(self, team: str):
        tot = self.total[team]
        w = self.wins[team]
        rate = (w / tot) if tot >= 2 else 0.50
        return {'wins': w, 'total': tot, 'win_rate': float(rate)}


class Ratings:
    """Dynamic pre-match team strength tracker incorporating Elo, form, and H2H."""
    def __init__(self):
        self.elo = defaultdict(default_elo)
        self.form = defaultdict(default_form)
        self.h2h = defaultdict(default_h2h) # [team1_wins, draws, team2_wins] for (min(t1,t2), max(t1,t2))

    def _team_form(self, team: str):
        g = self.form[team]
        if not g:
            return 1.3, 1.2, 1.2, 0.3
        ppg = np.mean([3 if res == 1.0 else 1 if res == 0.5 else 0 for res, hs, as_ in g])
        gf = np.mean([hs for res, hs, as_ in g])
        ga = np.mean([as_ for res, hs, as_ in g])
        cs = np.mean([1 if as_ == 0 else 0 for res, hs, as_ in g])
        return ppg, gf, ga, cs

    def features(self, home: str, away: str, neutral: bool, fifa_rankings: FIFARankings, date: pd.Timestamp, tournament: str = 'FIFA World Cup'):
        e_diff = self.elo[home] - self.elo[away] + (0 if neutral else 65.0)

        h_pts, h_rnk = fifa_rankings.get_fifa(home, date) if fifa_rankings else (1200.0, 100.0)
        a_pts, a_rnk = fifa_rankings.get_fifa(away, date) if fifa_rankings else (1200.0, 100.0)

        fifa_pts_diff = h_pts - a_pts
        fifa_rnk_diff = np.log(a_rnk + 1.0) - np.log(h_rnk + 1.0)

        hf = self._team_form(home)
        af = self._team_form(away)

        pair_key = (min(home, away), max(home, away))
        prior_h2h = self.h2h[pair_key]
        tot_h2h = sum(prior_h2h)
        if tot_h2h > 0:
            h2h_win_rate = (prior_h2h[0] / tot_h2h) if home < away else (prior_h2h[2] / tot_h2h)
        else:
            h2h_win_rate = 0.40

        t_weight = TOURNAMENT_WEIGHTS.get(tournament, 25)

        return [
            float(e_diff),
            float(fifa_pts_diff),
            float(fifa_rnk_diff),
            float(hf[0] - af[0]), # PPG diff
            float(hf[1] - af[1]), # GF diff
            float(hf[2] - af[2]), # GA diff
            float(hf[3] - af[3]), # Clean sheet % diff
            int(not neutral),
            float(t_weight),
            float(h2h_win_rate)
        ]

    def update(self, home: str, away: str, home_score: float, away_score: float, neutral: bool):
        result = 1.0 if home_score > away_score else 0.0 if home_score < away_score else 0.5
        expected = 1.0 / (1.0 + 10.0 ** ((self.elo[away] - self.elo[home] - (0 if neutral else 65.0)) / 400.0))

        margin = abs(home_score - away_score)
        margin_mult = 1.0 if margin <= 1 else 1.5 if margin == 2 else (1.75 + (margin - 3) / 8.0)
        delta = 24.0 * margin_mult * (result - expected)

        self.elo[home] += delta
        self.elo[away] -= delta

        self.form[home].append((result, home_score, away_score))
        self.form[away].append((1.0 - result, away_score, home_score))

        pair_key = (min(home, away), max(home, away))
        if home_score > away_score:
            if home < away:
                self.h2h[pair_key][0] += 1
            else:
                self.h2h[pair_key][2] += 1
        elif home_score == away_score:
            self.h2h[pair_key][1] += 1
        else:
            if home < away:
                self.h2h[pair_key][2] += 1
            else:
                self.h2h[pair_key][0] += 1
