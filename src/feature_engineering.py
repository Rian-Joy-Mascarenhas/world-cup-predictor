from collections import defaultdict, deque
import numpy as np

def initial_elo():
    return 1500.

def recent_games():
    return deque(maxlen=10)

class Ratings:
    def __init__(self):
        self.elo = defaultdict(initial_elo)
        self.form = defaultdict(recent_games)

    def features(self, home, away, neutral):
        def form(team):
            games = self.form[team]
            return (np.mean([g[0] for g in games]), np.mean([g[1] for g in games]), np.mean([g[2] for g in games])) if games else (.5, 1.2, 1.2)
        a, b = form(home), form(away)
        return [self.elo[home]-self.elo[away], *(x-y for x,y in zip(a,b)), int(not neutral)]

    def update(self, home, away, hs, ass, neutral):
        result = 1. if hs > ass else 0. if hs < ass else .5
        expected = 1/(1+10**((self.elo[away]-self.elo[home]-(0 if neutral else 65))/400))
        delta = 24*(result-expected)
        self.elo[home] += delta
        self.elo[away] -= delta
        self.form[home].append((result,hs,ass))
        self.form[away].append((1-result,ass,hs))
