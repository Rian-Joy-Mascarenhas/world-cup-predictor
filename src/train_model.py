from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from .data_preprocessing import load_matches, build_features

MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'model.joblib'

def train():
    matches = load_matches()
    features,labels,dates,ratings = build_features(matches)
    x,y = np.asarray(features),np.asarray(labels)
    train_mask = dates < '2022-01-01'
    test_mask = (dates >= '2022-01-01') & (dates < '2026-01-01')
    def model():
        return make_pipeline(StandardScaler(),LogisticRegression(max_iter=500))
    validation = model().fit(x[train_mask],y[train_mask])
    probs = validation.predict_proba(x[test_mask])
    artifact = {'model':model().fit(x,y),'ratings':ratings,
                'metrics':{'accuracy':float(accuracy_score(y[test_mask],validation.predict(x[test_mask]))),
                           'log_loss':float(log_loss(y[test_mask],probs,labels=[0,1,2])),
                           'matches':int(test_mask.sum())},
                'teams':sorted(set(matches.home_team)|set(matches.away_team)),
                'last_result':str(matches.date.max().date())}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(artifact,MODEL_PATH)
    return artifact

if __name__ == '__main__':
    print(train()['metrics'])
