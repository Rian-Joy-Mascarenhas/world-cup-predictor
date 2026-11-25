import joblib
from .train_model import MODEL_PATH,train

def load_artifact():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else train()

def predict(home,away,neutral=True,artifact=None):
    artifact = artifact if artifact is not None else load_artifact()
    if home == away or home not in artifact['teams'] or away not in artifact['teams']:
        raise ValueError('Choose two different teams.')
    probs = artifact['model'].predict_proba([artifact['ratings'].features(home,away,neutral)])[0]
    return dict(zip(('Home win','Draw','Away win'),map(float,probs)))
