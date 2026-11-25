import sys
from pathlib import Path
import streamlit as st
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.predict import load_artifact,predict

st.set_page_config(page_title='Matchday | World Cup Predictor',page_icon='⚽',layout='wide')
st.markdown("""<style>
.stApp {background:#0b1425;color:#f6f8fc}
[data-testid="stSidebar"] {background:#101e33}
h1,h2,h3,p,label {color:#f6f8fc!important}
.hero {background:linear-gradient(120deg,#17395a,#112e36);border:1px solid #42637a;padding:2rem;border-radius:1rem;margin-bottom:1.5rem}
.eyebrow {color:#a6ebcb!important;font-weight:700;letter-spacing:.12em;text-transform:uppercase}
.muted {color:#c5d2e1!important}
.card {background:#14243b;border:1px solid #52657c;border-radius:.9rem;padding:1.3rem;margin:.7rem 0}
.bar {height:1rem;background:#344760;border-radius:1rem;overflow:hidden}
.fill {height:100%;background:#78e3b3;border-radius:1rem}
div.stButton button[kind="primary"] {background:#79e2b2;color:#09251d;font-weight:700}
</style>""",unsafe_allow_html=True)

@st.cache_resource(show_spinner='Preparing match ratings and model…')
def get_model():
    return load_artifact()

st.markdown('<div class="hero"><p class="eyebrow">Matchday / Football intelligence</p><h1>World Cup match predictor</h1><p class="muted">Explore match outcome probabilities from international football history.</p></div>',unsafe_allow_html=True)
try:
    artifact = get_model()
except Exception as exc:
    st.error(f'Could not prepare the model: {exc}')
    st.stop()
with st.sidebar:
    st.header('Model report')
    m = artifact['metrics']
    st.metric('Held-out accuracy',f"{m['accuracy']:.1%}")
    st.metric('Log loss',f"{m['log_loss']:.3f}")
    st.caption(f"{m['matches']:,} matches · 2022–2025 holdout")
    st.caption(f"Latest scored match: {artifact['last_result']}")
    st.caption('Evaluation includes international matches, not just World Cup finals.')

teams = artifact['teams']
st.subheader('Set up a fixture')
with st.form('fixture'):
    a,b = st.columns(2)
    home = a.selectbox('Home / first team',teams,index=teams.index('Brazil') if 'Brazil' in teams else 0)
    away = b.selectbox('Away / second team',teams,index=teams.index('France') if 'France' in teams else 1)
    neutral = st.checkbox('Neutral venue',value=True)
    submitted = st.form_submit_button('Predict match',type='primary',use_container_width=True)
if submitted:
    try:
        result = predict(home,away,neutral,artifact)
    except ValueError as exc:
        st.warning(str(exc))
    else:
        st.subheader(f'{home} vs {away}')
        winner = max(result,key=result.get)
        st.markdown(f'<div class="card"><p class="eyebrow">Most likely outcome</p><h2>{winner}</h2><p class="muted">{result[winner]:.1%} estimated probability</p></div>',unsafe_allow_html=True)
        for label,value in result.items():
            st.markdown(f'<div class="card"><strong>{label} · {value:.1%}</strong><div class="bar" role="img" aria-label="{label}: {value:.1%}"><div class="fill" style="width:{value*100:.1f}%"></div></div></div>',unsafe_allow_html=True)
        st.caption('Regulation-time outcomes. No extra time, penalties, lineups or injuries are modeled.')
