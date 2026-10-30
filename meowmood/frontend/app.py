"""
frontend/app.py  —  MeowMood: AI Cat Emotion Detector
Run (from meowmood/ directory):  streamlit run frontend/app.py
"""

import os, sys, tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(
    page_title="MeowMood — Cat Emotion AI",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import hashlib, re

MODEL_IMG  = ROOT / "models" / "image_model.keras"
MODEL_AUD  = ROOT / "models" / "audio_model.keras"
LABELS_AUD = ROOT / "models" / "audio_labels.json"
IMAGE_DIR  = ROOT / "data" / "raw" / "images"
AUDIO_DIR  = ROOT / "data" / "raw" / "audio"

from backend.utils import (
    IMAGE_CLASSES, AUDIO_LABEL_MAP,
    preprocess_image_array, preprocess_audio_bytes, load_audio_labels,
)

# ── Auth helpers ───────────────────────────────────────────────────────────────
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

if "users_db" not in st.session_state:
    st.session_state.users_db = {
        "demo": {"password": _hash("demo123"), "name": "Demo User", "email": "demo@meowmood.ai"}
    }
if "logged_in"    not in st.session_state: st.session_state.logged_in    = False
if "username"     not in st.session_state: st.session_state.username     = ""
if "auth_page"    not in st.session_state: st.session_state.auth_page    = "login"
if "show_landing" not in st.session_state: st.session_state.show_landing = True

EMOTION_META = {
    "Angry":     {"emoji":"😾","color":"#FF4D6D","bg":"rgba(255,77,109,0.12)","tip":"Give your cat space and avoid sudden movements."},
    "Disgusted": {"emoji":"🙀","color":"#FF8C42","bg":"rgba(255,140,66,0.12)","tip":"Something nearby may be bothering your cat."},
    "Happy":     {"emoji":"😸","color":"#06D6A0","bg":"rgba(6,214,160,0.12)","tip":"Your cat is content! Great time for play or cuddles."},
    "Normal":    {"emoji":"😺","color":"#4CC9F0","bg":"rgba(76,201,240,0.12)","tip":"Your cat is calm and relaxed. All is well!"},
    "Sad":       {"emoji":"😿","color":"#A78BFA","bg":"rgba(167,139,250,0.12)","tip":"Your cat may need attention or comfort."},
    "Scared":    {"emoji":"🙀","color":"#FBBF24","bg":"rgba(251,191,36,0.12)","tip":"Create a safe, quiet space for your cat."},
    "Surprised": {"emoji":"😲","color":"#38BDF8","bg":"rgba(56,189,248,0.12)","tip":"Something caught your cat off guard."},
}

# ── CSS + Animations ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&display=swap');

*, html, body, [class*="css"] { font-family: 'Poppins', sans-serif !important; box-sizing: border-box; }

/* ═══ BACKGROUND ═══ */
.stApp {
  background: linear-gradient(160deg, #0D0D1A 0%, #111128 35%, #0A1628 65%, #0D0D1A 100%) !important;
  min-height: 100vh;
}
header[data-testid="stHeader"]  { background: transparent !important; }
section[data-testid="stSidebar"]{ display: none !important; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

/* ═══ KEYFRAME ANIMATIONS ═══ */
@keyframes fadeUp {
  from { opacity:0; transform:translateY(30px); }
  to   { opacity:1; transform:translateY(0);    }
}
@keyframes fadeIn {
  from { opacity:0; } to { opacity:1; }
}
@keyframes float {
  0%,100% { transform: translateY(0px);   }
  50%      { transform: translateY(-12px); }
}
@keyframes pulse-ring {
  0%   { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(139,92,246,0.5); }
  70%  { transform: scale(1);    box-shadow: 0 0 0 16px rgba(139,92,246,0);  }
  100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(139,92,246,0);    }
}
@keyframes shimmer {
  0%   { background-position: -200% center; }
  100% { background-position:  200% center; }
}
@keyframes spin-slow {
  from { transform: rotate(0deg);   }
  to   { transform: rotate(360deg); }
}
@keyframes blob {
  0%,100% { border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%; }
  50%      { border-radius: 30% 60% 70% 40% / 50% 60% 30% 60%; }
}
@keyframes floatEmoji {
  0%,100% { transform: translateY(0px) rotate(0deg); }
  33%      { transform: translateY(-18px) rotate(5deg); }
  66%      { transform: translateY(-8px) rotate(-3deg); }
}

/* ═══ LANDING PAGE ═══ */
.landing-wrap {
  position: relative;
  min-height: 90vh;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  text-align: center;
  padding: 3rem 1.5rem 4rem;
  overflow: hidden;
}
.landing-float-emoji {
  position: absolute;
  font-size: 2.2rem;
  pointer-events: none;
  user-select: none;
  animation: floatEmoji 4s ease-in-out infinite;
}
.landing-title {
  font-size: 5.5rem; font-weight: 900; line-height: 1.0;
  margin-bottom: 0.8rem;
  background: linear-gradient(90deg, #F472B6, #A78BFA, #60A5FA, #34D399, #F472B6);
  background-size: 300% auto;
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: fadeUp 0.6s 0.2s ease both, shimmer 5s linear infinite;
}
.landing-sub {
  font-size: 1.25rem; color: #94A3B8; font-weight: 500;
  max-width: 500px; margin: 0 auto 2.5rem; line-height: 1.7;
  animation: fadeUp 0.6s 0.35s ease both;
}
.cat-photos-row {
  display: flex; gap: 1.5rem; justify-content: center; flex-wrap: wrap;
  margin-bottom: 2.5rem;
  animation: fadeUp 0.6s 0.5s ease both;
}
.cat-photo {
  width: 280px; height: 200px;
  border-radius: 20px; object-fit: cover;
  box-shadow: 0 12px 40px rgba(0,0,0,0.5);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.cat-photo:hover {
  transform: translateY(-8px) rotate(0deg) !important;
  box-shadow: 0 24px 60px rgba(139,92,246,0.35);
}
.cat-photo-1 { transform: rotate(-3deg); }
.cat-photo-2 { transform: rotate(0deg); }
.cat-photo-3 { transform: rotate(3deg); }
.landing-features {
  display: flex; gap: 1.5rem; justify-content: center; flex-wrap: wrap;
  margin-bottom: 2.5rem;
  animation: fadeUp 0.6s 0.65s ease both;
}
.landing-feat-item {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 16px; padding: 1rem 1.5rem;
  display: flex; align-items: center; gap: 0.6rem;
  backdrop-filter: blur(10px);
}
.landing-feat-icon { font-size: 1.4rem; }
.landing-feat-title { font-size: 0.95rem; font-weight: 700; color: #F1F5F9; }
.cta-btn {
  display: inline-block;
  background: linear-gradient(135deg, #7C3AED, #4F46E5, #06D6A0);
  background-size: 200% auto;
  color: #fff !important;
  border: none; border-radius: 50px;
  padding: 1rem 3rem; font-size: 1.1rem; font-weight: 700;
  cursor: pointer; letter-spacing: 0.5px;
  box-shadow: 0 12px 40px rgba(124,58,237,0.45);
  transition: transform 0.2s, box-shadow 0.2s;
  animation: fadeUp 0.6s 0.8s ease both, shimmer 4s linear infinite;
  text-decoration: none;
}
.cta-btn:hover {
  transform: translateY(-3px);
  box-shadow: 0 20px 60px rgba(124,58,237,0.6);
}

/* ═══ HERO ═══ */
.hero-wrap {
  position: relative;
  background: linear-gradient(135deg, #13131F 0%, #1A1A35 50%, #0F2040 100%);
  border-radius: 32px;
  padding: 4rem 2rem 3.5rem;
  text-align: center;
  overflow: hidden;
  margin-bottom: 2rem;
  border: 1px solid rgba(139,92,246,0.2);
  animation: fadeIn 0.8s ease both;
}
.blob1 {
  position:absolute; top:-80px; right:-80px;
  width:320px; height:320px;
  background: radial-gradient(circle, rgba(139,92,246,0.3) 0%, transparent 70%);
  animation: blob 8s ease-in-out infinite, spin-slow 20s linear infinite;
  border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%;
}
.blob2 {
  position:absolute; bottom:-100px; left:-60px;
  width:280px; height:280px;
  background: radial-gradient(circle, rgba(6,214,160,0.2) 0%, transparent 70%);
  animation: blob 10s ease-in-out infinite reverse;
  border-radius: 30% 60% 70% 40% / 50% 60% 30% 60%;
}
.blob3 {
  position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
  width:500px; height:200px;
  background: radial-gradient(ellipse, rgba(76,201,240,0.06) 0%, transparent 70%);
  pointer-events:none;
}
.hero-badge {
  display:inline-block;
  background: linear-gradient(90deg, rgba(139,92,246,0.2), rgba(6,214,160,0.2));
  border: 1px solid rgba(139,92,246,0.4);
  color: #C4B5FD;
  font-size: 0.72rem; font-weight:600; letter-spacing:2.5px;
  text-transform:uppercase; padding:0.35rem 1.2rem;
  border-radius:30px; margin-bottom:1.2rem;
  animation: fadeUp 0.6s 0.1s ease both;
}
.hero-title {
  font-size: 4.8rem; font-weight:900; line-height:1.0;
  margin-bottom: 0.8rem;
  background: linear-gradient(90deg, #F472B6, #A78BFA, #60A5FA, #34D399);
  background-size: 200% auto;
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: fadeUp 0.6s 0.2s ease both, shimmer 4s linear infinite;
}
.hero-cat {
  font-size: 5rem; display:block; margin-bottom:0.5rem;
  animation: float 3s ease-in-out infinite;
  -webkit-text-fill-color: initial;
}
.hero-sub {
  font-size: 1.1rem; color: #94A3B8; font-weight:500;
  max-width:520px; margin:0 auto 2.5rem; line-height:1.7;
  animation: fadeUp 0.6s 0.3s ease both;
}
.hero-stats {
  display:flex; justify-content:center; gap:3rem; flex-wrap:wrap;
  animation: fadeUp 0.6s 0.4s ease both;
}
.stat-box {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius:16px; padding:1rem 1.5rem; text-align:center;
  backdrop-filter: blur(10px);
  transition: transform 0.2s, border-color 0.2s;
}
.stat-box:hover { transform:translateY(-4px); border-color:rgba(139,92,246,0.4); }
.stat-num { font-size:2rem; font-weight:800; color:#fff; line-height:1; }
.stat-label { font-size:0.7rem; color:#64748B; font-weight:600;
              letter-spacing:1.5px; text-transform:uppercase; margin-top:0.3rem; }

/* ═══ FEATURE CARDS ═══ */
.feat-row { display:flex; gap:1rem; margin-bottom:2rem; }
.feat-card {
  flex:1;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius:20px; padding:1.6rem 1.2rem; text-align:center;
  backdrop-filter: blur(12px);
  transition: transform 0.25s, border-color 0.25s, box-shadow 0.25s;
  animation: fadeUp 0.6s ease both;
}
.feat-card:hover {
  transform: translateY(-6px);
  border-color: rgba(139,92,246,0.5);
  box-shadow: 0 16px 40px rgba(139,92,246,0.15);
}
.feat-icon-wrap {
  width:56px; height:56px; border-radius:16px;
  display:flex; align-items:center; justify-content:center;
  font-size:1.6rem; margin:0 auto 1rem;
}
.feat-title { font-size:0.95rem; font-weight:700; color:#F1F5F9; }

/* ═══ TABS ═══ */
.stTabs [data-baseweb="tab-list"] {
  gap:0.4rem;
  background: rgba(255,255,255,0.05);
  border-radius:14px; padding:0.4rem;
  border: 1px solid rgba(255,255,255,0.08);
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border-radius:10px !important;
  color: #94A3B8 !important;
  font-weight:600 !important; font-size:0.88rem !important;
  padding:0.5rem 1.4rem !important; border:none !important;
  transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg,#7C3AED,#4F46E5) !important;
  color: #ffffff !important; font-weight:700 !important;
  box-shadow: 0 4px 16px rgba(124,58,237,0.4) !important;
}

/* ═══ SECTION HEADING ═══ */
.sec-head { font-size:1.4rem; font-weight:800; color:#F1F5F9; margin-bottom:0.2rem; }
.sec-sub  { font-size:0.88rem; color:#64748B; margin-bottom:1.4rem; }

/* ═══ UPLOAD ZONE ═══ */
div[data-testid="stFileUploadDropzone"] {
  background: rgba(255,255,255,0.03) !important;
  border: 2px dashed rgba(139,92,246,0.5) !important;
  border-radius:16px !important;
  transition: all 0.2s !important;
}
div[data-testid="stFileUploadDropzone"]:hover {
  border-color: rgba(139,92,246,0.9) !important;
  background: rgba(139,92,246,0.05) !important;
}

/* ═══ RESULT CARD ═══ */
.result-card {
  border-radius:24px; padding:2rem 1.5rem; text-align:center;
  border: 1px solid rgba(255,255,255,0.1);
  animation: fadeUp 0.5s ease both;
}
.result-emoji { font-size:4.5rem; line-height:1; margin-bottom:0.6rem;
                animation: float 2.5s ease-in-out infinite; display:block; }
.result-pill {
  display:inline-block; border-radius:30px;
  padding:0.2rem 0.9rem; font-size:0.7rem; font-weight:700;
  letter-spacing:1.5px; text-transform:uppercase; margin-bottom:0.6rem;
}
.result-label { font-size:2.4rem; font-weight:900; margin-bottom:0.3rem; }
.result-conf  { font-size:0.95rem; color:#94A3B8; font-weight:600; margin-bottom:0.8rem; }
.result-tip   {
  font-size:0.85rem; color:#CBD5E1; font-style:italic; line-height:1.6;
  background:rgba(255,255,255,0.05); border-radius:12px; padding:0.7rem 1rem;
}

/* ═══ PLACEHOLDER ═══ */
.placeholder {
  border: 2px dashed rgba(255,255,255,0.1);
  border-radius:20px; padding:3rem; text-align:center;
  background: rgba(255,255,255,0.02);
}

/* ═══ DASHBOARD CARDS ═══ */
.dash-card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius:20px; padding:1.5rem;
  backdrop-filter:blur(10px);
  transition: transform 0.2s, border-color 0.2s;
}
.dash-card:hover { transform:translateY(-3px); border-color:rgba(139,92,246,0.3); }
.dash-metric-num { font-size:2.4rem; font-weight:800; color:#fff; line-height:1; }
.dash-metric-label { font-size:0.78rem; color:#64748B; font-weight:600;
                     letter-spacing:1px; text-transform:uppercase; margin-top:0.3rem; }
.dash-metric-sub { font-size:0.8rem; color:#94A3B8; margin-top:0.4rem; }

/* ═══ HOW IT WORKS ═══ */
.how-step {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 20px; padding: 1.5rem;
  text-align: center;
  backdrop-filter: blur(12px);
  transition: transform 0.2s, border-color 0.2s;
}
.how-step:hover { transform: translateY(-4px); border-color: rgba(139,92,246,0.4); }
.step-num {
  font-size: 2.8rem; font-weight: 900; line-height: 1;
  background: linear-gradient(135deg, #A78BFA, #60A5FA);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; margin-bottom: 0.5rem;
}
.step-icon { font-size: 2rem; margin-bottom: 0.5rem; display: block; }
.step-title { font-size: 1rem; font-weight: 700; color: #F1F5F9; margin-bottom: 0.4rem; }
.step-desc  { font-size: 0.85rem; color: #94A3B8; line-height: 1.6; }

/* ═══ PROFILE ═══ */
.profile-avatar {
  width:100px; height:100px; border-radius:50%;
  background: linear-gradient(135deg,#7C3AED,#06D6A0);
  display:flex; align-items:center; justify-content:center;
  font-size:3rem; margin:0 auto 1rem;
  animation: pulse-ring 2.5s ease-in-out infinite;
}
.profile-card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius:24px; padding:2rem; text-align:center;
  backdrop-filter:blur(12px);
}
.profile-name { font-size:1.6rem; font-weight:800; color:#F1F5F9; margin-bottom:0.2rem; }
.profile-role { font-size:0.88rem; color:#A78BFA; font-weight:600; margin-bottom:1rem; }
.skill-tag {
  display:inline-block; background:rgba(139,92,246,0.15);
  border:1px solid rgba(139,92,246,0.3); color:#C4B5FD;
  border-radius:20px; padding:0.25rem 0.8rem;
  font-size:0.75rem; font-weight:600; margin:0.2rem;
}
.info-row {
  display:flex; align-items:center; gap:0.8rem;
  padding:0.7rem 0; border-bottom:1px solid rgba(255,255,255,0.06);
  color:#CBD5E1; font-size:0.88rem;
}
.info-icon { font-size:1.1rem; width:28px; text-align:center; }

/* ═══ GLOBAL TEXT ═══ */
p, li, label, .stMarkdown p { color:#CBD5E1 !important; }
h1,h2,h3,h4 { color:#F1F5F9 !important; }
.stWarning { background:rgba(251,191,36,0.1) !important;
             color:#FCD34D !important; border-radius:12px !important;
             border:1px solid rgba(251,191,36,0.3) !important; }
strong { color:#F1F5F9 !important; }

/* ═══ DIVIDER ═══ */
.divider { border:none; border-top:1px solid rgba(255,255,255,0.07); margin:2rem 0; }

/* ═══ FOOTER ═══ */
.footer { text-align:center; color:#334155; font-size:0.8rem; padding:1.5rem 0 0.5rem; }

/* ═══ AUTH PAGE ═══ */
.auth-wrap {
  min-height: 100vh;
  display: flex; align-items: center; justify-content: center;
  padding: 2rem;
}
.auth-card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(139,92,246,0.25);
  border-radius: 28px;
  padding: 3rem 2.5rem;
  width: 100%; max-width: 440px;
  backdrop-filter: blur(20px);
  box-shadow: 0 24px 80px rgba(0,0,0,0.5);
  animation: fadeUp 0.5s ease both;
}
.auth-logo {
  text-align:center; margin-bottom:1.5rem;
  animation: float 3s ease-in-out infinite;
  font-size:3.5rem; line-height:1;
}
.auth-title {
  text-align:center; font-size:1.8rem; font-weight:900;
  background: linear-gradient(90deg,#F472B6,#A78BFA,#60A5FA);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  background-clip:text; margin-bottom:0.3rem;
}
.auth-sub {
  text-align:center; color:#64748B; font-size:0.88rem;
  margin-bottom:2rem;
}
.auth-card .stTextInput input {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 12px !important;
  color: #F1F5F9 !important;
  padding: 0.7rem 1rem !important;
  font-size: 0.9rem !important;
  transition: border-color 0.2s !important;
}
.auth-card .stTextInput input:focus {
  border-color: rgba(139,92,246,0.7) !important;
  box-shadow: 0 0 0 3px rgba(139,92,246,0.15) !important;
}
.auth-card .stTextInput label { color:#94A3B8 !important; font-size:0.82rem !important; font-weight:600 !important; }
.auth-card .stButton > button {
  width:100% !important;
  background: linear-gradient(135deg,#7C3AED,#4F46E5) !important;
  color:#fff !important; border:none !important;
  border-radius:14px !important; padding:0.75rem !important;
  font-size:0.95rem !important; font-weight:700 !important;
  letter-spacing:0.5px !important;
  box-shadow:0 8px 24px rgba(124,58,237,0.35) !important;
  transition:opacity 0.2s, transform 0.2s !important;
  margin-top:0.5rem !important;
}
.auth-card .stButton > button:hover { opacity:0.9 !important; transform:translateY(-1px) !important; }
.auth-footer-link {
  text-align:center; color:#64748B; font-size:0.83rem; margin-top:1.2rem;
}
.auth-footer-link span { color:#A78BFA; font-weight:600; cursor:pointer; }
/* Navbar */
.navbar {
  display:flex; align-items:center; justify-content:space-between;
  background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
  border-radius:16px; padding:0.7rem 1.5rem; margin-bottom:1.5rem;
  backdrop-filter:blur(12px);
}
.nav-brand { font-size:1.2rem; font-weight:800; color:#F1F5F9; }
.nav-user  { font-size:0.85rem; color:#94A3B8; font-weight:500; }
.nav-user span { color:#A78BFA; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── Model loaders ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_image_model():
    if not MODEL_IMG.exists(): return None
    import tensorflow as tf
    return tf.keras.models.load_model(str(MODEL_IMG))

@st.cache_resource(show_spinner=False)
def load_audio_model():
    if not MODEL_AUD.exists() or not LABELS_AUD.exists(): return None, None
    import tensorflow as tf
    model  = tf.keras.models.load_model(str(MODEL_AUD))
    labels = load_audio_labels(str(LABELS_AUD))
    return model, labels

def predict_image(model, img):
    return model.predict(preprocess_image_array(img), verbose=0)[0], IMAGE_CLASSES

def predict_audio(model, labels, wav_bytes):
    return model.predict(preprocess_audio_bytes(wav_bytes), verbose=0)[0], labels

def confidence_chart(probs, labels):
    colors = [EMOTION_META.get(l,{}).get("color","#7C3AED") for l in labels]
    fig = go.Figure(go.Bar(
        x=probs, y=labels, orientation="h",
        marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
        text=[f"<b>{p*100:.1f}%</b>" for p in probs],
        textposition="outside",
        textfont=dict(size=12, color="#CBD5E1"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#CBD5E1", family="Poppins", size=12),
        xaxis=dict(showgrid=False, showticklabels=False, range=[0,1.3],
                   showline=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(color="#CBD5E1", size=12), showline=False),
        margin=dict(l=10, r=60, t=10, b=10), height=290, bargap=0.35,
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
def show_landing():
    st.markdown("""
    <div class="landing-wrap">
      <!-- Floating background emojis -->
      <div class="landing-float-emoji" style="top:8%;left:5%;opacity:0.2;animation-delay:0s;">🐱</div>
      <div class="landing-float-emoji" style="top:15%;right:7%;opacity:0.25;animation-delay:0.8s;">😸</div>
      <div class="landing-float-emoji" style="top:55%;left:3%;opacity:0.15;animation-delay:1.5s;">🐾</div>
      <div class="landing-float-emoji" style="top:70%;right:5%;opacity:0.2;animation-delay:0.4s;">🐱</div>
      <div class="landing-float-emoji" style="top:35%;left:8%;opacity:0.18;animation-delay:2s;">🐾</div>
      <div class="landing-float-emoji" style="top:80%;left:15%;opacity:0.15;animation-delay:1.2s;">😸</div>

      <div class="landing-title">MeowMood</div>
      <div class="landing-sub">Decode your cat's emotions with AI</div>

      <div class="cat-photos-row">
        <img class="cat-photo cat-photo-1"
             src="https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400"
             alt="Orange cat" />
        <img class="cat-photo cat-photo-2"
             src="https://images.unsplash.com/photo-1573865526739-10659fec78a5?w=400"
             alt="Cute cat" />
        <img class="cat-photo cat-photo-3"
             src="https://images.unsplash.com/photo-1495360010541-f48722b34f7d?w=400"
             alt="Fluffy cat" />
      </div>

      <div class="landing-features">
        <div class="landing-feat-item">
          <span class="landing-feat-icon">🖼️</span>
          <span class="landing-feat-title">Facial Analysis</span>
        </div>
        <div class="landing-feat-item">
          <span class="landing-feat-icon">🎙️</span>
          <span class="landing-feat-title">Vocal Analysis</span>
        </div>
        <div class="landing-feat-item">
          <span class="landing-feat-icon">💡</span>
          <span class="landing-feat-title">Smart Tips</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        if st.button("Get Started →", key="cta_btn", use_container_width=True):
            st.session_state.show_landing = False
            st.rerun()


# ── Gate: show landing page first ─────────────────────────────────────────────
if st.session_state.show_landing:
    show_landing()
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  AUTH PAGE
# ══════════════════════════════════════════════════════════════════════════════
def show_auth():
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown('<div class="auth-logo">🐱</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">MeowMood</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">AI-Powered Cat Emotion Recognition</div>', unsafe_allow_html=True)

        # Toggle buttons — NO st.rerun(), just flip the flag and let Streamlit re-render
        t1, t2 = st.columns(2)
        with t1:
            if st.button("🔑  Log In", key="tog_login",
                         type="primary" if st.session_state.auth_page == "login" else "secondary",
                         use_container_width=True):
                st.session_state.auth_page = "login"
        with t2:
            if st.button("✨  Sign Up", key="tog_signup",
                         type="primary" if st.session_state.auth_page == "signup" else "secondary",
                         use_container_width=True):
                st.session_state.auth_page = "signup"

        st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

        # ── LOGIN ──────────────────────────────────────────────────────────────
        if st.session_state.auth_page == "login":
            with st.container(border=True):
                st.markdown("#### Welcome back 👋")
                # Use session_state defaults so values survive reruns
                li_user = st.text_input("Username", placeholder="Enter your username",
                                        key="li_user",
                                        value=st.session_state.get("li_user_val", ""))
                li_pass = st.text_input("Password", placeholder="Enter your password",
                                        type="password", key="li_pass")

                if st.button("Log In →", key="btn_login", use_container_width=True):
                    username = li_user.strip()
                    password = li_pass
                    db = st.session_state.users_db
                    if username in db and db[username]["password"] == _hash(password):
                        st.session_state.logged_in = True
                        st.session_state.username  = username
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password.")

                st.markdown("<div style='text-align:center;color:#475569;font-size:0.8rem;margin-top:0.5rem'>"
                            "Demo account: <b style='color:#A78BFA'>demo</b> / "
                            "<b style='color:#A78BFA'>demo123</b></div>", unsafe_allow_html=True)

        # ── SIGN UP ────────────────────────────────────────────────────────────
        else:
            with st.container(border=True):
                st.markdown("#### Create your account ✨")

                # Render all fields and capture return values directly
                name     = st.text_input("Full Name",        placeholder="Your name",         key="su_name")
                email    = st.text_input("Email",            placeholder="you@example.com",   key="su_email")
                username = st.text_input("Username",         placeholder="Choose a username", key="su_user")
                password = st.text_input("Password",         placeholder="Min 6 characters",  key="su_pass", type="password")
                confirm  = st.text_input("Confirm Password", placeholder="Repeat password",   key="su_conf", type="password")

                # Persist values into dedicated storage keys so they survive any rerun
                st.session_state["_su_name"]  = name
                st.session_state["_su_email"] = email
                st.session_state["_su_user"]  = username
                st.session_state["_su_pass"]  = password
                st.session_state["_su_conf"]  = confirm

                if st.button("Create Account →", key="btn_signup", use_container_width=True):
                    # Always read from the persisted storage keys
                    n  = st.session_state.get("_su_name",  "").strip()
                    e  = st.session_state.get("_su_email", "").strip()
                    u  = st.session_state.get("_su_user",  "").strip()
                    p  = st.session_state.get("_su_pass",  "")
                    c  = st.session_state.get("_su_conf",  "")
                    db = st.session_state.users_db

                    if not all([n, e, u, p, c]):
                        st.error("❌ Please fill in all fields.")
                    elif len(p) < 6:
                        st.error("❌ Password must be at least 6 characters.")
                    elif p != c:
                        st.error("❌ Passwords do not match.")
                    elif u in db:
                        st.error("❌ Username already taken. Choose another.")
                    elif not re.match(r"[^@]+@[^@]+\.[^@]+", e):
                        st.error("❌ Please enter a valid email address.")
                    else:
                        db[u] = {"password": _hash(p), "name": n, "email": e}
                        # Clear stored signup values
                        for k in ["_su_name","_su_email","_su_user","_su_pass","_su_conf"]:
                            st.session_state.pop(k, None)
                        st.success("✅ Account created! You can now log in.")
                        st.session_state.auth_page = "login"
                        st.rerun()

# ── Gate: show auth if not logged in ──────────────────────────────────────────
if not st.session_state.logged_in:
    show_auth()
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  NAVBAR (shown only when logged in)
# ══════════════════════════════════════════════════════════════════════════════
user_info = st.session_state.users_db.get(st.session_state.username, {})
nav_l, nav_r = st.columns([6, 1])
with nav_l:
    st.markdown(f"""
    <div class="navbar">
      <div class="nav-brand">🐱 MeowMood</div>
      <div class="nav-user">Logged in as <span>{user_info.get('name', st.session_state.username)}</span></div>
    </div>""", unsafe_allow_html=True)
with nav_r:
    if st.button("Log Out", key="logout_btn"):
        st.session_state.logged_in    = False
        st.session_state.username     = ""
        st.session_state.show_landing = True
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
  <div class="blob1"></div>
  <div class="blob2"></div>
  <div class="blob3"></div>
  <div class="hero-badge">✦ Deep Learning &nbsp;·&nbsp; Computer Vision &nbsp;·&nbsp; Audio AI</div>
  <span class="hero-cat">🐱</span>
  <div class="hero-title">MeowMood</div>
  <div class="hero-sub">
    Understand what your cat is feeling — upload a photo or audio clip
    and let AI decode their emotions in seconds.
  </div>
  <div class="hero-stats">
    <div class="stat-box"><div class="stat-num">7</div><div class="stat-label">Emotions</div></div>
    <div class="stat-box"><div class="stat-num">CNN</div><div class="stat-label">Image Model</div></div>
    <div class="stat-box"><div class="stat-num">🎙️</div><div class="stat-label">Audio Model</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Feature cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4, gap="medium")
cards = [
    ("🖼️", "linear-gradient(135deg,#7C3AED22,#4F46E522)", "#7C3AED", "Facial Analysis",    "0.1s"),
    ("🎙️", "linear-gradient(135deg,#E8439322,#FF6B3522)", "#E84393", "Vocal Analysis",     "0.2s"),
    ("⚡",  "linear-gradient(135deg,#06D6A022,#4CC9F022)", "#06D6A0", "Instant Results",    "0.3s"),
    ("💡",  "linear-gradient(135deg,#FBBF2422,#F4724422)", "#FBBF24", "Smart Tips",         "0.4s"),
]
for col, (icon, bg, accent, title, delay) in zip([c1,c2,c3,c4], cards):
    with col:
        st.markdown(f"""
        <div class="feat-card" style="animation-delay:{delay}">
          <div class="feat-icon-wrap" style="background:{bg};border:1px solid {accent}33;">
            <span>{icon}</span>
          </div>
          <div class="feat-title">{title}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:1.8rem'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_img, tab_aud, tab_dash, tab_profile = st.tabs([
    "🖼️   Image Analysis",
    "🎙️   Audio Analysis",
    "📊   Dashboard",
    "👤   Profile",
])

# ─── TAB 1: IMAGE ─────────────────────────────────────────────────────────────
with tab_img:
    st.markdown('<div class="sec-head">Cat Facial Emotion Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Upload a clear photo of your cat\'s face</div>', unsafe_allow_html=True)

    img_model = load_image_model()
    if img_model is None:
        st.warning("⚠️ Image model not trained yet. Run: `python backend/train/image_model.py`")
    else:
        uploaded_img = st.file_uploader("img", type=["jpg","jpeg","png"],
                                        key="img_upload", label_visibility="collapsed")
        if uploaded_img:
            col_img, col_res = st.columns([1,1], gap="large")
            with col_img:
                img = Image.open(uploaded_img).convert("RGB")
                st.image(img, caption="Your cat 🐱", use_container_width=True)
            with col_res:
                with st.spinner("Analyzing facial expression..."):
                    probs, classes = predict_image(img_model, img)
                top_idx   = int(np.argmax(probs))
                top_label = classes[top_idx]
                meta = EMOTION_META.get(top_label, {"emoji":"🐱","color":"#7C3AED","bg":"rgba(124,58,237,0.12)","tip":""})
                conf = probs[top_idx]*100
                st.markdown(f"""
                <div class="result-card" style="background:{meta['bg']};border-color:{meta['color']}40;">
                  <span class="result-emoji">{meta['emoji']}</span>
                  <div class="result-pill" style="background:{meta['color']}22;color:{meta['color']};">DETECTED EMOTION</div>
                  <div class="result-label" style="color:{meta['color']}">{top_label}</div>
                  <div class="result-conf">Confidence: <strong style="color:{meta['color']}">{conf:.1f}%</strong></div>
                  <div class="result-tip">💡 {meta['tip']}</div>
                </div>""", unsafe_allow_html=True)
                st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
                st.markdown("**Confidence Breakdown**")
                st.plotly_chart(confidence_chart(probs, classes),
                                use_container_width=True, config={"displayModeBar":False})
        else:
            st.markdown("""
            <div class="placeholder">
              <div style="font-size:3rem;margin-bottom:0.6rem">🐱</div>
              <div style="color:#A78BFA;font-weight:600;font-size:1rem">Drop a cat photo to get started</div>
              <div style="color:#475569;font-size:0.85rem;margin-top:0.3rem">Supports JPG and PNG</div>
            </div>""", unsafe_allow_html=True)

# ─── TAB 2: AUDIO ─────────────────────────────────────────────────────────────
with tab_aud:
    st.markdown('<div class="sec-head">Cat Vocal Emotion Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Upload a WAV recording of your cat\'s vocalization</div>', unsafe_allow_html=True)

    aud_model, aud_labels = load_audio_model()
    if aud_model is None:
        st.warning("⚠️ Audio model not trained yet. Run: `python backend/train/audio_model.py`")
    else:
        uploaded_aud = st.file_uploader("aud", type=["wav"],
                                        key="aud_upload", label_visibility="collapsed")
        if uploaded_aud:
            col_aud, col_ares = st.columns([1,1], gap="large")
            with col_aud:
                st.audio(uploaded_aud, format="audio/wav")
                try:
                    import librosa, librosa.display, matplotlib.pyplot as plt
                    wav_bytes = uploaded_aud.read(); uploaded_aud.seek(0)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(wav_bytes); tmp_path = tmp.name
                    y_w, sr_w = librosa.load(tmp_path, sr=22050)
                    os.unlink(tmp_path)

                    fig_w, ax_w = plt.subplots(figsize=(6,2))
                    fig_w.patch.set_facecolor("#0D0D1A"); ax_w.set_facecolor("#111128")
                    librosa.display.waveshow(y_w, sr=sr_w, ax=ax_w, color="#A78BFA", alpha=0.9)
                    ax_w.set_xlabel("Time (s)", color="#64748B", fontsize=9)
                    ax_w.tick_params(colors="#64748B", labelsize=8)
                    for sp in ax_w.spines.values(): sp.set_edgecolor("#1E293B")
                    ax_w.set_title("Waveform", color="#F1F5F9", fontsize=10, fontweight="bold")
                    plt.tight_layout(); st.pyplot(fig_w); plt.close()

                    S    = librosa.feature.melspectrogram(y=y_w, sr=sr_w, n_mels=128)
                    S_db = librosa.power_to_db(S, ref=np.max)
                    fig_s, ax_s = plt.subplots(figsize=(6,2.5))
                    fig_s.patch.set_facecolor("#0D0D1A"); ax_s.set_facecolor("#111128")
                    img_s = librosa.display.specshow(S_db, sr=sr_w, x_axis="time",
                                                     y_axis="mel", ax=ax_s, cmap="magma")
                    fig_s.colorbar(img_s, ax=ax_s, format="%+2.0f dB")
                    ax_s.set_title("Mel-Spectrogram", color="#F1F5F9", fontsize=10, fontweight="bold")
                    ax_s.tick_params(colors="#64748B", labelsize=8)
                    plt.tight_layout(); st.pyplot(fig_s); plt.close()
                except Exception as e:
                    st.info(f"Waveform preview unavailable: {e}")

            with col_ares:
                uploaded_aud.seek(0); wav_bytes = uploaded_aud.read()
                with st.spinner("Analyzing vocalization..."):
                    probs, labels = predict_audio(aud_model, aud_labels, wav_bytes)
                top_idx   = int(np.argmax(probs))
                top_label = labels[top_idx]
                meta = EMOTION_META.get(top_label, {"emoji":"🐱","color":"#7C3AED","bg":"rgba(124,58,237,0.12)","tip":""})
                conf = probs[top_idx]*100
                st.markdown(f"""
                <div class="result-card" style="background:{meta['bg']};border-color:{meta['color']}40;">
                  <span class="result-emoji">{meta['emoji']}</span>
                  <div class="result-pill" style="background:{meta['color']}22;color:{meta['color']};">DETECTED EMOTION</div>
                  <div class="result-label" style="color:{meta['color']}">{top_label}</div>
                  <div class="result-conf">Confidence: <strong style="color:{meta['color']}">{conf:.1f}%</strong></div>
                  <div class="result-tip">💡 {meta['tip']}</div>
                </div>""", unsafe_allow_html=True)
                st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
                st.markdown("**Confidence Breakdown**")
                st.plotly_chart(confidence_chart(probs, labels),
                                use_container_width=True, config={"displayModeBar":False})
        else:
            st.markdown("""
            <div class="placeholder">
              <div style="font-size:3rem;margin-bottom:0.6rem">🎙️</div>
              <div style="color:#F472B6;font-weight:600;font-size:1rem">Drop a WAV audio file to get started</div>
              <div style="color:#475569;font-size:0.85rem;margin-top:0.3rem">Supports WAV format</div>
            </div>""", unsafe_allow_html=True)

# ─── TAB 3: DASHBOARD ─────────────────────────────────────────────────────────
with tab_dash:
    st.markdown('<div class="sec-head">Project Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Overview of datasets, model insights, and emotion classes</div>', unsafe_allow_html=True)

    # ── Charts row ──────────────────────────────────────────────────────────────
    ch1, ch2 = st.columns(2, gap="large")

    with ch1:
        st.markdown("**Image Dataset — Class Distribution**")
        try:
            import pandas as pd
            counts = {}
            for split in ["train","valid"]:
                for cls in IMAGE_CLASSES:
                    p = IMAGE_DIR / split / cls
                    if p.exists():
                        counts.setdefault(cls,{})[split] = len(list(p.glob("*.jpg")))
            df = pd.DataFrame(counts).T.fillna(0).astype(int)
            colors_t = [EMOTION_META.get(c,{}).get("color","#7C3AED") for c in df.index]
            fig_i = go.Figure()
            fig_i.add_trace(go.Bar(name="Train", x=df.index.tolist(),
                                   y=df.get("train", pd.Series()).tolist(),
                                   marker=dict(color=colors_t, opacity=0.9, line=dict(width=0))))
            fig_i.add_trace(go.Bar(name="Valid", x=df.index.tolist(),
                                   y=df.get("valid", pd.Series()).tolist(),
                                   marker=dict(color=colors_t, opacity=0.4, line=dict(width=0))))
            fig_i.update_layout(
                barmode="group", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.03)",
                font=dict(color="#CBD5E1", family="Poppins", size=11),
                legend=dict(font=dict(color="#CBD5E1"), bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(showgrid=False, tickfont=dict(color="#94A3B8"),
                           showline=True, linecolor="rgba(255,255,255,0.08)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                           tickfont=dict(color="#94A3B8"), showline=False),
                margin=dict(l=10,r=10,t=10,b=10), height=320, bargap=0.2,
            )
            st.plotly_chart(fig_i, use_container_width=True, config={"displayModeBar":False})
        except Exception as e:
            st.error(f"Could not load image stats: {e}")

    with ch2:
        st.markdown("**Audio Dataset — Label Distribution**")
        try:
            audio_counts: dict = {}
            for wav in AUDIO_DIR.glob("*.wav"):
                prefix = wav.stem[0]
                if prefix in AUDIO_LABEL_MAP:
                    lbl = AUDIO_LABEL_MAP[prefix]
                    audio_counts[lbl] = audio_counts.get(lbl,0)+1
            labels_a = list(audio_counts.keys())
            vals_a   = list(audio_counts.values())
            colors_a = [EMOTION_META.get(l,{}).get("color","#7C3AED") for l in labels_a]
            fig_a = go.Figure(go.Pie(
                labels=labels_a, values=vals_a,
                marker=dict(colors=colors_a, line=dict(color="#0D0D1A", width=3)),
                hole=0.52, textinfo="label+percent",
                textfont=dict(color="#F1F5F9", size=12, family="Poppins"),
                pull=[0.05]*len(labels_a),
            ))
            fig_a.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1", family="Poppins"),
                legend=dict(font=dict(color="#CBD5E1"), bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=10,r=10,t=10,b=10), height=320,
            )
            st.plotly_chart(fig_a, use_container_width=True, config={"displayModeBar":False})
        except Exception as e:
            st.error(f"Could not load audio stats: {e}")

    # ── How It Works ─────────────────────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**How It Works**")
    st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)
    hw1, hw2, hw3 = st.columns(3, gap="large")
    how_steps = [
        ("1", "📤", "Upload Media",   "Upload a cat photo or WAV audio recording"),
        ("2", "🧠", "AI Analysis",    "Our deep learning models analyze facial features or vocal patterns"),
        ("3", "💡", "Get Results",    "Receive the detected emotion with confidence score and care tips"),
    ]
    for col, (num, icon, title, desc) in zip([hw1, hw2, hw3], how_steps):
        with col:
            st.markdown(f"""
            <div class="how-step">
              <div class="step-num">{num}</div>
              <span class="step-icon">{icon}</span>
              <div class="step-title">{title}</div>
              <div class="step-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # ── Emotion reference grid ───────────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**Emotion Classes**")
    st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)
    ref_cols = st.columns(7, gap="small")
    for i, (emotion, meta) in enumerate(EMOTION_META.items()):
        with ref_cols[i]:
            st.markdown(f"""
            <div style="background:{meta['bg']};border:1px solid {meta['color']}30;
                        border-radius:16px;padding:1rem 0.5rem;text-align:center;
                        transition:transform 0.2s;">
              <div style="font-size:2rem;margin-bottom:0.4rem">{meta['emoji']}</div>
              <div style="color:{meta['color']};font-weight:800;font-size:0.82rem">{emotion}</div>
            </div>""", unsafe_allow_html=True)

# ─── TAB 4: PROFILE ───────────────────────────────────────────────────────────
with tab_profile:
    st.markdown('<div class="sec-head">My Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Your account details and preferences</div>', unsafe_allow_html=True)

    uinfo = st.session_state.users_db.get(st.session_state.username, {})
    p_name  = uinfo.get("name",  st.session_state.username)
    p_email = uinfo.get("email", "—")

    pc1, pc2 = st.columns([1, 2], gap="large")

    with pc1:
        initials = "".join([w[0].upper() for w in p_name.split()[:2]])
        st.markdown(f"""
        <div class="profile-card">
          <div class="profile-avatar">{initials}</div>
          <div class="profile-name">{p_name}</div>
          <div class="profile-role">🐱 MeowMood User</div>
          <div style="margin:1rem 0">
            <span class="skill-tag">Cat Lover</span>
            <span class="skill-tag">AI Explorer</span>
            <span class="skill-tag">Pet Owner</span>
          </div>
        </div>""", unsafe_allow_html=True)

    with pc2:
        st.markdown("""
        <div class="dash-card" style="margin-bottom:1rem">
          <div style="color:#A78BFA;font-weight:700;font-size:0.9rem;
                      margin-bottom:1rem;letter-spacing:1px;text-transform:uppercase">
            Account Information
          </div>""", unsafe_allow_html=True)

        st.markdown(f"""
          <div class="info-row">
            <span class="info-icon">👤</span>
            <div><div style="color:#64748B;font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:1px">Full Name</div>
            <div style="color:#F1F5F9;font-weight:600">{p_name}</div></div>
          </div>
          <div class="info-row">
            <span class="info-icon">📧</span>
            <div><div style="color:#64748B;font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:1px">Email</div>
            <div style="color:#F1F5F9;font-weight:600">{p_email}</div></div>
          </div>
          <div class="info-row">
            <span class="info-icon">🔑</span>
            <div><div style="color:#64748B;font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:1px">Username</div>
            <div style="color:#F1F5F9;font-weight:600">@{st.session_state.username}</div></div>
          </div>
          <div class="info-row" style="border-bottom:none">
            <span class="info-icon">🛡️</span>
            <div><div style="color:#64748B;font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:1px">Account Status</div>
            <div style="color:#06D6A0;font-weight:600">● Active</div></div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="dash-card" style="margin-top:1rem">
          <div style="color:#34D399;font-weight:700;font-size:0.9rem;
                      margin-bottom:1rem;letter-spacing:1px;text-transform:uppercase">
            Available Features
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:0.5rem">
            <span class="skill-tag" style="background:rgba(6,214,160,0.12);border-color:rgba(6,214,160,0.3);color:#34D399">🖼️ Image Analysis</span>
            <span class="skill-tag" style="background:rgba(244,114,182,0.12);border-color:rgba(244,114,182,0.3);color:#F472B6">🎙️ Audio Analysis</span>
            <span class="skill-tag" style="background:rgba(96,165,250,0.12);border-color:rgba(96,165,250,0.3);color:#60A5FA">📊 Dashboard</span>
            <span class="skill-tag" style="background:rgba(251,191,36,0.12);border-color:rgba(251,191,36,0.3);color:#FBBF24">💡 Smart Tips</span>
            <span class="skill-tag" style="background:rgba(167,139,250,0.12);border-color:rgba(167,139,250,0.3);color:#A78BFA">7 Emotion Classes</span>
          </div>
        </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
  🐱 <strong style="color:#475569">MeowMood</strong> — AI Cat Emotion Recognition &nbsp;·&nbsp; Built with Streamlit &amp; TensorFlow
</div>
""", unsafe_allow_html=True)
