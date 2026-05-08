import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import plotly.express as px
import plotly.graph_objects as go
import base64
import os

def get_base64(bin_file):
    if not os.path.exists(bin_file):
        return ""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TrendVault • YouTube Trending",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
import streamlit.components.v1


st.components.v1.html('''
<script src="https://unpkg.com/lucide@latest"></script>
<script>
  const observer = new MutationObserver(() => {
    if (window.parent.lucide) {
      window.parent.lucide.createIcons();
    }
  });
  observer.observe(window.parent.document.body, { childList: true, subtree: true });
  if (window.parent.lucide) {
    window.parent.lucide.createIcons();
  }
</script>
''', height=0)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg:          #f8fafc;
  --surface:     #ffffff;
  --card:        #ffffff;
  --sidebar-bg:  #ffffff;
  --border:      #e2e8f0;
  --text:        #111827;
  --text-muted:  #475569;
  --text-light:  #64748b;
  --accent:      #6366f1;
  --accent2:     #8b5cf6;
  --pink:        #ec4899;
  --teal:        #14b8a6;
  --amber:       #f59e0b;
  --green:       #10b981;
  --shadow-sm:   0 1px 3px rgba(0,0,0,0.07), 0 1px 2px rgba(0,0,0,0.04);
  --shadow-md:   0 4px 16px rgba(0,0,0,0.08), 0 2px 6px rgba(0,0,0,0.04);
  --shadow-lg:   0 10px 40px rgba(0,0,0,0.10);
}

html, body, [data-testid="stAppViewContainer"] {
  background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e0e7ff 100%) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--text) !important;
}

[data-testid="stSidebar"] {
  background: #ffffff !important;
  border-right: 1px solid #e2e8f0 !important;
  box-shadow: 2px 0 20px rgba(0,0,0,0.05);
}
[data-testid="stSidebar"] * { color: #111827 !important; }

header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }

.stButton > button {
  width: 100%;
  background: #6366f1;
  border: none;
  color: #ffffff !important;
  border-radius: 8px;
  padding: 0 14px;
  height: 44px;
  font-family: 'Inter', sans-serif;
  font-size: 13.5px;
  font-weight: 600;
  transition: all 0.15s ease;
  text-align: center;
  margin-bottom: 4px;
  box-shadow: 0 2px 4px rgba(99,102,241,0.2) !important;
  box-sizing: border-box;
}
.stButton > button:hover {
  background: #4f46e5 !important;
  color: #ffffff !important;
  box-shadow: 0 0 10px rgba(99,102,241,0.5), 0 4px 6px rgba(99,102,241,0.3) !important;
}

/* ── SIDEBAR: force every button to identical full width ── */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
  width: 100% !important;
}
[data-testid="stSidebar"] .stButton {
  width: 100% !important;
  display: block !important;
}
[data-testid="stSidebar"] .stButton > button {
  width: 100% !important;
  min-width: 100% !important;
  max-width: 100% !important;
  display: block !important;
}

[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div > input,
[data-testid="stNumberInput"] > div > div > input {
  background: #ffffff !important;
  border: 1.5px solid #e2e8f0 !important;
  border-radius: 8px !important;
  color: #111827 !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 14px !important;
  box-shadow: var(--shadow-sm) !important;
}

[data-testid="metric-container"] {
  background: #ffffff;
  border: 1px solid transparent;
  background-clip: padding-box;
  position: relative;
  border-radius: 12px;
  padding: 20px;
  box-shadow: var(--shadow-sm);
}
[data-testid="metric-container"]::after {
  content: '';
  position: absolute;
  top: -1px; bottom: -1px; left: -1px; right: -1px;
  background: linear-gradient(135deg, #6366f1, #ec4899);
  z-index: -1;
  border-radius: 13px;
  opacity: 0.3;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }

@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes floatY {
  0%,100% { transform: translateY(0px); }
  50%      { transform: translateY(-8px); }
}

.float-icon { animation: floatY 3s ease-in-out infinite; display: inline-block; }

.kpi-card {
  animation: fadeSlideUp 0.4s ease forwards;
  border-radius: 14px; padding: 24px; text-align: center;
  background: #ffffff;
  border: 1px solid transparent;
  background-clip: padding-box;
  position: relative;
  box-shadow: var(--shadow-sm);
  transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card::before {
  content: '';
  position: absolute;
  top: -1.5px; bottom: -1.5px; left: -1.5px; right: -1.5px;
  background: linear-gradient(135deg, #6366f1, #ec4899, #14b8a6);
  z-index: -1;
  border-radius: 15px;
  opacity: 0.3;
  transition: opacity 0.2s;
}
.kpi-card:hover {
  transform: translateY(-3px);
  box-shadow: var(--shadow-md);
}
.kpi-card:hover::before {
  opacity: 0.8;
}

.video-card {
  background: #ffffff;
  border: 1px solid transparent;
  background-clip: padding-box;
  position: relative;
  border-radius: 12px; overflow: hidden;
  box-shadow: var(--shadow-sm);
  transition: transform 0.2s, box-shadow 0.2s;
}
.video-card::before {
  content: '';
  position: absolute;
  top: -1.5px; bottom: -1.5px; left: -1.5px; right: -1.5px;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  z-index: -1;
  border-radius: 13.5px;
  opacity: 0.2;
  transition: opacity 0.2s;
}
.video-card:hover {
  transform: translateY(-3px);
  box-shadow: var(--shadow-md);
}
.video-card:hover::before {
  opacity: 0.6;
}

div, span, p, h1, h2, h3, h4, h5, h6 {
  text-shadow: none !important;
}

[data-testid="stTabs"] button {
  color: #475569 !important;
  border-bottom-color: transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: #111827 !important;
  border-bottom-color: #6366f1 !important;
  box-shadow: 0 2px 10px rgba(99,102,241,0.2) !important;
  font-weight: 700 !important;
}

.shimmer-text {
  background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899, #8b5cf6, #6366f1);
  background-size: 300% auto;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 800;
}

/* Force Streamlit widget labels and text to dark mode to prevent blending */
label, p, .stMarkdown p, .stText, [data-baseweb="radio"] div {
  color: #111827 !important;
}

/* Fix radio button group container backgrounds if any */
[data-testid="stRadio"] label {
  color: #111827 !important;
}


/* Force Streamlit Dropdown menus to be light */
ul[data-testid="stVirtualDropdown"] {
    background-color: #ffffff !important;
}
ul[data-testid="stVirtualDropdown"] li {
    color: #111827 !important;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Decision Tree"

# ─────────────────────────────────────────────
#  CATEGORY MAP
# ─────────────────────────────────────────────
CATEGORY_MAP = {
    1:"Film & Animation", 2:"Autos & Vehicles", 10:"Music",
    15:"Pets & Animals", 17:"Sports", 19:"Travel & Events",
    20:"Gaming", 22:"People & Blogs", 23:"Comedy",
    24:"Entertainment", 25:"News & Politics", 26:"Howto & Style",
    27:"Education", 28:"Science & Technology", 29:"Nonprofits & Activism"
}

# ─────────────────────────────────────────────
#  DATA LOADER
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("youtube_trending.csv")
        df.rename(columns={
            "video_title":            "title",
            "video_view_count":       "views",
            "video_like_count":       "likes",
            "video_comment_count":    "comment_count",
            "video_category_id":      "category_name",   # ← contains TEXT like "Music","Gaming" etc.
            "video_trending_country": "country",
            "video_trending__date":   "trending_date",
            "video_default_thumbnail":"thumbnail_link",
        }, inplace=True)
    except FileNotFoundError:
        np.random.seed(42)
        n = 1500
        countries = ["US","GB","CA","IN","DE","FR","JP","BR","KR","AU","MX","RU"]
        channels  = ["MrBeast","PewDiePie","T-Series","Cocomelon","SET India",
                     "5-Minute Crafts","WWE","Zee Music","Like Nastya","BLACKPINK",
                     "Dude Perfect","Markiplier","jacksepticeye","MKBHD","Veritasium"]
        titles_pool = [
            "I Spent 24 Hours In","World's Most Extreme","Top 10 Most Amazing",
            "New Official Music Video","Epic Battle Season","Reacting To The Worst",
            "How To Make Perfect","Unboxing The New","I Challenged My Friends",
            "Last To Leave Wins","Ultimate Survival","Building The World's",
            "100 Days In","I Bought Every","Overnight Challenge"
        ]
        cat_ids = [1,2,10,15,17,19,20,22,23,24,25,26,27,28]
        dates   = pd.date_range("2022-01-01","2023-12-31", periods=n)
        df = pd.DataFrame({
            "title":         [f"{np.random.choice(titles_pool)} #{i}" for i in range(n)],
            "channel_title": np.random.choice(channels, n),
            "views":         np.random.randint(50_000, 25_000_000, n),
            "likes":         np.random.randint(1_000,  2_000_000, n),
            "dislikes":      np.random.randint(100,    200_000, n),
            "comment_count": np.random.randint(200,    500_000, n),
            "category_id":   np.random.choice(cat_ids, n),
            "trending_date": [d.strftime("%Y-%m-%d") for d in dates],
            "thumbnail_link":[f"https://picsum.photos/seed/{i+10}/320/180" for i in range(n)],
            "country":       np.random.choice(countries, n),
        })

    df.columns = df.columns.str.strip().str.lower().str.replace(" ","_")
    df.dropna(subset=["title","views"], inplace=True)
    # Ensure channel_title exists
    if "channel_title" not in df.columns:
        df["channel_title"] = "Unknown"
    if "thumbnail_link" not in df.columns:
        df["thumbnail_link"] = [f"https://picsum.photos/seed/{i+10}/320/180" for i in range(len(df))]
    df["views"]         = pd.to_numeric(df["views"],         errors="coerce").fillna(0).astype(int)
    df["likes"]         = pd.to_numeric(df["likes"],         errors="coerce").fillna(0).astype(int)
    df["dislikes"]      = pd.to_numeric(df.get("dislikes",   pd.Series([0]*len(df))), errors="coerce").fillna(0).astype(int)
    df["comment_count"] = pd.to_numeric(df["comment_count"], errors="coerce").fillna(0).astype(int)
    # ── category_name: video_category_id already contains text names ──────────
    if "category_name" in df.columns:
        df["category_name"] = df["category_name"].astype(str).str.strip()
    else:
        df["category_name"] = "Entertainment"
    # Remove junk rows
    df = df[~df["category_name"].isin(["Grand Total","nan","None","","(blank)"])]
    df["category_name"] = df["category_name"].fillna("Entertainment")
    df["engagement_rate"] = ((df["likes"] + df["comment_count"]) / df["views"].replace(0,1) * 100).clip(upper=100.0).round(2)
    df["trending_date"] = pd.to_datetime(df["trending_date"], errors="coerce")
    return df

df = load_data()

# ───────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:24px 0 12px;'>
      <div class='float-icon' style='display:flex; justify-content:center;'><i data-lucide='video' style='width:40px; height:40px; color:#374151;'></i></div>
      <div class='shimmer-text' style='font-family:Inter;font-weight:800;font-size:20px;margin-top:6px;'>
        TrendVault
      </div>
      <div style='color:#475569;font-size:11px;margin-top:4px;font-weight:500;letter-spacing:0.5px;'>YouTube Intelligence</div>
    </div>
    <hr style='border:0;border-top:1px solid rgba(255,255,255,0.08);margin:10px 0 14px;'>
    """, unsafe_allow_html=True)

    for p in ["Home","Search & Explore","Trending Analytics",
              "World Map","Leaderboard","Country Compare",
              "Virality Predictor","Monetization"]:
        if st.button(p, key=f"nav_{p}", use_container_width=True):
            st.session_state.page = p
    
    st.markdown("<hr style='border:0;border-top:1px solid rgba(255,255,255,0.08);margin:14px 0;'>", unsafe_allow_html=True)
    st.markdown(f"""<div style='font-size:11px;color:#475569;text-align:center;line-height:1.8;'>
      <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='box' style='width:18px; height:18px; color:#374151;'></i></span> {len(df):,} videos &nbsp;|&nbsp; <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span> {df['country'].nunique()} countries<br>
      <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='layout-dashboard' style='width:18px; height:18px; color:#374151;'></i></span> {df['category_name'].nunique()} categories</div>""", unsafe_allow_html=True)

page = st.session_state.page

# ══════════════════════════════════════════════
#  PAGE 1 — HOME
# ══════════════════════════════════════════════
if page == "Home":

    # 3D Hero Section
    hero_img_path = r"C:\Users\ADMIN\.gemini\antigravity\brain\edd51759-fc94-47d6-843c-bb31ffb9083f\youtube_trending_analysis_hero_1777396767026.png"
    hero_img_b64 = get_base64(hero_img_path)
    bg_url = f"data:image/png;base64,{hero_img_b64}" if hero_img_b64 else "https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=2070&auto=format&fit=crop"

    st.markdown(f"""
    <div class='dot-grid-bg' style='position:relative;overflow:hidden;border-radius:16px;
         background: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), 
                     url("{bg_url}");
         background-size: cover; background-position: center;
         padding:56px 48px 52px;margin-bottom:28px;
         box-shadow:0 8px 40px rgba(99,102,241,0.35), 0 2px 8px rgba(99,102,241,0.2);'>
      <div style='position:relative;z-index:2;max-width:80%;'>
        <div style='font-size:11px;font-weight:600;color:#e2e8f0;letter-spacing:2px;
             text-transform:uppercase;margin-bottom:12px;'>
          ✦ YouTube Trending Intelligence Platform
        </div>
        <h1 style='font-family:Inter;font-weight:800;
             font-size:56px;margin:0 0 14px;line-height:1.05;color:#ffffff;
             letter-spacing:-1px;'>TrendVault</h1>
        <p style='font-size:16px;color:#ffffff;margin:0 0 10px;font-weight:500;
            letter-spacing:0.3px;'>Discover · Analyse · Predict</p>
        <p style='font-size:14px;color:#cbd5e1;max-width:480px;line-height:1.8;'>
          Explore what's trending globally across countries, categories and channels.
          Powered by real YouTube data with virality predictions.</p>
        <div style='margin-top:24px;display:flex;gap:12px;flex-wrap:wrap;'>
          <div style='background:rgba(124,58,237,0.3);backdrop-filter:blur(4px);border:1px solid #7c3aed;border-radius:20px;
               padding:6px 16px;font-size:13px;color:#ffffff;'><span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='flame' style='width:18px; height:18px; color:#ffffff;'></i></span> {len(df):,} Videos</div>
          <div style='background:rgba(236,72,153,0.3);backdrop-filter:blur(4px);border:1px solid #ec4899;border-radius:20px;
               padding:6px 16px;font-size:13px;color:#ffffff;'><span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#ffffff;'></i></span> {df["country"].nunique()} Countries</div>
          <div style='background:rgba(245,158,11,0.3);backdrop-filter:blur(4px);border:1px solid #f59e0b;border-radius:20px;
               padding:6px 16px;font-size:13px;color:#ffffff;'><span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='layout-dashboard' style='width:18px; height:18px; color:#ffffff;'></i></span> {df["category_name"].nunique()} Categories</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    c1,c2,c3,c4 = st.columns(4)
    kpi_data = [
        (c1,"video",f"{len(df):,}","Total Trending Videos","#5046e5","#818cf8"),
        (c2,"eye",f"{df['views'].sum()/1e9:.2f}B","Total Views","#6d28d9","#c4b5fd"),
        (c3,"thumbs-up",f"{df['likes'].sum()/1e6:.1f}M","Total Likes","#9333ea","#e879f9"),
        (c4,"trending-up",f"{df['engagement_rate'].mean():.2f}%","Avg Engagement","#7c2d12","#fcd34d"),
    ]
    for col,icon,val,label,_,color in kpi_data:
        with col:
            st.markdown(f"""
            <div class='kpi-card' style='background:#f8fafc;'>
              <div style='display:flex; justify-content:center; align-items:center; margin-bottom:8px;'><i data-lucide='{icon}' style='width:18px; height:18px; color:#374151;'></i></div>
              <div style='font-size:30px;font-weight:800;color:{color};'>{val}</div>
              <div style='color:#475569;font-size:13px;margin-top:4px;'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Top 5
    st.markdown("""<div style='border-left:4px solid #7c3aed;padding-left:14px;margin-bottom:20px;'>
      <h2 style='font-family:Outfit;font-weight:700;font-size:26px;margin:0;
          background:linear-gradient(90deg,#a855f7,#ec4899);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='flame' style='width:18px; height:18px; color:#374151;'></i></span> Top 5 Trending Right Now</h2>
      <div style='color:#64748b;font-size:13px;margin-top:4px;'>Most viewed videos globally</div>
    </div>""", unsafe_allow_html=True)

    top5 = df.nlargest(5,"views")
    rank_colors = ["#f59e0b","#b0b8d4","#cd7f32","#7c3aed","#6366f1"]
    cols = st.columns(5)
    for i,(col,(_,row)) in enumerate(zip(cols, top5.iterrows())):
        with col:
            st.markdown(f"""
            <div class='video-card' style='animation:fadeSlideUp {0.2+i*0.1}s ease forwards;'>
              <div style='position:relative;'>
                <img src='{row["thumbnail_link"]}' style='width:100%;height:115px;object-fit:cover;'
                     onerror="this.src='https://picsum.photos/seed/{i}42/320/180'">
                <div style='position:absolute;top:8px;left:8px;background:{rank_colors[i]};
                    color:#000;font-weight:900;font-size:12px;border-radius:6px;padding:2px 8px;'>
                    #{i+1}</div>
                <div style='position:absolute;bottom:0;left:0;right:0;
                    background:linear-gradient(transparent,rgba(255,255,255,0.95));padding:6px 8px;'>
                  <div style='font-size:10px;color:#6d28d9;font-weight:600;'>{row["category_name"]}</div>
                </div>
              </div>
              <div style='padding:10px;'>
                <div style='font-size:12px;font-weight:600;color:#111827;
                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>
                    {row["title"][:38]}...</div>
                <div style='font-size:11px;color:#6d28d9;margin:4px 0;'><span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='tv' style='width:18px; height:18px; color:#374151;'></i></span> {row["channel_title"]}</div>
                <div style='font-size:10px;color:#64748b;margin-bottom:6px;'>
                    <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='calendar' style='width:18px; height:18px; color:#374151;'></i></span> {str(row["trending_date"])[:10]} • <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span> {row["country"]}</div>
                <div style='background:#f8fafc;border-radius:8px;padding:7px;'>
                  <div style='display:flex;justify-content:space-between;font-size:11px;'>
                    <span style='color:#60a5fa;'><span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='eye' style='width:18px; height:18px; color:#374151;'></i></span> {row["views"]/1e6:.1f}M</span>
                    <span style='color:#4ade80;'><span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='thumbs-up' style='width:18px; height:18px; color:#374151;'></i></span> {row["likes"]/1e3:.0f}K</span>
                  </div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)



# ══════════════════════════════════════════════
#  PAGE 2 — SEARCH & EXPLORE
# ══════════════════════════════════════════════
elif page == "Search & Explore":
    st.markdown("""<h1 style='font-family:Inter;font-weight:700;font-size:32px;
        color:#0f172a;'>
        Search & Explore</h1>
        <p style='color:#64748b;font-size:15px;margin-top:0;'>Find videos by title, channel, country or category</p>""", unsafe_allow_html=True)

    st.markdown("""<div style='background:linear-gradient(135deg,#ede9fe,#ddd6fe);
        border:1px solid #5046e5;border-radius:18px;padding:28px;margin-bottom:24px;'>""",
        unsafe_allow_html=True)

    col1,col2,col3,col4 = st.columns([3,1,1,1])
    with col1:
        query = st.text_input("", placeholder="Search by title or channel...",
                              label_visibility="collapsed")
    with col2:
        sel_country = st.selectbox("Country", ["All Countries"]+sorted(df["country"].dropna().unique().tolist()),
                                   label_visibility="collapsed")
    with col3:
        # ALL categories from actual data
        all_cats = sorted(df["category_name"].dropna().unique().tolist())
        sel_cat  = st.selectbox("Category", ["All Categories"]+all_cats,
                                label_visibility="collapsed")
    with col4:
        min_d = df["trending_date"].min()
        sel_date = st.date_input("From date", value=min_d if pd.notna(min_d) else None,
                                 label_visibility="collapsed")

    sr1,sr2 = st.columns([5,1])
    with sr1:
        sort_by = st.radio("Sort:", ["Views","Likes","Comments","Engagement"],
                           horizontal=True)
    with sr2:
        st.markdown("<div style='margin-top:6px;'>", unsafe_allow_html=True)
        search_clicked = st.button("Search", key="search_btn",
                                   help="Click to search")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    sort_map = {"Views": "views", "Likes": "likes",
                "Comments": "comment_count", "Engagement": "engagement_rate"}

    filtered = df.copy()
    if query:
        filtered = filtered[
            filtered["title"].str.contains(query, case=False, na=False) |
            filtered["channel_title"].str.contains(query, case=False, na=False)
        ]
    if sel_country != "All Countries":
        filtered = filtered[filtered["country"] == sel_country]
    if sel_cat != "All Categories":
        filtered = filtered[filtered["category_name"] == sel_cat]
    if sel_date:
        filtered = filtered[filtered["trending_date"] >= pd.Timestamp(sel_date)]
    filtered = filtered.sort_values(sort_map[sort_by], ascending=False)

    show_results = search_clicked or query or sel_country != "All Countries" or sel_cat != "All Categories"

    if show_results:
        if len(filtered) > 0:
            st.markdown(f"""<div style='background:linear-gradient(135deg,#ede9fe,#ddd6fe);
                border:1px solid #6d28d9;border-radius:12px;padding:12px 20px;margin-bottom:20px;'>
                <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='flame' style='width:18px; height:18px; color:#374151;'></i></span> <b style='color:#4338ca;font-size:17px;'>
                Top {min(5,len(filtered))} Results</b>
                <span style='color:#64748b;font-size:13px;margin-left:8px;'>
                — {len(filtered):,} total matches</span></div>""",
                unsafe_allow_html=True)

            top_res = filtered.head(5)
            cols = st.columns(min(5,len(top_res)))
            for i,(col,(_,row)) in enumerate(zip(cols, top_res.iterrows())):
                with col:
                    st.markdown(f"""
                    <div class='video-card'>
                      <div style='position:relative;'>
                        <img src='{row["thumbnail_link"]}' style='width:100%;height:110px;object-fit:cover;'
                             onerror="this.src='https://picsum.photos/seed/{i}99/320/180'">
                        <div style='position:absolute;bottom:0;left:0;right:0;
                            background:linear-gradient(transparent,rgba(255,255,255,0.95));
                            padding:6px 8px;font-size:11px;color:#fcd34d;font-weight:700;'>
                            #{i+1} Trending</div>
                      </div>
                      <div style='padding:10px;'>
                        <div style='font-size:12px;font-weight:600;color:#111827;
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>
                            {row["title"][:38]}...</div>
                        <div style='font-size:11px;color:#6d28d9;margin:4px 0;'>
                            <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='tv' style='width:18px; height:18px; color:#374151;'></i></span> {row["channel_title"]}</div>
                        <div style='font-size:10px;color:#64748b;'>
                            <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='layout-dashboard' style='width:18px; height:18px; color:#374151;'></i></span> {row["category_name"]} • <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span> {row["country"]}</div>
                        <div style='font-size:10px;color:#475569;margin:3px 0 6px;'>
                            <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='calendar' style='width:18px; height:18px; color:#374151;'></i></span> {str(row["trending_date"])[:10]}</div>
                        <div style='background:#f8fafc;border-radius:8px;padding:8px;'>
                          <div style='display:flex;justify-content:space-between;font-size:11px;'>
                            <span style='color:#60a5fa;'><span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='eye' style='width:18px; height:18px; color:#374151;'></i></span> {row["views"]/1e6:.2f}M</span>
                            <span style='color:#4ade80;'><span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='thumbs-up' style='width:18px; height:18px; color:#374151;'></i></span> {row["likes"]/1e3:.1f}K</span>
                          </div>
                          <div style='display:flex;justify-content:space-between;font-size:11px;margin-top:4px;'>
                            <span style='color:#f472b6;'><span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='message-circle' style='width:18px; height:18px; color:#374151;'></i></span> {row["comment_count"]/1e3:.1f}K</span>
                            <span style='color:#fbbf24;'><span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='trending-up' style='width:18px; height:18px; color:#374151;'></i></span> {row["engagement_rate"]:.1f}%</span>
                          </div>
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)

            if len(filtered) > 5:
                st.markdown("<br>##### <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='clipboard-list' style='width:18px; height:18px; color:#374151;'></i> All Results</span>", unsafe_allow_html=True)
                disp_cols = ["title","channel_title","category_name","country",
                             "trending_date","views","likes","comment_count","engagement_rate"]
                st.dataframe(
                    filtered[disp_cols].head(50)
                        .style
                        .background_gradient(subset=["views"], cmap="Blues")
                        .format({"views":"{:,.0f}","likes":"{:,.0f}",
                                 "comment_count":"{:,.0f}","engagement_rate":"{:.2f}%"}),
                    use_container_width=True, height=380
                )
        else:
            st.markdown("""<div style='text-align:center;padding:60px;color:#64748b;'>
                <div style='display:flex; justify-content:center;'><i data-lucide='search' style='width:60px; height:60px; color:#64748b;'></i></div>
                <div style='font-size:18px;margin-top:16px;'>No videos found.</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.markdown("##### <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='flame' style='width:18px; height:18px; color:#374151;'></i> Top 5 Overall</span>", unsafe_allow_html=True)
        top5 = df.nlargest(5,"views")
        cols = st.columns(5)
        for i,(col,(_,row)) in enumerate(zip(cols, top5.iterrows())):
            with col:
                st.markdown(f"""
                <div class='video-card'>
                  <img src='{row["thumbnail_link"]}' style='width:100%;height:110px;object-fit:cover;'
                       onerror="this.src='https://picsum.photos/seed/{i}33/320/180'">
                  <div style='padding:10px;'>
                    <div style='font-size:12px;font-weight:600;color:#111827;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>
                        {row["title"][:38]}...</div>
                    <div style='font-size:11px;color:#6d28d9;margin:4px 0;'>
                        {row["channel_title"]}</div>
                    <div style='font-size:10px;color:#64748b;'>
                        <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='calendar' style='width:18px; height:18px; color:#374151;'></i></span> {str(row["trending_date"])[:10]} • <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span> {row["country"]}</div>
                    <div style='font-size:11px;color:#60a5fa;margin-top:6px;'>
                        <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='eye' style='width:18px; height:18px; color:#374151;'></i></span> {row["views"]/1e6:.1f}M views</div>
                  </div>
                </div>""", unsafe_allow_html=True)

    # Top 5 by genre
    st.markdown("<hr style='border-color:#3a4080;margin:30px 0;'>", unsafe_allow_html=True)
    st.markdown("""<div style='border-left:4px solid #ec4899;padding-left:14px;margin-bottom:16px;'>
      <h2 style='font-family:Outfit;font-weight:700;font-size:22px;margin:0;
          background:linear-gradient(90deg,#ec4899,#f59e0b);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span> Top 5 by Genre — Worldwide</h2></div>""", unsafe_allow_html=True)

    sel_genre = st.selectbox("Pick a Genre:", all_cats)
    genre_top5 = df[df["category_name"]==sel_genre].nlargest(5,"views")
    if len(genre_top5)>0:
        gcols = st.columns(min(5,len(genre_top5)))
        for i,(col,(_,row)) in enumerate(zip(gcols, genre_top5.iterrows())):
            with col:
                st.markdown(f"""
                <div class='video-card'>
                  <img src='{row["thumbnail_link"]}' style='width:100%;height:100px;object-fit:cover;'
                       onerror="this.src='https://picsum.photos/seed/{i}77/320/180'">
                  <div style='padding:10px;'>
                    <div style='font-size:12px;color:#111827;font-weight:600;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>
                        {row["title"][:35]}...</div>
                    <div style='font-size:11px;color:#6d28d9;margin:3px 0;'>
                        {row["channel_title"]}</div>
                    <div style='font-size:10px;color:#64748b;'>
                        <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='calendar' style='width:18px; height:18px; color:#374151;'></i></span> {str(row["trending_date"])[:10]}</div>
                    <div style='display:flex;justify-content:space-between;font-size:11px;
                        color:#475569;margin-top:4px;'>
                      <span><span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span> {row["country"]}</span>
                      <span><span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='eye' style='width:18px; height:18px; color:#374151;'></i></span> {row["views"]/1e6:.1f}M</span>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PAGE 3 — ANALYTICS (PER-CATEGORY)
# ══════════════════════════════════════════════
elif page == "Trending Analytics":
    st.markdown("""<h1 style='font-family:Outfit;font-weight:900;font-size:38px;
        background:linear-gradient(90deg,#7c3aed,#a855f7);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        Trending Analytics</h1>""", unsafe_allow_html=True)

    # ── Global KPIs ──────────────────────────────────────────────────────────
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Avg Views",    f"{df['views'].mean()/1e6:.2f}M")
    k2.metric("Avg Likes",    f"{df['likes'].mean()/1e3:.1f}K")
    k3.metric("Avg Comments", f"{df['comment_count'].mean()/1e3:.1f}K")
    k4.metric("Avg Engage",   f"{df['engagement_rate'].mean():.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Category selector ────────────────────────────────────────────────────
    all_cats_analytics = ["🌐 All Categories"] + sorted(df["category_name"].dropna().unique().tolist())
    sel_analytics_cat = st.selectbox(
        "Select a Category to Analyse:",
        all_cats_analytics,
        key="analytics_cat_selector"
    )

    if sel_analytics_cat == "🌐 All Categories":
        adf = df.copy()
        cat_label = "All Categories"
    else:
        adf = df[df["category_name"] == sel_analytics_cat].copy()
        cat_label = sel_analytics_cat

    if len(adf) == 0:
        st.warning("No data found for this category.")
    else:
        # Per-category KPIs
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#ede9fe,#ddd6fe);
            border:1px solid #5046e5;border-radius:16px;padding:20px 28px;margin:16px 0 24px;'>
          <div style='font-size:13px;color:#6d28d9;letter-spacing:2px;
              text-transform:uppercase;margin-bottom:12px;font-weight:600;'>
            <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='bar-chart-2' style='width:18px; height:18px; color:#374151;'></i></span> {cat_label} — Performance Snapshot
          </div>
          <div style='display:flex;gap:32px;flex-wrap:wrap;'>
            <div><div style='font-size:26px;font-weight:800;color:#4338ca;'>{len(adf):,}</div>
              <div style='font-size:12px;color:#475569;'>Total Videos</div></div>
            <div><div style='font-size:26px;font-weight:800;color:#f9a8d4;'>{adf["views"].mean()/1e6:.2f}M</div>
              <div style='font-size:12px;color:#475569;'>Avg Views</div></div>
            <div><div style='font-size:26px;font-weight:800;color:#059669;'>{adf["likes"].mean()/1e3:.1f}K</div>
              <div style='font-size:12px;color:#475569;'>Avg Likes</div></div>
            <div><div style='font-size:26px;font-weight:800;color:#fcd34d;'>{adf["engagement_rate"].mean():.2f}%</div>
              <div style='font-size:12px;color:#475569;'>Avg Engagement</div></div>
            <div><div style='font-size:26px;font-weight:800;color:#60a5fa;'>{adf["country"].nunique()}</div>
              <div style='font-size:12px;color:#475569;'>Countries</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Virality Score (ML-powered, user-friendly) ──────────────────────
        @st.cache_data
        def compute_virality_score(dataframe):
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import train_test_split
            ml = dataframe.copy()
            le_cat  = LabelEncoder()
            le_ctry = LabelEncoder()
            ml["cat_enc"]     = le_cat.fit_transform(ml["category_name"].astype(str))
            ml["country_enc"] = le_ctry.fit_transform(ml["country"].astype(str))
            ml["viral"]       = (ml["views"] > ml["views"].quantile(0.75)).astype(int)
            feats = ["cat_enc","country_enc","likes","comment_count","engagement_rate"]
            X = ml[feats]; y = ml["viral"]
            X_tr,_,y_tr,_ = train_test_split(X,y,test_size=0.2,random_state=42)
            clf = RandomForestClassifier(n_estimators=100,random_state=42)
            clf.fit(X_tr,y_tr)
            ml["virality_score"] = (clf.predict_proba(X)[:,1]*100).round(1)
            return ml

        scored_df = compute_virality_score(df)
        if sel_analytics_cat == "🌐 All Categories":
            scored_adf = scored_df.copy()
        else:
            scored_adf = scored_df[scored_df["category_name"] == sel_analytics_cat].copy()

        avg_virality = scored_adf["virality_score"].mean()
        heat_color   = "#10b981" if avg_virality >= 60 else "#f59e0b" if avg_virality >= 40 else "#ef4444"

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#ede9fe,#e0e7ff);
            border:1px solid {heat_color}55;border-radius:16px;padding:20px 28px;
            margin-bottom:24px;display:flex;align-items:center;gap:28px;'>
          <div style='text-align:center;'>
            <div style='font-size:48px;font-weight:900;color:{heat_color};'>
                {avg_virality:.0f}</div>
            <div style='font-size:12px;color:#475569;'>/ 100</div>
            <div style='font-size:13px;font-weight:600;color:{heat_color};margin-top:4px;'>
                Virality Score</div>
          </div>
          <div>
            <div style='font-size:18px;font-weight:700;color:#111827;'>
                {"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='flame' style='width:18px; height:18px; color:#374151;'></i></span> Hot Category — High Viral Potential" if avg_virality>=60
                 else "<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='zap' style='width:18px; height:18px; color:#374151;'></i></span> Moderate Reach — Growing Audience" if avg_virality>=40
                 else "<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='snowflake' style='width:18px; height:18px; color:#374151;'></i></span> Niche Category — Lower Trending Rate"}
            </div>
            <div style='font-size:13px;color:#475569;margin-top:6px;max-width:500px;'>
                Based on {len(scored_adf):,} videos analysed across views, likes, comments,
                and engagement patterns. Higher score = more likely to trend globally.
            </div>
            <div style='background:#7c3aed;border-radius:6px;height:8px;width:300px;margin-top:12px;'>
              <div style='background:linear-gradient(90deg,{heat_color}88,{heat_color});
                  height:100%;width:{avg_virality}%;border-radius:6px;'></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Charts row 1 ─────────────────────────────────────────────────────
        if sel_analytics_cat == "🌐 All Categories":
            st.markdown("#### <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='layout-dashboard' style='width:18px; height:18px; color:#374151;'></i> Views by Category</span>", unsafe_allow_html=True)
            cat_v = adf.groupby("category_name")["views"].sum().nlargest(12).reset_index()
            fig2  = px.pie(cat_v, values="views", names="category_name",
                           color_discrete_sequence=px.colors.sequential.Purples_r, hole=0.45)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#111827",
                               height=300,
                               legend=dict(font_size=10),margin=dict(t=10,b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown(f"#### <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i> Views by Country — {{sel_analytics_cat}}</span>", unsafe_allow_html=True)
            ctry_v = adf.groupby("country")["views"].sum().nlargest(10).reset_index()
            fig2   = px.pie(ctry_v, values="views", names="country",
                            color_discrete_sequence=px.colors.sequential.Purples_r, hole=0.45)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#111827",
                               height=300,
                               legend=dict(font_size=10),margin=dict(t=10,b=10))
            st.plotly_chart(fig2, use_container_width=True)

        # ── Scatter ──────────────────────────────────────────────────────────
        st.markdown("#### <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='trending-down' style='width:18px; height:18px; color:#374151;'></i> Views vs Likes Correlation</span>", unsafe_allow_html=True)
        sample = adf.sample(min(500,len(adf)))
        color_col = "category_name" if sel_analytics_cat == "🌐 All Categories" else "country"
        fig3 = px.scatter(sample, x="views", y="likes", color=color_col,
                          size="comment_count",
                          hover_data=["title","channel_title","country","trending_date"],
                          color_discrete_sequence=px.colors.qualitative.Bold)
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#111827",height=320,
                           xaxis=dict(gridcolor="#6600cc"),yaxis=dict(gridcolor="#6600cc"),
                           margin=dict(t=10,b=10))
        st.plotly_chart(fig3, use_container_width=True)


        # ── Engagement health meter ───────────────────────────────────────────
        st.markdown("#### 💡 Content Health Insights")
        avg_eng    = adf["engagement_rate"].mean()
        avg_views  = adf["views"].mean()
        peak_day   = (adf.dropna(subset=["trending_date"])
                        .groupby(adf["trending_date"].dt.day_name())["views"].mean()
                        .idxmax() if adf["trending_date"].notna().sum()>5 else "N/A")
        top_ch     = adf.groupby("channel_title")["views"].sum().idxmax() if len(adf)>0 else "N/A"

        ins1,ins2,ins3 = st.columns(3)
        for col,icon,title,val,tip in [
            (ins1,"🎯","Best Peak Day",peak_day,
             "Most views land on this day — plan uploads accordingly"),
            (ins2,"🏅","Top Performing Channel",str(top_ch)[:22],
             "This channel dominates this category's views"),
            (ins3,"🌍","Engagement vs Global",
             f"{avg_eng:.2f}% {'↑ Above' if avg_eng>df['engagement_rate'].mean() else '↓ Below'} avg",
             "Compared to the global dataset average engagement rate"),
        ]:
            with col:
                st.markdown(f"""
                <div style='background:#ffffff;border:1.5px solid #e2e8f0;border-radius:14px;
                    padding:18px;text-align:center;'>
                  <div style='font-size:30px;'>{icon}</div>
                  <div style='font-size:12px;color:#475569;margin:6px 0 4px;'>{title}</div>
                  <div style='font-size:16px;font-weight:700;color:#4338ca;'>{val}</div>
                  <div style='font-size:11px;color:#64748b;margin-top:6px;'>{tip}</div>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PAGE 4 — WORLD MAP (Google Maps–style pins)
# ══════════════════════════════════════════════
elif page == "World Map":
    st.markdown("""<h1 style='font-family:Outfit;font-weight:900;font-size:38px;
        background:linear-gradient(90deg,#06b6d4,#7c3aed);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span> Trending Videos World Map</h1>
        <p style='color:#475569;margin-bottom:20px;'>
        Hover over any country pin — see top category, top channel, views and engagement.</p>
    """, unsafe_allow_html=True)

    metric_choice = st.radio("Colour pins by:",
        ["Total Views","Video Count","Avg Engagement Rate"], horizontal=True)

    # ── Master lookup: handles BOTH full names AND 2-letter codes ─────────────
    # Format: key → (lat, lon, iso3, display_name)
    COUNTRY_MASTER = {
        # ── Every country name format your Kaggle dataset might use ──────────
        # Covers: 2-letter ISO codes, full English names, common variants
        # Format: key → (lat, lon, ISO3, display_name)

        # A
        "AF":(33.93,67.71,"AFG","Afghanistan"),
        "Afghanistan":(33.93,67.71,"AFG","Afghanistan"),
        "AL":(41.15,20.17,"ALB","Albania"),
        "Albania":(41.15,20.17,"ALB","Albania"),
        "DZ":(28.03,1.66,"DZA","Algeria"),
        "Algeria":(28.03,1.66,"DZA","Algeria"),
        "AO":(-11.20,17.87,"AGO","Angola"),
        "Angola":(-11.20,17.87,"AGO","Angola"),
        "AR":(-38.41,-63.61,"ARG","Argentina"),
        "Argentina":(-38.41,-63.61,"ARG","Argentina"),
        "AM":(40.07,45.04,"ARM","Armenia"),
        "Armenia":(40.07,45.04,"ARM","Armenia"),
        "AU":(-25.27,133.77,"AUS","Australia"),
        "Australia":(-25.27,133.77,"AUS","Australia"),
        "AT":(47.51,14.55,"AUT","Austria"),
        "Austria":(47.51,14.55,"AUT","Austria"),
        "AZ":(40.14,47.58,"AZE","Azerbaijan"),
        "Azerbaijan":(40.14,47.58,"AZE","Azerbaijan"),

        # B
        "BD":(23.68,90.35,"BGD","Bangladesh"),
        "Bangladesh":(23.68,90.35,"BGD","Bangladesh"),
        "BY":(53.71,27.95,"BLR","Belarus"),
        "Belarus":(53.71,27.95,"BLR","Belarus"),
        "BE":(50.50,4.46,"BEL","Belgium"),
        "Belgium":(50.50,4.46,"BEL","Belgium"),
        "BJ":(9.31,2.32,"BEN","Benin"),
        "Benin":(9.31,2.32,"BEN","Benin"),
        "BO":(-16.29,-63.59,"BOL","Bolivia"),
        "Bolivia":(-16.29,-63.59,"BOL","Bolivia"),
        "BA":(43.91,17.68,"BIH","Bosnia and Herzegovina"),
        "Bosnia and Herzegovina":(43.91,17.68,"BIH","Bosnia and Herzegovina"),
        "Bosnia":(43.91,17.68,"BIH","Bosnia and Herzegovina"),
        "BW":(-22.32,24.68,"BWA","Botswana"),
        "Botswana":(-22.32,24.68,"BWA","Botswana"),
        "BR":(-14.23,-51.92,"BRA","Brazil"),
        "Brazil":(-14.23,-51.92,"BRA","Brazil"),
        "BN":(4.53,114.73,"BRN","Brunei"),
        "Brunei":(4.53,114.73,"BRN","Brunei"),
        "BG":(42.73,25.49,"BGR","Bulgaria"),
        "Bulgaria":(42.73,25.49,"BGR","Bulgaria"),
        "BF":(12.36,-1.53,"BFA","Burkina Faso"),
        "Burkina Faso":(12.36,-1.53,"BFA","Burkina Faso"),

        # C
        "KH":(12.56,104.99,"KHM","Cambodia"),
        "Cambodia":(12.56,104.99,"KHM","Cambodia"),
        "CM":(3.84,11.50,"CMR","Cameroon"),
        "Cameroon":(3.84,11.50,"CMR","Cameroon"),
        "CA":(56.13,-106.34,"CAN","Canada"),
        "Canada":(56.13,-106.34,"CAN","Canada"),
        "CF":(6.61,20.94,"CAF","Central African Republic"),
        "Central African Republic":(6.61,20.94,"CAF","Central African Republic"),
        "TD":(15.45,18.73,"TCD","Chad"),
        "Chad":(15.45,18.73,"TCD","Chad"),
        "CL":(-35.67,-71.54,"CHL","Chile"),
        "Chile":(-35.67,-71.54,"CHL","Chile"),
        "CN":(35.86,104.19,"CHN","China"),
        "China":(35.86,104.19,"CHN","China"),
        "CO":(4.57,-74.29,"COL","Colombia"),
        "Colombia":(4.57,-74.29,"COL","Colombia"),
        "CD":(-4.03,21.75,"COD","Congo DR"),
        "Congo DR":(-4.03,21.75,"COD","Congo DR"),
        "Democratic Republic of the Congo":(-4.03,21.75,"COD","Congo DR"),
        "CG":(-0.23,15.83,"COG","Congo Republic"),
        "Congo Republic":(-0.23,15.83,"COG","Congo Republic"),
        "CR":(9.75,-83.75,"CRI","Costa Rica"),
        "Costa Rica":(9.75,-83.75,"CRI","Costa Rica"),
        "CI":(7.54,-5.55,"CIV","Cote d'Ivoire"),
        "Cote d'Ivoire":(7.54,-5.55,"CIV","Cote d'Ivoire"),
        "Ivory Coast":(7.54,-5.55,"CIV","Cote d'Ivoire"),
        "HR":(45.10,15.20,"HRV","Croatia"),
        "Croatia":(45.10,15.20,"HRV","Croatia"),
        "CU":(21.52,-77.78,"CUB","Cuba"),
        "Cuba":(21.52,-77.78,"CUB","Cuba"),
        "CY":(35.13,33.43,"CYP","Cyprus"),
        "Cyprus":(35.13,33.43,"CYP","Cyprus"),
        "CZ":(49.82,15.47,"CZE","Czech Republic"),
        "Czech Republic":(49.82,15.47,"CZE","Czech Republic"),
        "Czechia":(49.82,15.47,"CZE","Czech Republic"),

        # D
        "DK":(56.26,9.50,"DNK","Denmark"),
        "Denmark":(56.26,9.50,"DNK","Denmark"),
        "DO":(18.74,-70.16,"DOM","Dominican Republic"),
        "Dominican Republic":(18.74,-70.16,"DOM","Dominican Republic"),

        # E
        "EC":(-1.83,-78.18,"ECU","Ecuador"),
        "Ecuador":(-1.83,-78.18,"ECU","Ecuador"),
        "EG":(26.82,30.80,"EGY","Egypt"),
        "Egypt":(26.82,30.80,"EGY","Egypt"),
        "SV":(13.79,-88.89,"SLV","El Salvador"),
        "El Salvador":(13.79,-88.89,"SLV","El Salvador"),
        "ET":(9.14,40.49,"ETH","Ethiopia"),
        "Ethiopia":(9.14,40.49,"ETH","Ethiopia"),

        # F
        "FI":(61.92,25.74,"FIN","Finland"),
        "Finland":(61.92,25.74,"FIN","Finland"),
        "FR":(46.22,2.21,"FRA","France"),
        "France":(46.22,2.21,"FRA","France"),

        # G
        "GE":(42.31,43.36,"GEO","Georgia"),
        "Georgia":(42.31,43.36,"GEO","Georgia"),
        "DE":(51.16,10.45,"DEU","Germany"),
        "Germany":(51.16,10.45,"DEU","Germany"),
        "GH":(7.95,-1.02,"GHA","Ghana"),
        "Ghana":(7.95,-1.02,"GHA","Ghana"),
        "GR":(39.07,21.82,"GRC","Greece"),
        "Greece":(39.07,21.82,"GRC","Greece"),
        "GT":(15.78,-90.23,"GTM","Guatemala"),
        "Guatemala":(15.78,-90.23,"GTM","Guatemala"),
        "GN":(11.00,-10.90,"GIN","Guinea"),
        "Guinea":(11.00,-10.90,"GIN","Guinea"),

        # H
        "HT":(18.97,-72.29,"HTI","Haiti"),
        "Haiti":(18.97,-72.29,"HTI","Haiti"),
        "HN":(15.20,-86.24,"HND","Honduras"),
        "Honduras":(15.20,-86.24,"HND","Honduras"),
        "HK":(22.39,114.10,"HKG","Hong Kong"),
        "Hong Kong":(22.39,114.10,"HKG","Hong Kong"),
        "HU":(47.16,19.50,"HUN","Hungary"),
        "Hungary":(47.16,19.50,"HUN","Hungary"),

        # I
        "IS":(64.96,-19.02,"ISL","Iceland"),
        "Iceland":(64.96,-19.02,"ISL","Iceland"),
        "IN":(20.59,78.96,"IND","India"),
        "India":(20.59,78.96,"IND","India"),
        "ID":(-0.78,113.92,"IDN","Indonesia"),
        "Indonesia":(-0.78,113.92,"IDN","Indonesia"),
        "IR":(32.43,53.69,"IRN","Iran"),
        "Iran":(32.43,53.69,"IRN","Iran"),
        "IQ":(33.22,43.68,"IRQ","Iraq"),
        "Iraq":(33.22,43.68,"IRQ","Iraq"),
        "IE":(53.41,-8.24,"IRL","Ireland"),
        "Ireland":(53.41,-8.24,"IRL","Ireland"),
        "IL":(31.04,34.85,"ISR","Israel"),
        "Israel":(31.04,34.85,"ISR","Israel"),
        "IT":(41.87,12.56,"ITA","Italy"),
        "Italy":(41.87,12.56,"ITA","Italy"),

        # J
        "JM":(18.10,-77.30,"JAM","Jamaica"),
        "Jamaica":(18.10,-77.30,"JAM","Jamaica"),
        "JP":(36.20,138.25,"JPN","Japan"),
        "Japan":(36.20,138.25,"JPN","Japan"),
        "JO":(30.59,36.24,"JOR","Jordan"),
        "Jordan":(30.59,36.24,"JOR","Jordan"),

        # K
        "KZ":(48.02,66.92,"KAZ","Kazakhstan"),
        "Kazakhstan":(48.02,66.92,"KAZ","Kazakhstan"),
        "KE":(0.02,37.91,"KEN","Kenya"),
        "Kenya":(0.02,37.91,"KEN","Kenya"),
        "KW":(29.31,47.48,"KWT","Kuwait"),
        "Kuwait":(29.31,47.48,"KWT","Kuwait"),
        "KG":(41.20,74.76,"KGZ","Kyrgyzstan"),
        "Kyrgyzstan":(41.20,74.76,"KGZ","Kyrgyzstan"),

        # L
        "LA":(19.86,102.49,"LAO","Laos"),
        "Laos":(19.86,102.49,"LAO","Laos"),
        "LV":(56.88,24.60,"LVA","Latvia"),
        "Latvia":(56.88,24.60,"LVA","Latvia"),
        "LB":(33.85,35.86,"LBN","Lebanon"),
        "Lebanon":(33.85,35.86,"LBN","Lebanon"),
        "LY":(26.34,17.23,"LBY","Libya"),
        "Libya":(26.34,17.23,"LBY","Libya"),
        "LT":(55.17,23.88,"LTU","Lithuania"),
        "Lithuania":(55.17,23.88,"LTU","Lithuania"),
        "LU":(49.81,6.13,"LUX","Luxembourg"),
        "Luxembourg":(49.81,6.13,"LUX","Luxembourg"),

        # M
        "MK":(41.61,21.74,"MKD","North Macedonia"),
        "North Macedonia":(41.61,21.74,"MKD","North Macedonia"),
        "Macedonia":(41.61,21.74,"MKD","North Macedonia"),
        "MG":(-18.77,46.87,"MDG","Madagascar"),
        "Madagascar":(-18.77,46.87,"MDG","Madagascar"),
        "MW":(-13.25,34.30,"MWI","Malawi"),
        "Malawi":(-13.25,34.30,"MWI","Malawi"),
        "MY":(4.21,108.96,"MYS","Malaysia"),
        "Malaysia":(4.21,108.96,"MYS","Malaysia"),
        "ML":(17.57,-3.99,"MLI","Mali"),
        "Mali":(17.57,-3.99,"MLI","Mali"),
        "MT":(35.94,14.38,"MLT","Malta"),
        "Malta":(35.94,14.38,"MLT","Malta"),
        "MR":(21.01,-10.94,"MRT","Mauritania"),
        "Mauritania":(21.01,-10.94,"MRT","Mauritania"),
        "MX":(23.63,-102.55,"MEX","Mexico"),
        "Mexico":(23.63,-102.55,"MEX","Mexico"),
        "MD":(47.41,28.37,"MDA","Moldova"),
        "Moldova":(47.41,28.37,"MDA","Moldova"),
        "MN":(46.86,103.85,"MNG","Mongolia"),
        "Mongolia":(46.86,103.85,"MNG","Mongolia"),
        "ME":(42.71,19.37,"MNE","Montenegro"),
        "Montenegro":(42.71,19.37,"MNE","Montenegro"),
        "MA":(31.79,-7.09,"MAR","Morocco"),
        "Morocco":(31.79,-7.09,"MAR","Morocco"),
        "MZ":(-18.67,35.53,"MOZ","Mozambique"),
        "Mozambique":(-18.67,35.53,"MOZ","Mozambique"),
        "MM":(21.91,95.96,"MMR","Myanmar"),
        "Myanmar":(21.91,95.96,"MMR","Myanmar"),
        "Burma":(21.91,95.96,"MMR","Myanmar"),

        # N
        "NP":(28.39,84.12,"NPL","Nepal"),
        "Nepal":(28.39,84.12,"NPL","Nepal"),
        "NL":(52.13,5.29,"NLD","Netherlands"),
        "Netherlands":(52.13,5.29,"NLD","Netherlands"),
        "NZ":(-40.90,174.88,"NZL","New Zealand"),
        "New Zealand":(-40.90,174.88,"NZL","New Zealand"),
        "NI":(12.87,-85.21,"NIC","Nicaragua"),
        "Nicaragua":(12.87,-85.21,"NIC","Nicaragua"),
        "NE":(17.61,8.08,"NER","Niger"),
        "Niger":(17.61,8.08,"NER","Niger"),
        "NG":(9.08,8.67,"NGA","Nigeria"),
        "Nigeria":(9.08,8.67,"NGA","Nigeria"),
        "NO":(60.47,8.47,"NOR","Norway"),
        "Norway":(60.47,8.47,"NOR","Norway"),

        # O
        "OM":(21.51,55.92,"OMN","Oman"),
        "Oman":(21.51,55.92,"OMN","Oman"),

        # P
        "PK":(30.37,69.34,"PAK","Pakistan"),
        "Pakistan":(30.37,69.34,"PAK","Pakistan"),
        "PS":(31.95,35.23,"PSE","Palestine"),
        "Palestine":(31.95,35.23,"PSE","Palestine"),
        "PA":(8.54,-80.78,"PAN","Panama"),
        "Panama":(8.54,-80.78,"PAN","Panama"),
        "PY":(-23.44,-58.44,"PRY","Paraguay"),
        "Paraguay":(-23.44,-58.44,"PRY","Paraguay"),
        "PE":(-9.18,-75.01,"PER","Peru"),
        "Peru":(-9.18,-75.01,"PER","Peru"),
        "PH":(12.87,121.77,"PHL","Philippines"),
        "Philippines":(12.87,121.77,"PHL","Philippines"),
        "PL":(51.92,19.14,"POL","Poland"),
        "Poland":(51.92,19.14,"POL","Poland"),
        "PT":(39.39,-8.22,"PRT","Portugal"),
        "Portugal":(39.39,-8.22,"PRT","Portugal"),

        # Q
        "QA":(25.35,51.18,"QAT","Qatar"),
        "Qatar":(25.35,51.18,"QAT","Qatar"),

        # R
        "RO":(45.94,24.96,"ROU","Romania"),
        "Romania":(45.94,24.96,"ROU","Romania"),
        "RU":(61.52,105.31,"RUS","Russia"),
        "Russia":(61.52,105.31,"RUS","Russia"),
        "RW":(-1.94,29.87,"RWA","Rwanda"),
        "Rwanda":(-1.94,29.87,"RWA","Rwanda"),

        # S
        "SA":(23.88,45.07,"SAU","Saudi Arabia"),
        "Saudi Arabia":(23.88,45.07,"SAU","Saudi Arabia"),
        "SN":(14.50,-14.45,"SEN","Senegal"),
        "Senegal":(14.50,-14.45,"SEN","Senegal"),
        "RS":(44.02,21.01,"SRB","Serbia"),
        "Serbia":(44.02,21.01,"SRB","Serbia"),
        "SL":(8.46,-11.78,"SLE","Sierra Leone"),
        "Sierra Leone":(8.46,-11.78,"SLE","Sierra Leone"),
        "SG":(1.35,103.81,"SGP","Singapore"),
        "Singapore":(1.35,103.81,"SGP","Singapore"),
        "SK":(48.67,19.70,"SVK","Slovakia"),
        "Slovakia":(48.67,19.70,"SVK","Slovakia"),
        "SI":(46.15,14.99,"SVN","Slovenia"),
        "Slovenia":(46.15,14.99,"SVN","Slovenia"),
        "SO":(5.15,46.20,"SOM","Somalia"),
        "Somalia":(5.15,46.20,"SOM","Somalia"),
        "ZA":(-30.55,22.93,"ZAF","South Africa"),
        "South Africa":(-30.55,22.93,"ZAF","South Africa"),
        "SS":(6.87,31.31,"SSD","South Sudan"),
        "South Sudan":(6.87,31.31,"SSD","South Sudan"),
        "ES":(40.46,-3.74,"ESP","Spain"),
        "Spain":(40.46,-3.74,"ESP","Spain"),
        "LK":(7.87,80.77,"LKA","Sri Lanka"),
        "Sri Lanka":(7.87,80.77,"LKA","Sri Lanka"),
        "SD":(12.86,30.22,"SDN","Sudan"),
        "Sudan":(12.86,30.22,"SDN","Sudan"),
        "SE":(60.12,18.64,"SWE","Sweden"),
        "Sweden":(60.12,18.64,"SWE","Sweden"),
        "CH":(46.81,8.22,"CHE","Switzerland"),
        "Switzerland":(46.81,8.22,"CHE","Switzerland"),
        "SY":(34.80,38.99,"SYR","Syria"),
        "Syria":(34.80,38.99,"SYR","Syria"),

        # T
        "TW":(23.70,120.96,"TWN","Taiwan"),
        "Taiwan":(23.70,120.96,"TWN","Taiwan"),
        "TJ":(38.86,71.27,"TJK","Tajikistan"),
        "Tajikistan":(38.86,71.27,"TJK","Tajikistan"),
        "TZ":(-6.37,34.89,"TZA","Tanzania"),
        "Tanzania":(-6.37,34.89,"TZA","Tanzania"),
        "TH":(15.87,100.99,"THA","Thailand"),
        "Thailand":(15.87,100.99,"THA","Thailand"),
        "TG":(8.62,0.82,"TGO","Togo"),
        "Togo":(8.62,0.82,"TGO","Togo"),
        "TT":(10.69,-61.22,"TTO","Trinidad and Tobago"),
        "Trinidad and Tobago":(10.69,-61.22,"TTO","Trinidad and Tobago"),
        "TN":(33.89,9.54,"TUN","Tunisia"),
        "Tunisia":(33.89,9.54,"TUN","Tunisia"),
        "TR":(38.96,35.24,"TUR","Turkey"),
        "Turkey":(38.96,35.24,"TUR","Turkey"),
        "Turkiye":(38.96,35.24,"TUR","Turkey"),
        "TM":(38.97,59.56,"TKM","Turkmenistan"),
        "Turkmenistan":(38.97,59.56,"TKM","Turkmenistan"),

        # U
        "UG":(1.37,32.29,"UGA","Uganda"),
        "Uganda":(1.37,32.29,"UGA","Uganda"),
        "UA":(48.37,31.16,"UKR","Ukraine"),
        "Ukraine":(48.37,31.16,"UKR","Ukraine"),
        "AE":(23.42,53.84,"ARE","UAE"),
        "UAE":(23.42,53.84,"ARE","UAE"),
        "United Arab Emirates":(23.42,53.84,"ARE","UAE"),
        "GB":(55.37,-3.43,"GBR","United Kingdom"),
        "UK":(55.37,-3.43,"GBR","United Kingdom"),
        "United Kingdom":(55.37,-3.43,"GBR","United Kingdom"),
        "US":(37.09,-95.71,"USA","United States"),
        "USA":(37.09,-95.71,"USA","United States"),
        "United States":(37.09,-95.71,"USA","United States"),
        "UY":(-32.52,-55.77,"URY","Uruguay"),
        "Uruguay":(-32.52,-55.77,"URY","Uruguay"),
        "UZ":(41.38,64.59,"UZB","Uzbekistan"),
        "Uzbekistan":(41.38,64.59,"UZB","Uzbekistan"),

        # V
        "VE":(6.42,-66.59,"VEN","Venezuela"),
        "Venezuela":(6.42,-66.59,"VEN","Venezuela"),
        "VN":(14.05,108.27,"VNM","Vietnam"),
        "Vietnam":(14.05,108.27,"VNM","Vietnam"),
        "Viet Nam":(14.05,108.27,"VNM","Vietnam"),

        # Y
        "YE":(15.55,48.52,"YEM","Yemen"),
        "Yemen":(15.55,48.52,"YEM","Yemen"),

        # Z
        "ZM":(-13.13,27.85,"ZMB","Zambia"),
        "Zambia":(-13.13,27.85,"ZMB","Zambia"),
        "ZW":(-19.02,29.15,"ZWE","Zimbabwe"),
        "Zimbabwe":(-19.02,29.15,"ZWE","Zimbabwe"),

        # Extra common variants
        "KR":(35.90,127.76,"KOR","South Korea"),
        "Korea":(35.90,127.76,"KOR","South Korea"),
        "South Korea":(35.90,127.76,"KOR","South Korea"),
        "Republic of Korea":(35.90,127.76,"KOR","South Korea"),
        "KP":(40.34,127.51,"PRK","North Korea"),
        "North Korea":(40.34,127.51,"PRK","North Korea"),
        "TZ":(-6.37,34.89,"TZA","Tanzania"),
        "Tanzania, United Republic of":(-6.37,34.89,"TZA","Tanzania"),
        "Bolivia (Plurinational State of)":(-16.29,-63.59,"BOL","Bolivia"),
        "Venezuela (Bolivarian Republic of)":(6.42,-66.59,"VEN","Venezuela"),
        "Iran (Islamic Republic of)":(32.43,53.69,"IRN","Iran"),
        "Syrian Arab Republic":(34.80,38.99,"SYR","Syria"),
        "Lao PDR":(19.86,102.49,"LAO","Laos"),
        "Viet Nam":(14.05,108.27,"VNM","Vietnam"),
        "Republic of Moldova":(47.41,28.37,"MDA","Moldova"),
        "Russian Federation":(61.52,105.31,"RUS","Russia"),
        "Democratic People's Republic of Korea":(40.34,127.51,"PRK","North Korea"),
        "United States of America":(37.09,-95.71,"USA","United States"),
        "United Republic of Tanzania":(-6.37,34.89,"TZA","Tanzania"),
    }

    # ── Build per-country stats ───────────────────────────────────────────────
    country_stats = df.groupby("country").agg(
        total_views    =("views","sum"),
        video_count    =("title","count"),
        avg_engagement =("engagement_rate","mean"),
        avg_views      =("views","mean"),
        avg_likes      =("likes","mean"),
        avg_comments   =("comment_count","mean"),
    ).reset_index()

    top_cat = (df.groupby(["country","category_name"])["views"].sum().reset_index()
               .sort_values("views",ascending=False).drop_duplicates("country")
               [["country","category_name"]].rename(columns={"category_name":"top_category"}))
    top_chan = (df.groupby(["country","channel_title"])["views"].sum().reset_index()
                .sort_values("views",ascending=False).drop_duplicates("country")
                [["country","channel_title"]].rename(columns={"channel_title":"top_channel"}))

    cat_breakdown = {}
    for ctry, grp in df.groupby("country"):
        cat_sum = grp.groupby("category_name")["views"].sum().nlargest(3)
        total   = cat_sum.sum()
        cat_breakdown[ctry] = " | ".join([f"{c} ({v/total*100:.0f}%)" for c,v in cat_sum.items()])

    country_stats = country_stats.merge(top_cat,  on="country", how="left")
    country_stats = country_stats.merge(top_chan,  on="country", how="left")
    country_stats["cat_breakdown"]  = country_stats["country"].map(cat_breakdown).fillna("—")
    country_stats["avg_engagement"] = country_stats["avg_engagement"].round(2)
    country_stats["avg_views"]      = country_stats["avg_views"].round(0).astype(int)
    country_stats["avg_likes"]      = country_stats["avg_likes"].round(0).astype(int)
    country_stats["avg_comments"]   = country_stats["avg_comments"].round(0).astype(int)

    # Map each country → coordinates (try exact, then title-case, then strip)
    def get_coords(c):
        c = str(c).strip()
        return (COUNTRY_MASTER.get(c) or
                COUNTRY_MASTER.get(c.title()) or
                COUNTRY_MASTER.get(c.upper()) or
                None)

    country_stats["_coords"] = country_stats["country"].map(get_coords)
    country_stats["lat"]     = country_stats["_coords"].map(lambda x: x[0] if x else None)
    country_stats["lon"]     = country_stats["_coords"].map(lambda x: x[1] if x else None)
    country_stats["iso3"]    = country_stats["_coords"].map(lambda x: x[2] if x else None)
    country_stats["disp"]    = country_stats["_coords"].map(lambda x: x[3] if x else country_stats["country"])

    mcol = {"Total Views":"total_views","Video Count":"video_count",
            "Avg Engagement Rate":"avg_engagement"}[metric_choice]

    # Debug: show what was found / not found
    pins     = country_stats[country_stats["lat"].notna()].copy().reset_index(drop=True)
    no_match = country_stats[country_stats["lat"].isna()]["country"].tolist()

    if len(pins) == 0:
        st.error("❌ No country coordinates matched. Please check what format your country column uses — full names like 'India' or codes like 'IN'.")
        st.write("Countries in your dataset:", df["country"].dropna().unique().tolist()[:20])
    else:
        max_v = float(pins[mcol].max()) if float(pins[mcol].max()) > 0 else 1.0
        min_v = float(pins[mcol].min())

        norm     = [(float(v)-min_v)/(max_v-min_v+1e-9) for v in pins[mcol]]
        sizes    = [int(16 + n*26) for n in norm]

        def interp_color(t):
            r = int(67  + t*(236-67))
            g = int(56  + t*(72 -56))
            b = int(237 + t*(153-237))
            return f"rgb({r},{g},{b})"
        colors = [interp_color(n) for n in norm]

        hover_texts = []
        for _, row in pins.iterrows():
            hover_texts.append(
                f"<b>📍 {row['country']}</b><br>"
                f"─────────────────────<br>"
                f"🥇 Top Category: <b>{row['top_category']}</b><br>"
                f"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='tv' style='width:18px; height:18px; color:#374151;'></i></span> Top Channel: <b>{row['top_channel']}</b><br>"
                f"─────────────────────<br>"
                f"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='eye' style='width:18px; height:18px; color:#374151;'></i></span> Total Views: <b>{int(row['total_views']):,}</b><br>"
                f"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='video' style='width:18px; height:18px; color:#374151;'></i></span> Videos: <b>{int(row['video_count'])}</b><br>"
                f"Avg Views: <b>{int(row['avg_views']):,}</b><br>"
                f"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='thumbs-up' style='width:18px; height:18px; color:#374151;'></i></span> Avg Likes: <b>{int(row['avg_likes']):,}</b><br>"
                f"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='message-circle' style='width:18px; height:18px; color:#374151;'></i></span> Avg Comments: <b>{int(row['avg_comments']):,}</b><br>"
                f"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='trending-up' style='width:18px; height:18px; color:#374151;'></i></span> Engagement: <b>{row['avg_engagement']:.2f}%</b><br>"
                f"─────────────────────<br>"
                f"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='layout-dashboard' style='width:18px; height:18px; color:#374151;'></i></span> {row['cat_breakdown']}"
            )

        fig = go.Figure()

        # Choropleth fill layer
        choro = pins.dropna(subset=["iso3"])
        choro = choro[choro["iso3"] != ""]
        if len(choro) > 0:
            fig.add_trace(go.Choropleth(
                locations=choro["iso3"].tolist(),
                z=choro[mcol].tolist(),
                colorscale=[[0,"#4400bb"],[0.3,"#06b6d4"],[0.6,"#a855f7"],[1,"#ec4899"]],
                showscale=False,
                marker=dict(line=dict(color="#3a4080", width=0.5)),
                hoverinfo="skip",
                name="",
            ))

        # Scatter pin layer
        fig.add_trace(go.Scattergeo(
            lat=pins["lat"].tolist(),
            lon=pins["lon"].tolist(),
            mode="markers+text",
            text=pins["country"].tolist(),
            textposition="top center",
            textfont=dict(size=10, color="#111827", family="Arial"),
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=0, color="#111827"),
                opacity=1.0,
            ),
            hovertext=hover_texts,
            hoverinfo="text",
            hoverlabel=dict(
                bgcolor="#ffffff",
                bordercolor="#e2e8f0",
                font=dict(size=13, color="#111827", family="Arial"),
            ),
            name="",
        ))

        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True, coastlinecolor="#94a3b8",
                showland=True,       landcolor="#e2e8f0",
                showocean=True,      oceancolor="#f8fafc",
                showcountries=True,  countrycolor="#cbd5e1",
                bgcolor="#ffffff",
                projection_type="natural earth",
                showlakes=False,
            ),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            margin=dict(t=0, b=0, l=0, r=0),
            height=550,
            showlegend=False,
            font=dict(color="#b0b8d4"),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div style='display:flex;justify-content:center;align-items:center;gap:16px;
            margin:6px 0 20px;flex-wrap:wrap;'>
          <span style='font-size:12px;color:#64748b;'>Pin size & colour = {metric_choice}</span>
          <span style='font-size:11px;color:#475569;'>Low</span>
          <div style='width:80px;height:6px;border-radius:3px;
              background:linear-gradient(90deg,#5046e5,#7c3aed,#ec4899);'></div>
          <span style='font-size:11px;color:#475569;'>High</span>
          <span style='font-size:12px;color:#64748b;'>({len(pins)} countries mapped)</span>
        </div>""", unsafe_allow_html=True)

    # ── Category drilldown ────────────────────────────────────────────────────
    st.markdown("<hr style='border-color:#3a4080;margin:20px 0;'>", unsafe_allow_html=True)
    st.markdown("#### <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='search' style='width:18px; height:18px; color:#374151;'></i></span> Category Performance by Country", unsafe_allow_html=True)
    map_cat_filter = st.selectbox(
        "Filter by category:",
        ["All Categories"] + sorted(df["category_name"].dropna().unique().tolist()),
        key="map_cat_filter"
    )
    if map_cat_filter != "All Categories":
        cat_country = (df[df["category_name"]==map_cat_filter]
                       .groupby("country").agg(
                           total_views=("views","sum"),
                           video_count=("title","count"),
                           avg_engagement=("engagement_rate","mean"),
                           avg_views=("views","mean"),
                       ).reset_index().sort_values("total_views",ascending=False))

        max_views = cat_country["total_views"].max() if len(cat_country) > 0 else 1
        VIBRANT_GRADIENTS = [
            "linear-gradient(135deg,#7c3aed,#ec4899)",
            "linear-gradient(135deg,#06b6d4,#7c3aed)",
            "linear-gradient(135deg,#f59e0b,#ef4444)",
            "linear-gradient(135deg,#10b981,#06b6d4)",
            "linear-gradient(135deg,#ec4899,#f59e0b)",
            "linear-gradient(135deg,#6366f1,#06b6d4)",
            "linear-gradient(135deg,#a855f7,#3b82f6)",
            "linear-gradient(135deg,#f43f5e,#a855f7)",
        ]
        cols_per_row = 4
        rows = [cat_country.iloc[i:i+cols_per_row] for i in range(0, len(cat_country), cols_per_row)]
        for row_df in rows:
            cols_c = st.columns(len(row_df))
            for col_c, (_, crow) in zip(cols_c, row_df.iterrows()):
                bar_pct = int(crow["total_views"] / max_views * 100)
                rank    = cat_country.index[cat_country["country"] == crow["country"]].tolist()
                rank_n  = cat_country.reset_index(drop=True).index[cat_country.reset_index(drop=True)["country"] == crow["country"]].tolist()
                rk      = rank_n[0] if rank_n else 0
                grad    = VIBRANT_GRADIENTS[rk % len(VIBRANT_GRADIENTS)]
                medal   = ["🥇","🥈","🥉"][rk] if rk < 3 else f"#{rk+1}"
                with col_c:
                    st.markdown(f"""
                    <div style='background:#ffffff;border:1.5px solid #e2e8f0;border-radius:16px;
                        padding:16px 14px;text-align:center;margin-bottom:10px;
                        transition:transform 0.2s;'>
                      <div style='font-size:22px;margin-bottom:4px;'>{medal}</div>
                      <div style='font-size:15px;font-weight:800;color:#111827;margin-bottom:8px;
                          background:{grad};-webkit-background-clip:text;
                          -webkit-text-fill-color:transparent;'>
                        {crow["country"]}
                      </div>
                      <div style='background:#7c3aed;border-radius:6px;height:6px;margin-bottom:8px;'>
                        <div style='background:{grad};height:100%;width:{bar_pct}%;
                            border-radius:6px;'></div>
                      </div>
                      <div style='font-size:13px;font-weight:700;color:#fcd34d;'>
                        {crow["total_views"]/1e6:.1f}M views</div>
                      <div style='font-size:11px;color:#475569;margin-top:3px;'>
                        <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='video' style='width:18px; height:18px; color:#374151;'></i></span> {int(crow["video_count"])} videos &nbsp;|&nbsp;
                        <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='trending-up' style='width:18px; height:18px; color:#374151;'></i></span> {crow["avg_engagement"]:.1f}%</div>
                    </div>""", unsafe_allow_html=True)






# ══════════════════════════════════════════════
#  PAGE 5 — LEADERBOARD
# ══════════════════════════════════════════════
elif page == "Leaderboard":
    st.markdown("""<h1 style='font-family:Outfit;font-weight:900;font-size:38px;
        background:linear-gradient(90deg,#f59e0b,#ef4444);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
         Trending YouTubers Leaderboard</h1>""", unsafe_allow_html=True)

    tab1,tab2 = st.tabs(["Most Viewed", "Most Trending"])

    with tab1:
        ch = df.groupby("channel_title").agg(
            total_views=("views","sum"), total_videos=("title","count"),
            avg_likes=("likes","mean"),
            countries=("country",lambda x:", ".join(x.unique()[:3]))
        ).sort_values("total_views",ascending=False).head(15).reset_index()
        for i,row in ch.iterrows():
            medals = ["🥇","🥈","🥉"]
            medal  = medals[i] if i<3 else f"#{i+1}"
            bar_w  = int(row["total_views"]/ch["total_views"].max()*100)
            gold   = ("border-color:#f59e0b;box-shadow:0 0 20px #f59e0b44;" if i==0 else
                      "border-color:#475569;" if i==1 else
                      "border-color:#cd7f32;" if i==2 else "")
            st.markdown(f"""
            <div style='background:#ffffff;border:1.5px solid #e2e8f0;{gold}
                border-radius:12px;padding:14px 20px;margin-bottom:10px;'>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div style='display:flex;align-items:center;gap:14px;'>
                  <div style='font-size:28px;min-width:40px;text-align:center;'>{medal}</div>
                  <div>
                    <div style='font-weight:700;font-size:16px;color:#111827;'>{row["channel_title"]}</div>
                    <div style='font-size:12px;color:#475569;'>{row["total_videos"]} videos • <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span> {row["countries"]}</div>
                    <div style='background:#f8fafc;border-radius:4px;height:4px;width:200px;margin-top:8px;'>
                      <div style='background:linear-gradient(90deg,#7c3aed,#ec4899);
                          height:100%;width:{bar_w}%;border-radius:4px;'></div>
                    </div>
                  </div>
                </div>
                <div style='text-align:right;'>
                  <div style='font-size:22px;font-weight:700;color:#4338ca;'>{row["total_views"]/1e6:.1f}M</div>
                  <div style='font-size:11px;color:#64748b;'>total views</div>
                  <div style='font-size:12px;color:#4ade80;'><span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='thumbs-up' style='width:18px; height:18px; color:#374151;'></i></span> {row["avg_likes"]/1e3:.1f}K avg</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    with tab2:
        cf = df.groupby("channel_title").agg(
            trending_count=("title","count"),total_views=("views","sum")
        ).sort_values("trending_count",ascending=False).head(10).reset_index()
        fig = px.bar(cf,x="channel_title",y="trending_count",
                     color="total_views",color_continuous_scale=["#06b6d4","#a855f7","#ec4899","#f59e0b"],
                     labels={"channel_title":"Channel","trending_count":"Times Trending"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#111827",coloraxis_showscale=False,
                          xaxis=dict(tickangle=-30,gridcolor="#6600cc"),
                          yaxis=dict(gridcolor="#6600cc"),margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE 6 — COUNTRY COMPARE
# ══════════════════════════════════════════════
elif page == "Country Compare":
    st.markdown("""<h1 style='font-family:Outfit;font-weight:900;font-size:38px;
        background:linear-gradient(90deg,#06b6d4,#ec4899);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
         Country vs Country</h1>""", unsafe_allow_html=True)

    countries = sorted(df["country"].unique().tolist())
    c1,c2 = st.columns(2)
    with c1: country1 = st.selectbox("🔵 Country 1", countries, index=0)
    with c2: country2 = st.selectbox("🔴 Country 2", countries, index=min(1,len(countries)-1))

    df1 = df[df["country"]==country1]
    df2 = df[df["country"]==country2]

    st.markdown("<br>", unsafe_allow_html=True)
    metrics = [
        ("<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='video' style='width:18px; height:18px; color:#374151;'></i></span> Videos",     len(df1),                     len(df2),                     ""),
        ("<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='eye' style='width:18px; height:18px; color:#374151;'></i></span> Avg Views",  df1["views"].mean(),           df2["views"].mean(),           ".0f"),
        ("<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='thumbs-up' style='width:18px; height:18px; color:#374151;'></i></span> Avg Likes",  df1["likes"].mean(),           df2["likes"].mean(),           ".0f"),
        ("<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='trending-up' style='width:18px; height:18px; color:#374151;'></i></span> Avg Engage", df1["engagement_rate"].mean(), df2["engagement_rate"].mean(), ".2f"),
    ]
    cols = st.columns(4)
    for col,(label,v1,v2,fmt) in zip(cols,metrics):
        winner = "🔵" if v1>v2 else "🔴"
        fmt_str = f"{{:{fmt}}}" if fmt else "{:.0f}"
        with col:
            st.markdown(f"""
            <div style='background:#ffffff;border:1.5px solid #e2e8f0;border-radius:14px;
                padding:18px;text-align:center;'>
              <div style='font-size:13px;color:#475569;margin-bottom:8px;'>{label}</div>
              <div style='display:flex;justify-content:space-around;align-items:center;'>
                <div>
                  <div style='font-size:18px;font-weight:700;color:#60a5fa;'>{fmt_str.format(v1)}</div>
                  <div style='font-size:11px;color:#475569;'>{country1}</div>
                </div>
                <div style='font-size:22px;'>{winner}</div>
                <div>
                  <div style='font-size:18px;font-weight:700;color:#dc2626;'>{fmt_str.format(v2)}</div>
                  <div style='font-size:11px;color:#475569;'>{country2}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ca1,ca2 = st.columns(2)
    with ca1:
        st.markdown(f"#### 🔵 {country1}")
        c1c = df1.groupby("category_name")["views"].sum().nlargest(8).reset_index()
        f1  = px.bar(c1c,x="views",y="category_name",orientation="h",
                     color="views",color_continuous_scale=["#1e40af","#60a5fa"])
        f1.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                         font_color="#111827",coloraxis_showscale=False,
                         yaxis=dict(autorange="reversed",gridcolor="#6600cc"),
                         xaxis=dict(gridcolor="#6600cc"),margin=dict(t=10,b=10))
        st.plotly_chart(f1, use_container_width=True)
    with ca2:
        st.markdown(f"#### 🔴 {country2}")
        c2c = df2.groupby("category_name")["views"].sum().nlargest(8).reset_index()
        f2  = px.bar(c2c,x="views",y="category_name",orientation="h",
                     color="views",color_continuous_scale=["#991b1b","#f87171"])
        f2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                         font_color="#111827",coloraxis_showscale=False,
                         yaxis=dict(autorange="reversed",gridcolor="#6600cc"),
                         xaxis=dict(gridcolor="#6600cc"),margin=dict(t=10,b=10))
        st.plotly_chart(f2, use_container_width=True)

    st.markdown("#### <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='video' style='width:18px; height:18px; color:#374151;'></i></span> Top 3 Videos Each", unsafe_allow_html=True)
    v1c,v2c = st.columns(2)
    for col,cdf,clr in [(v1c,df1,"#1d4ed8"),(v2c,df2,"#991b1b")]:
        with col:
            for _,row in cdf.nlargest(3,"views").iterrows():
                st.markdown(f"""
                <div style='background:#ffffff;border:1px solid {clr};border-radius:10px;
                    padding:12px;margin-bottom:8px;display:flex;gap:12px;align-items:center;'>
                  <img src='{row["thumbnail_link"]}' style='width:80px;height:50px;
                      object-fit:cover;border-radius:6px;flex-shrink:0;'
                       onerror="this.src='https://picsum.photos/80/50'">
                  <div>
                    <div style='font-size:12px;font-weight:600;color:#111827;'>
                        {row["title"][:38]}...</div>
                    <div style='font-size:11px;color:#6d28d9;'>{row["channel_title"]}</div>
                    <div style='font-size:10px;color:#475569;'>
                        <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='calendar' style='width:18px; height:18px; color:#374151;'></i></span> {str(row["trending_date"])[:10]} • <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='eye' style='width:18px; height:18px; color:#374151;'></i></span> {row["views"]/1e6:.2f}M</div>
                  </div>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PAGE 7 — VIRALITY PREDICTOR
# ══════════════════════════════════════════════
elif page == "Virality Predictor":
    st.markdown("""<h1 style='font-family:Outfit;font-weight:900;font-size:38px;
        background:linear-gradient(90deg,#10b981,#7c3aed);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
         Virality Predictor</h1>
        <p style='color:#475569;'>Enter your video details and find out if it will trend on YouTube!</p>
    """, unsafe_allow_html=True)

    # ── ML runs silently in backend (Random Forest + Linear Regression) ───────
    def prepare_ml(dataframe):
        ml = dataframe.copy()
        le_cat  = LabelEncoder()
        le_ctry = LabelEncoder()
        le_chan  = LabelEncoder()
        ml["cat_enc"]     = le_cat.fit_transform(ml["category_name"].astype(str))
        ml["country_enc"] = le_ctry.fit_transform(ml["country"].astype(str))
        ml["channel_enc"] = le_chan.fit_transform(ml["channel_title"].astype(str))
        ml["viral"]       = (ml["views"] > ml["views"].quantile(0.75)).astype(int)
        feats = ["cat_enc","country_enc","channel_enc","likes","comment_count","engagement_rate"]
        return ml[feats], ml["viral"], ml["views"], le_cat, le_ctry, le_chan

    X, y_clf, y_reg, le_cat, le_ctry, le_chan = prepare_ml(df)
    X_tr,X_te,yc_tr,yc_te = train_test_split(X,y_clf,test_size=0.2,random_state=42)
    _,_,yr_tr,yr_te        = train_test_split(X,y_reg,test_size=0.2,random_state=42)

    st.markdown("""<div style='background:linear-gradient(135deg,#ede9fe,#ddd6fe);
        border:1px solid #5046e5;border-radius:16px;padding:28px;margin:20px 0;'>""",
        unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        inp_cat     = st.selectbox("Category", sorted(df["category_name"].unique()))
        inp_country = st.selectbox("Country",  sorted(df["country"].unique()))
        inp_channel = st.selectbox("Channel",  sorted(df["channel_title"].unique()))
        inp_date    = st.date_input("Target Trending Date")
    with col2:
        inp_likes    = st.slider("Expected Likes",    0, 2_000_000, 50_000, step=1000)
        inp_comments = st.slider("Expected Comments", 0, 500_000,   5_000,  step=500)
        inp_views_e  = st.slider("Expected Views",    10_000, 10_000_000, 500_000, step=10_000)

    inp_eng = round(min((inp_likes+inp_comments)/max(inp_views_e,1)*100, 100.0), 2)
    st.markdown(f"<div style='color:#6d28d9;font-size:14px;margin-top:8px;'>"
                f"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='trending-up' style='width:18px; height:18px; color:#374151;'></i></span> Engagement Rate: <b>{inp_eng:.2f}%</b></div>", unsafe_allow_html=True)

    # ── Video Duration Input (HH:MM:SS timer style) ───────────────────────────
    st.markdown("""
    <div style='margin-top:20px;margin-bottom:6px;'>
      <div style='font-size:13px;font-weight:600;color:#4b5563;margin-bottom:2px;'>⏱️ Video Duration</div>
      <div style='font-size:11px;color:#6366f1;font-style:italic;margin-bottom:8px;'>
        Set your video length in HH : MM : SS format
      </div>
    </div>
    """, unsafe_allow_html=True)

    dur_col_h, dur_sep1, dur_col_m, dur_sep2, dur_col_s, dur_pad = st.columns([2, 0.35, 2, 0.35, 2, 3])
    with dur_col_h:
        inp_hours = st.number_input(
            "HH", min_value=0, max_value=23, value=0, step=1,
            key="dur_hours", format="%02d"
        )
    with dur_sep1:
        st.markdown(
            "<div style='text-align:center;font-size:30px;font-weight:800;"
            "color:#6366f1;padding-top:26px;line-height:1;'>:</div>",
            unsafe_allow_html=True
        )
    with dur_col_m:
        inp_minutes = st.number_input(
            "MM", min_value=0, max_value=59, value=5, step=1,
            key="dur_minutes", format="%02d"
        )
    with dur_sep2:
        st.markdown(
            "<div style='text-align:center;font-size:30px;font-weight:800;"
            "color:#6366f1;padding-top:26px;line-height:1;'>:</div>",
            unsafe_allow_html=True
        )
    with dur_col_s:
        inp_seconds = st.number_input(
            "SS", min_value=0, max_value=59, value=0, step=1,
            key="dur_seconds", format="%02d"
        )

    inp_duration = inp_hours * 3600 + inp_minutes * 60 + inp_seconds
    st.markdown(
        f"<div style='font-size:12px;color:#6d28d9;margin-top:4px;margin-bottom:14px;'>"
        f"⏳ Total: <b>{inp_hours:02d}:{inp_minutes:02d}:{inp_seconds:02d}</b>"
        f" &nbsp;·&nbsp; <b>{inp_duration:,}</b> seconds</div>",
        unsafe_allow_html=True
    )

    # ── Licensed Content + Video Dimension ────────────────────────────────────
    lic_col, dim_col = st.columns(2)
    with lic_col:
        inp_licensed  = st.checkbox("Licensed Content", value=False)
    with dim_col:
        inp_dimension = st.selectbox("Video Dimension", ["2D", "3D", "hd", "sd"])

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚀 Predict Now!", use_container_width=True):
        cat_e  = int(le_cat.transform([inp_cat])[0])      if inp_cat     in le_cat.classes_  else 0
        ctry_e = int(le_ctry.transform([inp_country])[0]) if inp_country in le_ctry.classes_ else 0
        chan_e  = int(le_chan.transform([inp_channel])[0]) if inp_channel in le_chan.classes_  else 0
        inp_arr = np.array([[cat_e, ctry_e, chan_e, inp_likes, inp_comments, inp_eng]])

        # Backend: Random Forest for trend classification
        clf   = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_tr, yc_tr)
        pred  = clf.predict(inp_arr)[0]
        proba = clf.predict_proba(inp_arr)[0]
        viral = (pred == 1)
        conf  = proba[1] if viral else proba[0]

        # Backend: Linear Regression for view count estimate
        reg        = LinearRegression()
        reg.fit(X_tr, yr_tr)
        pred_views = max(0, int(reg.predict(inp_arr)[0]))

        # Backend: Decision Tree for key factors
        dt = DecisionTreeClassifier(max_depth=8, random_state=42)
        dt.fit(X_tr, yc_tr)
        imp = dt.feature_importances_

        # Translate importance to user-friendly insight
        feat_names  = ["Category","Country","Channel","Likes","Comments","Engagement"]
        top_factor  = feat_names[imp.argmax()]
        factor_tips = {
            "Likes":       "💡 Focus on getting more likes — it's your biggest virality driver!",
            "Engagement":  "💡 Your engagement rate matters most — reply to comments & interact!",
            "Comments":    "💡 Videos that spark discussion trend faster — ask questions in your video!",
            "Category":    "💡 Your content category heavily influences trending potential.",
            "Country":     "💡 Targeting the right audience country boosts your trending chances.",
            "Channel":     "💡 Building your channel brand improves trending frequency.",
        }
        tip_text = factor_tips.get(top_factor, "💡 Keep creating consistent, engaging content!")

        # Trend score (0-100 user-facing)
        trend_score = int(conf * 100)

        if viral:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#ecfdf5,#d1fae5);
                border:1.5px solid #34d399;border-radius:18px;padding:34px;
                text-align:center;box-shadow:0 4px 16px rgba(16,185,129,0.2);margin-top:20px;
                animation:glowPulse 2s ease infinite;'>
              <div style='font-size:70px;animation:floatY 2s ease infinite;'><span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='flame' style='width:18px; height:18px; color:#374151;'></i></span></div>
              <div style='font-size:34px;font-weight:900;color:#059669;margin:10px 0;'>WILL TREND!</div>
              <div style='font-size:16px;color:#065f46;'>High viral potential on YouTube</div>
              <div style='font-size:48px;font-weight:800;color:#047857;margin:14px 0 4px;'>
                {trend_score}%</div>
              <div style='font-size:13px;color:#059669;'>Trend Score · <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='calendar' style='width:18px; height:18px; color:#374151;'></i></span> {inp_date}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#fef2f2,#fee2e2);
                border:1.5px solid #f87171;border-radius:18px;padding:34px;
                text-align:center;box-shadow:0 4px 16px rgba(239,68,68,0.2);margin-top:20px;'>
              <div style='font-size:70px;'><span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='snowflake' style='width:18px; height:18px; color:#374151;'></i></span></div>
              <div style='font-size:34px;font-weight:900;color:#b91c1c;margin:10px 0;'>WON'T TREND</div>
              <div style='font-size:16px;color:#991b1b;'>Below trending threshold — boost engagement!</div>
              <div style='font-size:48px;font-weight:800;color:#dc2626;margin:14px 0 4px;'>
                {trend_score}%</div>
              <div style='font-size:13px;color:#b91c1c;'>Trend Score · <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='calendar' style='width:18px; height:18px; color:#374151;'></i></span> {inp_date}</div>
            </div>""", unsafe_allow_html=True)

        # Stats row
        st.markdown(f"""
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:20px;'>
          <div style='background:#ffffff;border:1.5px solid #e2e8f0;border-radius:14px;
              padding:18px;text-align:center;'>
            <div style='font-size:12px;color:#475569;margin-bottom:6px;'><span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='bar-chart-2' style='width:18px; height:18px; color:#374151;'></i></span> Predicted Views</div>
            <div style='font-size:28px;font-weight:800;color:#4338ca;'>{pred_views:,}</div>
          </div>
          <div style='background:#ffffff;border:1.5px solid #e2e8f0;border-radius:14px;
              padding:18px;text-align:center;'>
            <div style='font-size:12px;color:#475569;margin-bottom:6px;'>🎯 Trend Score</div>
            <div style='font-size:28px;font-weight:800;color:#{"34d399" if viral else "f87171"};'>{trend_score}/100</div>
          </div>
          <div style='background:#ffffff;border:1.5px solid #e2e8f0;border-radius:14px;
              padding:18px;text-align:center;'>
            <div style='font-size:12px;color:#475569;margin-bottom:6px;'>🔑 Top Success Factor</div>
            <div style='font-size:20px;font-weight:800;color:#fcd34d;'>{top_factor}</div>
          </div>
        </div>
        <div style='background:#f8fafc;border:1px solid #5046e5;border-radius:12px;
            padding:16px;margin-top:14px;font-size:14px;color:#4338ca;'>
          {tip_text}
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PAGE 8 — MONETIZATION GUIDE
# ══════════════════════════════════════════════
elif page == "Monetization":
    st.markdown("""<h1 style='font-family:Inter;font-weight:700;font-size:32px;
        color:#0f172a;'>
         Monetization Intelligence</h1>
        <p style='color:#64748b;font-size:12px;
            letter-spacing:2px;'>Data-driven insights to maximise your YouTube revenue potential.</p>
    """, unsafe_allow_html=True)

    # ── CPM benchmarks by category ────────────────────────────────────────────
    CPM_MAP = {
        "Film & Animation": 3.5, "Autos & Vehicles": 7.2, "Music": 2.0,
        "Pets & Animals": 3.0, "Sports": 4.0, "Travel & Events": 5.5,
        "Gaming": 3.2, "People & Blogs": 2.5, "Comedy": 2.8,
        "Entertainment": 2.4, "News & Politics": 6.5, "Howto & Style": 6.0,
        "Education": 5.8, "Science & Technology": 8.5, "Nonprofits & Activism": 1.5,
        "Grand Total": 4.0,
    }
    DEFAULT_CPM = 3.0

    cat_mon = df.groupby("category_name").agg(
        total_views=("views","sum"),
        avg_views=("views","mean"),
        avg_engagement=("engagement_rate","mean"),
        video_count=("title","count"),
    ).reset_index()
    cat_mon["cpm"]         = cat_mon["category_name"].map(CPM_MAP).fillna(DEFAULT_CPM)
    cat_mon["est_revenue"] = (cat_mon["total_views"] / 1000 * cat_mon["cpm"]).round(0).astype(int)
    cat_mon["rev_per_vid"] = (cat_mon["est_revenue"] / cat_mon["video_count"]).round(0).astype(int)

    best_cat   = cat_mon.sort_values("est_revenue",ascending=False).iloc[0]
    best_cpm   = cat_mon.sort_values("cpm",ascending=False).iloc[0]
    total_est  = cat_mon["est_revenue"].sum()
    best_per_v = cat_mon.sort_values("rev_per_vid",ascending=False).iloc[0]

    km1,km2,km3,km4 = st.columns(4)
    # Standardized accent colors for cards
    _card_cols = ["#6366f1", "#ec4899", "#14b8a6", "#f59e0b"]
    
    kpi_items = [
        (km1, f"${total_est/1e6:.1f}M", "Total Est. Revenue", _card_cols[0]),
        (km2, best_cat["category_name"], "Highest Revenue Category", _card_cols[1]),
        (km3, f"${best_cpm['cpm']:.1f}", "Highest CPM ($/1K views)", _card_cols[2]),
        (km4, f"${best_per_v['rev_per_vid']:,}", "Best Revenue/Video", _card_cols[3]),
    ]
    
    for col, val, label, color in kpi_items:
        with col:
            st.markdown(f"""
            <div class='kpi-card' style='background:#ffffff; border-radius:14px; padding:24px; text-align:center; box-shadow:var(--shadow-sm); border:1px solid #e2e8f0;'>
              <div style='font-size:22px; font-weight:800; color:{color}; font-family:Inter, sans-serif;'>{val}</div>
              <div style='color:#64748b; font-size:12px; margin-top:6px; font-weight:500; letter-spacing:0.5px; text-transform:uppercase;'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Revenue Calculator ────────────────────────────────────────────────────
    st.markdown("<hr style='border:0;border-top:1px solid #bf00ff33;margin:24px 0;'>", unsafe_allow_html=True)
    st.markdown("""<div style='border-left:4px solid #f59e0b;
        box-shadow:-4px 0 12px #ffe60044;padding-left:14px;margin-bottom:20px;'>
      <h2 style='font-family:Inter;font-weight:700;font-size:24px;margin:0;
          color:#f59e0b;letter-spacing:2px;'>
        🧮 REVENUE CALCULATOR — ESTIMATE YOUR EARNINGS</h2>
      <div style='color:#ffaa00;font-size:12px;margin-top:4px;font-family:'JetBrains Mono',monospace;'>
        Enter your video stats to get a personalised earnings estimate</div>
    </div>""", unsafe_allow_html=True)

    rc1,rc2,rc3 = st.columns(3)
    with rc1:
        calc_cat   = st.selectbox("Your Category", sorted(df["category_name"].unique()), key="calc_cat")
        calc_views = st.number_input("Expected Monthly Views", min_value=1000,
                                      max_value=500_000_000, value=100_000, step=10_000)
    with rc2:
        calc_videos = st.number_input("Videos Per Month", min_value=1, max_value=100, value=4)
        calc_subs   = st.number_input("Subscribers", min_value=0, max_value=100_000_000,
                                       value=10_000, step=1_000)
    with rc3:
        calc_country = st.selectbox("Primary Audience Country",
                                     sorted(df["country"].unique()), key="calc_country")

    # ── Country currency + CPM multiplier (defined BEFORE button so always available) ──
    COUNTRY_CONFIG = {
        # Tier 1 — Premium English markets
        "US":{"sym":"$",   "name":"USD","flag":"🇺🇸","mult":1.50,"rate":1.00},
        "GB":{"sym":"£",   "name":"GBP","flag":"🇬🇧","mult":1.40,"rate":0.79},
        "CA":{"sym":"CA$", "name":"CAD","flag":"🇨🇦","mult":1.30,"rate":1.36},
        "AU":{"sym":"A$",  "name":"AUD","flag":"🇦🇺","mult":1.30,"rate":1.53},
        "NZ":{"sym":"NZ$", "name":"NZD","flag":"🇳🇿","mult":1.20,"rate":1.63},
        # Tier 2 — Western Europe
        "DE":{"sym":"€",   "name":"EUR","flag":"🇩🇪","mult":1.20,"rate":0.92},
        "FR":{"sym":"€",   "name":"EUR","flag":"🇫🇷","mult":1.10,"rate":0.92},
        "NL":{"sym":"€",   "name":"EUR","flag":"🇳🇱","mult":1.10,"rate":0.92},
        "SE":{"sym":"kr",  "name":"SEK","flag":"🇸🇪","mult":1.05,"rate":10.5},
        "NO":{"sym":"kr",  "name":"NOK","flag":"🇳🇴","mult":1.10,"rate":10.7},
        "CH":{"sym":"CHF", "name":"CHF","flag":"🇨🇭","mult":1.15,"rate":0.90},
        "AT":{"sym":"€",   "name":"EUR","flag":"🇦🇹","mult":1.05,"rate":0.92},
        "BE":{"sym":"€",   "name":"EUR","flag":"🇧🇪","mult":1.00,"rate":0.92},
        "DK":{"sym":"kr",  "name":"DKK","flag":"🇩🇰","mult":1.05,"rate":6.9},
        "FI":{"sym":"€",   "name":"EUR","flag":"🇫🇮","mult":1.00,"rate":0.92},
        "IE":{"sym":"€",   "name":"EUR","flag":"🇮🇪","mult":1.05,"rate":0.92},
        "ES":{"sym":"€",   "name":"EUR","flag":"🇪🇸","mult":0.80,"rate":0.92},
        "IT":{"sym":"€",   "name":"EUR","flag":"🇮🇹","mult":0.80,"rate":0.92},
        "PT":{"sym":"€",   "name":"EUR","flag":"🇵🇹","mult":0.65,"rate":0.92},
        # Tier 3 — East Asia
        "JP":{"sym":"¥",   "name":"JPY","flag":"🇯🇵","mult":0.70,"rate":149.5},
        "KR":{"sym":"₩",   "name":"KRW","flag":"🇰🇷","mult":0.65,"rate":1325.0},
        "SG":{"sym":"S$",  "name":"SGD","flag":"🇸🇬","mult":0.80,"rate":1.34},
        "HK":{"sym":"HK$", "name":"HKD","flag":"🇭🇰","mult":0.75,"rate":7.82},
        "TW":{"sym":"NT$", "name":"TWD","flag":"🇹🇼","mult":0.55,"rate":31.5},
        # Tier 4 — Latin America
        "BR":{"sym":"R$",  "name":"BRL","flag":"🇧🇷","mult":0.40,"rate":4.97},
        "MX":{"sym":"MX$", "name":"MXN","flag":"🇲🇽","mult":0.35,"rate":17.1},
        "AR":{"sym":"$",   "name":"ARS","flag":"🇦🇷","mult":0.25,"rate":870.0},
        "CO":{"sym":"$",   "name":"COP","flag":"🇨🇴","mult":0.28,"rate":3900.0},
        "CL":{"sym":"$",   "name":"CLP","flag":"🇨🇱","mult":0.32,"rate":900.0},
        "PE":{"sym":"S/",  "name":"PEN","flag":"🇵🇪","mult":0.25,"rate":3.7},
        # Tier 5 — Eastern Europe / Russia
        "RU":{"sym":"₽",   "name":"RUB","flag":"🇷🇺","mult":0.30,"rate":90.0},
        "PL":{"sym":"zł",  "name":"PLN","flag":"🇵🇱","mult":0.50,"rate":4.0},
        "CZ":{"sym":"Kč",  "name":"CZK","flag":"🇨🇿","mult":0.50,"rate":22.5},
        "RO":{"sym":"lei", "name":"RON","flag":"🇷🇴","mult":0.40,"rate":4.6},
        "HU":{"sym":"Ft",  "name":"HUF","flag":"🇭🇺","mult":0.40,"rate":355.0},
        "UA":{"sym":"₴",   "name":"UAH","flag":"🇺🇦","mult":0.20,"rate":37.0},
        "TR":{"sym":"₺",   "name":"TRY","flag":"🇹🇷","mult":0.25,"rate":27.0},
        "GR":{"sym":"€",   "name":"EUR","flag":"🇬🇷","mult":0.55,"rate":0.92},
        # Tier 6 — South / Southeast Asia
        "IN":{"sym":"₹",   "name":"INR","flag":"🇮🇳","mult":0.25,"rate":83.2},
        "PK":{"sym":"₨",   "name":"PKR","flag":"🇵🇰","mult":0.18,"rate":278.0},
        "BD":{"sym":"৳",   "name":"BDT","flag":"🇧🇩","mult":0.15,"rate":110.0},
        "PH":{"sym":"₱",   "name":"PHP","flag":"🇵🇭","mult":0.28,"rate":55.0},
        "MY":{"sym":"RM",  "name":"MYR","flag":"🇲🇾","mult":0.35,"rate":4.7},
        "TH":{"sym":"฿",   "name":"THB","flag":"🇹🇭","mult":0.30,"rate":35.0},
        "VN":{"sym":"₫",   "name":"VND","flag":"🇻🇳","mult":0.20,"rate":24500.0},
        "ID":{"sym":"Rp",  "name":"IDR","flag":"🇮🇩","mult":0.22,"rate":15500.0},
        "LK":{"sym":"₨",   "name":"LKR","flag":"🇱🇰","mult":0.15,"rate":320.0},
        "NP":{"sym":"₨",   "name":"NPR","flag":"🇳🇵","mult":0.15,"rate":132.0},
        # Tier 7 — Middle East
        "SA":{"sym":"﷼",   "name":"SAR","flag":"🇸🇦","mult":0.45,"rate":3.75},
        "AE":{"sym":"د.إ", "name":"AED","flag":"🇦🇪","mult":0.55,"rate":3.67},
        "QA":{"sym":"﷼",   "name":"QAR","flag":"🇶🇦","mult":0.50,"rate":3.64},
        "KW":{"sym":"د.ك", "name":"KWD","flag":"🇰🇼","mult":0.50,"rate":0.31},
        "IL":{"sym":"₪",   "name":"ILS","flag":"🇮🇱","mult":0.60,"rate":3.7},
        "IQ":{"sym":"ع.د", "name":"IQD","flag":"🇮🇶","mult":0.20,"rate":1310.0},
        "JO":{"sym":"د.ا", "name":"JOD","flag":"🇯🇴","mult":0.25,"rate":0.71},
        # Tier 8 — Africa (low CPM markets)
        "NG":{"sym":"₦",   "name":"NGN","flag":"🇳🇬","mult":0.15,"rate":1550.0},
        "EG":{"sym":"£",   "name":"EGP","flag":"🇪🇬","mult":0.18,"rate":30.9},
        "ZA":{"sym":"R",   "name":"ZAR","flag":"🇿🇦","mult":0.28,"rate":18.5},
        "KE":{"sym":"KSh", "name":"KES","flag":"🇰🇪","mult":0.15,"rate":130.0},
        "GH":{"sym":"₵",   "name":"GHS","flag":"🇬🇭","mult":0.15,"rate":12.0},
        "MA":{"sym":"د.م.","name":"MAD","flag":"🇲🇦","mult":0.20,"rate":10.0},
        "DZ":{"sym":"د.ج", "name":"DZD","flag":"🇩🇿","mult":0.18,"rate":134.0},  # Algeria
        "TN":{"sym":"د.ت", "name":"TND","flag":"🇹🇳","mult":0.18,"rate":3.1},
        "ET":{"sym":"Br",  "name":"ETB","flag":"🇪🇹","mult":0.10,"rate":55.0},
        "TZ":{"sym":"Sh",  "name":"TZS","flag":"🇹🇿","mult":0.10,"rate":2500.0},
        # Tier 9 — Full country name aliases (dataset uses full names)
        "Algeria":       {"sym":"د.ج","name":"DZD","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.18,"rate":134.0},
        "Argentina":     {"sym":"$",  "name":"ARS","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.25,"rate":870.0},
        "Australia":     {"sym":"A$", "name":"AUD","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":1.30,"rate":1.53},
        "Bangladesh":    {"sym":"৳",  "name":"BDT","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.15,"rate":110.0},
        "Belgium":       {"sym":"€",  "name":"EUR","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":1.00,"rate":0.92},
        "Brazil":        {"sym":"R$", "name":"BRL","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.40,"rate":4.97},
        "Canada":        {"sym":"CA$","name":"CAD","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":1.30,"rate":1.36},
        "Chile":         {"sym":"$",  "name":"CLP","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.32,"rate":900.0},
        "Colombia":      {"sym":"$",  "name":"COP","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.28,"rate":3900.0},
        "Egypt":         {"sym":"£",  "name":"EGP","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.18,"rate":30.9},
        "Ethiopia":      {"sym":"Br", "name":"ETB","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.10,"rate":55.0},
        "France":        {"sym":"€",  "name":"EUR","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":1.10,"rate":0.92},
        "Germany":       {"sym":"€",  "name":"EUR","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":1.20,"rate":0.92},
        "Ghana":         {"sym":"₵",  "name":"GHS","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.15,"rate":12.0},
        "Greece":        {"sym":"€",  "name":"EUR","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.55,"rate":0.92},
        "Iceland":       {"sym":"kr", "name":"ISK","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.90,"rate":136.0},
        "India":         {"sym":"₹",  "name":"INR","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.25,"rate":83.2},
        "Indonesia":     {"sym":"Rp", "name":"IDR","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.22,"rate":15500.0},
        "Iraq":          {"sym":"ع.د","name":"IQD","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.20,"rate":1310.0},
        "Israel":        {"sym":"₪",  "name":"ILS","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.60,"rate":3.7},
        "Italy":         {"sym":"€",  "name":"EUR","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.80,"rate":0.92},
        "Japan":         {"sym":"¥",  "name":"JPY","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.70,"rate":149.5},
        "Jordan":        {"sym":"د.ا","name":"JOD","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.25,"rate":0.71},
        "Kenya":         {"sym":"KSh","name":"KES","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.15,"rate":130.0},
        "Kuwait":        {"sym":"د.ك","name":"KWD","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.50,"rate":0.31},
        "Libya":         {"sym":"ل.د","name":"LYD","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.18,"rate":4.8},
        "Malaysia":      {"sym":"RM", "name":"MYR","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.35,"rate":4.7},
        "Mexico":        {"sym":"MX$","name":"MXN","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.35,"rate":17.1},
        "Morocco":       {"sym":"د.م.","name":"MAD","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.20,"rate":10.0},
        "Netherlands":   {"sym":"€",  "name":"EUR","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":1.10,"rate":0.92},
        "New Zealand":   {"sym":"NZ$","name":"NZD","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":1.20,"rate":1.63},
        "Nigeria":       {"sym":"₦",  "name":"NGN","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.15,"rate":1550.0},
        "Norway":        {"sym":"kr", "name":"NOK","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":1.10,"rate":10.7},
        "Pakistan":      {"sym":"₨",  "name":"PKR","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.18,"rate":278.0},
        "Peru":          {"sym":"S/", "name":"PEN","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.25,"rate":3.7},
        "Philippines":   {"sym":"₱",  "name":"PHP","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.28,"rate":55.0},
        "Poland":        {"sym":"zł", "name":"PLN","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.50,"rate":4.0},
        "Portugal":      {"sym":"€",  "name":"EUR","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.65,"rate":0.92},
        "Qatar":         {"sym":"﷼",  "name":"QAR","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.50,"rate":3.64},
        "Romania":       {"sym":"lei","name":"RON","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.40,"rate":4.6},
        "Russia":        {"sym":"₽",  "name":"RUB","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.30,"rate":90.0},
        "Saudi Arabia":  {"sym":"﷼",  "name":"SAR","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.45,"rate":3.75},
        "Singapore":     {"sym":"S$", "name":"SGD","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.80,"rate":1.34},
        "South Africa":  {"sym":"R",  "name":"ZAR","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.28,"rate":18.5},
        "South Korea":   {"sym":"₩",  "name":"KRW","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.65,"rate":1325.0},
        "Spain":         {"sym":"€",  "name":"EUR","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.80,"rate":0.92},
        "Sweden":        {"sym":"kr", "name":"SEK","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":1.05,"rate":10.5},
        "Switzerland":   {"sym":"CHF","name":"CHF","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":1.15,"rate":0.90},
        "Thailand":      {"sym":"฿",  "name":"THB","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.30,"rate":35.0},
        "Tunisia":       {"sym":"د.ت","name":"TND","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.18,"rate":3.1},
        "Turkey":        {"sym":"₺",  "name":"TRY","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.25,"rate":27.0},
        "Ukraine":       {"sym":"₴",  "name":"UAH","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.20,"rate":37.0},
        "United Kingdom":{"sym":"£",  "name":"GBP","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":1.40,"rate":0.79},
        "United States": {"sym":"$",  "name":"USD","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":1.50,"rate":1.00},
        "Vietnam":       {"sym":"₫",  "name":"VND","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.20,"rate":24500.0},
    }
    cc        = COUNTRY_CONFIG.get(calc_country,
                {"sym":"$","name":"USD","flag":"<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='globe' style='width:18px; height:18px; color:#374151;'></i></span>","mult":0.30,"rate":1.0})
    curr_sym  = cc["sym"]
    curr_name = cc["name"]
    curr_flag = cc["flag"]
    curr_rate = cc["rate"]
    cpm_mult  = cc["mult"]
    calc_cpm  = CPM_MAP.get(calc_cat, DEFAULT_CPM) * cpm_mult

    # Country's actual top category from dataset
    ctry_df      = df[df["country"] == calc_country]
    ctry_top_cat = (ctry_df.groupby("category_name")["views"].sum().idxmax()
                    if len(ctry_df) > 0 else calc_cat)
    ctry_cat_bd  = (ctry_df.groupby("category_name")["views"].sum()
                    .nlargest(5).reset_index() if len(ctry_df) > 0 else pd.DataFrame())

    # Country insight card (always visible)
    st.markdown(f"""
    <div style='background:#ffffff;
        border:1.5px solid #6366f144;border-radius:8px;padding:14px 20px;margin:14px 0;
        display:flex;gap:16px;align-items:center;flex-wrap:wrap;
        box-shadow:0 2px 10px rgba(0,0,0,0.05);'>
      <div style='font-size:36px;'>{curr_flag}</div>
      <div>
        <div style='font-size:13px;font-weight:700;color:#0f172a;
            font-family:Inter,sans-serif;letter-spacing:0.3px;'>
          {calc_country} Audience Insights</div>
        <div style='font-size:12px;color:#6d28d9;margin-top:4px;font-family:'JetBrains Mono',monospace;'>
          🥇 Top Trending: <b style='color:#ec4899;'>{ctry_top_cat}</b> &nbsp;|&nbsp;
          💱 Currency: <b style='color:#f59e0b;'>{curr_sym} ({curr_name})</b> &nbsp;|&nbsp;
          <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='bar-chart-2' style='width:18px; height:18px; color:#374151;'></i></span> CPM multiplier: <b style='color:#10b981;'>{cpm_mult:.2f}x</b>
        </div>
        <div style='font-size:11px;color:#94a3b8;margin-top:5px;font-family:'JetBrains Mono',monospace;'>
          <span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='zap' style='width:18px; height:18px; color:#374151;'></i></span> <b style='color:#0f766eaa;'>{calc_cat}</b> base CPM:
          <b style='color:#10b981;'>${CPM_MAP.get(calc_cat, DEFAULT_CPM):.2f}</b>
          → effective CPM <b style='color:#f59e0b;'>${calc_cpm:.2f}</b> after country multiplier.
          {f"<br>💡 <b style='color:#ec4899;'>Note:</b> <span style='color:#6d28d9;'>'{calc_cat}' CPM (${CPM_MAP.get(calc_cat, DEFAULT_CPM):.2f}) differs from '{ctry_top_cat}' CPM (${CPM_MAP.get(ctry_top_cat, DEFAULT_CPM):.2f}) — <b style='color:#f59e0b;'>trending ≠ highest earning.</b> CPM varies by advertiser demand per niche.</span>" if calc_cat != ctry_top_cat else ""}
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    if st.button("Calculate My Earnings", use_container_width=True, key="calc_btn"):
        calc_rpm       = calc_cpm * 0.55
        monthly_ad_rev = (calc_views / 1000) * calc_rpm
        sponsorship    = calc_subs * 0.02 * calc_videos / 4 if calc_subs >= 10_000 else 0
        membership     = calc_subs * 0.001 * 4.99          if calc_subs >= 30_000 else 0
        annual_est     = (monthly_ad_rev + sponsorship + membership) * 12

        def fmt_local(usd_val):
            return f"{curr_sym}{usd_val * curr_rate:,.0f}"

        st.markdown(f"""
        <div style='background:#f8fafc;
            border:1.5px solid #f59e0b55;border-radius:8px;padding:28px;margin-top:16px;
            box-shadow:0 4px 16px rgba(0,0,0,0.05);'>
          <div style='display:flex;justify-content:space-between;align-items:center;
              margin-bottom:20px;flex-wrap:wrap;gap:8px;'>
            <div style='font-size:12px;color:#6366f1;letter-spacing:1px;
                text-transform:uppercase;font-weight:700;font-family:Inter,sans-serif;'>
              {curr_flag} {calc_country} · {calc_cat} — Earnings Estimate
            </div>
            <div style='font-size:11px;color:#64748b;'>
              💱 {curr_name} &nbsp;(1 USD = {curr_sym}{curr_rate:,.1f})
            </div>
          </div>
          <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:14px;'>
            <div style='background:rgba(255,255,255,0.95);border:1.5px solid #6366f1;border-radius:12px;
                padding:16px;text-align:center;box-shadow:var(--shadow-sm);'>
              <div style='font-size:10px;color:#6366f1;margin-bottom:6px;
                  font-family:Inter,sans-serif;letter-spacing:1px;font-weight:700;'>💵 AD REV/MONTH</div>
              <div style='font-size:26px;font-weight:700;color:#111827;
                  font-family:Inter;'>{fmt_local(monthly_ad_rev)}</div>
              <div style='font-size:10px;color:#475569;margin-top:4px;'>
                ≈ ${monthly_ad_rev:,.0f} USD &nbsp;|&nbsp; RPM {curr_sym}{calc_rpm*curr_rate:.2f}</div>
            </div>
            <div style='background:rgba(255,255,255,0.95);border:1.5px solid #10b981;border-radius:12px;
                padding:16px;text-align:center;box-shadow:var(--shadow-sm);'>
              <div style='font-size:10px;color:#10b981;margin-bottom:6px;
                  font-family:Inter,sans-serif;letter-spacing:1px;font-weight:700;'>🤝 SPONSORSHIP/MO</div>
              <div style='font-size:26px;font-weight:700;color:#111827;
                  font-family:Inter;'>{fmt_local(sponsorship)}</div>
              <div style='font-size:10px;color:#475569;margin-top:4px;'>
                {'✅ Eligible' if calc_subs>=10000 else '⚠️ Need 10K subs'}</div>
            </div>
            <div style='background:rgba(255,255,255,0.95);border:1.5px solid #ec4899;border-radius:12px;
                padding:16px;text-align:center;box-shadow:var(--shadow-sm);'>
              <div style='font-size:10px;color:#ec4899;margin-bottom:6px;
                  font-family:Inter,sans-serif;letter-spacing:1px;font-weight:700;'>🌟 MEMBERSHIPS/MO</div>
              <div style='font-size:26px;font-weight:700;color:#111827;
                  font-family:Inter;'>{fmt_local(membership)}</div>
              <div style='font-size:10px;color:#475569;margin-top:4px;'>
                {'✅ Eligible' if calc_subs>=30000 else '⚠️ Need 30K subs'}</div>
            </div>
            <div style='background:linear-gradient(135deg,#6366f1,#ec4899);
                border:1.5px solid #ffffff;border-radius:12px;padding:16px;text-align:center;
                box-shadow:var(--shadow-md);'>
              <div style='font-size:10px;color:#ffffff;margin-bottom:6px;
                  font-family:Inter,sans-serif;letter-spacing:1px;font-weight:700;
                  '> ANNUAL TOTAL</div>
              <div style='font-size:28px;font-weight:700;color:#ffffff;
                  font-family:Inter;'>{fmt_local(annual_est)}</div>
              <div style='font-size:10px;color:rgba(255,255,255,0.9);margin-top:4px;font-family:Inter,sans-serif;'>≈ ${annual_est:,.0f} USD/yr</div>
              <div style='font-size:10px;color:rgba(255,255,255,0.7);margin-top:6px;border-top:1px solid rgba(255,255,255,0.2);
                  padding-top:6px;font-family:Inter,sans-serif;'>
                (${monthly_ad_rev:,.0f} + ${sponsorship:,.0f} + ${membership:,.0f}) × 12</div>
            </div>
          </div>
          <div style='margin-top:16px;padding:12px 16px;background:#fef3c7;
              border:1px solid #f59e0b33;border-radius:6px;font-size:11px;
              color:#ffcc00;font-family:'JetBrains Mono',monospace;line-height:1.8;'>
            📡 CPM used: <b style='color:#f59e0b;'>${calc_cpm:.2f}</b> &nbsp;|&nbsp;
            RPM (after YT 45% cut): <b style='color:#10b981;'>${calc_rpm:.2f}</b> &nbsp;|&nbsp;
            Base CPM for <b style='color:#6366f1;'>{calc_cat}</b>: <b style='color:#6366f1;'>${CPM_MAP.get(calc_cat, DEFAULT_CPM):.2f}</b>
            {f" &nbsp;vs&nbsp; top-trending <b style='color:#ec4899;'>{ctry_top_cat}</b> CPM: <b style='color:#ec4899;'>${CPM_MAP.get(ctry_top_cat, DEFAULT_CPM) * cpm_mult:.2f}</b> — <b style='color:#f59e0b;'>higher CPM = more revenue per 1K views even with fewer viewers</b>" if calc_cat != ctry_top_cat else ""}
          </div>
        </div>""", unsafe_allow_html=True)

        # Country top categories breakdown from REAL dataset
        if len(ctry_cat_bd) > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""<div style='border-left:4px solid #6366f1;
              padding-left:14px;margin-bottom:14px;'>
              <h3 style='font-family:Inter;font-weight:700;font-size:18px;margin:0;
                color:#0f172a;'>
                {curr_flag} Top Categories in {calc_country}</h3>
            </div>""", unsafe_allow_html=True)
            total_v  = ctry_cat_bd["views"].sum()
            cat_cols = st.columns(min(5, len(ctry_cat_bd)))
            for col_c, (_, crow) in zip(cat_cols, ctry_cat_bd.iterrows()):
                share  = crow["views"] / total_v * 100
                is_top = crow["category_name"] == ctry_top_cat
                with col_c:
                    st.markdown(f"""
                    <div style='background:{"#f8fafc" if is_top else "#ffffff"};
                        border:{"2px solid #ff00c8" if is_top else "1px solid #bf00ff33"};
                        border-radius:8px;padding:14px;text-align:center;
                        {"box-shadow:0 2px 8px rgba(99,102,241,0.2);" if is_top else ""}'>
                      {"<div style='font-size:10px;color:#f59e0b;font-weight:700;margin-bottom:4px;font-family:'JetBrains Mono',monospace;'>👑 #1 IN COUNTRY</div>" if is_top else ""}
                      <div style='font-size:12px;font-weight:700;color:#111827;margin-bottom:6px;
                          font-family:Inter,sans-serif;'>{crow["category_name"]}</div>
                      <div style='font-size:22px;font-weight:700;
                          color:{"#ff00c8" if is_top else "#00fff7"};
                          text-shadow:none;
                          font-family:Inter,sans-serif;'>{share:.1f}%</div>
                      <div style='font-size:10px;color:#64748b;'>of views</div>
                      <div style='background:#f8fafc;border-radius:4px;height:3px;margin-top:8px;'>
                        <div style='background:linear-gradient(90deg,{"#ff00c8,#bf00ff" if is_top else "#00fff7,#00ff88"});
                            height:100%;width:{int(share)}%;border-radius:4px;
                            box-shadow:0 0 6px {"#ff00c8" if is_top else "#00fff7"};'></div>
                      </div>
                    </div>""", unsafe_allow_html=True)

        # Strategy cards
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div style='border-left:4px solid #6366f1;padding-left:14px;margin-bottom:20px;'><h2 style='font-family:Inter;font-weight:700;font-size:20px;margin:0;color:#0f172a;'>🚀 Best Revenue Strategies for Your Category</h2></div>""", unsafe_allow_html=True)
        STRATEGIES = {
            "Science & Technology":["💻 Tech affiliate links","🤝 SaaS sponsorships ($500–5K)","📱 App review deals","<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='box' style='width:18px; height:18px; color:#374151;'></i></span> Unboxing partnerships"],
            "Education":           ["📚 Online course sales","🎓 Patreon memberships","🤝 EdTech sponsorships","📝 Premium download packs"],
            "Howto & Style":       ["💄 Affiliate marketing","🤝 Brand deals & hauls","🛍️ Merch store","<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='box' style='width:18px; height:18px; color:#374151;'></i></span> PR packages from brands"],
            "Gaming":              ["🎮 Twitch + YouTube","💎 Viewer donations","🤝 Publisher sponsorships","🛒 Gaming gear affiliates"],
            "Music":               ["🎵 Streaming royalties","🎤 Live performance","🛒 Merch + physical sales","💿 Sync licensing"],
            "News & Politics":     ["📰 Newsletter subscription","🤝 Think tank sponsorships","🌟 Super Thanks","Premium research reports"],
            "Autos & Vehicles":    ["🚗 Dealership partnerships","🔧 Parts affiliate links","🤝 Insurance deals","<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='box' style='width:18px; height:18px; color:#374151;'></i></span> Auto gear sponsorships"],
            "Travel & Events":     ["✈️ Booking affiliate","🏨 Hotel partnerships","📸 Photography gear affiliates","🗺️ Travel agency collab"],
        }
        tips = STRATEGIES.get(calc_cat, ["📣 Brand sponsorships","🛒 Affiliate marketing","🌟 Channel memberships","<span style='display:inline-flex; align-items:center; gap:8px;'><i data-lucide='box' style='width:18px; height:18px; color:#374151;'></i></span> Merchandise store"])
        _strat_colors = ["#00fff7","#ff00c8","#00ff88","#ffe600"]
        scols = st.columns(4)
        for col, tip, sc in zip(scols, tips, _strat_colors):
            with col:
                st.markdown(f"""
                <div style='background:#f1f5f9;border:1px solid {sc}44;border-radius:8px;
                    padding:16px;text-align:center;height:80px;display:flex;align-items:center;
                    justify-content:center;box-shadow:0 0 12px {sc}22;
                    transition:all 0.2s;'>
                  <div style='font-size:13px;font-weight:600;color:{sc};
                      font-family:Inter,sans-serif;letter-spacing:0.5px;
                      text-shadow:none;'>{tip}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("""<div style='font-size:10px;color:#8b5cf6;margin-top:8px;
      font-family:'JetBrains Mono',monospace;border-top:1px solid #e2e8f0;padding-top:8px;'>
      ⚠ Revenue estimates use industry-average CPM benchmarks. Actual earnings depend on
      audience geography, ad formats, seasonality, and YouTube's revenue share (45% retained).
      Sponsorship and membership estimates are illustrative projections.
    </div>""", unsafe_allow_html=True)
