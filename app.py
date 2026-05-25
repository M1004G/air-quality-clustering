# =============================================================================
# AirCluster — Global Air Quality Risk Zones
# Run: python -m streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings, os
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AirCluster · Global Risk Zones",
    page_icon="🌫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# STYLES
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], * {
    font-family: 'Inter', system-ui, sans-serif !important;
    box-sizing: border-box;
}

/* ── Backgrounds ── */
[data-testid="stAppViewContainer"] { background: #F4F6F9 !important; }
[data-testid="stMain"], section.main { background: #F4F6F9 !important; }

/* ── Hide Streamlit chrome ── */
[data-testid="stHeader"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }

/* ── Main content ── */
.block-container { padding: 1.5rem 1.5rem 1rem !important; max-width: 100% !important; }

/* ── Sidebar shell ── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
    min-width: 300px !important;
    max-width: 300px !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] { width: 300px !important; }

/* Kill the gap — Streamlit injects a header row with the collapse button
   that sits above sidebar content. Zero out every wrapper. */
[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebar"] > div:first-child > div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
[data-testid="stSidebarContent"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
/* The actual collapse button row */
[data-testid="stSidebarHeader"] {
    padding: 0 !important;
    min-height: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
}
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
    display: none !important;
}
[data-testid="stSidebar"] .stVerticalBlock { gap: 0 !important; }

/* ── Sidebar text ── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #334155; }

/* ── Slider ── */
[data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"] {
    background: #2563EB !important; border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stSliderTrack"] > div:first-child {
    background: #E2E8F0 !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stSliderTrack"] > div:nth-child(2) {
    background: #2563EB !important;
}

/* ── Download button ── */
[data-testid="stSidebar"] .stDownloadButton > button {
    background: #2563EB !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-size: 13px !important; font-weight: 600 !important;
    width: 100% !important; padding: 0.55rem 1rem !important;
}
[data-testid="stSidebar"] .stDownloadButton > button:hover { background: #1D4ED8 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #E2E8F0 !important;
    gap: 0 !important; padding: 0 1rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #94A3B8 !important;
    font-size: 12px !important; font-weight: 500 !important;
    padding: 0.6rem 0.9rem !important;
    border: none !important; border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] { color: #2563EB !important; border-bottom: 2px solid #2563EB !important; }
.stTabs [data-baseweb="tab-panel"] { padding: 1rem !important; background: transparent !important; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #fff !important; border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important; color: #1E293B !important; font-size: 13px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #E2E8F0 !important; border-radius: 10px !important; }

/* ── Cards ── */
.ac-map-wrap {
    border: 1px solid #E2E8F0; border-radius: 12px; overflow: hidden;
    background: #fff; box-shadow: 0 1px 8px rgba(0,0,0,0.06); margin-bottom: 0.75rem;
}
.ac-panel {
    background: #fff; border: 1px solid #E2E8F0; border-radius: 12px;
    overflow: hidden; box-shadow: 0 1px 8px rgba(0,0,0,0.04); margin-bottom: 0.75rem;
}
.ac-panel-hdr {
    padding: 0.8rem 1.25rem; border-bottom: 1px solid #F1F5F9;
    background: #FAFBFC; display: flex; align-items: center; gap: 8px;
}
.ac-page-header { display: flex; align-items: center; margin-bottom: 1rem; }
.ac-stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 0 1.25rem 1rem; }
.ac-stat { background: #F8FAFC; border: 1px solid #F1F5F9; border-radius: 8px; padding: 9px 12px; }
.risk-row { display: flex; align-items: center; gap: 10px; padding: 10px 1.25rem; border-bottom: 1px solid #F8FAFC; }
.risk-row:hover { background: #F8FAFC; }
.section-label {
    font-size: 10.5px; font-weight: 600; color: #94A3B8;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 0.9rem 1.25rem 0.4rem; border-top: 1px solid #F1F5F9; display: block;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
DATASET_PATH = "global air pollution dataset.csv"

RISK_COLORS = {0:"#22C55E", 1:"#84CC16", 2:"#F59E0B", 3:"#F97316", 4:"#EF4444", 5:"#DC2626", 6:"#9333EA", 7:"#6B21A8"}
RISK_LABELS = {0:"Low", 1:"Low-Mod", 2:"Moderate", 3:"Mod-High", 4:"High", 5:"Severe", 6:"Critical", 7:"Extreme"}
RISK_FULL   = {0:"Low Risk", 1:"Low-Moderate", 2:"Moderate Risk", 3:"Mod-High Risk", 4:"High Risk", 5:"Severe Risk", 6:"Critical Risk", 7:"Extreme Risk"}

CLUSTER_FEATURES = ["aqi_value","co_aqi_value","ozone_aqi_value",
                    "no2_aqi_value","pm2.5_aqi_value","pollution_spread","pm25_dominance"]
FEATURE_LABELS = {
    "aqi_value":"AQI","co_aqi_value":"CO","ozone_aqi_value":"Ozone",
    "no2_aqi_value":"NO₂","pm2.5_aqi_value":"PM2.5",
    "pollution_spread":"Spread","pm25_dominance":"PM2.5 Share"
}

# =============================================================================
# COORDINATE LOOKUP
# Two structured DataFrames joined on ISO alpha-2 — same logic as
# air_quality_clustering.py, using bundled data instead of a runtime HTTP fetch.
# Sources: Google DSPL canonical countries + ISO-3166
# =============================================================================
_CENTROIDS = pd.DataFrame([
    ("AF",33.93,67.71),("AL",41.15,20.17),("DZ",28.03,1.66),("AO",-11.20,17.87),
    ("AR",-38.42,-63.62),("AM",40.07,45.04),("AU",-25.27,133.78),("AT",47.52,14.55),
    ("AZ",40.14,47.58),("BH",26.02,50.55),("BD",23.68,90.36),("BY",53.71,27.95),
    ("BE",50.50,4.47),("BO",-16.29,-63.59),("BA",43.92,17.68),("BR",-14.24,-51.93),
    ("BG",42.73,25.49),("KH",12.57,104.99),("CM",3.85,11.50),("CA",56.13,-106.35),
    ("CL",-35.68,-71.54),("CN",35.86,104.20),("CO",4.57,-74.30),("CG",-0.23,15.83),
    ("CR",9.75,-83.75),("HR",45.10,15.20),("CU",21.52,-77.78),("CY",35.13,33.43),
    ("CZ",49.82,15.47),("DK",56.26,9.50),("DO",18.74,-70.16),("EC",-1.83,-78.18),
    ("EG",26.82,30.80),("SV",13.79,-88.90),("ET",9.15,40.49),("FI",61.92,25.75),
    ("FR",46.23,2.21),("GE",42.32,43.36),("DE",51.17,10.45),("GH",7.95,-1.02),
    ("GR",39.07,21.82),("GT",15.78,-90.23),("HN",15.20,-86.24),("HU",47.16,19.50),
    ("IN",20.59,78.96),("ID",-0.79,113.92),("IR",32.43,53.69),("IQ",33.22,43.68),
    ("IE",53.41,-8.24),("IL",31.05,34.85),("IT",41.87,12.57),("CI",7.54,-5.55),
    ("JM",18.11,-77.30),("JP",36.20,138.25),("JO",30.59,36.24),("KZ",48.02,66.92),
    ("KE",-0.02,37.91),("XK",42.60,20.90),("KW",29.31,47.48),("KG",41.20,74.77),
    ("LA",19.86,102.50),("LV",56.88,24.60),("LB",33.85,35.86),("LY",26.34,17.23),
    ("LT",55.17,23.88),("LU",49.82,6.13),("MY",4.21,108.00),("ML",17.57,-3.99),
    ("MX",23.63,-102.55),("MD",47.41,28.37),("MN",46.86,103.85),("MA",31.79,-7.09),
    ("MZ",-18.67,35.53),("MM",21.91,95.96),("NP",28.39,84.12),("NL",52.13,5.29),
    ("NZ",-40.90,174.89),("NI",12.87,-85.21),("NE",17.61,8.08),("NG",9.08,8.68),
    ("KP",40.34,127.51),("MK",41.61,21.75),("NO",60.47,8.47),("OM",21.51,55.92),
    ("PK",30.38,69.35),("PS",31.95,35.23),("PA",8.54,-80.78),("PY",-23.44,-58.44),
    ("PE",-9.19,-75.02),("PH",12.88,121.77),("PL",51.92,19.15),("PT",39.40,-8.22),
    ("QA",25.35,51.18),("RO",45.94,24.97),("RU",61.52,105.32),("SA",23.89,45.08),
    ("SN",14.50,-14.45),("RS",44.02,21.01),("SG",1.35,103.82),("SK",48.67,19.70),
    ("SI",46.15,14.99),("ZA",-30.56,22.94),("KR",35.91,127.77),("ES",40.46,-3.75),
    ("LK",7.87,80.77),("SD",12.86,30.22),("SE",60.13,18.64),("CH",46.82,8.23),
    ("SY",34.80,38.00),("TW",23.70,120.96),("TJ",38.86,71.28),("TZ",-6.37,34.89),
    ("TH",15.87,100.99),("TN",33.89,9.54),("TR",38.96,35.24),("TM",38.97,59.56),
    ("UG",1.37,32.29),("UA",48.38,31.17),("AE",23.42,53.85),("GB",55.38,-3.44),
    ("US",37.09,-95.71),("UY",-32.52,-55.77),("UZ",41.38,64.59),("VE",6.42,-66.59),
    ("VN",14.06,108.28),("YE",15.55,48.52),("ZM",-13.13,27.85),("ZW",-19.02,29.15),
    ("CD",-4.04,21.76),("TT",10.69,-61.22),
], columns=["alpha2","lat","lon"])

_ISO_NAMES = pd.DataFrame([
    ("AF","Afghanistan"),("AL","Albania"),("DZ","Algeria"),("AO","Angola"),
    ("AR","Argentina"),("AM","Armenia"),("AU","Australia"),("AT","Austria"),
    ("AZ","Azerbaijan"),("BH","Bahrain"),("BD","Bangladesh"),("BY","Belarus"),
    ("BE","Belgium"),("BO","Bolivia"),("BA","Bosnia and Herzegovina"),("BR","Brazil"),
    ("BG","Bulgaria"),("KH","Cambodia"),("CM","Cameroon"),("CA","Canada"),
    ("CL","Chile"),("CN","China"),("CO","Colombia"),("CG","Congo"),
    ("CR","Costa Rica"),("HR","Croatia"),("CU","Cuba"),("CY","Cyprus"),
    ("CZ","Czech Republic"),("CZ","Czechia"),("DK","Denmark"),
    ("DO","Dominican Republic"),("EC","Ecuador"),("EG","Egypt"),("SV","El Salvador"),
    ("ET","Ethiopia"),("FI","Finland"),("FR","France"),("GE","Georgia"),
    ("DE","Germany"),("GH","Ghana"),("GR","Greece"),("GT","Guatemala"),
    ("HN","Honduras"),("HU","Hungary"),("IN","India"),("ID","Indonesia"),
    ("IR","Iran"),("IQ","Iraq"),("IE","Ireland"),("IL","Israel"),("IT","Italy"),
    ("CI","Ivory Coast"),("JM","Jamaica"),("JP","Japan"),("JO","Jordan"),
    ("KZ","Kazakhstan"),("KE","Kenya"),("XK","Kosovo"),("KW","Kuwait"),
    ("KG","Kyrgyzstan"),("LA","Laos"),("LV","Latvia"),("LB","Lebanon"),
    ("LY","Libya"),("LT","Lithuania"),("LU","Luxembourg"),("MY","Malaysia"),
    ("ML","Mali"),("MX","Mexico"),("MD","Moldova"),("MN","Mongolia"),
    ("MA","Morocco"),("MZ","Mozambique"),("MM","Myanmar"),("NP","Nepal"),
    ("NL","Netherlands"),("NZ","New Zealand"),("NI","Nicaragua"),("NE","Niger"),
    ("NG","Nigeria"),("KP","North Korea"),("MK","North Macedonia"),("NO","Norway"),
    ("OM","Oman"),("PK","Pakistan"),("PS","Palestine"),("PA","Panama"),
    ("PY","Paraguay"),("PE","Peru"),("PH","Philippines"),("PL","Poland"),
    ("PT","Portugal"),("QA","Qatar"),("RO","Romania"),("RU","Russia"),
    ("SA","Saudi Arabia"),("SN","Senegal"),("RS","Serbia"),("SG","Singapore"),
    ("SK","Slovakia"),("SI","Slovenia"),("ZA","South Africa"),("KR","South Korea"),
    ("KR","Korea, South"),("ES","Spain"),("LK","Sri Lanka"),("SD","Sudan"),
    ("SE","Sweden"),("CH","Switzerland"),("SY","Syria"),("SY","Syrian Arab Republic"),
    ("TW","Taiwan"),("TJ","Tajikistan"),("TZ","Tanzania"),("TH","Thailand"),
    ("TN","Tunisia"),("TR","Turkey"),("TM","Turkmenistan"),("UG","Uganda"),
    ("UA","Ukraine"),("AE","United Arab Emirates"),("GB","United Kingdom"),
    ("US","United States"),("UY","Uruguay"),("UZ","Uzbekistan"),("VE","Venezuela"),
    ("VN","Vietnam"),("VN","Viet Nam"),("YE","Yemen"),("ZM","Zambia"),
    ("ZW","Zimbabwe"),("CD","Democratic Republic of the Congo"),
    ("TT","Trinidad and Tobago"),
], columns=["alpha2","country_name"])


@st.cache_data(show_spinner=False)
def load_coords_lookup() -> dict:
    merged = _ISO_NAMES.merge(_CENTROIDS, on="alpha2", how="inner")
    return dict(zip(merged["country_name"], zip(merged["lat"], merged["lon"])))


# =============================================================================
# DATA
# =============================================================================
@st.cache_data(show_spinner=False)
def load_and_cluster(k=4):
    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.str.strip().str.replace(" ","_").str.lower()
    df = df.groupby(["country","city"], as_index=False).mean(numeric_only=True)
    df = df.dropna(subset=["aqi_value","co_aqi_value","ozone_aqi_value","no2_aqi_value","pm2.5_aqi_value"])
    df["pollution_spread"] = df[["co_aqi_value","ozone_aqi_value","no2_aqi_value","pm2.5_aqi_value"]].std(axis=1)
    total = df["co_aqi_value"]+df["ozone_aqi_value"]+df["no2_aqi_value"]+df["pm2.5_aqi_value"]+1e-6
    df["pm25_dominance"] = df["pm2.5_aqi_value"]/total
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X)
    rank_map = {c:r for r,c in enumerate(df.groupby("cluster")["aqi_value"].mean().sort_values().index)}
    df["risk_level"] = df["cluster"].map(rank_map)
    df["risk_label"] = df["risk_level"].map(RISK_LABELS)
    pca = PCA(n_components=2, random_state=42)
    c2d = pca.fit_transform(X)
    df["pc1"], df["pc2"] = c2d[:,0], c2d[:,1]
    return df, X, pca, silhouette_score(X, df["cluster"])


# =============================================================================
# MAP
# =============================================================================
def build_map(df, coords_lookup, k):
    fmap = folium.Map(location=[20,10], zoom_start=2, tiles="CartoDB positron", prefer_canvas=True)
    csub = df.groupby("country").agg(
        avg_aqi=("aqi_value","mean"),
        risk_level=("risk_level", lambda x: x.mode()[0]),
        n_cities=("city","count")
    ).reset_index()

    for _, row in csub.iterrows():
        coords = coords_lookup.get(row["country"])
        if not coords: continue
        lvl   = int(row["risk_level"])
        color = RISK_COLORS.get(lvl, "#888")
        aqi   = row["avg_aqi"]
        rl    = RISK_FULL.get(lvl, f"Zone {lvl}")
        folium.CircleMarker(
            location=coords, radius=max(5, min(28, aqi/10)),
            color=color, weight=1.5, fill=True, fill_color=color, fill_opacity=0.75,
            tooltip=folium.Tooltip(
                f"""<div style='font-family:Inter,sans-serif;background:#fff;color:#1E293B;
                    padding:12px 16px;border-radius:10px;border:1px solid #E2E8F0;
                    box-shadow:0 4px 20px rgba(0,0,0,0.12);min-width:160px;line-height:1.9'>
                    <b style='font-size:14px'>{row['country']}</b><br>
                    <span style='display:inline-block;background:{color}18;color:{color};
                    border:1px solid {color}44;border-radius:20px;padding:0 10px;
                    font-size:11px;font-weight:600;margin:4px 0 6px'>{rl}</span><br>
                    <span style='color:#94A3B8;font-size:11px'>Avg AQI </span>
                    <span style='color:{color};font-size:22px;font-weight:700'>{aqi:.0f}</span><br>
                    <span style='color:#CBD5E1;font-size:11px'>{int(row['n_cities'])} cities</span>
                    </div>""", sticky=True),
        ).add_to(fmap)

    legend = f"""<div style='position:fixed;bottom:24px;left:24px;z-index:1000;
        background:#fff;padding:14px 18px;border-radius:10px;border:1px solid #E2E8F0;
        font-family:Inter,sans-serif;box-shadow:0 4px 16px rgba(0,0,0,0.08)'>
        <div style='font-size:11px;font-weight:700;color:#1E293B;margin-bottom:8px;
             letter-spacing:0.05em;text-transform:uppercase'>Risk Zone</div>
        {''.join([f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:5px'><div style='width:10px;height:10px;border-radius:50%;background:{RISK_COLORS[i]}'></div><span style='font-size:12px;color:#475569'>{RISK_FULL[i]}</span></div>" for i in range(k) if i in RISK_FULL])}
    </div>"""
    fmap.get_root().html.add_child(folium.Element(legend))
    return fmap


# =============================================================================
# CHARTS
# =============================================================================
BG = "#FFFFFF"; GRID = "#F1F5F9"; MUTED = "#94A3B8"; DARK = "#1E293B"

def make_pca_fig(df, pca, k):
    var = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for level in sorted(df["risk_level"].unique()):
        mask = df["risk_level"] == level
        ax.scatter(df.loc[mask,"pc1"], df.loc[mask,"pc2"],
                   c=RISK_COLORS.get(level,"#888"), alpha=0.55, s=14,
                   edgecolors="none", zorder=3)
    ax.set_xlabel(f"PC1 · {var[0]*100:.1f}% var", fontsize=8.5, color=MUTED)
    ax.set_ylabel(f"PC2 · {var[1]*100:.1f}% var", fontsize=8.5, color=MUTED)
    handles = [mpatches.Patch(color=RISK_COLORS.get(l,"#888"), label=RISK_FULL.get(l))
               for l in sorted(df["risk_level"].unique())]
    leg = ax.legend(handles=handles, fontsize=8, frameon=True,
                    facecolor=BG, edgecolor=GRID, labelcolor="#475569")
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout(pad=0.8)
    return fig

def make_heatmap_fig(df):
    ordered = [RISK_LABELS[i] for i in range(4) if RISK_LABELS[i] in df["risk_label"].values]
    profile = df.groupby("risk_label")[CLUSTER_FEATURES].mean().loc[ordered]
    norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-6)
    norm.columns = [FEATURE_LABELS.get(c,c) for c in norm.columns]
    profile.columns = [FEATURE_LABELS.get(c,c) for c in profile.columns]
    fig, ax = plt.subplots(figsize=(8, 2.6))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    sns.heatmap(norm, annot=profile.round(0), fmt=".0f", cmap="YlOrRd", ax=ax,
                linewidths=2, linecolor="#FFFFFF", cbar=False,
                annot_kws={"size":9, "color":"#1E293B"})
    ax.tick_params(axis="x", labelsize=9, colors=MUTED, bottom=False, rotation=0)
    ax.tick_params(axis="y", labelsize=9, colors=MUTED, left=False, rotation=0)
    rev = {v:k for k,v in RISK_LABELS.items()}
    for tick, label in zip(ax.get_yticklabels(), ordered):
        tick.set_color(RISK_COLORS.get(rev.get(label,0), MUTED))
        tick.set_fontweight("600")
    ax.set_xlabel(""); ax.set_ylabel("")
    fig.tight_layout(pad=0.4)
    return fig

def make_distribution_fig(df, k):
    pollutants  = ["co_aqi_value","ozone_aqi_value","no2_aqi_value","pm2.5_aqi_value"]
    poll_labels = ["CO","Ozone","NO₂","PM2.5"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor(BG)
    zones_present = sorted(df["risk_level"].unique())

    ax1 = axes[0]; ax1.set_facecolor(BG)
    for sp in ax1.spines.values(): sp.set_color(GRID)
    ax1.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax1.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
    counts = [df[df["risk_level"]==i].shape[0] for i in zones_present]
    labels_list = [RISK_FULL.get(i,f"Zone {i}") for i in zones_present]
    colors = [RISK_COLORS.get(i,"#888") for i in zones_present]
    bars = ax1.barh(labels_list, counts, color=colors, alpha=0.85, height=0.52, zorder=3)
    for bar, val in zip(bars, counts):
        ax1.text(val + max(counts)*0.01, bar.get_y()+bar.get_height()/2,
                 f"{val:,}", va="center", ha="left", fontsize=8.5, color=MUTED)
    ax1.set_xlabel("Number of Cities", fontsize=8.5, color=MUTED)
    ax1.set_title("City Distribution by Zone", fontsize=10, fontweight="600", color=DARK, pad=10)
    ax1.invert_yaxis()

    ax2 = axes[1]; ax2.set_facecolor(BG)
    for sp in ax2.spines.values(): sp.set_color(GRID)
    ax2.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax2.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    x = np.arange(len(pollutants)); width = 0.18
    n = len(zones_present); offsets = np.linspace(-(n-1)*width/2, (n-1)*width/2, n)
    for lvl, off in zip(zones_present, offsets):
        means = [df[df["risk_level"]==lvl][p].mean() for p in pollutants]
        ax2.bar(x+off, means, width, color=RISK_COLORS.get(lvl,"#888"),
                alpha=0.82, zorder=3, label=RISK_FULL.get(lvl,f"Zone {lvl}"))
    ax2.set_xticks(x); ax2.set_xticklabels(poll_labels, fontsize=9, color=MUTED)
    ax2.set_ylabel("Mean AQI Value", fontsize=8.5, color=MUTED)
    ax2.set_title("Pollutant Levels by Risk Zone", fontsize=10, fontweight="600", color=DARK, pad=10)
    ax2.axhline(y=15, color="#EF4444", linewidth=1, linestyle="--", alpha=0.5, zorder=2)
    ax2.text(len(pollutants)-0.5, 17, "WHO guideline", fontsize=7.5, color="#EF4444", ha="right")
    leg2 = ax2.legend(fontsize=8, frameon=True, facecolor=BG, edgecolor=GRID,
                      labelcolor="#475569", ncol=2)
    leg2.get_frame().set_linewidth(0.8)
    fig.tight_layout(pad=0.8)
    return fig


# =============================================================================
# GUARD
# =============================================================================
if not os.path.exists(DATASET_PATH):
    st.error(f"Dataset not found: **{DATASET_PATH}** — place it next to app.py and restart.")
    st.stop()

coords_lookup = load_coords_lookup()

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style="height:60px;display:flex;align-items:center;padding:0 1.25rem;
         border-bottom:1px solid #F1F5F9;gap:10px;margin-top:0">
      <div style="width:34px;height:34px;background:#2563EB;border-radius:9px;
           display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0">🌫</div>
      <div>
        <div style="font-size:16px;font-weight:800;line-height:1.15;letter-spacing:-0.02em">
          <span style="color:#1E293B">air</span><span style="color:#2563EB">cluster</span>
        </div>
        <div style="font-size:10px;color:#94A3B8;margin-top:1px">K-Means · PCA · Risk Zones</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="section-label" style="border-top:none">Risk Zones (K)</span>',
                unsafe_allow_html=True)
    k_val = st.slider("", min_value=2, max_value=8, value=4,
                      key="_k_slider", label_visibility="collapsed")

    df, X, pca, sil = load_and_cluster(k=k_val)

    st.markdown('<span class="section-label">Zones Detected</span>', unsafe_allow_html=True)
    for level in range(k_val):
        subset = df[df["risk_level"]==level]
        if subset.empty: continue
        color = RISK_COLORS.get(level,"#888")
        avg   = subset["aqi_value"].mean()
        count = len(subset)
        pct   = count / len(df) * 100
        st.markdown(f"""
        <div class="risk-row">
          <div style="width:10px;height:10px;border-radius:50%;background:{color};flex-shrink:0"></div>
          <div style="flex:1;min-width:0">
            <div style="font-size:12.5px;font-weight:600;color:#1E293B">{RISK_FULL[level]}</div>
            <div style="font-size:10.5px;color:#94A3B8;margin-top:1px">{count:,} cities · avg {avg:.0f}</div>
            <div style="height:3px;background:#F1F5F9;border-radius:2px;margin-top:5px">
              <div style="height:3px;width:{pct:.1f}%;background:{color};border-radius:2px"></div>
            </div>
          </div>
          <b style="font-size:13.5px;color:{color};min-width:38px;text-align:right">{count:,}</b>
        </div>""", unsafe_allow_html=True)

    st.markdown('<span class="section-label">Overview</span>', unsafe_allow_html=True)
    stats = [("Cities",f"{len(df):,}"),("Countries",f"{df['country'].nunique()}"),
             ("Max AQI",f"{df['aqi_value'].max():.0f}"),("Mean AQI",f"{df['aqi_value'].mean():.0f}"),
             ("Silhouette",f"{sil:.3f}")]
    grid_html = '<div class="ac-stat-grid">'
    for key, val in stats:
        grid_html += f'<div class="ac-stat"><div style="font-size:10px;color:#94A3B8;font-weight:500">{key}</div><div style="font-size:16px;font-weight:700;color:#1E293B;margin-top:2px">{val}</div></div>'
    grid_html += "</div>"
    st.markdown(grid_html, unsafe_allow_html=True)

    st.markdown('<div style="padding:0 1.25rem 0.75rem">', unsafe_allow_html=True)
    st.download_button("↓  Export Results",
                       data=df.to_csv(index=False).encode("utf-8"),
                       file_name="air_quality_clustered.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0.75rem 1.25rem;font-size:10px;color:#CBD5E1;border-top:1px solid #F1F5F9;line-height:2">
      Data · WHO / Kaggle &nbsp;·&nbsp; hasibalmuzdadid
    </div>""", unsafe_allow_html=True)

# =============================================================================
# MAIN
# =============================================================================
st.markdown("""
<div class="ac-page-header">
  <div>
    <div style="font-size:18px;font-weight:700;color:#1E293B;line-height:1.2">Global Air Quality Distribution</div>
    <div style="font-size:12px;color:#94A3B8;margin-top:3px">Hover any country marker for details</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="ac-map-wrap">', unsafe_allow_html=True)
st_folium(build_map(df, coords_lookup, k_val), width="100%", height=440, returned_objects=[])
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="ac-panel">
  <div class="ac-panel-hdr">
    <div style="width:4px;height:16px;background:#2563EB;border-radius:2px"></div>
    <span style="font-size:13px;font-weight:600;color:#1E293B">Analysis</span>
  </div>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "PCA · Cluster Separation", "Pollutant Distribution", "Pollution Profile", "City Explorer",
])

with tab1:
    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        st.caption("Each dot = one city. 7 features compressed to 2D via PCA.")
        st.pyplot(make_pca_fig(df, pca, k_val), use_container_width=True)
    with c2:
        st.markdown('<p style="font-size:11px;font-weight:600;color:#94A3B8;text-transform:uppercase;letter-spacing:0.08em">Top cities per zone</p>', unsafe_allow_html=True)
        for level in range(k_val):
            subset = df[df["risk_level"]==level]
            if subset.empty: continue
            color = RISK_COLORS.get(level,"#888")
            st.markdown(f'<div style="font-size:11px;font-weight:700;color:{color};margin:12px 0 5px;padding-bottom:4px;border-bottom:2px solid {color}22">{RISK_FULL[level]}</div>', unsafe_allow_html=True)
            for _, r in subset.nlargest(2,"aqi_value")[["city","country","aqi_value"]].iterrows():
                st.markdown(f'<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #F8FAFC;align-items:center"><div><div style="font-size:12px;color:#1E293B;font-weight:500">{r["city"]}</div><div style="font-size:10.5px;color:#94A3B8">{r["country"]}</div></div><b style="font-size:14px;color:{color}">{r["aqi_value"]:.0f}</b></div>', unsafe_allow_html=True)

with tab2:
    st.caption("City counts and mean pollutant AQI per risk zone. Dashed = WHO guideline.")
    st.pyplot(make_distribution_fig(df, k_val), use_container_width=True)

with tab3:
    st.caption("Mean AQI per pollutant. Color intensity = relative severity.")
    st.pyplot(make_heatmap_fig(df), use_container_width=True)

with tab4:
    fc, sc, _ = st.columns([1,1,2])
    with fc:
        zone_filter = st.selectbox("Zone", ["All"] + [RISK_FULL.get(i,f"Zone {i}") for i in range(k_val)])
    with sc:
        sort_by = st.selectbox("Sort by", ["AQI ↓","PM2.5 ↓","NO₂ ↓"])
    rev_full = {RISK_FULL.get(k2,f"Zone {k2}"):k2 for k2 in range(8)}
    filtered = df if zone_filter=="All" else df[df["risk_level"]==rev_full.get(zone_filter,0)]
    sort_col = {"AQI ↓":"aqi_value","PM2.5 ↓":"pm2.5_aqi_value","NO₂ ↓":"no2_aqi_value"}[sort_by]
    top = (filtered.nlargest(20,sort_col)
           [["city","country","aqi_value","pm2.5_aqi_value","no2_aqi_value","risk_label"]]
           .rename(columns={"city":"City","country":"Country","aqi_value":"AQI",
                            "pm2.5_aqi_value":"PM2.5","no2_aqi_value":"NO₂","risk_label":"Zone"})
           .reset_index(drop=True))
    top.index += 1
    st.dataframe(top, use_container_width=True, height=260)

st.markdown("""
<div style="display:flex;justify-content:space-between;border-top:1px solid #E2E8F0;padding-top:0.65rem;margin-top:0.5rem">
  <span style="font-size:11px;color:#CBD5E1">AirCluster · K-Means + PCA · scikit-learn</span>
  <span style="font-size:11px;color:#CBD5E1">Data · WHO / Kaggle</span>
</div>""", unsafe_allow_html=True)
