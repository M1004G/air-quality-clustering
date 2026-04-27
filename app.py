# =============================================================================
# Air Quality Clustering — Streamlit Dashboard
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Air Quality Risk Zones",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
RISK_COLORS  = {0: "#4ade80", 1: "#facc15", 2: "#fb923c", 3: "#f43f5e"}
RISK_LABELS  = {0: "Low Risk", 1: "Moderate", 2: "High Risk", 3: "Severe"}
RISK_ICONS   = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}

CLUSTER_FEATURES = [
    "aqi_value", "co_aqi_value", "ozone_aqi_value",
    "no2_aqi_value", "pm2.5_aqi_value",
    "pollution_spread", "pm25_dominance",
]

FEATURE_LABELS = {
    "aqi_value":        "AQI",
    "co_aqi_value":     "CO",
    "ozone_aqi_value":  "Ozone",
    "no2_aqi_value":    "NO₂",
    "pm2.5_aqi_value":  "PM2.5",
    "pollution_spread": "Spread",
    "pm25_dominance":   "PM2.5 Dom.",
}

# ── Global stylesheet ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp { background: #0f1117; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #16181f;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * { color: #a0aec0 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #e2e8f0 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #16181f;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 1rem 1.25rem;
}
[data-testid="metric-container"] label { color: #64748b !important; font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.6rem; font-weight: 600; }

/* ── Subheaders ── */
h2, h3 { color: #f1f5f9 !important; font-weight: 500 !important; letter-spacing: -0.02em; }

/* ── Dividers ── */
hr { border-color: #1e2130 !important; margin: 1.5rem 0 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Selectbox / file uploader ── */
[data-testid="stSelectbox"] select,
[data-baseweb="select"] { background: #16181f !important; border-color: #1e2130 !important; color: #e2e8f0 !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #4ade80 !important; }

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: #16181f !important;
    border: 1px solid #1e2130 !important;
    color: #a0aec0 !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: #4ade80 !important;
    color: #4ade80 !important;
}

/* ── Info box ── */
[data-testid="stInfo"] {
    background: #16181f !important;
    border-left: 3px solid #4ade80 !important;
    border-radius: 8px !important;
    color: #a0aec0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#16181f",
    "axes.facecolor":    "#16181f",
    "axes.edgecolor":    "#1e2130",
    "axes.labelcolor":   "#64748b",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "text.color":        "#a0aec0",
    "grid.color":        "#1e2130",
    "grid.linewidth":    0.8,
    "font.family":       "sans-serif",
    "font.size":         10,
    "legend.frameon":    False,
    "legend.labelcolor": "#a0aec0",
})

# =============================================================================
# DATA — cached
# =============================================================================

@st.cache_data
def load_and_cluster(filepath, k: int = 4):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    df = df.groupby(["country", "city"], as_index=False).mean(numeric_only=True)
    df = df.dropna(subset=["aqi_value", "co_aqi_value", "ozone_aqi_value",
                            "no2_aqi_value", "pm2.5_aqi_value"])

    df["pollution_spread"] = df[["co_aqi_value", "ozone_aqi_value",
                                  "no2_aqi_value", "pm2.5_aqi_value"]].std(axis=1)
    total = (df["co_aqi_value"] + df["ozone_aqi_value"] +
             df["no2_aqi_value"] + df["pm2.5_aqi_value"] + 1e-6)
    df["pm25_dominance"] = df["pm2.5_aqi_value"] / total

    scaler = StandardScaler()
    X = scaler.fit_transform(df[CLUSTER_FEATURES])

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["kmeans_cluster"] = km.fit_predict(X)
    mean_aqi = df.groupby("kmeans_cluster")["aqi_value"].mean().sort_values()
    rank_map  = {c: r for r, c in enumerate(mean_aqi.index)}
    df["risk_level"] = df["kmeans_cluster"].map(rank_map)
    df["risk_label"]  = df["risk_level"].map(RISK_LABELS)

    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(X)
    df["pc1"] = coords_2d[:, 0]
    df["pc2"] = coords_2d[:, 1]

    silhouette = silhouette_score(X, df["kmeans_cluster"])
    return df, X, pca, silhouette


@st.cache_data
def load_coords():
    import requests, io
    try:
        url  = "https://raw.githubusercontent.com/google/dspl/master/samples/google/canonical/countries.csv"
        url2 = "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv"
        coords  = pd.read_csv(io.StringIO(requests.get(url,  timeout=10).text))
        mapping = pd.read_csv(io.StringIO(requests.get(url2, timeout=10).text))[["name", "alpha-2"]]
        mapping.columns = ["country_name", "country_code"]
        merged = mapping.merge(coords, left_on="country_code", right_on="country", how="inner")
        return dict(zip(merged["country_name"], zip(merged["latitude"], merged["longitude"])))
    except Exception:
        return {}

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## 🌍 Air Quality")
    st.markdown("<p style='color:#4a5568;font-size:0.8rem;margin-top:-0.5rem;'>Risk Zone Explorer</p>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded = st.file_uploader(
        "Dataset CSV", type="csv",
        help="Download from Kaggle: hasibalmuzdadid/global-air-pollution-dataset",
    )
    k = st.slider("Clusters (K)", min_value=2, max_value=8, value=4)

    st.markdown("---")
    st.markdown("""
<p style='font-size:0.78rem; color:#4a5568; line-height:1.6;'>
Clusters 23 k+ cities by pollution patterns using <b style='color:#64748b'>K-Means</b>.<br><br>
<b style='color:#64748b'>Stack</b><br>
scikit-learn · Folium · Streamlit
</p>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div style='margin-bottom:1.5rem;'>
  <h1 style='color:#f1f5f9; font-size:1.9rem; font-weight:600; letter-spacing:-0.03em; margin:0;'>
    Air Quality Risk Zones
  </h1>
  <p style='color:#4a5568; font-size:0.9rem; margin-top:0.3rem;'>
    Unsupervised clustering · global pollution patterns · health risk classification
  </p>
</div>
""", unsafe_allow_html=True)

if uploaded is None:
    st.info("Upload a dataset CSV in the sidebar to begin.")
    st.markdown("""
**Getting the dataset**
1. Visit [Kaggle — Global Air Pollution Dataset](https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset)
2. Download and unzip
3. Upload `global air pollution dataset.csv` via the sidebar
""")
    st.stop()

# =============================================================================
# LOAD
# =============================================================================

with st.spinner("Running K-Means clustering…"):
    df, X, pca, silhouette = load_and_cluster(uploaded, k=k)
    coords_lookup = load_coords()

# =============================================================================
# KPI ROW
# =============================================================================

c1, c2, c3, c4 = st.columns(4)
c1.metric("Cities", f"{len(df):,}")
c2.metric("Countries", f"{df['country'].nunique()}")
c3.metric("Clusters", k)
c4.metric("Silhouette", f"{silhouette:.3f}")

st.markdown("---")

# =============================================================================
# RISK CARDS
# =============================================================================

st.markdown("#### Risk Zone Breakdown")
cols = st.columns(min(k, 4))
for level in range(min(k, 4)):
    subset = df[df["risk_level"] == level]
    color  = RISK_COLORS.get(level, "#888")
    label  = RISK_LABELS.get(level, f"Cluster {level}")
    icon   = RISK_ICONS.get(level, "⚪")
    avg    = subset["aqi_value"].mean()
    with cols[level]:
        st.markdown(f"""
<div style='
  background:#16181f;
  border:1px solid #1e2130;
  border-top:3px solid {color};
  border-radius:10px;
  padding:1rem 1.1rem;
'>
  <div style='font-size:0.7rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;'>
    {icon} {label}
  </div>
  <div style='font-size:1.8rem;font-weight:600;color:#f1f5f9;line-height:1;'>{len(subset):,}</div>
  <div style='font-size:0.75rem;color:#4a5568;margin-top:0.3rem;'>cities · avg AQI {avg:.0f}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# MAP + PCA
# =============================================================================

left, right = st.columns([3, 2], gap="large")

# ── MAP ────────────────────────────────────────────────────────────────────────
with left:
    st.markdown("#### 🗺 World Map")

    fmap = folium.Map(location=[20, 10], zoom_start=2,
                      tiles="CartoDB dark_matter")

    legend_html = """
<div style="
  position:fixed;bottom:20px;left:20px;z-index:1000;
  background:rgba(22,24,31,0.92);backdrop-filter:blur(6px);
  padding:10px 14px;border-radius:8px;border:1px solid #1e2130;
  font-family:'DM Sans',sans-serif;font-size:11px;color:#a0aec0;">
  <div style="font-weight:600;margin-bottom:6px;color:#e2e8f0;">Health Risk Zone</div>
  <span style="color:#4ade80">●</span> Low &nbsp;&nbsp;
  <span style="color:#facc15">●</span> Moderate<br>
  <span style="color:#fb923c">●</span> High &nbsp;&nbsp;
  <span style="color:#f43f5e">●</span> Severe
</div>"""
    fmap.get_root().html.add_child(folium.Element(legend_html))

    country_summary = (
        df.groupby("country")
        .agg(avg_aqi=("aqi_value", "mean"),
             risk_level=("risk_level", lambda x: x.mode()[0]),
             n_cities=("city", "count"))
        .reset_index()
    )

    for _, row in country_summary.iterrows():
        coords = coords_lookup.get(row["country"])
        if coords is None:
            continue
        color = RISK_COLORS.get(int(row["risk_level"]), "#888")
        folium.CircleMarker(
            location=coords,
            radius=max(4, min(16, row["avg_aqi"] / 15)),
            color=color, fill=True, fill_color=color, fill_opacity=0.7,
            weight=1,
            tooltip=f"{row['country']} · AQI {row['avg_aqi']:.0f}",
            popup=folium.Popup(
                f"<div style='font-family:sans-serif;font-size:12px;'>"
                f"<b>{row['country']}</b><br>"
                f"Avg AQI: <b>{row['avg_aqi']:.0f}</b><br>"
                f"Zone: {RISK_LABELS.get(int(row['risk_level']))}<br>"
                f"Cities: {int(row['n_cities'])}</div>",
                max_width=180,
            ),
        ).add_to(fmap)

    st_folium(fmap, width=None, height=400, use_container_width=True)

# ── PCA ────────────────────────────────────────────────────────────────────────
with right:
    st.markdown("#### 🔬 Cluster Separation (PCA)")
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(5, 4.2))

    for level in sorted(df["risk_level"].unique()):
        mask  = df["risk_level"] == level
        color = RISK_COLORS.get(level, "#888")
        label = RISK_LABELS.get(level, f"Cluster {level}")
        ax.scatter(df.loc[mask, "pc1"], df.loc[mask, "pc2"],
                   c=color, label=label, alpha=0.55, s=8, linewidths=0)

    ax.set_xlabel(f"PC1  {var[0]*100:.1f}% var", fontsize=9)
    ax.set_ylabel(f"PC2  {var[1]*100:.1f}% var", fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8, markerscale=2.5,
              facecolor="#16181f", edgecolor="#1e2130",
              labelcolor="#a0aec0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2130")
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.markdown("---")

# =============================================================================
# HEATMAP + TOP CITIES
# =============================================================================

left2, right2 = st.columns([3, 2], gap="large")

with left2:
    st.markdown("#### 🔥 Pollutant Profile by Risk Zone")
    profile      = df.groupby("risk_label")[CLUSTER_FEATURES].mean()
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-6)
    ordered      = [RISK_LABELS[i] for i in range(4) if RISK_LABELS[i] in profile_norm.index]
    profile_norm = profile_norm.loc[ordered]
    profile_norm.columns = [FEATURE_LABELS.get(c, c) for c in profile_norm.columns]
    profile_disp = profile.loc[ordered].copy()
    profile_disp.columns = [FEATURE_LABELS.get(c, c) for c in profile_disp.columns]

    fig2, ax2 = plt.subplots(figsize=(8, 2.8))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "aq", ["#16181f", "#f43f5e"], N=256
    )
    sns.heatmap(
        profile_norm, annot=profile_disp.round(0), fmt=".0f",
        cmap=cmap, ax=ax2,
        linewidths=0.5, linecolor="#0f1117",
        cbar_kws={"label": "normalised", "shrink": 0.8},
        annot_kws={"fontsize": 8, "color": "#e2e8f0"},
    )
    ax2.set_xlabel("", fontsize=0)
    ax2.set_ylabel("", fontsize=0)
    ax2.tick_params(axis="x", labelsize=8, colors="#64748b")
    ax2.tick_params(axis="y", labelsize=8, colors="#64748b")
    cbar = ax2.collections[0].colorbar
    cbar.ax.tick_params(colors="#64748b", labelsize=7)
    cbar.set_label("normalised", color="#4a5568", fontsize=8)
    plt.tight_layout(pad=0.4)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

with right2:
    st.markdown("#### 🏙 Most Polluted Cities")
    risk_filter = st.selectbox(
        "Filter by zone",
        ["All"] + [RISK_LABELS[i] for i in range(min(k, 4))],
        label_visibility="collapsed",
    )
    filtered = df if risk_filter == "All" else df[df["risk_label"] == risk_filter]
    top = (
        filtered.nlargest(10, "aqi_value")
        [["city", "country", "aqi_value", "risk_label"]]
        .rename(columns={"aqi_value": "AQI", "risk_label": "Zone"})
        .reset_index(drop=True)
    )
    st.dataframe(top, use_container_width=True, height=290,
                 column_config={"AQI": st.column_config.NumberColumn(format="%.0f")})

st.markdown("---")

# =============================================================================
# DOWNLOAD
# =============================================================================

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "↓  Download clustered dataset (CSV)",
    data=csv_bytes,
    file_name="air_quality_clustered.csv",
    mime="text/csv",
)
