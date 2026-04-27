# =============================================================================
# Air Quality Clustering — Streamlit Dashboard
# =============================================================================
# Run with:  streamlit run app.py
# Deploy free at: https://streamlit.io/cloud  (connect your GitHub repo)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Air Quality Risk Zones",
    page_icon="🌍",
    layout="wide",
)

RISK_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e67e22", 3: "#e74c3c"}
RISK_LABELS = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk", 3: "Severe Risk"}

CLUSTER_FEATURES = [
    "aqi_value", "co_aqi_value", "ozone_aqi_value",
    "no2_aqi_value", "pm2.5_aqi_value",
    "pollution_spread", "pm25_dominance",
]

# =============================================================================
# DATA LOADING — cached so it only runs once
# =============================================================================

@st.cache_data
def load_and_cluster(filepath: str, k: int = 4):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    df = df.groupby(["country", "city"], as_index=False).mean(numeric_only=True)
    df = df.dropna(subset=["aqi_value", "co_aqi_value", "ozone_aqi_value",
                            "no2_aqi_value", "pm2.5_aqi_value"])

    # Feature engineering
    df["pollution_spread"] = df[["co_aqi_value", "ozone_aqi_value",
                                  "no2_aqi_value", "pm2.5_aqi_value"]].std(axis=1)
    total = (df["co_aqi_value"] + df["ozone_aqi_value"] +
             df["no2_aqi_value"] + df["pm2.5_aqi_value"] + 1e-6)
    df["pm25_dominance"] = df["pm2.5_aqi_value"] / total

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(df[CLUSTER_FEATURES])

    # K-Means
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["kmeans_cluster"] = km.fit_predict(X)
    mean_aqi = df.groupby("kmeans_cluster")["aqi_value"].mean().sort_values()
    rank_map = {c: r for r, c in enumerate(mean_aqi.index)}
    df["risk_level"] = df["kmeans_cluster"].map(rank_map)
    df["risk_label"] = df["risk_level"].map(RISK_LABELS)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(X)
    df["pc1"] = coords_2d[:, 0]
    df["pc2"] = coords_2d[:, 1]

    silhouette = silhouette_score(X, df["kmeans_cluster"])
    return df, X, pca, silhouette


@st.cache_data
def load_coords():
    """Download country centroids — no hardcoding."""
    import requests, io
    try:
        url = "https://raw.githubusercontent.com/google/dspl/master/samples/google/canonical/countries.csv"
        resp = requests.get(url, timeout=10)
        coords = pd.read_csv(io.StringIO(resp.text))

        url2 = "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv"
        resp2 = requests.get(url2, timeout=10)
        mapping = pd.read_csv(io.StringIO(resp2.text))[["name", "alpha-2"]]
        mapping.columns = ["country_name", "country_code"]

        merged = mapping.merge(coords, left_on="country_code", right_on="country", how="inner")
        return dict(zip(merged["country_name"], zip(merged["latitude"], merged["longitude"])))
    except Exception:
        return {}


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("⚙️ Controls")
uploaded = st.sidebar.file_uploader(
    "Upload dataset CSV", type="csv",
    help="Download from: kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset"
)
k = st.sidebar.slider("Number of clusters (K)", min_value=2, max_value=8, value=4)
st.sidebar.markdown("---")
st.sidebar.markdown("""
**About**
This dashboard clusters 23,000+ cities by air pollution patterns using K-Means and displays health risk zones on an interactive world map.

**Tech:** scikit-learn · Folium · Streamlit
""")

# =============================================================================
# MAIN CONTENT
# =============================================================================

st.title("🌍 Air Quality Clustering & Health Risk Zones")
st.markdown("Unsupervised ML project — clustering cities by pollution patterns into health risk zones")

if uploaded is None:
    st.info("👈 Upload the dataset CSV using the sidebar to get started.")
    st.markdown("""
    **How to get the dataset:**
    1. Go to [kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset](https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset)
    2. Click Download
    3. Unzip and upload `global air pollution dataset.csv` above
    """)
    st.stop()

# ── Load & cluster ────────────────────────────────────────────────────────────
with st.spinner("Clustering cities..."):
    df, X, pca, silhouette = load_and_cluster(uploaded, k=k)
    coords_lookup = load_coords()

# =============================================================================
# METRICS ROW
# =============================================================================

col1, col2, col3, col4 = st.columns(4)
col1.metric("Cities analysed", f"{len(df):,}")
col2.metric("Countries", f"{df['country'].nunique()}")
col3.metric("Clusters (K)", k)
col4.metric("Silhouette Score", f"{silhouette:.3f}", help="Closer to 1.0 = better separated clusters")

st.markdown("---")

# =============================================================================
# RISK ZONE BREAKDOWN
# =============================================================================

st.subheader("📊 Risk Zone Breakdown")
cols = st.columns(4)
for level in range(min(k, 4)):
    label = RISK_LABELS.get(level, f"Cluster {level}")
    subset = df[df["risk_level"] == level]
    color = RISK_COLORS.get(level, "#888")
    with cols[level]:
        st.markdown(
            f"<div style='background:{color}22; border-left: 4px solid {color}; "
            f"padding: 12px; border-radius: 6px;'>"
            f"<b style='color:{color}'>{label}</b><br>"
            f"<span style='font-size:24px; font-weight:600'>{len(subset):,}</span> cities<br>"
            f"<span style='color:#666'>Avg AQI: {subset['aqi_value'].mean():.0f}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

st.markdown("---")

# =============================================================================
# TWO COLUMN LAYOUT — Map + PCA
# =============================================================================

left, right = st.columns([3, 2])

with left:
    st.subheader("🗺️ Interactive World Map")

    fmap = folium.Map(location=[20, 10], zoom_start=2, tiles="CartoDB positron")

    # Legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
         background: white; padding: 10px 14px; border-radius: 8px;
         border: 1px solid #ccc; font-family: Arial; font-size: 12px;">
      <b>Health Risk Zone</b><br>
      <span style="color:#2ecc71">●</span> Low Risk &nbsp;
      <span style="color:#f39c12">●</span> Moderate<br>
      <span style="color:#e67e22">●</span> High Risk &nbsp;
      <span style="color:#e74c3c">●</span> Severe
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
            radius=max(5, min(18, row["avg_aqi"] / 15)),
            color=color, fill=True, fill_color=color, fill_opacity=0.75,
            tooltip=row["country"],
            popup=folium.Popup(
                f"<b>{row['country']}</b><br>"
                f"Avg AQI: <b>{row['avg_aqi']:.0f}</b><br>"
                f"Zone: <b>{RISK_LABELS.get(int(row['risk_level']))}</b><br>"
                f"Cities: {int(row['n_cities'])}",
                max_width=200
            ),
        ).add_to(fmap)

    st_folium(fmap, width=700, height=420)

with right:
    st.subheader("🔬 PCA — Cluster Separation")
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f9f9f9")
    var = pca.explained_variance_ratio_

    for level in sorted(df["risk_level"].unique()):
        mask = df["risk_level"] == level
        label = RISK_LABELS.get(level, f"Cluster {level}")
        color = RISK_COLORS.get(level, "#888")
        ax.scatter(df.loc[mask, "pc1"], df.loc[mask, "pc2"],
                   c=color, label=label, alpha=0.6, s=10, edgecolors="none")

    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)", fontsize=10)
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(True, color="#e0e0e0")
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

# =============================================================================
# HEATMAP + TOP CITIES
# =============================================================================

left2, right2 = st.columns([2, 1])

with left2:
    st.subheader("🔥 Pollutant Profile per Risk Zone")
    profile = df.groupby("risk_label")[CLUSTER_FEATURES].mean()
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-6)
    ordered = [RISK_LABELS[i] for i in range(4) if RISK_LABELS[i] in profile_norm.index]
    profile_norm = profile_norm.loc[ordered]

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    fig2.patch.set_facecolor("white")
    sns.heatmap(profile_norm, annot=profile.loc[ordered].round(1),
                fmt=".1f", cmap="YlOrRd", ax=ax2,
                linewidths=0.5, linecolor="#ddd",
                cbar_kws={"label": "Normalised"})
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig2)

with right2:
    st.subheader("🏙️ Most Polluted Cities")
    risk_filter = st.selectbox("Filter by zone", ["All"] + list(RISK_LABELS.values()))
    filtered = df if risk_filter == "All" else df[df["risk_label"] == risk_filter]
    top = (filtered.nlargest(10, "aqi_value")[["city", "country", "aqi_value", "risk_label"]]
           .rename(columns={"aqi_value": "AQI", "risk_label": "Zone"})
           .reset_index(drop=True))
    st.dataframe(top, use_container_width=True, height=300)

st.markdown("---")

# =============================================================================
# DOWNLOAD
# =============================================================================

st.subheader("💾 Download Results")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download clustered dataset (CSV)",
    data=csv,
    file_name="air_quality_clustered.csv",
    mime="text/csv",
)
