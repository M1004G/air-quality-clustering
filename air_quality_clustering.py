# =============================================================================
# Air Quality Clustering & Health Risk Zones
# Resume ML Project | Unsupervised Learning
# =============================================================================
# Dataset: Global Air Pollution Dataset
# Source:  https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset
#
# What this project does:
#   1. Loads and cleans global city-level air quality data
#   2. Engineers features from AQI values (PM2.5, NO2, O3, CO)
#   3. Applies K-Means AND DBSCAN — compares both
#   4. Uses PCA to visualize clusters in 2D
#   5. Labels clusters as health risk zones (Low / Moderate / High / Severe)
#   6. Plots an interactive choropleth world map with Folium
#
# How to run:
#   pip install pandas numpy scikit-learn matplotlib seaborn folium requests
#   python air_quality_clustering.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import requests
import io
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import folium

# ── Plotting style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f9f9f9",
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "font.family": "DejaVu Sans",
})

RISK_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e67e22", 3: "#e74c3c"}
RISK_LABELS = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk", 3: "Severe Risk"}


# =============================================================================
# 1. LOAD & INSPECT DATA
# =============================================================================

def load_data(filepath: str = "global air pollution dataset.csv") -> pd.DataFrame:
    """Load the Kaggle dataset."""
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print("\nColumns:", df.columns.tolist())
    print("\nMissing values:\n", df.isnull().sum())
    return df


# =============================================================================
# 2. LOAD COUNTRY COORDINATES (proper approach — no hardcoding)
# =============================================================================

def load_country_coords() -> pd.DataFrame:
    """
    Downloads a free country centroids CSV from a public dataset on GitHub.
    Source: datasets/country-capitals (public domain)
    Falls back to a minimal inline set if the download fails (e.g. no internet).
    """
    url = "https://raw.githubusercontent.com/google/dspl/master/samples/google/canonical/countries.csv"
    try:
        print("Fetching country coordinates...")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        coords_df = pd.read_csv(io.StringIO(resp.text))
        # This CSV has columns: country, latitude, longitude
        coords_df = coords_df.rename(columns={"country": "country_code",
                                               "latitude": "lat",
                                               "longitude": "lon"})
        print(f"Loaded coordinates for {len(coords_df)} countries.")
        return coords_df
    except Exception as e:
        print(f"Could not fetch coordinates online ({e}). Using fallback set.")
        # Minimal fallback — only used if internet is unavailable
        fallback = {
            "India": (20.59, 78.96), "China": (35.86, 104.19),
            "United States of America": (37.09, -95.71),
            "Pakistan": (30.37, 69.34), "Bangladesh": (23.68, 90.35),
            "Indonesia": (-0.79, 113.92), "Nigeria": (9.08, 8.67),
            "Brazil": (14.23, -51.92), "Russia": (61.52, 105.31),
            "Germany": (51.16, 10.45),
        }
        rows = [{"name": k, "lat": v[0], "lon": v[1]} for k, v in fallback.items()]
        return pd.DataFrame(rows)


def load_country_name_mapping() -> pd.DataFrame:
    """
    Downloads a country name ↔ ISO code mapping so we can join our dataset
    (which uses full country names) with the coordinates CSV (which uses codes).

    Source: lukes/ISO-3166-Countries-with-Regional-Codes (public domain)
    """
    url = "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        mapping = pd.read_csv(io.StringIO(resp.text))
        # Keep only the columns we need
        mapping = mapping[["name", "alpha-2"]].rename(
            columns={"name": "country_name", "alpha-2": "country_code"}
        )
        print(f"Loaded name mapping for {len(mapping)} countries.")
        return mapping
    except Exception as e:
        print(f"Could not fetch name mapping ({e}).")
        return pd.DataFrame(columns=["country_name", "country_code"])


def build_coords_lookup(coords_df: pd.DataFrame,
                         mapping_df: pd.DataFrame) -> dict:
    """
    Merges the two DataFrames to produce a dict of {country_name: (lat, lon)}.
    This replaces the 80-line hardcoded dictionary entirely.
    """
    if mapping_df.empty or "country_code" not in coords_df.columns:
        # Fallback: coords_df already has a 'name' column (from fallback path)
        if "name" in coords_df.columns:
            return dict(zip(coords_df["name"],
                            zip(coords_df["lat"], coords_df["lon"])))
        return {}

    merged = mapping_df.merge(coords_df, on="country_code", how="inner")
    lookup = dict(zip(merged["country_name"],
                      zip(merged["lat"], merged["lon"])))
    print(f"Coordinate lookup built for {len(lookup)} countries.")
    return lookup


# =============================================================================
# 3. DATA CLEANING & FEATURE ENGINEERING
# =============================================================================

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset and engineer features for clustering.

    Original columns (Kaggle dataset):
        Country, City, AQI Value, AQI Category,
        CO AQI Value, Ozone AQI Value, NO2 AQI Value, PM2.5 AQI Value
    """
    df = df.copy()

    # Standardise column names
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

    # Drop duplicates — keep one row per city (take the mean if duplicated)
    df = df.groupby(["country", "city"], as_index=False).mean(numeric_only=True)

    # Drop rows with any missing AQI values
    feature_cols = ["aqi_value", "co_aqi_value", "ozone_aqi_value",
                    "no2_aqi_value", "pm2.5_aqi_value"]
    df = df.dropna(subset=feature_cols)

    # ── Extra engineered features ──────────────────────────────────────────────
    # Pollution diversity: how spread out are the individual pollutant AQIs?
    df["pollution_spread"] = df[["co_aqi_value", "ozone_aqi_value",
                                  "no2_aqi_value", "pm2.5_aqi_value"]].std(axis=1)

    # PM2.5 dominance: share of total AQI explained by PM2.5
    total_pollutants = (df["co_aqi_value"] + df["ozone_aqi_value"] +
                        df["no2_aqi_value"] + df["pm2.5_aqi_value"] + 1e-6)
    df["pm25_dominance"] = df["pm2.5_aqi_value"] / total_pollutants

    print(f"\nClean dataset: {df.shape[0]} cities across {df['country'].nunique()} countries")
    return df


# =============================================================================
# 4. SCALING
# =============================================================================

CLUSTER_FEATURES = [
    "aqi_value", "co_aqi_value", "ozone_aqi_value",
    "no2_aqi_value", "pm2.5_aqi_value",
    "pollution_spread", "pm25_dominance",
]

def scale_features(df: pd.DataFrame):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[CLUSTER_FEATURES])
    print(f"\nFeature matrix shape: {X.shape}")
    return X, scaler


# =============================================================================
# 5. OPTIMAL K — ELBOW + SILHOUETTE
# =============================================================================

def find_optimal_k(X: np.ndarray, k_range: range = range(2, 11)) -> int:
    inertias, silhouettes = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(list(k_range), inertias, "o-", color="#3498db", linewidth=2)
    ax1.set_title("Elbow Method — Inertia vs K")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia")

    ax2.plot(list(k_range), silhouettes, "o-", color="#e74c3c", linewidth=2)
    ax2.set_title("Silhouette Score vs K")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score")

    plt.suptitle("Choosing Optimal K for K-Means", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("01_optimal_k.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: 01_optimal_k.png")

    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"\nBest K by silhouette: {best_k}  (score={max(silhouettes):.3f})")
    return best_k


# =============================================================================
# 6. K-MEANS CLUSTERING
# =============================================================================

def run_kmeans(df: pd.DataFrame, X: np.ndarray, k: int = 4) -> pd.DataFrame:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["kmeans_cluster"] = km.fit_predict(X)

    # Map clusters to risk levels by mean AQI
    mean_aqi = df.groupby("kmeans_cluster")["aqi_value"].mean().sort_values()
    rank_map = {cluster: rank for rank, cluster in enumerate(mean_aqi.index)}
    df["risk_level"] = df["kmeans_cluster"].map(rank_map)
    df["risk_label"] = df["risk_level"].map(RISK_LABELS)

    silhouette = silhouette_score(X, df["kmeans_cluster"])
    print(f"\nK-Means (k={k}) — Silhouette Score: {silhouette:.3f}")
    print("\nCluster sizes:\n", df["risk_label"].value_counts())

    print("\nTop 5 most polluted cities per risk zone:")
    for level in sorted(df["risk_level"].unique(), reverse=True):
        subset = df[df["risk_level"] == level].nlargest(5, "aqi_value")[["city", "country", "aqi_value"]]
        print(f"\n  {RISK_LABELS[level]}:")
        print(subset.to_string(index=False))

    return df


# =============================================================================
# 7. DBSCAN CLUSTERING (comparison)
# =============================================================================

def run_dbscan(df: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    """
    DBSCAN finds dense clusters without requiring K upfront.
    Noise points are labelled -1.
    """
    db = DBSCAN(eps=1.2, min_samples=5)
    df["dbscan_cluster"] = db.fit_predict(X)

    n_clusters = len(set(df["dbscan_cluster"])) - (1 if -1 in df["dbscan_cluster"].values else 0)
    n_noise = (df["dbscan_cluster"] == -1).sum()
    print(f"\nDBSCAN — {n_clusters} clusters found, {n_noise} noise points")

    if n_clusters > 1:
        mask = df["dbscan_cluster"] != -1
        score = silhouette_score(X[mask], df.loc[mask, "dbscan_cluster"])
        print(f"DBSCAN Silhouette Score (excl. noise): {score:.3f}")

    return df


# =============================================================================
# 8. PCA VISUALISATION
# =============================================================================

def plot_pca(df: pd.DataFrame, X: np.ndarray):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, method, col in zip(
        axes,
        ["K-Means Risk Zones", "DBSCAN Clusters"],
        ["risk_level", "dbscan_cluster"],
    ):
        unique_vals = sorted(df[col].unique())
        cmap = plt.cm.get_cmap("RdYlGn_r", len(unique_vals))

        for i, val in enumerate(unique_vals):
            mask = df[col] == val
            label = RISK_LABELS.get(val, f"Cluster {val}" if val != -1 else "Noise")
            color = RISK_COLORS.get(val, cmap(i)) if col == "risk_level" else cmap(i)
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[color], label=label, alpha=0.7, s=30, edgecolors="none")

        ax.set_title(f"PCA — {method}", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)")
        ax.legend(fontsize=8, markerscale=1.5)

    plt.suptitle("2D PCA Projection of Air Quality Clusters", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("02_pca_clusters.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: 02_pca_clusters.png")


# =============================================================================
# 9. FEATURE HEATMAP
# =============================================================================

def plot_cluster_profiles(df: pd.DataFrame):
    profile = df.groupby("risk_label")[CLUSTER_FEATURES].mean()
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-6)

    ordered = [RISK_LABELS[i] for i in range(4) if RISK_LABELS[i] in profile_norm.index]
    profile_norm = profile_norm.loc[ordered]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(profile_norm, annot=profile.loc[ordered].round(1),
                fmt=".1f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, linecolor="#ddd",
                cbar_kws={"label": "Normalised value"})
    ax.set_title("Pollution Profile per Risk Zone\n(values = mean AQI, color = normalised intensity)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig("03_cluster_profiles.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: 03_cluster_profiles.png")


# =============================================================================
# 10. FOLIUM MAP — with proper geocoding (no hardcoded coords)
# =============================================================================

def build_folium_map(df: pd.DataFrame, coords_lookup: dict) -> folium.Map:
    fmap = folium.Map(location=[20, 10], zoom_start=2, tiles="CartoDB positron")

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
         background: white; padding: 12px 16px; border-radius: 8px;
         border: 1px solid #ccc; font-family: Arial; font-size: 13px;">
      <b>Health Risk Zone</b><br>
      <span style="color:#2ecc71">&#9679;</span> Low Risk<br>
      <span style="color:#f39c12">&#9679;</span> Moderate Risk<br>
      <span style="color:#e67e22">&#9679;</span> High Risk<br>
      <span style="color:#e74c3c">&#9679;</span> Severe Risk
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    country_summary = (
        df.groupby("country")
        .agg(avg_aqi=("aqi_value", "mean"),
             risk_level=("risk_level", lambda x: x.mode()[0]),
             n_cities=("city", "count"))
        .reset_index()
    )

    matched, skipped = 0, 0
    for _, row in country_summary.iterrows():
        # ── Professional lookup — uses the downloaded mapping, not hardcoding ──
        coords = coords_lookup.get(row["country"])
        if coords is None:
            skipped += 1
            continue

        matched += 1
        color = RISK_COLORS.get(int(row["risk_level"]), "#888")
        popup_text = (
            f"<b>{row['country']}</b><br>"
            f"Avg AQI: <b>{row['avg_aqi']:.0f}</b><br>"
            f"Risk Zone: <b>{RISK_LABELS.get(int(row['risk_level']))}</b><br>"
            f"Cities tracked: {int(row['n_cities'])}"
        )

        folium.CircleMarker(
            location=coords,
            radius=max(5, min(18, row["avg_aqi"] / 15)),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            tooltip=row["country"],
            popup=folium.Popup(popup_text, max_width=200),
        ).add_to(fmap)

    print(f"Map: {matched} countries plotted, {skipped} skipped (name mismatch)")
    fmap.save("04_risk_zone_map.html")
    print("Saved: 04_risk_zone_map.html  (open in any browser)")
    return fmap


# =============================================================================
# 11. SUMMARY REPORT
# =============================================================================

def print_summary(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  AIR QUALITY CLUSTERING — SUMMARY REPORT")
    print("="*60)
    for level in range(4):
        subset = df[df["risk_level"] == level]
        if subset.empty:
            continue
        top_countries = subset.groupby("country")["aqi_value"].mean().nlargest(3).index.tolist()
        print(f"\n{RISK_LABELS[level]} (n={len(subset)} cities)")
        print(f"  Avg AQI      : {subset['aqi_value'].mean():.1f}")
        print(f"  Avg PM2.5 AQI: {subset['pm2.5_aqi_value'].mean():.1f}")
        print(f"  Top countries: {', '.join(top_countries)}")
    print("\n" + "="*60)
    print("Output files:")
    print("  01_optimal_k.png        — Elbow + Silhouette plots")
    print("  02_pca_clusters.png     — 2D PCA cluster view (K-Means vs DBSCAN)")
    print("  03_cluster_profiles.png — Heatmap of pollutant profiles per zone")
    print("  04_risk_zone_map.html   — Interactive Folium world map")
    print("  air_quality_clustered.csv — Enriched dataset with cluster labels")
    print("="*60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  AIR QUALITY CLUSTERING & HEALTH RISK ZONES")
    print("=" * 60)

    # ── Step 1: Load data ──────────────────────────────────────────────────────
    df = load_data("global air pollution dataset.csv")

    # ── Step 2: Load coordinates properly (no hardcoding) ─────────────────────
    coords_df  = load_country_coords()
    mapping_df = load_country_name_mapping()
    coords_lookup = build_coords_lookup(coords_df, mapping_df)

    # ── Step 3: Clean & feature engineer ──────────────────────────────────────
    df = clean_and_engineer(df)

    # ── Step 4: Scale ─────────────────────────────────────────────────────────
    X, scaler = scale_features(df)

    # ── Step 5: Find optimal K ────────────────────────────────────────────────
    find_optimal_k(X)
    k = 4   # Fixed at 4 → maps to Low / Moderate / High / Severe

    # ── Step 6: K-Means ───────────────────────────────────────────────────────
    df = run_kmeans(df, X, k=k)

    # ── Step 7: DBSCAN ────────────────────────────────────────────────────────
    df = run_dbscan(df, X)

    # ── Step 8: PCA visualisation ─────────────────────────────────────────────
    plot_pca(df, X)

    # ── Step 9: Cluster profiles heatmap ─────────────────────────────────────
    plot_cluster_profiles(df)

    # ── Step 10: Folium map ───────────────────────────────────────────────────
    build_folium_map(df, coords_lookup)

    # ── Step 11: Summary ──────────────────────────────────────────────────────
    print_summary(df)

    # Save enriched CSV
    df.to_csv("air_quality_clustered.csv", index=False)
    print("\nEnriched dataset saved: air_quality_clustered.csv")


if __name__ == "__main__":
    main()