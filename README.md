# Air Quality Clustering & Health Risk Zones

An unsupervised machine learning project that clusters cities worldwide by air pollution patterns and assigns them to health risk zones — using real WHO-level data.

## What it does

- Loads the [Global Air Pollution Dataset](https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset) from Kaggle (6,000+ cities, 4 pollutants)
- Engineers features: pollution spread, PM2.5 dominance, per-pollutant AQI
- Runs **K-Means** (with elbow + silhouette tuning) and **DBSCAN** side-by-side
- Reduces dimensions with **PCA** and plots both clustering results in 2D
- Maps cluster centroids to interpretable risk labels: Low / Moderate / High / Severe
- Generates an **interactive Folium world map** where bubble size = average AQI

## Outputs

| File | Description |
|---|---|
| `01_optimal_k.png` | Elbow method + silhouette score to choose K |
| `02_pca_clusters.png` | 2D PCA view comparing K-Means vs DBSCAN |
| `03_cluster_profiles.png` | Heatmap — what defines each risk zone? |
| `04_risk_zone_map.html` | Interactive world map (open in browser) |
| `air_quality_clustered.csv` | Enriched dataset with cluster labels |

## Tech stack

- **scikit-learn** — KMeans, DBSCAN, PCA, StandardScaler, silhouette_score
- **pandas / numpy** — data wrangling and feature engineering
- **matplotlib / seaborn** — static visualisations
- **folium** — interactive choropleth map

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn folium kaggle

# Download dataset (requires Kaggle API key)
kaggle datasets download -d hasibalmuzdadid/global-air-pollution-dataset
unzip global-air-pollution-dataset.zip

python air_quality_clustering.py
```

## Key findings (example)

| Risk Zone | Avg AQI | Top Countries |
|---|---|---|
| Low Risk | ~30 | Finland, New Zealand, Sweden |
| Moderate Risk | ~75 | Germany, Brazil, Mexico |
| High Risk | ~130 | India, China, Pakistan |
| Severe Risk | ~200+ | Bangladesh, Nepal, Afghanistan |

## Skills demonstrated

- Unsupervised learning (K-Means, DBSCAN)
- Hyperparameter tuning via silhouette score
- Dimensionality reduction (PCA)
- Feature engineering on domain data
- Geospatial visualisation (Folium)
- Clean, modular, documented Python code
