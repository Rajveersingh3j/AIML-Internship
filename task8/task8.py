  import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# --------------- Helper / config ----------------
CSV_PATH = "Mall_Customers.csv"
PLOT_PCA = "pca_view.png"
PLOT_ELBOW = "elbow.png"
PLOT_CLUSTERS = "clusters_pca.png"
OUT_CSV = "Mall_Customers_with_clusters.csv"


def main():
    # 1. Load and inspect dataset
    df = pd.read_csv(CSV_PATH)
    print("Columns:", df.columns.tolist())
    print(df.head())

    # Preferred features for customer segmentation (fall back to numeric cols)
    preferred = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    if all(col in df.columns for col in preferred):
        X = df[preferred].copy()
    else:
        # Under-specification: pick numeric columns except CustomerID if present
        nums = df.select_dtypes(include=[np.number]).columns.tolist()
        nums = [c for c in nums if c.lower() != 'customerid']
        X = df[nums].copy()

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optional: PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], s=30)
    plt.title("PCA view of dataset")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(PLOT_PCA)
    plt.close()
    print(f"Saved PCA view to {PLOT_PCA}")

    # 3. Elbow Method (inertia) and silhouette scores to help choose K
    inertia = []
    sil_scores = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)
        labels_k = km.fit_predict(X_scaled)
        inertia.append(km.inertia_)
        try:
            sil = silhouette_score(X_scaled, labels_k)
        except Exception:
            sil = np.nan
        sil_scores.append(sil)
        print(f"K={k}: inertia={km.inertia_:.2f}, silhouette={sil:.4f}")

    # plot elbow
    plt.figure(figsize=(6, 4))
    plt.plot(list(K_range), inertia, marker='o')
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.xticks(list(K_range))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_ELBOW)
    plt.close()
    print(f"Saved elbow plot to {PLOT_ELBOW}")

    # choose best K by silhouette (fallback to 3 if all NaN)
    if all(np.isnan(sil_scores)):
        best_k = 3
    else:
        best_k = int(2 + int(np.nanargmax(sil_scores)))
    print(f"Selected best_k={best_k} based on silhouette scores")

    # 2 & 4. Fit KMeans with chosen K and assign labels
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    df['Cluster'] = labels

    # Visualize clusters on PCA projection
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', s=40, edgecolor='k')
    plt.title(f"K-Means Clusters (PCA view) — K={best_k}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(PLOT_CLUSTERS)
    plt.close()
    print(f"Saved cluster PCA plot to {PLOT_CLUSTERS}")

    # 5. Evaluate clustering using Silhouette Score
    final_sil = silhouette_score(X_scaled, labels)
    print("Final Silhouette Score:", round(final_sil, 4))

    # Save dataframe with cluster labels
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved clustered data to {OUT_CSV}")


if __name__ == '__main__':
    main()
