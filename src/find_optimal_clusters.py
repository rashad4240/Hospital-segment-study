
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
VISUAL_DIR = BASE_DIR / "visualizations"

INPUT_FILE = PROCESSED_DIR / "patient_clustering_scaled.csv"
METRICS_FILE = PROCESSED_DIR / "cluster_evaluation.csv"

VISUAL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("\nSTARTING CLUSTER EVALUATION")

    df = pd.read_csv(INPUT_FILE)

    # Patient ID is not used for clustering
    X = df.drop(columns=["patient_id"])

    # Use a representative sample for faster model evaluation
    sample = X.sample(n=min(20000, len(X)), random_state=42)

    results = []

    for k in range(2, 9):
        print(f"\nTesting {k} clusters...")

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(sample)

        silhouette = silhouette_score(
            sample,
            model.labels_,
            sample_size=min(5000, len(sample)),
            random_state=42
        )

        results.append({
            "clusters": k,
            "inertia": model.inertia_,
            "silhouette_score": silhouette
        })

        print(f"Inertia: {model.inertia_:.2f}")
        print(f"Silhouette Score: {silhouette:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(METRICS_FILE, index=False)

    # Elbow chart
    plt.figure(figsize=(9, 5))
    plt.plot(
        results_df["clusters"],
        results_df["inertia"],
        marker="o"
    )
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.xticks(range(2, 9))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(VISUAL_DIR / "08_elbow_method.png", dpi=300)
    plt.close()

    # Silhouette chart
    plt.figure(figsize=(9, 5))
    plt.plot(
        results_df["clusters"],
        results_df["silhouette_score"],
        marker="o"
    )
    plt.title("Silhouette Score Comparison")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.xticks(range(2, 9))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(VISUAL_DIR / "09_silhouette_scores.png", dpi=300)
    plt.close()

    print("\nCLUSTER EVALUATION COMPLETED!")
    print(results_df.to_string(index=False))
    print(f"\nSaved: {METRICS_FILE}")
    print("Saved: 08_elbow_method.png")
    print("Saved: 09_silhouette_scores.png")


if __name__ == "__main__":
    main()
