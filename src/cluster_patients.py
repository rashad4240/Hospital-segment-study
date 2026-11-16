
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "patient_clustering_scaled.csv"
OUTPUT_FILE = PROCESSED_DIR / "patient_segments.csv"


def main():
    print("\nSTARTING PATIENT CLUSTERING")

    df = pd.read_csv(INPUT_FILE)

    patient_ids = df["patient_id"]
    X = df.drop(columns=["patient_id"])

    print(f"Patients loaded: {len(df):,}")
    print("Training K-means model with 3 clusters...")

    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    cluster_labels = model.fit_predict(X)

    results = pd.DataFrame({
        "patient_id": patient_ids,
        "segment": cluster_labels
    })

    results.to_csv(OUTPUT_FILE, index=False)

    print("\nPATIENT CLUSTERING COMPLETED!")
    print("\nPATIENT DISTRIBUTION")

    distribution = (
        results["segment"]
        .value_counts()
        .sort_index()
        .rename_axis("segment")
        .reset_index(name="patient_count")
    )

    distribution["percentage"] = (
        distribution["patient_count"] / len(results) * 100
    ).round(2)

    print(distribution.to_string(index=False))

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
