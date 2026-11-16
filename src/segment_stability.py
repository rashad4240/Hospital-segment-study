
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
VISUAL_DIR = BASE_DIR / "visualizations"

OPD_FILE = PROCESSED_DIR / "opd_visits_clean.csv"
ADMISSIONS_FILE = PROCESSED_DIR / "admissions_clean.csv"
BILLING_FILE = PROCESSED_DIR / "billing_clean.csv"
FEATURES_FILE = PROCESSED_DIR / "patient_features.csv"

MIGRATION_FILE = PROCESSED_DIR / "segment_migration.csv"
STABILITY_FILE = PROCESSED_DIR / "segment_stability_summary.csv"

VISUAL_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    print("\nLOADING DATA FOR STABILITY ANALYSIS")

    opd = pd.read_csv(OPD_FILE, parse_dates=["visit_date"])
    admissions = pd.read_csv(
        ADMISSIONS_FILE,
        parse_dates=["admission_date", "discharge_date"]
    )
    billing = pd.read_csv(BILLING_FILE, parse_dates=["billing_date"])
    patients = pd.read_csv(FEATURES_FILE)

    return opd, admissions, billing, patients


def build_period_features(opd, admissions, billing, patients, start, end,period_years):
    # OPD visits per patient
    opd_period = opd[
        opd["visit_date"].between(start, end)
    ]

    opd_counts = (
        opd_period.groupby("patient_id")
        .size()
        .rename("opd_visits")
    )

    # Admissions and length of stay
    admission_period = admissions[
        admissions["admission_date"].between(start, end)
    ]

    admission_counts = (
        admission_period.groupby("patient_id")
        .size()
        .rename("admission_count")
    )

    los = (
        admission_period.groupby("patient_id")["length_of_stay"]
        .mean()
        .rename("average_length_of_stay")
    )

    # Billing revenue
    billing_period = billing[
        billing["billing_date"].between(start, end)
    ]

    revenue = (
        billing_period.groupby("patient_id")["total_amount"]
        .sum()
        .rename("monetary_value")
    )

    # Build one row for every patient
    result = pd.DataFrame({
        "patient_id": patients["patient_id"]
    }).set_index("patient_id")

    result = result.join(opd_counts)
    result = result.join(admission_counts)
    result = result.join(los)
    result = result.join(revenue)

    result = result.fillna({
        "opd_visits": 0,
        "admission_count": 0,
        "average_length_of_stay": 0,
        "monetary_value": 0,
    })

    # Convert activity and revenue into annual averages
    result["opd_visits"] /= period_years
    result["admission_count"] /= period_years
    result["monetary_value"] /= period_years
    return result.reset_index()


def assign_clusters(period_df, kmeans):
    features = period_df.drop(columns=["patient_id"]).copy()

    # Use the same transformation and scaler for both periods
    log_columns = [
        "opd_visits",
        "admission_count",
        "average_length_of_stay",
        "monetary_value",
    ]

    for column in log_columns:
        features[column] = features[column].clip(lower=0)
        features[column] = __import__("numpy").log1p(features[column])

    return kmeans.predict(features)


def main():
    print("\nSTARTING SEGMENT STABILITY ANALYSIS")

    opd, admissions, billing, patients = load_data()

    print("\nBUILDING PERIOD FEATURES")

    period_1 = build_period_features(
        opd, admissions, billing, patients,
        "2023-01-01", "2024-12-31",2
    )

    period_2 = build_period_features(
        opd, admissions, billing, patients,
        "2025-01-01", "2025-12-31",1
    )

    feature_columns = [
        "opd_visits",
        "admission_count",
        "average_length_of_stay",
        "monetary_value",
    ]

    # Fit a common scaler using both periods together
    combined = pd.concat(
        [period_1[feature_columns], period_2[feature_columns]],
        ignore_index=True
    )

    import numpy as np

    combined_scaled_features = combined.copy()

    for column in feature_columns:
        combined_scaled_features[column] = np.log1p(
            combined_scaled_features[column].clip(lower=0)
        )

    scaler = StandardScaler()
    scaler.fit(combined_scaled_features)

    period_1_scaled = scaler.transform(
        np.log1p(period_1[feature_columns].clip(lower=0))
    )

    period_2_scaled = scaler.transform(
        np.log1p(period_2[feature_columns].clip(lower=0))
    )

    # Fit scaling using the baseline period only
    import numpy as np

    scaler = StandardScaler()

    period_1_scaled = scaler.fit_transform(
        np.log1p(period_1[feature_columns].clip(lower=0))
    )

    period_2_scaled = scaler.transform(
        np.log1p(period_2[feature_columns].clip(lower=0))
    )

    # Define segments using the baseline period
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    period_1["segment_2023_2024"] = kmeans.fit_predict(
        period_1_scaled
    )

    # Apply the same segment definitions to 2025
    period_2["segment_2025"] = kmeans.predict(
        period_2_scaled
    )   
    migration = period_1[
        ["patient_id", "segment_2023_2024"]
    ].merge(
        period_2[["patient_id", "segment_2025"]],
        on="patient_id",
        how="inner"
    )

    migration.to_csv(MIGRATION_FILE, index=False)

    # Migration table
    migration_table = pd.crosstab(
        migration["segment_2023_2024"],
        migration["segment_2025"],
        rownames=["2023-2024"],
        colnames=["2025"]
    )

    print("\nSEGMENT MIGRATION TABLE")
    print(migration_table)

    # Calculate percentage staying in same segment
    same_segment = (
        migration["segment_2023_2024"]
        == migration["segment_2025"]
    )

    stability = same_segment.mean() * 100

    summary = pd.DataFrame({
        "metric": ["Patients compared", "Same-segment stability (%)"],
        "value": [len(migration), round(stability, 2)]
    })

    summary.to_csv(STABILITY_FILE, index=False)

    print("\nSTABILITY SUMMARY")
    print(summary.to_string(index=False))

    # Migration heatmap
    plt.figure(figsize=(8, 6))

    plt.imshow(migration_table, cmap="Blues")

    plt.xticks(
        range(len(migration_table.columns)),
        migration_table.columns
    )
    plt.yticks(
        range(len(migration_table.index)),
        migration_table.index
    )

    plt.xlabel("2025 Segment")
    plt.ylabel("2023-2024 Segment")
    plt.title("Patient Segment Migration")

    for i in range(len(migration_table.index)):
        for j in range(len(migration_table.columns)):
            plt.text(
                j, i,
                migration_table.iloc[i, j],
                ha="center",
                va="center"
            )

    plt.colorbar(label="Number of Patients")
    plt.tight_layout()

    plt.savefig(
        VISUAL_DIR / "10_segment_migration.png",
        dpi=300
    )
    plt.close()

    print("\nSEGMENT STABILITY ANALYSIS COMPLETED!")
    print(f"Saved: {MIGRATION_FILE}")
    print(f"Saved: {STABILITY_FILE}")
    print("Saved: 10_segment_migration.png")


if __name__ == "__main__":
    main()

