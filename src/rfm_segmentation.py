
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

FEATURE_FILE = DATA_DIR / "patient_features.csv"
KMEANS_FILE = DATA_DIR / "patient_segments.csv"

RFM_FILE = DATA_DIR / "patient_rfm_segments.csv"
SUMMARY_FILE = DATA_DIR / "rfm_segment_summary.csv"


def load_data():
    print("\nLOADING PATIENT DATA")

    features = pd.read_csv(FEATURE_FILE)
    kmeans = pd.read_csv(KMEANS_FILE)

    return features, kmeans


def calculate_rfm(features):
    print("\nCALCULATING RFM SCORES")

    df = features.copy()

    # Missing activity means no recorded activity or billing
    df["recency_days"] = df["recency_days"].fillna(1096)
    df["opd_visits"] = df["opd_visits"].fillna(0)
    df["admission_count"] = df["admission_count"].fillna(0)
    df["monetary_value"] = df["monetary_value"].fillna(0)

    # Frequency combines OPD visits and admissions
    df["frequency"] = (
        df["opd_visits"] + df["admission_count"]
    )

    # Assign scores from 1 to 5
    df["R_score"] = pd.qcut(
        df["recency_days"].rank(method="first"),
        5,
        labels=[5, 4, 3, 2, 1]
    ).astype(int)

    df["F_score"] = pd.qcut(
        df["frequency"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    df["M_score"] = pd.qcut(
        df["monetary_value"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    df["RFM_score"] = (
        df["R_score"] + df["F_score"] + df["M_score"]
    )

    return df



def assign_segments(df):
    print("\nASSIGNING RFM SEGMENTS")

    df["rfm_segment"] = pd.qcut(
        df["RFM_score"].rank(method="first"),
        q=4,
        labels=[
            "Low Engagement Patients",
            "Developing Patients",
            "Regular Patients",
            "High Value Patients"
        ]
    )

    return df


def create_summary(df):
    print("\nCREATING RFM SEGMENT SUMMARY")

    summary = (
        df.groupby("rfm_segment", observed=True)
        .agg(
            patient_count=("patient_id", "count"),
            average_recency=("recency_days", "mean"),
            average_frequency=("frequency", "mean"),
            average_revenue=("monetary_value", "mean"),
            total_revenue=("monetary_value", "sum"),
            average_rfm_score=("RFM_score", "mean")
        )
        .reset_index()
    )

    summary["patient_share_pct"] = (
        summary["patient_count"] / len(df) * 100
    ).round(2)

    summary.to_csv(SUMMARY_FILE, index=False)

    return summary


def compare_with_kmeans(df, kmeans):
    print("\nCOMPARING RFM WITH K-MEANS")

    comparison = df[
        ["patient_id", "rfm_segment"]
    ].merge(
        kmeans,
        on="patient_id",
        how="inner"
    )

    comparison_table = pd.crosstab(
        comparison["rfm_segment"],
        comparison["segment"]
    )

    print(comparison_table)

    comparison_table.to_csv(
        DATA_DIR / "rfm_kmeans_comparison.csv"
    )


def main():
    print("\nSTARTING RFM SEGMENTATION")

    features, kmeans = load_data()

    df = calculate_rfm(features)
    df = assign_segments(df)

    summary = create_summary(df)

    df.to_csv(RFM_FILE, index=False)

    compare_with_kmeans(df, kmeans)

    print("\nRFM SEGMENT SUMMARY")
    print(summary.round(2).to_string(index=False))

    print("\nRFM SEGMENTATION COMPLETED!")
    print(f"Saved: {RFM_FILE}")
    print(f"Saved: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
