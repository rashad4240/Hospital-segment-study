
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

FEATURE_FILE = DATA_DIR / "patient_features.csv"
KMEANS_FILE = DATA_DIR / "patient_segments.csv"
RFM_FILE = DATA_DIR / "patient_rfm_segments.csv"
ADMISSIONS_FILE = DATA_DIR / "admissions_clean.csv"

OUTPUT_FILE = DATA_DIR / "segmentation_method_comparison.csv"


def load_data():
    print("\nLOADING DATA FOR SEGMENTATION COMPARISON")

    features = pd.read_csv(FEATURE_FILE)
    kmeans = pd.read_csv(KMEANS_FILE)
    rfm = pd.read_csv(RFM_FILE)
    admissions = pd.read_csv(ADMISSIONS_FILE)

    return features, kmeans, rfm, admissions


def prepare_readmissions(admissions):
    print("\nCALCULATING 30-DAY READMISSIONS")

    admissions["admission_date"] = pd.to_datetime(
        admissions["admission_date"]
    )

    admissions["discharge_date"] = pd.to_datetime(
        admissions["discharge_date"]
    )

    admissions = admissions.sort_values(
        ["patient_id", "admission_date"]
    ).copy()

    admissions["next_admission"] = (
        admissions.groupby("patient_id")["admission_date"]
        .shift(-1)
    )

    days_to_next = (
        admissions["next_admission"]
        - admissions["discharge_date"]
    ).dt.days

    admissions["readmission_event"] = (
        days_to_next.between(0, 30)
    ).astype(int)

    admissions["eligible_discharge"] = (
        admissions["discharge_date"]
        <= pd.Timestamp("2025-12-01")
    ).astype(int)

    return admissions


def create_method_summary(
    features,
    assignments,
    admissions,
    method_name,
    segment_column
):
    df = features.merge(
        assignments[["patient_id", segment_column]],
        on="patient_id",
        how="inner"
    )

    df["monetary_value"] = df["monetary_value"].fillna(0)
    df["opd_visits"] = df["opd_visits"].fillna(0)
    df["admission_count"] = df["admission_count"].fillna(0)

    summary = (
        df.groupby(segment_column, observed=True)
        .agg(
            patient_count=("patient_id", "count"),
            total_revenue=("monetary_value", "sum"),
            average_revenue=("monetary_value", "mean"),
            average_opd_visits=("opd_visits", "mean"),
            average_admissions=("admission_count", "mean"),
            average_length_of_stay=(
                "average_length_of_stay", "mean"
            )
        )
        .reset_index()
    )

    summary["patient_share_pct"] = (
        summary["patient_count"] / len(df) * 100
    )

    # Attach patient segments to admission records
    admission_segments = admissions.merge(
        assignments[["patient_id", segment_column]],
        on="patient_id",
        how="inner"
    )

    readmission_summary = (
        admission_segments.groupby(segment_column)
        .agg(
            eligible_discharges=("eligible_discharge", "sum"),
            readmission_events=("readmission_event", "sum")
        )
        .reset_index()
    )

    readmission_summary["readmission_rate_30d"] = (
        readmission_summary["readmission_events"]
        / readmission_summary["eligible_discharges"].replace(0, pd.NA)
        * 100
    )

    summary = summary.merge(
        readmission_summary[
            [segment_column, "readmission_rate_30d"]
        ],
        on=segment_column,
        how="left"
    )

    summary.insert(0, "method", method_name)

    summary = summary.rename(
        columns={segment_column: "segment"}
    )

    return summary


def main():
    print("\nSTARTING SEGMENTATION METHOD COMPARISON")

    features, kmeans, rfm, admissions = load_data()

    admissions = prepare_readmissions(admissions)

    kmeans_summary = create_method_summary(
        features,
        kmeans,
        admissions,
        "K-means",
        "segment"
    )

    rfm_summary = create_method_summary(
        features,
        rfm,
        admissions,
        "RFM",
        "rfm_segment"
    )

    comparison = pd.concat(
        [kmeans_summary, rfm_summary],
        ignore_index=True
    )

    comparison = comparison.round(2)

    comparison.to_csv(OUTPUT_FILE, index=False)

    print("\nSEGMENTATION METHOD COMPARISON")
    print(comparison.to_string(index=False))

    print("\nCOMPARISON COMPLETED!")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
