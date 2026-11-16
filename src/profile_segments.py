
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

SEGMENTS_FILE = PROCESSED_DIR / "patient_segments.csv"
FEATURES_FILE = PROCESSED_DIR / "patient_features.csv"
ADMISSIONS_FILE = PROCESSED_DIR / "admissions_clean.csv"
HOSPITALS_FILE = PROCESSED_DIR / "hospitals_clean.csv"

OUTPUT_FILE = PROCESSED_DIR / "segment_profiles.csv"

START_DATE = pd.Timestamp("2023-01-01")
END_DATE = pd.Timestamp("2025-12-31")
PERIOD_DAYS = (END_DATE - START_DATE).days + 1


def main():
    print("\nSTARTING SEGMENT PROFILING")

    segments = pd.read_csv(SEGMENTS_FILE)
    features = pd.read_csv(FEATURES_FILE)
    admissions = pd.read_csv(ADMISSIONS_FILE)
    hospitals = pd.read_csv(HOSPITALS_FILE)

    # Combine patient features and segment assignments
    df = features.merge(segments, on="patient_id", how="inner")

    # Treat patients without billing records as zero revenue
    df["monetary_value"] = df["monetary_value"].fillna(0)

    # Prepare admission dates
    admissions["admission_date"] = pd.to_datetime(
        admissions["admission_date"], errors="coerce"
    )
    admissions["discharge_date"] = pd.to_datetime(
        admissions["discharge_date"], errors="coerce"
    )

    admissions = admissions.dropna(
        subset=["patient_id", "admission_date", "discharge_date"]
    ).copy()

    admissions = admissions.merge(
        segments, on="patient_id", how="inner"
    )

    admissions = admissions.sort_values(
        ["patient_id", "admission_date"]
    ).copy()

    # Calculate next admission after each discharge
    admissions["next_admission_date"] = (
        admissions.groupby("patient_id")["admission_date"].shift(-1)
    )

    admissions["days_to_readmission"] = (
        admissions["next_admission_date"]
        - admissions["discharge_date"]
    ).dt.days

    # Allow a full 30-day follow-up period
    admissions["eligible_discharge"] = (
        admissions["discharge_date"] <= END_DATE - pd.Timedelta(days=30)
    )

    admissions["readmission_event"] = (
        admissions["eligible_discharge"]
        & admissions["days_to_readmission"].between(0, 30)
    )

    # Segment-level patient KPIs
    profiles = df.groupby("segment").agg(
        patient_count=("patient_id", "nunique"),
        average_age=("age_years", "mean"),
        average_opd_visits=("opd_visits", "mean"),
        average_admissions=("admission_count", "mean"),
        total_revenue=("monetary_value", "sum"),
        average_revenue_per_patient=("monetary_value", "mean"),
        average_length_of_stay=("average_length_of_stay", "mean"),
    ).reset_index()

    profiles["patient_share_pct"] = (
        profiles["patient_count"] / len(df) * 100
    )

    # Readmission KPIs
    readmission = admissions.groupby("segment").agg(
        eligible_discharges=("eligible_discharge", "sum"),
        readmission_events=("readmission_event", "sum"),
    ).reset_index()

    readmission["readmission_rate_30d"] = (
        readmission["readmission_events"]
        / readmission["eligible_discharges"].replace(0, pd.NA)
        * 100
    )

    profiles = profiles.merge(
        readmission, on="segment", how="left"
    )

    # Calculate bed days within the study period
    admissions["stay_start"] = admissions["admission_date"].clip(
        lower=START_DATE
    )

    admissions["stay_end"] = admissions["discharge_date"].clip(
        upper=END_DATE
    )

    admissions["bed_days"] = (
        admissions["stay_end"] - admissions["stay_start"]
    ).dt.days.clip(lower=0)

    bed_days = admissions.groupby("segment")[
        "bed_days"
    ].sum().reset_index(name="total_bed_days")

    # Network capacity across all four hospitals
    total_network_beds = hospitals["bed_capacity"].sum()
    total_available_bed_days = total_network_beds * PERIOD_DAYS

    profiles = profiles.merge(
        bed_days, on="segment", how="left"
    )

    profiles["total_bed_days"] = (
        profiles["total_bed_days"].fillna(0)
    )

    # Segment contribution to total network capacity
    profiles["bed_occupancy_contribution_pct"] = (
        profiles["total_bed_days"]
        / total_available_bed_days * 100
    )

    profiles = profiles.round(2)

    profiles.to_csv(OUTPUT_FILE, index=False)

    print("\nCORRECTED SEGMENT PROFILE RESULTS")
    print(profiles.to_string(index=False))

    print(f"\nSaved: {OUTPUT_FILE}")
    print("\nSEGMENT PROFILING COMPLETED!")


if __name__ == "__main__":
    main()
