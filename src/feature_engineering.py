
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = DATA_DIR / "patient_features.csv"

REFERENCE_DATE = pd.Timestamp("2025-12-31")


# --------------------------------------------------
# LOAD CLEANED DATA
# --------------------------------------------------

def load_data():
    print("\nLOADING CLEANED DATASETS")

    patients = pd.read_csv(DATA_DIR / "patients_clean.csv")
    opd = pd.read_csv(DATA_DIR / "opd_visits_clean.csv")
    admissions = pd.read_csv(DATA_DIR / "admissions_clean.csv")
    billing = pd.read_csv(DATA_DIR / "billing_clean.csv")
    labs = pd.read_csv(DATA_DIR / "lab_tests_clean.csv")

    return patients, opd, admissions, billing, labs


# --------------------------------------------------
# PREPARE DATE COLUMNS
# --------------------------------------------------

def prepare_dates(opd, admissions, billing, labs):

    opd["visit_date"] = pd.to_datetime(
        opd["visit_date"], errors="coerce"
    )

    admissions["admission_date"] = pd.to_datetime(
        admissions["admission_date"], errors="coerce"
    )

    billing["billing_date"] = pd.to_datetime(
        billing["billing_date"], errors="coerce"
    )

    labs["test_date"] = pd.to_datetime(
        labs["test_date"], errors="coerce"
    )

    return opd, admissions, billing, labs


# --------------------------------------------------
# FREQUENCY FEATURES
# --------------------------------------------------

def create_frequency_features(opd, admissions):

    opd_frequency = (
        opd.groupby("patient_id")
        .size()
        .rename("opd_visits")
    )

    admission_frequency = (
        admissions.groupby("patient_id")
        .size()
        .rename("admission_count")
    )

    features = pd.concat(
        [opd_frequency, admission_frequency],
        axis=1
    ).fillna(0)

    features["total_encounters"] = (
        features["opd_visits"]
        + features["admission_count"]
    )

    return features


# --------------------------------------------------
# RECENCY FEATURES
# --------------------------------------------------

def create_recency_features(opd, admissions):

    opd_dates = opd[["patient_id", "visit_date"]].rename(
        columns={"visit_date": "activity_date"}
    )

    admission_dates = admissions[
        ["patient_id", "admission_date"]
    ].rename(columns={"admission_date": "activity_date"})

    all_dates = pd.concat(
        [opd_dates, admission_dates],
        ignore_index=True
    )

    last_visit = (
        all_dates.groupby("patient_id")["activity_date"]
        .max()
        .rename("last_activity_date")
    )

    features = last_visit.to_frame()

    features["recency_days"] = (
        REFERENCE_DATE - features["last_activity_date"]
    ).dt.days

    return features


# --------------------------------------------------
# MONETARY FEATURES
# --------------------------------------------------

def create_monetary_features(billing):

    billing["total_amount"] = pd.to_numeric(
        billing["total_amount"], errors="coerce"
    )

    features = (
        billing.groupby("patient_id")["total_amount"]
        .agg(
            monetary_value="sum",
            average_bill="mean",
            billing_count="count"
        )
    )

    return features


# --------------------------------------------------
# GROWTH FEATURES
# --------------------------------------------------

def create_growth_features(opd, admissions):

    opd_year = (
        opd.dropna(subset=["visit_date"])
        .groupby([
            "patient_id",
            opd["visit_date"].dt.year
        ])
        .size()
        .unstack(fill_value=0)
    )

    admission_year = (
        admissions.dropna(subset=["admission_date"])
        .groupby([
            "patient_id",
            admissions["admission_date"].dt.year
        ])
        .size()
        .unstack(fill_value=0)
    )

    yearly = opd_year.add(admission_year, fill_value=0)

    yearly = yearly.reindex(columns=[2024, 2025], fill_value=0)

    features = pd.DataFrame(index=yearly.index)

    features["activity_2024"] = yearly[2024]
    features["activity_2025"] = yearly[2025]

    features["activity_growth"] = (
        features["activity_2025"]
        - features["activity_2024"]
    )

    features["activity_growth_pct"] = np.where(
        features["activity_2024"] > 0,
        (
            features["activity_growth"]
            / features["activity_2024"]
        ) * 100,
        np.nan
    )

    return features


# --------------------------------------------------
# SEASONALITY FEATURES
# --------------------------------------------------

def create_seasonality_features(opd, admissions):

    opd_monthly = (
        opd.dropna(subset=["visit_date"])
        .assign(month=lambda x: x["visit_date"].dt.month)
        .groupby(["patient_id", "month"])
        .size()
    )

    admission_monthly = (
        admissions.dropna(subset=["admission_date"])
        .assign(month=lambda x: x["admission_date"].dt.month)
        .groupby(["patient_id", "month"])
        .size()
    )

    monthly = opd_monthly.add(
        admission_monthly, fill_value=0
    ).unstack(fill_value=0)

    monthly = monthly.reindex(columns=range(1, 13), fill_value=0)

    features = pd.DataFrame(index=monthly.index)

    features["active_months"] = (
        monthly.gt(0).sum(axis=1)
    )

    features["peak_activity_month"] = (
        monthly.idxmax(axis=1)
    )

    monthly_mean = monthly.mean(axis=1)

    features["seasonality_index"] = np.where(
        monthly_mean > 0,
        monthly.max(axis=1) / monthly_mean,
        0
    )

    return features


# --------------------------------------------------
# OTHER BEHAVIOURAL FEATURES
# --------------------------------------------------

def create_other_features(admissions, labs):

    admission_features = (
        admissions.groupby("patient_id")["length_of_stay"]
        .agg(average_length_of_stay="mean")
    )

    lab_features = (
        labs.groupby("patient_id")
        .size()
        .rename("lab_test_count")
    )

    return admission_features, lab_features


# --------------------------------------------------
# BUILD FINAL PATIENT DATASET
# --------------------------------------------------

def build_patient_features():

    print("\nSTARTING FEATURE ENGINEERING")

    patients, opd, admissions, billing, labs = load_data()

    opd, admissions, billing, labs = prepare_dates(
        opd, admissions, billing, labs
    )

    frequency = create_frequency_features(opd, admissions)
    recency = create_recency_features(opd, admissions)
    monetary = create_monetary_features(billing)
    growth = create_growth_features(opd, admissions)
    seasonality = create_seasonality_features(opd, admissions)

    admission_features, lab_features = create_other_features(
        admissions, labs
    )

    # Combine all patient-level features
    features = patients.set_index("patient_id")

    feature_groups = [
        frequency,
        recency,
        monetary,
        growth,
        seasonality,
        admission_features,
        lab_features
    ]

    for group in feature_groups:
        features = features.join(group, how="left")

    # Fill missing activity values
    count_columns = [
        "opd_visits",
        "admission_count",
        "total_encounters",
        "billing_count",
        "activity_2024",
        "activity_2025",
        "active_months",
        "lab_test_count"
    ]

    for column in count_columns:
        features[column] = features[column].fillna(0)

    # Save final dataset
    features = features.reset_index()

    features.to_csv(OUTPUT_FILE, index=False)

    print("\nFEATURE ENGINEERING COMPLETED!")
    print(f"Total patients: {len(features):,}")
    print(f"Total features: {len(features.columns)}")
    print(f"Saved: {OUTPUT_FILE}")

    print("\nFEATURE COLUMNS:")
    print(features.columns.tolist())

    print("\nDATASET PREVIEW:")
    print(features.head())

    return features


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    build_patient_features()
