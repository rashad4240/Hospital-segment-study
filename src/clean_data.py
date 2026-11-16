
from pathlib import Path

from duckdb import df
import pandas as pd
import numpy as np


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CLEANING_REPORT = []


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def record_result(dataset, operation, before, after):
    CLEANING_REPORT.append({
        "dataset": dataset,
        "operation": operation,
        "records_before": before,
        "records_after": after,
        "records_changed": before - after
    })


def load_data(filename):
    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pd.read_csv(path)


def save_data(df, filename):
    path = PROCESSED_DIR / filename

    df.to_csv(path, index=False)

    print(f"Saved: {filename} ({len(df):,} rows)")


def clean_diagnosis(value):
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().lower()

    if "diabet" in value or value == "dm":
        return "Diabetes"

    if (
        "hypertension" in value
        or value == "htn"
        or "high bp" in value
        or "high blood pressure" in value
    ):
        return "Hypertension"

    if "heart" in value or "cardiac" in value:
        return "Heart Disease"

    if (
        "respiratory" in value
        or "resp infection" in value
        or value == "rti"
    ):
        return "Respiratory Infection"

    if (
        "arthritis" in value
        or "joint pain" in value
    ):
        return "Arthritis"

    if "kidney" in value or "renal" in value:
        return "Kidney Disease"

    if "fever" in value:
        return "Fever"

    if "asthma" in value:
        return "Asthma"

    return "Other"


def clean_text_column(df, column):
    if column in df.columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

        df[column] = df[column].replace(
            "", pd.NA
        )


# --------------------------------------------------
# 1. PATIENT DATA
# --------------------------------------------------

def clean_patients():

    df = load_data("patients.csv")
    before = len(df)

    df = df.drop_duplicates(
        subset="patient_id"
    )

    df["date_of_birth"] = pd.to_datetime(
        df["date_of_birth"],
        errors="coerce"
    )

    # numeric_age = pd.to_numeric(numeric_age, errors="coerce")
    # age_from_dob = pd.to_numeric(age_from_dob, errors="coerce")

    

    # Check whether stored age agrees with DOB.
    calculated_age = (
        pd.Timestamp("2026-01-01")
        - df["date_of_birth"]
    ).dt.days // 365

    df["age_years"] = df["age_years"].fillna(
        calculated_age
    )

    df.loc[
        ~df["age_years"].between(0, 120),
        "age_years"
    ] = np.nan

    df["age_years"] = df["age_years"].round()

    for column in ["gender", "city"]:
        clean_text_column(df, column)

        df[column] = df[column].fillna("Unknown")

    record_result(
        "patients",
        "Remove duplicate patient IDs",
        before,
        len(df)
    )

    save_data(df, "patients_clean.csv")

    return df


# --------------------------------------------------
# 2. OPD VISITS
# --------------------------------------------------

def clean_opd():

    df = load_data("opd_visits.csv")
    before = len(df)

    # Remove exact duplicate records.
    df = df.drop_duplicates()

    # Remove repeated encounter IDs.
    df = df.drop_duplicates(
        subset="encounter_id",
        keep="first"
    )

    df["visit_date"] = pd.to_datetime(
        df["visit_date"],
        errors="coerce"
    )

    # Convert mixed age / DOB field.
    values = df["age_or_dob"].astype("string").str.strip()

    numeric_age = pd.to_numeric(
        values,
        errors="coerce"
    )

    birth_dates = pd.to_datetime(
        values.where(numeric_age.isna()),
        errors="coerce"
    )

    age_from_dob = (
        df["visit_date"] - birth_dates
    ).dt.days / 365.2425

    numeric_age = pd.to_numeric(numeric_age, errors="coerce")
    age_from_dob = pd.to_numeric(age_from_dob, errors="coerce")

    df["age_years"] = numeric_age.astype("float64").fillna(
        age_from_dob.astype("float64")
    )

    df.loc[
        ~df["age_years"].between(0, 120),
        "age_years"
    ] = np.nan

    df["age_years"] = df["age_years"].round()

    df["diagnosis_text"] = (
        df["diagnosis_text"].apply(clean_diagnosis)
    )

    for column in ["department", "hospital_id"]:
        clean_text_column(df, column)

        df[column] = df[column].fillna("Unknown")

    df["department"] = df["department"].str.title()

    record_result(
        "opd_visits",
        "Remove duplicate encounters",
        before,
        len(df)
    )

    save_data(df, "opd_visits_clean.csv")

    return df


# --------------------------------------------------
# 3. ADMISSIONS
# --------------------------------------------------

def clean_admissions():

    df = load_data("admissions.csv")
    before = len(df)

    df = df.drop_duplicates(
        subset="admission_id"
    )

    for column in ["admission_date", "discharge_date"]:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )

    invalid_dates = (
        df["admission_date"].isna()
        | df["discharge_date"].isna()
        | (
            df["discharge_date"]
            < df["admission_date"]
        )
    )

    df = df.loc[~invalid_dates].copy()

    df["length_of_stay"] = (
        df["discharge_date"]
        - df["admission_date"]
    ).dt.days

    df["diagnosis_text"] = (
        df["diagnosis_text"].apply(clean_diagnosis)
    )

    for column in ["department", "hospital_id"]:
        clean_text_column(df, column)

        df[column] = df[column].fillna("Unknown")

    df["department"] = df["department"].str.title()

    record_result(
        "admissions",
        "Remove duplicates and invalid dates",
        before,
        len(df)
    )

    save_data(df, "admissions_clean.csv")

    return df


# --------------------------------------------------
# 4. ENCOUNTER MASTER
# --------------------------------------------------

def create_encounter_master(opd, admissions):

    opd_map = opd[
        ["encounter_id", "patient_id", "hospital_id"]
    ].copy()

    admission_map = admissions[
        ["admission_id", "patient_id", "hospital_id"]
    ].rename(
        columns={"admission_id": "encounter_id"}
    )

    return pd.concat(
        [opd_map, admission_map],
        ignore_index=True
    ).drop_duplicates("encounter_id")


# --------------------------------------------------
# 5. LABORATORY DATA
# --------------------------------------------------

def clean_labs(encounters):

    df = load_data("lab_tests.csv")
    before = len(df)

    df = df.drop_duplicates(
        subset="lab_id"
    )

    df["test_date"] = pd.to_datetime(
        df["test_date"],
        errors="coerce"
    )

    # Link records to their actual encounter.
    df = df.merge(
        encounters.rename(columns={
            "patient_id": "linked_patient_id",
            "hospital_id": "linked_hospital_id"
        }),
        left_on="source_encounter_id",
        right_on="encounter_id",
        how="left",
        validate="many_to_one"
    )

    df["patient_id"] = df["linked_patient_id"]
    df["hospital_id"] = df["linked_hospital_id"]

    df = df.drop(
        columns=[
            "encounter_id",
            "linked_patient_id",
            "linked_hospital_id"
        ]
    )

    df["test_result"] = pd.to_numeric(
        df["test_result"],
        errors="coerce"
    )

    clean_text_column(df, "test_name")

    # Fill missing test results using the
    # median for the same test.
    df["test_result"] = df.groupby(
        "test_name"
    )["test_result"].transform(
        lambda x: x.fillna(x.median())
    )

    df = df.dropna(
        subset=["patient_id", "test_date"]
    )

    df["test_name"] = df["test_name"].fillna(
        "Unknown"
    )

    record_result(
        "lab_tests",
        "Remove duplicates and link encounters",
        before,
        len(df)
    )

    save_data(df, "lab_tests_clean.csv")

    return df


# --------------------------------------------------
# 6. BILLING DATA
# --------------------------------------------------

def clean_billing(encounters):

    df = load_data("billing.csv")
    before = len(df)

    df = df.drop_duplicates(
        subset="billing_id"
    )

    df["billing_date"] = pd.to_datetime(
        df["billing_date"],
        errors="coerce"
    )

    for column in [
        "base_amount",
        "gst_rate",
        "gst_amount",
        "total_amount"
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Link patient and hospital using encounter ID.
    df = df.merge(
        encounters.rename(columns={
            "patient_id": "linked_patient_id",
            "hospital_id": "linked_hospital_id"
        }),
        on="encounter_id",
        how="left",
        validate="many_to_one"
    )

    df["patient_id"] = df["linked_patient_id"]
    df["hospital_id"] = df["linked_hospital_id"]

    df = df.drop(
        columns=[
            "linked_patient_id",
            "linked_hospital_id"
        ]
    )

    # Validate billing values.
    invalid = (
        df["base_amount"].isna()
        | df["gst_rate"].isna()
        | (df["base_amount"] < 0)
        | ~df["gst_rate"].between(0, 1)
        | df["billing_date"].isna()
    )

    df = df.loc[~invalid].copy()

    # Recalculate GST using the original base amount.
    df["gst_amount"] = (
        df["base_amount"] * df["gst_rate"]
    ).round(2)

    df["total_amount"] = (
        df["base_amount"] + df["gst_amount"]
    ).round(2)

    df = df.dropna(
        subset=["patient_id"]
    )

    record_result(
        "billing",
        "Correct GST and link encounters",
        before,
        len(df)
    )

    save_data(df, "billing_clean.csv")

    return df


# --------------------------------------------------
# 7. HOSPITAL DATA
# --------------------------------------------------

def clean_hospitals():

    df = load_data("hospitals.csv")

    df = df.drop_duplicates(
        subset="hospital_id"
    )

    df["bed_capacity"] = pd.to_numeric(
        df["bed_capacity"],
        errors="coerce"
    )

    save_data(df, "hospitals_clean.csv")

    return df


# --------------------------------------------------
# 8. CLEANING REPORT
# --------------------------------------------------

def create_cleaning_report():

    report = pd.DataFrame(CLEANING_REPORT)

    report.to_csv(
        PROCESSED_DIR / "cleaning_report.csv",
        index=False
    )

    print("\nCLEANING REPORT")
    print(report.to_string(index=False))


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    print("\nSTARTING HOSPITAL DATA CLEANING\n")

    patients = clean_patients()

    opd = clean_opd()

    admissions = clean_admissions()

    encounters = create_encounter_master(
        opd,
        admissions
    )

    labs = clean_labs(encounters)

    billing = clean_billing(encounters)

    hospitals = clean_hospitals()

    create_cleaning_report()

    print("\nFINAL DATASET SIZES")

    for name, df in {
        "Patients": patients,
        "OPD Visits": opd,
        "Admissions": admissions,
        "Lab Tests": labs,
        "Billing": billing,
        "Hospitals": hospitals
    }.items():

        print(f"{name}: {len(df):,} rows")

    print("\nDATA CLEANING COMPLETED!")


if __name__ == "__main__":
    main()
