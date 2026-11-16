
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SEED = 42
rng = np.random.default_rng(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2023-01-01"
TOTAL_DAYS = 1095

PATIENT_COUNT = 80_000

RECORD_COUNTS = {
    "opd_visits": 250_000,
    "admissions": 40_000,
    "lab_tests": 150_000,
    "billing": 60_000,
}

HOSPITALS = [
    "HYD001",
    "HYD002",
    "HYD003",
    "HYD004",
]

DEPARTMENTS = [
    "Cardiology",
    "Orthopedics",
    "Neurology",
    "General Medicine",
    "Pediatrics",
    "Dermatology",
    "Emergency",
    "Gynecology",
]

DIAGNOSES = [
    "Diabetes",
    "Hypertension",
    "Heart Disease",
    "Respiratory Infection",
    "Arthritis",
    "Kidney Disease",
    "Fever",
    "Asthma",
]

DIAGNOSIS_VARIATIONS = {
    "Diabetes": ["Diabetes", "diabetes", "Type 2 Diabetes",
                 "DM", "Diabetic"],
    "Hypertension": ["Hypertension", "HTN", "High BP",
                     "high blood pressure"],
    "Heart Disease": ["Heart Disease", "Cardiac Disease",
                      "heart disease", "Cardiac"],
    "Respiratory Infection": ["Respiratory Infection",
                              "Resp infection", "RTI"],
    "Arthritis": ["Arthritis", "Joint Pain", "arthritis"],
    "Kidney Disease": ["Kidney Disease", "Renal Disease",
                       "kidney disorder"],
    "Fever": ["Fever", "High Fever", "fever"],
    "Asthma": ["Asthma", "Bronchial Asthma", "asthma"],
}


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def random_dates(count):
    offsets = rng.integers(0, TOTAL_DAYS, count)

    return (
        pd.Timestamp(START_DATE)
        + pd.to_timedelta(offsets, unit="D")
    )


def random_patients(count):
    return rng.integers(1, PATIENT_COUNT + 1, count)


def random_diagnoses(count):
    diagnoses = rng.choice(DIAGNOSES, count)

    return [
        rng.choice(DIAGNOSIS_VARIATIONS[d])
        for d in diagnoses
    ]


def introduce_missing_values(df, columns, percentage=0.01):
    for column in columns:
        count = int(len(df) * percentage)

        if count > 0:
            indexes = rng.choice(
                df.index.to_numpy(),
                size=count,
                replace=False
            )

            df.loc[indexes, column] = np.nan

    return df


def save_dataset(df, filename):
    path = RAW_DIR / filename

    df.to_csv(path, index=False)

    print(
        f"Created {filename}: "
        f"{len(df):,} records"
    )


# --------------------------------------------------
# 1. PATIENT MASTER DATA
# --------------------------------------------------

def generate_patients():

    count = PATIENT_COUNT

    ages = rng.integers(1, 91, count)

    birth_years = 2025 - ages

    birth_dates = pd.to_datetime(
        birth_years.astype(str)
        + "-"
        + rng.integers(1, 13, count).astype(str)
        + "-"
        + rng.integers(1, 28, count).astype(str),
        errors="coerce"
    )

    patients = pd.DataFrame({
        "patient_id": [
            f"PAT{i:06d}"
            for i in range(1, count + 1)
        ],
        "date_of_birth": birth_dates,
        "age_years": ages,
        "gender": rng.choice(
            ["Male", "Female", "Other"],
            count,
            p=[0.50, 0.48, 0.02]
        ),
        "city": rng.choice(
            ["Hyderabad", "Secunderabad",
             "Medchal", "Ranga Reddy"],
            count
        )
    })

    save_dataset(patients, "patients.csv")

    return patients


# --------------------------------------------------
# 2. OPD VISITS
# --------------------------------------------------

def generate_opd(patients):

    count = RECORD_COUNTS["opd_visits"] - 500 # Reserve 500 for duplicates

    patient_numbers = random_patients(count)

    dates = random_dates(count)

    ages = patients["age_years"].to_numpy()[patient_numbers - 1]

    opd = pd.DataFrame({
        "encounter_id": [
            f"OPD{i:07d}"
            for i in range(1, count + 1)
        ],
        "patient_id": [
            f"PAT{i:06d}"
            for i in patient_numbers
        ],
        "hospital_id": rng.choice(HOSPITALS, count),
        "visit_date": dates,
        "department": rng.choice(DEPARTMENTS, count),
        "diagnosis_text": random_diagnoses(count),
        "age_or_dob": ages.astype(str),
        "visit_type": "OPD",
    })

    # Some records contain DOB instead of age.
    dob_indexes = rng.choice(
        opd.index.to_numpy(),
        size=int(count * 0.15),
        replace=False
    )

    opd.loc[dob_indexes, "age_or_dob"] = (
        pd.to_datetime(
            dates[dob_indexes]
        ) - pd.to_timedelta(
            rng.integers(18, 80, len(dob_indexes)) * 365,
            unit="D"
        )
    ).strftime("%Y-%m-%d")

    introduce_missing_values(
        opd,
        ["department", "diagnosis_text"],
        0.01
    )

    # Introduce a small number of duplicate records.
    opd = pd.concat(
        [opd, opd.sample(500, random_state=SEED)],
        ignore_index=True
    )

    save_dataset(opd, "opd_visits.csv")

    return opd


# --------------------------------------------------
# 3. HOSPITAL ADMISSIONS
# --------------------------------------------------

def generate_admissions():

    count = RECORD_COUNTS["admissions"]

    patient_numbers = random_patients(count)

    admission_dates = random_dates(count)

    length_of_stay = rng.integers(1, 15, count)

    discharge_dates = (
        admission_dates
        + pd.to_timedelta(length_of_stay, unit="D")
    )

    admissions = pd.DataFrame({
        "admission_id": [
            f"ADM{i:07d}"
            for i in range(1, count + 1)
        ],
        "patient_id": [
            f"PAT{i:06d}"
            for i in patient_numbers
        ],
        "hospital_id": rng.choice(HOSPITALS, count),
        "admission_date": admission_dates,
        "discharge_date": discharge_dates,
        "department": rng.choice(DEPARTMENTS, count),
        "diagnosis_text": random_diagnoses(count),
        "length_of_stay": length_of_stay,
    })

    introduce_missing_values(
        admissions,
        ["department", "diagnosis_text"],
        0.01
    )

    save_dataset(admissions, "admissions.csv")

    return admissions


# --------------------------------------------------
# 4. LABORATORY TESTS
# --------------------------------------------------

def generate_labs(opd, admissions):

    count = RECORD_COUNTS["lab_tests"]

    patient_numbers = random_patients(count)

    all_encounters = np.concatenate([
        opd["encounter_id"].to_numpy(),
        admissions["admission_id"].to_numpy()
    ])

    labs = pd.DataFrame({
        "lab_id": [
            f"LAB{i:07d}"
            for i in range(1, count + 1)
        ],
        "patient_id": [
            f"PAT{i:06d}"
            for i in patient_numbers
        ],
        "hospital_id": rng.choice(HOSPITALS, count),
        "test_date": random_dates(count),
        "test_name": rng.choice(
            ["CBC", "Blood Sugar", "Lipid Profile",
             "Liver Function", "Kidney Function",
             "Thyroid", "HbA1c"],
            count
        ),
        "test_result": np.round(
            rng.uniform(1, 250, count), 2
        ),
        "source_encounter_id": rng.choice(
            all_encounters, count
        )
    })

    introduce_missing_values(
        labs,
        ["test_result", "test_name"],
        0.01
    )

    save_dataset(labs, "lab_tests.csv")


# --------------------------------------------------
# 5. BILLING RECORDS
# --------------------------------------------------

def generate_billing(opd, admissions):

    count = RECORD_COUNTS["billing"]

    all_encounters = np.concatenate([
        opd["encounter_id"].to_numpy(),
        admissions["admission_id"].to_numpy()
    ])

    base_amount = np.round(
        rng.uniform(300, 50_000, count), 2
    )

    gst_rate = rng.choice(
        [0.05, 0.12, 0.18],
        count
    )

    # Normally GST is calculated once.
    gst_amount = base_amount * gst_rate

    # Introduce duplicate GST calculations in 8% of bills.
    duplicate_indexes = rng.choice(
        np.arange(count),
        size=int(count * 0.08),
        replace=False
    )

    gst_amount[duplicate_indexes] *= 2

    billing = pd.DataFrame({
        "billing_id": [
            f"BILL{i:07d}"
            for i in range(1, count + 1)
        ],
        "encounter_id": rng.choice(
            all_encounters, count
        ),
        "patient_id": [
            f"PAT{i:06d}"
            for i in random_patients(count)
        ],
        "hospital_id": rng.choice(HOSPITALS, count),
        "billing_date": random_dates(count),
        "base_amount": base_amount,
        "gst_rate": gst_rate,
        "gst_amount": np.round(gst_amount, 2),
        "total_amount": np.round(
            base_amount + gst_amount, 2
        )
    })

    introduce_missing_values(
        billing,
        ["gst_amount"],
        0.005
    )

    save_dataset(billing, "billing.csv")


# --------------------------------------------------
# 6. HOSPITAL REFERENCE DATA
# --------------------------------------------------

def generate_hospitals():

    hospitals = pd.DataFrame({
        "hospital_id": HOSPITALS,
        "hospital_name": [
            "Hyderabad Central Hospital",
            "Secunderabad Community Hospital",
            "Madhapur Care Hospital",
            "Kukatpally Medical Centre"
        ],
        "city": ["Hyderabad"] * 4,
        "bed_capacity": [250, 180, 150, 120]
    })

    save_dataset(hospitals, "hospitals.csv")


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    print("\nGenerating Hospital Dataset...\n")

    patients = generate_patients()

    opd = generate_opd(patients)

    admissions = generate_admissions()

    generate_labs(opd, admissions)

    generate_billing(opd, admissions)

    generate_hospitals()

    print("\n--------------------------------")
    print("DATASET GENERATION COMPLETED")
    print("--------------------------------")

    for filename in RAW_DIR.glob("*.csv"):
        df = pd.read_csv(filename)

        print(
            f"{filename.name}: "
            f"{len(df):,} rows"
        )

    print("\nAll files saved in data/raw/")


if __name__ == "__main__":
    main()
