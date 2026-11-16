
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "patient_features.csv"
INPUT_OUTPUT = PROCESSED_DIR / "patient_clustering_input.csv"
SCALED_OUTPUT = PROCESSED_DIR / "patient_clustering_scaled.csv"
REPORT_OUTPUT = PROCESSED_DIR / "clustering_preparation_report.csv"


def load_data():
    print("\nLOADING PATIENT FEATURES")
    return pd.read_csv(INPUT_FILE)


def prepare_features(df):
    print("\nPREPARING CLUSTERING FEATURES")

    # Select useful patient behaviour features
    selected_features = [
        "age_years",
        "recency_days",
        "opd_visits",
        "admission_count",
        "monetary_value",
        "active_months",
        "seasonality_index",
        "average_length_of_stay",
        "lab_test_count",
        "activity_growth",
    ]

    features = df[selected_features].copy()

    report = []

    # Missing values are handled according to their meaning
    fill_values = {
        "age_years": features["age_years"].median(),
        "recency_days": 1096,
        "opd_visits": 0,
        "admission_count": 0,
        "monetary_value": 0,
        "active_months": 0,
        "seasonality_index": 0,
        "average_length_of_stay": 0,
        "lab_test_count": 0,
        "activity_growth": 0,
    }

    for column, value in fill_values.items():
        missing_before = int(features[column].isna().sum())
        features[column] = features[column].fillna(value)

        report.append({
            "operation": "Missing value treatment",
            "column": column,
            "records_changed": missing_before,
            "details": f"Missing values filled with {value}",
        })

    # Prevent negative values in features that should not be negative
    non_negative_columns = [
        "age_years",
        "recency_days",
        "opd_visits",
        "admission_count",
        "monetary_value",
        "active_months",
        "seasonality_index",
        "average_length_of_stay",
        "lab_test_count",
    ]

    for column in non_negative_columns:
        features[column] = features[column].clip(lower=0)

    # Limit extreme values using the 1st and 99th percentiles
    # Activity growth is excluded because it can legitimately be negative.
    clip_columns = [
        "age_years",
        "recency_days",
        "opd_visits",
        "admission_count",
        "monetary_value",
        "active_months",
        "seasonality_index",
        "average_length_of_stay",
        "lab_test_count",
    ]

    for column in clip_columns:
        lower = features[column].quantile(0.01)
        upper = features[column].quantile(0.99)

        changed = int(
            ((features[column] < lower) | (features[column] > upper)).sum()
        )

        features[column] = features[column].clip(lower, upper)

        report.append({
            "operation": "Outlier treatment",
            "column": column,
            "records_changed": changed,
            "details": f"Values limited between {lower:.2f} and {upper:.2f}",
        })

    # Apply log transformation to reduce right-skewness
    log_columns = [
        "recency_days",
        "opd_visits",
        "admission_count",
        "monetary_value",
        "active_months",
        "average_length_of_stay",
        "lab_test_count",
    ]

    for column in log_columns:
        features[column] = np.log1p(features[column])

    return features, report


def scale_features(features):
    print("\nSCALING FEATURES")

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(features)

    scaled_df = pd.DataFrame(
        scaled_values,
        columns=features.columns,
        index=features.index,
    )

    return scaled_df


def main():
    print("\nSTARTING CLUSTERING DATA PREPARATION")

    df = load_data()

    features, report = prepare_features(df)

    # Keep patient IDs separate from model features
    input_df = pd.concat(
        [df[["patient_id"]].reset_index(drop=True),
         features.reset_index(drop=True)],
        axis=1,
    )

    scaled_df = pd.concat(
        [df[["patient_id"]].reset_index(drop=True),
         scale_features(features).reset_index(drop=True)],
        axis=1,
    )

    input_df.to_csv(INPUT_OUTPUT, index=False)
    scaled_df.to_csv(SCALED_OUTPUT, index=False)

    pd.DataFrame(report).to_csv(REPORT_OUTPUT, index=False)

    print("\nCLUSTERING PREPARATION COMPLETED!")
    print(f"Patients prepared: {len(scaled_df):,}")
    print(f"Features selected: {len(features.columns)}")
    print(f"Saved: {INPUT_OUTPUT}")
    print(f"Saved: {SCALED_OUTPUT}")
    print(f"Saved: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
