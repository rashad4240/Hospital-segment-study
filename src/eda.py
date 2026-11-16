
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
VISUAL_DIR = BASE_DIR / "visualizations"

VISUAL_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    print("\nLOADING DATA FOR EDA")

    features = pd.read_csv(
        DATA_DIR / "patient_features.csv"
    )

    opd = pd.read_csv(
        DATA_DIR / "opd_visits_clean.csv"
    )

    admissions = pd.read_csv(
        DATA_DIR / "admissions_clean.csv"
    )

    billing = pd.read_csv(
        DATA_DIR / "billing_clean.csv"
    )

    hospitals = pd.read_csv(
        DATA_DIR / "hospitals_clean.csv"
    )

    opd["visit_date"] = pd.to_datetime(
        opd["visit_date"], errors="coerce"
    )

    admissions["admission_date"] = pd.to_datetime(
        admissions["admission_date"], errors="coerce"
    )

    billing["billing_date"] = pd.to_datetime(
        billing["billing_date"], errors="coerce"
    )

    return features, opd, admissions, billing, hospitals


# --------------------------------------------------
# SAVE CHART
# --------------------------------------------------

def save_chart(filename):

    plt.tight_layout()

    plt.savefig(
        VISUAL_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved chart: {filename}")


# --------------------------------------------------
# PATIENT AGE DISTRIBUTION
# --------------------------------------------------

def age_distribution(features):

    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=features,
        x="age_years",
        bins=30,
        kde=True
    )

    plt.title("Patient Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Number of Patients")

    save_chart("01_age_distribution.png")


# --------------------------------------------------
# PATIENT VISIT FREQUENCY
# --------------------------------------------------

def visit_frequency(features):

    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=features,
        x="opd_visits",
        bins=30,
        kde=True
    )

    plt.title("OPD Visit Frequency")
    plt.xlabel("Number of OPD Visits")
    plt.ylabel("Number of Patients")

    save_chart("02_visit_frequency.png")


# --------------------------------------------------
# REVENUE DISTRIBUTION
# --------------------------------------------------

def revenue_distribution(features):

    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=features,
        x="monetary_value",
        bins=40,
        kde=True
    )

    plt.title("Patient Billing Distribution")
    plt.xlabel("Total Billing Amount")
    plt.ylabel("Number of Patients")

    save_chart("03_revenue_distribution.png")


# --------------------------------------------------
# HOSPITAL-WISE OPD ACTIVITY
# --------------------------------------------------

def hospital_activity(opd, hospitals):

    activity = (
        opd.groupby("hospital_id")
        .size()
        .reset_index(name="total_visits")
    )

    activity = activity.merge(
        hospitals[["hospital_id", "hospital_name"]],
        on="hospital_id",
        how="left"
    )

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=activity,
        x="hospital_name",
        y="total_visits"
    )

    plt.title("OPD Visits by Hospital")
    plt.xlabel("Hospital")
    plt.ylabel("Total OPD Visits")
    plt.xticks(rotation=15)

    save_chart("04_hospital_activity.png")


# --------------------------------------------------
# MONTHLY OPD TREND
# --------------------------------------------------

def monthly_trend(opd):

    monthly = (
        opd.dropna(subset=["visit_date"])
        .assign(
            month=lambda x: x["visit_date"].dt.to_period("M")
        )
        .groupby("month")
        .size()
        .reset_index(name="visits")
    )

    monthly["month"] = monthly["month"].astype(str)

    plt.figure(figsize=(13, 5))

    sns.lineplot(
        data=monthly,
        x="month",
        y="visits",
        marker="o"
    )

    plt.title("Monthly OPD Visit Trend")
    plt.xlabel("Month")
    plt.ylabel("Number of Visits")
    plt.xticks(rotation=60)

    save_chart("05_monthly_opd_trend.png")


# --------------------------------------------------
# ADMISSION LENGTH OF STAY
# --------------------------------------------------

def length_of_stay(admissions):

    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=admissions,
        x="length_of_stay",
        bins=30,
        kde=True
    )

    plt.title("Hospital Length of Stay Distribution")
    plt.xlabel("Length of Stay (Days)")
    plt.ylabel("Number of Admissions")

    save_chart("06_length_of_stay.png")


# --------------------------------------------------
# CORRELATION ANALYSIS
# --------------------------------------------------

def correlation_analysis(features):

    columns = [
        "age_years",
        "opd_visits",
        "admission_count",
        "total_encounters",
        "recency_days",
        "monetary_value",
        "average_bill",
        "active_months",
        "seasonality_index",
        "average_length_of_stay",
        "lab_test_count"
    ]

    correlation = features[columns].corr()

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Patient Feature Correlation")

    save_chart("07_correlation_heatmap.png")


# --------------------------------------------------
# SKEWNESS AND OUTLIER SUMMARY
# --------------------------------------------------

def statistical_summary(features):

    columns = [
        "age_years",
        "opd_visits",
        "admission_count",
        "monetary_value",
        "average_bill",
        "recency_days",
        "average_length_of_stay",
        "lab_test_count"
    ]

    summary = features[columns].describe().T

    summary["skewness"] = features[columns].skew()

    summary["missing_values"] = (
        features[columns].isna().sum()
    )

    summary["outlier_count"] = 0

    for column in columns:

        q1 = features[column].quantile(0.25)
        q3 = features[column].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        summary.loc[column, "outlier_count"] = (
            (features[column] < lower)
            | (features[column] > upper)
        ).sum()

    summary.to_csv(
        DATA_DIR / "eda_statistical_summary.csv"
    )

    print("\nSTATISTICAL SUMMARY")
    print(summary.round(2))

    print("\nSaved: eda_statistical_summary.csv")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("\nSTARTING EXPLORATORY DATA ANALYSIS")

    features, opd, admissions, billing, hospitals = load_data()

    age_distribution(features)
    visit_frequency(features)
    revenue_distribution(features)
    hospital_activity(opd, hospitals)
    monthly_trend(opd)
    length_of_stay(admissions)
    correlation_analysis(features)
    statistical_summary(features)

    print("\nEDA COMPLETED SUCCESSFULLY!")
    print(f"Charts saved in: {VISUAL_DIR}")


if __name__ == "__main__":
    main()
