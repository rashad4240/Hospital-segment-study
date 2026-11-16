
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

FEATURE_FILE = DATA_DIR / "patient_features.csv"
SEGMENT_FILE = DATA_DIR / "patient_segments.csv"

OUTPUT_FILE = DATA_DIR / "business_recommendations.csv"

# Illustrative planning assumptions, not measured outcomes
STRATEGIES = {
    0: {
        "segment_name": "High-Admission Patients",
        "action": "Introduce post-discharge follow-ups and care coordination.",
        "target_kpi": "30-day readmission rate",
        "program_cost_per_patient": 300,
        "expected_revenue_uplift_pct": 1.5
    },
    1: {
        "segment_name": "Low-Engagement Patients",
        "action": "Use appointment reminders and preventive-care outreach.",
        "target_kpi": "OPD visit frequency",
        "program_cost_per_patient": 100,
        "expected_revenue_uplift_pct": 3.0
    },
    2: {
        "segment_name": "Frequent OPD Patients",
        "action": "Improve appointment continuity and diagnostic service coordination.",
        "target_kpi": "Patient continuity and OPD engagement",
        "program_cost_per_patient": 150,
        "expected_revenue_uplift_pct": 2.0
    }
}


def main():
    print("\nSTARTING BUSINESS RECOMMENDATION ANALYSIS")

    print("\nLOADING DATA")

    features = pd.read_csv(FEATURE_FILE)
    segments = pd.read_csv(SEGMENT_FILE)

    df = features.merge(
        segments,
        on="patient_id",
        how="inner"
    )

    df["monetary_value"] = df["monetary_value"].fillna(0)
    df["opd_visits"] = df["opd_visits"].fillna(0)
    df["admission_count"] = df["admission_count"].fillna(0)

    print("\nGENERATING SEGMENT RECOMMENDATIONS")

    summary = (
        df.groupby("segment")
        .agg(
            patient_count=("patient_id", "count"),
            total_revenue=("monetary_value", "sum"),
            average_revenue=("monetary_value", "mean"),
            average_opd_visits=("opd_visits", "mean"),
            average_admissions=("admission_count", "mean")
        )
        .reset_index()
    )

    recommendations = []

    for _, row in summary.iterrows():

        segment_id = int(row["segment"])
        strategy = STRATEGIES[segment_id]

        program_cost = (
            row["patient_count"]
            * strategy["program_cost_per_patient"]
        )

        expected_revenue = (
            row["total_revenue"]
            * strategy["expected_revenue_uplift_pct"]
            / 100
        )

        net_benefit = expected_revenue - program_cost

        recommendations.append({
            "segment": strategy["segment_name"],
            "patient_count": int(row["patient_count"]),
            "patient_share_pct": round(
                row["patient_count"] / len(df) * 100, 2
            ),
            "total_revenue": round(row["total_revenue"], 2),
            "average_opd_visits": round(
                row["average_opd_visits"], 2
            ),
            "average_admissions": round(
                row["average_admissions"], 2
            ),
            "recommended_action": strategy["action"],
            "target_kpi": strategy["target_kpi"],
            "assumed_revenue_uplift_pct":
                strategy["expected_revenue_uplift_pct"],
            "estimated_program_cost": round(program_cost, 2),
            "estimated_incremental_revenue": round(
                expected_revenue, 2
            ),
            "estimated_net_benefit": round(net_benefit, 2)
        })

    result = pd.DataFrame(recommendations)

    result.to_csv(OUTPUT_FILE, index=False)

    print("\nBUSINESS RECOMMENDATIONS")
    print(result.to_string(index=False))

    print("\nIMPORTANT:")
    print("Financial projections use illustrative assumptions.")
    print("They are not measured savings or guaranteed revenue.")

    print("\nBUSINESS RECOMMENDATION ANALYSIS COMPLETED!")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
