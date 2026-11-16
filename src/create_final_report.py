
from pathlib import Path

import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_FILE = REPORT_DIR / "Hospital_Segmentation_Final_Report.xlsx"


def load_csv(filename):
    path = DATA_DIR / filename

    if path.exists():
        return pd.read_csv(path)

    print(f"Optional file not found: {filename}")
    return None


def create_workbook():
    print("\nSTARTING FINAL REPORT CREATION")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sheets = {
        "Business Recommendations": "business_recommendations.csv",
        "KMeans vs RFM": "segmentation_method_comparison.csv",
        "RFM Summary": "rfm_segment_summary.csv",
        "Segment Migration": "segment_migration.csv",
        "Stability Summary": "segment_stability_summary.csv",
        "EDA Statistics": "eda_statistical_summary.csv",
        "Data Cleaning": "cleaning_report.csv",
    }

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl"
    ) as writer:

        for sheet_name, filename in sheets.items():
            df = load_csv(filename)

            if df is not None:
                df.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False
                )
                print(f"Added sheet: {sheet_name}")

        # Combine patient-level K-means and RFM assignments
        kmeans = load_csv("patient_segments.csv")
        rfm = load_csv("patient_rfm_segments.csv")

        if kmeans is not None and rfm is not None:
            assignments = kmeans.merge(
                rfm[["patient_id", "rfm_segment", "RFM_score"]],
                on="patient_id",
                how="inner"
            )

            assignments.to_excel(
                writer,
                sheet_name="Patient Assignments",
                index=False
            )

            print("Added sheet: Patient Assignments")

    print("\nFORMATTING EXCEL WORKBOOK")

    workbook = pd.ExcelFile(OUTPUT_FILE)

    from openpyxl import load_workbook

    wb = load_workbook(OUTPUT_FILE)

    for ws in wb.worksheets:

        # Format header
        for cell in ws[1]:
            cell.font = Font(
                bold=True,
                color="FFFFFF",
                size=11
            )
            cell.fill = PatternFill(
                fill_type="solid",
                fgColor="17365D"
            )
            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True
            )

        ws.row_dimensions[1].height = 30

        # Adjust column widths
        for column_cells in ws.columns:
            max_length = 0
            column_letter = get_column_letter(
                column_cells[0].column
            )

            for cell in column_cells:
                if cell.value is not None:
                    max_length = max(
                        max_length,
                        len(str(cell.value))
                    )

            ws.column_dimensions[column_letter].width = min(
                max(max_length + 3, 14),
                45
            )

        # Make report easier to navigate
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        ws.sheet_view.showGridLines = False

    wb.save(OUTPUT_FILE)

    print("\nFINAL REPORT CREATED SUCCESSFULLY!")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_workbook()
