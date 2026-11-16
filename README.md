
# Hospital Segmentation Study

A data analytics and machine learning project that segments hospital patients based on healthcare usage, visit frequency, admissions, and billing behaviour.

## Project Overview

This project uses 500,000 synthetic healthcare event records from a fictional network of four hospitals in Hyderabad, India, covering 2023–2025.

The objective is to identify patient groups, understand their behaviour, and develop data-driven business recommendations.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- K-Means Clustering
- RFM Segmentation
- Excel and PowerPoint

## Project Workflow

1. Synthetic healthcare data generation
2. Data cleaning and preprocessing
3. Patient feature engineering
4. Exploratory Data Analysis (EDA)
5. Outlier treatment and scaling
6. K-Means clustering
7. Patient segment profiling
8. Segment stability and migration analysis
9. RFM segmentation
10. K-Means vs RFM comparison
11. Business recommendations
12. Final report generation

## Patient Segments

- High-Admission Patients
- Low-Engagement Patients
- Frequent OPD Patients

## Key Findings

- 80,000 synthetic patients analysed.
- Three patient segments identified using K-Means.
- K-Means and RFM approaches compared.
- Segment migration analysed across two time periods.
- Business recommendations developed for each segment.

## Project Deliverables

The `reports/` folder contains the final Excel workbook, Word project report, and business presentation.

The `visualizations/` folder contains charts generated during exploratory data analysis and clustering.

## Important Note

This project uses synthetic data for educational and analytical purposes. Financial benefits are illustrative estimates based on assumptions, not actual hospital results.

## How to Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the scripts in this order:

```bash
python src/generate_data.py
python src/clean_data.py
python src/feature_engineering.py
python src/eda.py
python src/prepare_clustering.py
python src/find_optimal_clusters.py
python src/cluster_patients.py
python src/profile_segments.py
python src/segment_stability.py
python src/rfm_segmentation.py
python src/compare_segmentation.py
python src/business_recommendations.py
python src/create_final_report.py
```
