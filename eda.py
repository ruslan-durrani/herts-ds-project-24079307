#!/usr/bin/env python3
"""Reproducible EDA pipeline for Bedfordshire UK Police street crime data.

This script performs exploratory/descriptive analysis only:
- data understanding
- data quality checks
- justified data cleaning
- descriptive statistics
- temporal analysis
- spatial analysis
- crime type analysis

No modelling, clustering, or prediction is performed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

UK_LONGITUDE_RANGE = (-8.0, 2.5)
UK_LATITUDE_RANGE = (49.5, 61.0)


def normalize_text(series: pd.Series) -> pd.Series:
    """Trim and collapse whitespace while preserving missing values."""
    s = series.astype("string")
    s = s.str.strip().str.replace(r"\s+", " ", regex=True)
    return s.replace({"": pd.NA})


def ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    paths = {
        "cleaned": output_dir / "cleaned",
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
        "report": output_dir / "report",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_dataset(dataset_dir: Path) -> Tuple[pd.DataFrame, List[Path]]:
    files = sorted(dataset_dir.glob("*/*-bedfordshire-street.csv"))
    if not files:
        raise FileNotFoundError(
            f"No monthly Bedfordshire street-crime files found in: {dataset_dir}"
        )

    frames: List[pd.DataFrame] = []
    for file_path in files:
        frame = pd.read_csv(file_path)
        frame["source_file"] = file_path.as_posix()
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    return combined, files


def save_data_understanding(df: pd.DataFrame, files: List[Path], table_dir: Path) -> pd.Series:
    month_dt = pd.to_datetime(df["Month"], format="%Y-%m", errors="coerce")
    structure = pd.Series(
        {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "tabular_structure_confirmed": True,
            "source_files_count": len(files),
            "time_coverage_start": month_dt.min().strftime("%Y-%m")
            if month_dt.notna().any()
            else pd.NA,
            "time_coverage_end": month_dt.max().strftime("%Y-%m")
            if month_dt.notna().any()
            else pd.NA,
        }
    )
    structure.to_frame("value").to_csv(table_dir / "data_structure_summary.csv", index_label="metric")

    feature_summary = pd.DataFrame(
        {
            "feature": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "non_null_count": df.notna().sum().values,
            "missing_count": df.isna().sum().values,
            "missing_percent": (df.isna().mean() * 100).round(2).values,
        }
    )
    feature_summary.to_csv(table_dir / "feature_summary.csv", index=False)

    month_coverage = (
        df["Month"]
        .value_counts(dropna=False)
        .rename_axis("month")
        .reset_index(name="crime_count")
        .sort_values("month")
    )
    month_coverage.to_csv(table_dir / "month_coverage_counts.csv", index=False)

    return structure


def save_quality_checks(df: pd.DataFrame, table_dir: Path) -> Dict[str, int]:
    missing = (
        pd.DataFrame(
            {
                "column": df.columns,
                "missing_count": df.isna().sum().values,
                "missing_percent": (df.isna().mean() * 100).round(2).values,
            }
        )
        .sort_values("missing_percent", ascending=False)
        .reset_index(drop=True)
    )
    missing.to_csv(table_dir / "missing_values_summary.csv", index=False)

    month_dt = pd.to_datetime(df["Month"], format="%Y-%m", errors="coerce")
    lon_num = pd.to_numeric(df["Longitude"], errors="coerce")
    lat_num = pd.to_numeric(df["Latitude"], errors="coerce")

    invalid_longitude_global = int((lon_num.notna() & ~lon_num.between(-180, 180)).sum())
    invalid_latitude_global = int((lat_num.notna() & ~lat_num.between(-90, 90)).sum())
    invalid_month_values = int(month_dt.isna().sum())

    in_uk_bbox = lon_num.between(*UK_LONGITUDE_RANGE) & lat_num.between(*UK_LATITUDE_RANGE)
    outside_uk_bbox = int(((lon_num.notna() & lat_num.notna()) & ~in_uk_bbox).sum())

    duplicate_exact = int(df.duplicated().sum())
    duplicate_crime_id_month = (
        int(df.duplicated(subset=["Crime ID", "Month"]).sum())
        if {"Crime ID", "Month"}.issubset(df.columns)
        else 0
    )

    consistency_rows: List[Dict[str, object]] = []
    categorical_cols = [col for col in df.columns if df[col].dtype == "object"]
    for col in categorical_cols:
        raw = df[col].astype("string")
        normalized = normalize_text(raw)
        formatting_issues = int(((raw.notna()) & (raw != normalized)).sum())
        consistency_rows.append(
            {
                "column": col,
                "raw_unique_count": int(raw.nunique(dropna=True)),
                "normalized_unique_count": int(normalized.nunique(dropna=True)),
                "formatting_issue_count": formatting_issues,
            }
        )

    pd.DataFrame(consistency_rows).to_csv(table_dir / "categorical_consistency.csv", index=False)

    quality_summary = pd.Series(
        {
            "exact_duplicate_rows": duplicate_exact,
            "duplicate_rows_by_crime_id_and_month": duplicate_crime_id_month,
            "invalid_month_values": invalid_month_values,
            "invalid_longitude_global_range": invalid_longitude_global,
            "invalid_latitude_global_range": invalid_latitude_global,
            "outside_uk_bounding_box": outside_uk_bbox,
        }
    )
    quality_summary.to_frame("value").to_csv(
        table_dir / "data_quality_summary.csv", index_label="metric"
    )

    return {
        "duplicate_exact": duplicate_exact,
        "invalid_month_values": invalid_month_values,
        "invalid_longitude_global": invalid_longitude_global,
        "invalid_latitude_global": invalid_latitude_global,
    }


def clean_dataset(df: pd.DataFrame, table_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = df.copy()
    cleaning_log: List[Dict[str, object]] = []

    # Standardize string columns before imputation.
    object_cols = cleaned.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        cleaned[col] = normalize_text(cleaned[col])
    cleaning_log.append(
        {
            "step": "Normalized categorical/string formatting",
            "before_count": int(df.shape[0]),
            "after_count": int(cleaned.shape[0]),
            "note": "Trimmed and standardized whitespace to improve categorical consistency.",
        }
    )

    cleaned["Month_dt"] = pd.to_datetime(cleaned["Month"], format="%Y-%m", errors="coerce")
    cleaned["Year"] = cleaned["Month_dt"].dt.year
    cleaned["Month_number"] = cleaned["Month_dt"].dt.month
    cleaned["Month_name"] = cleaned["Month_dt"].dt.month_name()
    cleaning_log.append(
        {
            "step": "Converted Month to datetime and extracted year/month features",
            "before_count": int(df.shape[0]),
            "after_count": int(cleaned.shape[0]),
            "note": "EDA-level time features for descriptive analysis only.",
        }
    )

    fill_map = {
        "Location": "Unknown location",
        "LSOA code": "Unknown LSOA code",
        "LSOA name": "Unknown LSOA name",
        "Crime type": "Unknown crime type",
        "Last outcome category": "Outcome not specified",
        "Context": "No context",
    }
    for col, default_value in fill_map.items():
        if col in cleaned.columns:
            before_missing = int(cleaned[col].isna().sum())
            cleaned[col] = cleaned[col].fillna(default_value)
            after_missing = int(cleaned[col].isna().sum())
            cleaning_log.append(
                {
                    "step": f"Handled missing values in {col}",
                    "before_count": before_missing,
                    "after_count": after_missing,
                    "note": f"Filled missing {col} to retain records without dropping data.",
                }
            )

    before_dupes = int(cleaned.shape[0])
    cleaned = cleaned.drop_duplicates().copy()
    after_dupes = int(cleaned.shape[0])
    cleaning_log.append(
        {
            "step": "Removed exact duplicate rows",
            "before_count": before_dupes,
            "after_count": after_dupes,
            "note": "Dropped only exact duplicates to avoid duplicate-event overcounting.",
        }
    )

    cleaned["Longitude"] = pd.to_numeric(cleaned["Longitude"], errors="coerce")
    cleaned["Latitude"] = pd.to_numeric(cleaned["Latitude"], errors="coerce")
    invalid_geo_mask = (
        (cleaned["Longitude"].notna() & ~cleaned["Longitude"].between(-180, 180))
        | (cleaned["Latitude"].notna() & ~cleaned["Latitude"].between(-90, 90))
    )
    invalid_geo_count = int(invalid_geo_mask.sum())
    cleaned.loc[invalid_geo_mask, ["Longitude", "Latitude"]] = np.nan
    cleaning_log.append(
        {
            "step": "Validated latitude/longitude ranges",
            "before_count": invalid_geo_count,
            "after_count": 0,
            "note": "Invalid global coordinates replaced with missing values for safe spatial plotting.",
        }
    )

    if "Crime type" in cleaned.columns:
        key = cleaned["Crime type"].str.lower()
        canonical = (
            cleaned.dropna(subset=["Crime type"])
            .groupby(key)["Crime type"]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
        )
        cleaned["Crime type"] = key.map(canonical).fillna("Unknown crime type")
        cleaning_log.append(
            {
                "step": "Standardized crime category naming",
                "before_count": int(df["Crime type"].nunique(dropna=True)),
                "after_count": int(cleaned["Crime type"].nunique(dropna=True)),
                "note": "Resolved case-format inconsistencies while preserving category meaning.",
            }
        )

    cleaning_log_df = pd.DataFrame(cleaning_log)
    cleaning_log_df.to_csv(table_dir / "cleaning_log.csv", index=False)
    return cleaned, cleaning_log_df


def save_descriptive_tables(cleaned: pd.DataFrame, table_dir: Path) -> Dict[str, pd.DataFrame]:
    total_crimes = pd.DataFrame({"metric": ["total_crimes"], "value": [int(cleaned.shape[0])]})
    total_crimes.to_csv(table_dir / "total_crimes.csv", index=False)

    crime_type_counts = (
        cleaned["Crime type"]
        .value_counts()
        .rename_axis("Crime type")
        .reset_index(name="Crime count")
    )
    crime_type_counts.to_csv(table_dir / "crime_type_counts.csv", index=False)
    crime_type_counts.head(10).to_csv(table_dir / "top_10_crime_types.csv", index=False)
    crime_type_counts.tail(10).to_csv(table_dir / "least_10_crime_types.csv", index=False)

    area_counts = (
        cleaned["LSOA name"]
        .value_counts()
        .rename_axis("LSOA name")
        .reset_index(name="Crime count")
    )
    area_counts.to_csv(table_dir / "area_level_crime_counts.csv", index=False)

    coordinate_summary = cleaned[["Longitude", "Latitude"]].describe().round(4)
    coordinate_summary.to_csv(table_dir / "coordinate_summary_stats.csv")

    area_month = (
        cleaned.dropna(subset=["Year", "Month_number"])
        .groupby(["Year", "Month_number", "LSOA name"], dropna=False)
        .size()
        .reset_index(name="Crime count")
        .sort_values(["Year", "Month_number", "Crime count"], ascending=[True, True, False])
    )
    area_month.to_csv(table_dir / "area_month_crime_aggregation.csv", index=False)

    return {
        "crime_type_counts": crime_type_counts,
        "area_counts": area_counts,
        "area_month": area_month,
    }


def plot_temporal(cleaned: pd.DataFrame, figure_dir: Path, table_dir: Path) -> None:
    monthly_counts = (
        cleaned.dropna(subset=["Month_dt"])
        .groupby("Month_dt")
        .size()
        .reset_index(name="Crime count")
        .sort_values("Month_dt")
    )
    monthly_counts.to_csv(table_dir / "monthly_crime_trend.csv", index=False)

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=monthly_counts, x="Month_dt", y="Crime count", marker="o")
    plt.title("Monthly Crime Trend in Bedfordshire (Dec 2023 to Dec 2025)")
    plt.xlabel("Month")
    plt.ylabel("Number of crimes")
    plt.tight_layout()
    plt.savefig(figure_dir / "monthly_trend_line.png", dpi=300)
    plt.close()

    yearly_counts = (
        cleaned.dropna(subset=["Year"])
        .groupby("Year")
        .size()
        .reset_index(name="Crime count")
        .sort_values("Year")
    )
    yearly_counts.to_csv(table_dir / "yearly_crime_trend.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=yearly_counts, x="Year", y="Crime count", hue="Year", legend=False)
    plt.title("Yearly Crime Counts in Bedfordshire")
    plt.xlabel("Year")
    plt.ylabel("Number of crimes")
    plt.tight_layout()
    plt.savefig(figure_dir / "yearly_trend_bar.png", dpi=300)
    plt.close()

    seasonal = (
        cleaned.dropna(subset=["Month_number", "Month_name"])
        .groupby(["Month_number", "Month_name"])
        .size()
        .reset_index(name="Crime count")
        .sort_values("Month_number")
    )
    seasonal["Month_name"] = pd.Categorical(
        seasonal["Month_name"],
        categories=[
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        ordered=True,
    )
    seasonal = seasonal.sort_values("Month_name")
    seasonal.to_csv(table_dir / "seasonal_month_comparison.csv", index=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=seasonal, x="Month_name", y="Crime count", color="#4c72b0")
    plt.title("Seasonal Crime Comparison by Calendar Month")
    plt.xlabel("Month")
    plt.ylabel("Total crimes across all years")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(figure_dir / "seasonal_comparison_bar.png", dpi=300)
    plt.close()

    monthly_counts["month_over_month_change"] = monthly_counts["Crime count"].diff()
    monthly_counts["month_over_month_pct_change"] = (
        monthly_counts["Crime count"].pct_change() * 100
    ).round(2)
    monthly_counts.to_csv(table_dir / "monthly_trend_with_volatility.csv", index=False)

    plt.figure(figsize=(14, 6))
    sns.lineplot(
        data=monthly_counts,
        x="Month_dt",
        y="month_over_month_pct_change",
        marker="o",
        color="#dd8452",
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Month-on-Month Crime Change (%)")
    plt.xlabel("Month")
    plt.ylabel("Percentage change")
    plt.tight_layout()
    plt.savefig(figure_dir / "monthly_volatility_line.png", dpi=300)
    plt.close()


def plot_spatial(cleaned: pd.DataFrame, figure_dir: Path, table_dir: Path) -> None:
    spatial = cleaned.dropna(subset=["Longitude", "Latitude"]).copy()
    spatial.to_csv(table_dir / "spatial_points_used.csv", index=False)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=spatial.sample(min(len(spatial), 30000), random_state=42),
        x="Longitude",
        y="Latitude",
        s=8,
        alpha=0.35,
        color="#2a9d8f",
        edgecolor=None,
    )
    plt.title("Geographic Distribution of Crimes (Scatter Plot)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(figure_dir / "geographic_scatter.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.hexbin(spatial["Longitude"], spatial["Latitude"], gridsize=60, cmap="YlOrRd", mincnt=1)
    cbar = plt.colorbar()
    cbar.set_label("Crime density")
    plt.title("Crime Density by Geographic Coordinates (Hexbin)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(figure_dir / "geographic_density_hexbin.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.hist2d(spatial["Longitude"], spatial["Latitude"], bins=70, cmap="magma")
    cbar = plt.colorbar()
    cbar.set_label("Crime count per grid")
    plt.title("Geographic Heatmap of Crime Concentration")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(figure_dir / "geographic_heatmap_hist2d.png", dpi=300)
    plt.close()

    top_areas = (
        cleaned["LSOA name"]
        .value_counts()
        .head(15)
        .rename_axis("LSOA name")
        .reset_index(name="Crime count")
    )
    top_areas.to_csv(table_dir / "top_15_areas_crime_counts.csv", index=False)

    plt.figure(figsize=(12, 7))
    sns.barplot(data=top_areas, y="LSOA name", x="Crime count", hue="LSOA name", legend=False)
    plt.title("Top 15 Areas by Crime Count")
    plt.xlabel("Number of crimes")
    plt.ylabel("LSOA area")
    plt.tight_layout()
    plt.savefig(figure_dir / "area_level_comparison_bar.png", dpi=300)
    plt.close()


def plot_crime_type_analysis(
    cleaned: pd.DataFrame, figure_dir: Path, table_dir: Path
) -> Dict[str, pd.DataFrame]:
    crime_type_counts = (
        cleaned["Crime type"]
        .value_counts()
        .rename_axis("Crime type")
        .reset_index(name="Crime count")
    )
    crime_type_counts["Crime proportion (%)"] = (
        (crime_type_counts["Crime count"] / crime_type_counts["Crime count"].sum()) * 100
    ).round(2)
    crime_type_counts.to_csv(table_dir / "crime_type_distribution_with_proportion.csv", index=False)

    plt.figure(figsize=(12, 6))
    top10 = crime_type_counts.head(10)
    sns.barplot(data=top10, y="Crime type", x="Crime count", hue="Crime type", legend=False)
    plt.title("Top 10 Crime Types by Frequency")
    plt.xlabel("Number of crimes")
    plt.ylabel("Crime type")
    plt.tight_layout()
    plt.savefig(figure_dir / "crime_type_distribution_bar.png", dpi=300)
    plt.close()

    top5_types = crime_type_counts.head(5)["Crime type"].tolist()
    top5_over_time = (
        cleaned[cleaned["Crime type"].isin(top5_types)]
        .dropna(subset=["Month_dt"])
        .groupby(["Month_dt", "Crime type"])
        .size()
        .reset_index(name="Crime count")
    )
    top5_over_time.to_csv(table_dir / "top5_crime_types_over_time.csv", index=False)

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=top5_over_time, x="Month_dt", y="Crime count", hue="Crime type", marker="o")
    plt.title("Top 5 Crime Types Over Time")
    plt.xlabel("Month")
    plt.ylabel("Number of crimes")
    plt.legend(title="Crime type")
    plt.tight_layout()
    plt.savefig(figure_dir / "top5_crime_types_trend_line.png", dpi=300)
    plt.close()

    area_crime = (
        cleaned.groupby(["LSOA name", "Crime type"])
        .size()
        .reset_index(name="Crime count")
        .sort_values("Crime count", ascending=False)
    )
    idx = area_crime.groupby("LSOA name")["Crime count"].idxmax()
    dominant_per_area = area_crime.loc[idx].sort_values("Crime count", ascending=False)
    dominant_per_area.to_csv(table_dir / "dominant_crime_type_per_area.csv", index=False)

    top_areas = dominant_per_area.head(20)["LSOA name"].tolist()
    heatmap_table = (
        cleaned[cleaned["LSOA name"].isin(top_areas)]
        .groupby(["LSOA name", "Crime type"])
        .size()
        .reset_index(name="Crime count")
        .pivot(index="LSOA name", columns="Crime type", values="Crime count")
        .fillna(0)
    )
    heatmap_table.to_csv(table_dir / "area_crime_type_heatmap_table.csv")

    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_table, cmap="Blues")
    plt.title("Crime Type Composition in Top 20 High-Crime Areas")
    plt.xlabel("Crime type")
    plt.ylabel("LSOA area")
    plt.tight_layout()
    plt.savefig(figure_dir / "area_crime_type_heatmap.png", dpi=300)
    plt.close()

    return {
        "crime_type_counts": crime_type_counts,
        "dominant_per_area": dominant_per_area,
    }


def write_summary_report(
    output_report_path: Path,
    structure: pd.Series,
    quality_metrics: Dict[str, int],
    cleaned: pd.DataFrame,
    crime_type_counts: pd.DataFrame,
    dominant_per_area: pd.DataFrame,
) -> None:
    monthly_counts = (
        cleaned.dropna(subset=["Month_dt"])
        .groupby("Month_dt")
        .size()
        .reset_index(name="Crime count")
        .sort_values("Month_dt")
    )
    peak_month = monthly_counts.loc[monthly_counts["Crime count"].idxmax()]
    low_month = monthly_counts.loc[monthly_counts["Crime count"].idxmin()]

    top_crime = crime_type_counts.iloc[0]
    least_crime = crime_type_counts.iloc[-1]
    top_area = dominant_per_area.iloc[0]

    report = f"""# Bedfordshire Crime Dataset - EDA Summary

## Scope and constraints
- EDA only: descriptive statistics and visualisation.
- No machine learning models were trained.
- No clustering algorithms were applied.
- No predictions were generated.

## Data understanding
- Total records: **{int(cleaned.shape[0]):,}**
- Total features after EDA preparation: **{int(cleaned.shape[1])}**
- Source monthly files: **{int(structure['source_files_count'])}**
- Time coverage: **{structure['time_coverage_start']}** to **{structure['time_coverage_end']}**
- Dataset structure: tabular CSV merged into one analysis table.

## Data quality and cleaning decisions
- Exact duplicate rows removed: **{quality_metrics['duplicate_exact']:,}**
- Invalid month values (unparseable): **{quality_metrics['invalid_month_values']:,}**
- Invalid global longitude values: **{quality_metrics['invalid_longitude_global']:,}**
- Invalid global latitude values: **{quality_metrics['invalid_latitude_global']:,}**
- Missing location/category fields were retained and imputed with explicit placeholders.
- No row-level removal was done except exact duplicate removal.

## Key descriptive findings
- Most frequent crime type: **{top_crime['Crime type']}** ({int(top_crime['Crime count']):,} records)
- Least frequent crime type: **{least_crime['Crime type']}** ({int(least_crime['Crime count']):,} records)
- Peak crime month: **{peak_month['Month_dt'].strftime('%Y-%m')}** ({int(peak_month['Crime count']):,})
- Lowest crime month: **{low_month['Month_dt'].strftime('%Y-%m')}** ({int(low_month['Crime count']):,})
- High-crime area example (dominant type): **{top_area['LSOA name']}** - {top_area['Crime type']} ({int(top_area['Crime count']):,})

## EDA-level feature preparation for later modelling
- Extracted time features: `Year`, `Month_number`, `Month_name`.
- Created area-month aggregation table: crimes per `LSOA name` per month.
- Candidate predictive variables identified (not modelled): `Crime type`, `LSOA name`, `Longitude`, `Latitude`, `Month_number`, `Year`, `Last outcome category`.

## Bias and limitation assessment
- Data represents **reported** crime, not all crimes committed.
- Reporting behavior and policing intensity may vary by area/time.
- Missing or anonymized geolocation fields may reduce spatial precision.
- Category imbalance exists (some crime types are much more frequent than others).

## Output artifacts
- Cleaned dataset: `outputs/cleaned/bedfordshire_crime_cleaned.csv`
- Tables: `outputs/tables/`
- Figures: `outputs/figures/`
- This summary: `outputs/report/eda_summary.md`
"""
    output_report_path.write_text(report, encoding="utf-8")


def run_eda(dataset_dir: Path, output_dir: Path) -> Dict[str, Path]:
    output_paths = ensure_output_dirs(output_dir)
    raw, files = load_dataset(dataset_dir)

    structure = save_data_understanding(raw, files, output_paths["tables"])
    quality_metrics = save_quality_checks(raw, output_paths["tables"])
    cleaned, _cleaning_log = clean_dataset(raw, output_paths["tables"])

    cleaned.to_csv(output_paths["cleaned"] / "bedfordshire_crime_cleaned.csv", index=False)
    _ = save_descriptive_tables(cleaned, output_paths["tables"])
    plot_temporal(cleaned, output_paths["figures"], output_paths["tables"])
    plot_spatial(cleaned, output_paths["figures"], output_paths["tables"])
    crime_type_outputs = plot_crime_type_analysis(
        cleaned, output_paths["figures"], output_paths["tables"]
    )

    write_summary_report(
        output_paths["report"] / "eda_summary.md",
        structure,
        quality_metrics,
        cleaned,
        crime_type_outputs["crime_type_counts"],
        crime_type_outputs["dominant_per_area"],
    )

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bedfordshire crime EDA pipeline.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Path to folder containing monthly Bedfordshire CSV subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Path where cleaned data, tables, figures, and report outputs are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_eda(args.dataset_dir, args.output_dir)
    print(f"EDA complete. Outputs saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
