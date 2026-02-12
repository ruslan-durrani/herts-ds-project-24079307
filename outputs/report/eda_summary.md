# Bedfordshire Crime Dataset - EDA Summary

## Scope and constraints
- EDA only: descriptive statistics and visualisation.
- No machine learning models were trained.
- No clustering algorithms were applied.
- No predictions were generated.

## Data understanding
- Total records: **125,307**
- Total features after EDA preparation: **17**
- Source monthly files: **25**
- Time coverage: **2023-12** to **2025-12**
- Dataset structure: tabular CSV merged into one analysis table.

## Data quality and cleaning decisions
- Exact duplicate rows removed: **5,583**
- Invalid month values (unparseable): **0**
- Invalid global longitude values: **0**
- Invalid global latitude values: **0**
- Missing location/category fields were retained and imputed with explicit placeholders.
- No row-level removal was done except exact duplicate removal.

## Key descriptive findings
- Most frequent crime type: **Violence and sexual offences** (44,127 records)
- Least frequent crime type: **Theft from the person** (1,013 records)
- Peak crime month: **2025-07** (5,723)
- Lowest crime month: **2023-12** (4,405)
- High-crime area example (dominant type): **Unknown LSOA name** - Violence and sexual offences (1,753)

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
