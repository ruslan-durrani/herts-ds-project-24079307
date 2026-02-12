# Bedfordshire Crime EDA

This project runs exploratory data analysis (EDA) on Bedfordshire UK Police street crime data.

## Project Structure

- `dataset/` - input monthly CSV files
- `outputs/` - generated cleaned data, tables, figures, report
- `eda.py` - main EDA pipeline script
- `requirements.txt` - Python dependencies
- `venv/` - virtual environment

## Run

1. Install dependencies (if not already installed):

```bash
venv/bin/pip install -r requirements.txt
```

2. Run the EDA pipeline (recommended, using project venv):

```bash
MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp venv/bin/python eda.py --dataset-dir dataset --output-dir outputs
```

3. Run the EDA pipeline (simple command, only if dependencies are installed globally):

```bash
python3 eda.py --dataset-dir dataset --output-dir outputs
```

## Main Outputs

- Cleaned dataset: `outputs/cleaned/bedfordshire_crime_cleaned.csv`
- Tables: `outputs/tables/`
- Figures: `outputs/figures/`
- EDA summary: `outputs/report/eda_summary.md`
