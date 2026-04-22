# Crime Analysis and Prediction using Machine Learning

This project analyses UK police crime data to identify patterns, detect hotspots, and predict future crime levels using machine learning techniques.

## Overview

Crime is influenced by both time and location. This project uses data analysis and machine learning to:

- Explore crime trends over time
- Detect high-crime hotspots
- Predict future crime levels
- Explain model predictions

The dataset was collected from Police.uk Open Crime Data and aggregated at LSOA-month level.

## Features

### 1. Exploratory Data Analysis (EDA)
- Monthly crime trends
- Seasonal patterns
- Top crime categories
- Spatial crime distribution

### 2. Hotspot Detection
Implemented clustering techniques:

- KMeans Clustering
- DBSCAN

### 3. Crime Prediction
Implemented regression models:

- Linear Regression
- Random Forest Regressor
- Regularised Random Forest
- Tuned Random Forest

Final tuned model achieved RMSE: 4.62

### 4. Model Explainability
Used SHAP to understand feature importance.

## Results

| Model | RMSE |
|-------|------:|
| Linear Regression | 5.35 |
| Random Forest | 4.84 |
| Regularised RF | 4.86 |
| Tuned RF | 4.62 |

## Technologies Used

Python, Pandas, NumPy, Matplotlib, Scikit-learn, SHAP, Jupyter Notebook

## Dataset Source

https://data.police.uk/data/

## Author

Developed as part of an MSc Data Analytics / Machine Learning project.