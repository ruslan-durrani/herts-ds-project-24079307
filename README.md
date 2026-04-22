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

### 2. Hotspot Detection / Clustering Results
Implemented clustering techniques:

#### KMeans Clustering
- Grouped crime regions into 4 clusters
- Provided general area segmentation
- Did not clearly isolate extreme hotspots
- Lower silhouette score compared to DBSCAN

#### DBSCAN
- Identified dense crime hotspots effectively
- Detected outliers/noise using label -1
- Cluster 2 represented very high crime areas
- Cluster 1 represented moderate crime areas
- Cluster 0 represented low crime regions
- Better suited for irregular real-world crime patterns

### 3. Crime Prediction
Implemented regression models:

- Linear Regression
- Random Forest Regressor
- Regularised Random Forest
- Tuned Random Forest

Final tuned model achieved RMSE: 4.62

### 4. Model Explainability
Used SHAP to understand feature importance.

Important features:
- lag1
- dominantTypeCount
- Latitude
- Longitude

## Results

### Prediction Performance

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

## Project

MSc Data Analytics / Machine Learning Project

## Author

**Muhammad Ruslan Babar**
