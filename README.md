🌍 AirQualityProject

An interactive Python application for exploring and predicting urban air quality worldwide (1958–2025). Users can visualize trends, filter by city/country/year/month, and predict PM2.5 and NO₂ levels using machine learning.

📂 Dataset

Filename: air_quality_global.csv

Rows x Columns: 6480 x 12

License: CC0

Metadata: metadata.json

Source: Urban Air Quality and Climate Dataset (1958–2025)

The metadata is used to handle missing values, quality flags, and understand dataset components.

🎯 Objective & Tasks Implemented

Data Cleaning & Preprocessing

Exploratory Data Analysis (EDA) with plots: PM2.5 vs NO₂, distributions, maps.

Predictive Modeling: Random Forest Regressor for PM2.5 prediction (with NO₂ as feature).

Model Evaluation: R², MAE, RMSE metrics.

Feature Importance visualization.

Interactive GUI using Tkinter for predictions and filtering.
