import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load data and model
# -------------------------------
df = pd.read_csv("../data/cleaned_air_quality.csv")
model = joblib.load("../models/random_forest_pm25.pkl")

# -------------------------------
# Functions
# -------------------------------
def plot_pm25_vs_no2():
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x='no2_ugm3', y='pm25_ugm3')
    plt.title("PM2.5 vs NO2")
    plt.xlabel("NO2 (µg/m³)")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.show()

def plot_feature_importance():
    features = ['latitude', 'longitude', 'year', 'month', 'no2_ugm3']
    importances = model.feature_importances_
    imp_df = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance', ascending=False)
    plt.figure(figsize=(6,4))
    sns.barplot(x='importance', y='feature', data=imp_df)
    plt.title("Feature Importance")
    plt.show()

def predict_pm25():
    try:
        lat = float(lat_entry.get())
        lon = float(lon_entry.get())
        year = int(year_entry.get())
        month = int(month_entry.get())
        no2 = float(no2_entry.get())
        pred = model.predict([[lat, lon, year, month, no2]])
        messagebox.showinfo("Prediction", f"Predicted PM2.5: {pred[0]:.2f} µg/m³")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values.")

def show_metadata():
    meta_text = """
Dataset: cleaned_air_quality.csv
Rows: 6480
Columns: 12
Features: city, country, latitude, longitude, year, month, pm25_ugm3, no2_ugm3, data_quality, measurement_method, data_source
License: CC0
    """
    messagebox.showinfo("Metadata", meta_text)

# -------------------------------
# GUI Layout
# -------------------------------
root = tk.Tk()
root.title("Urban Air Quality Explorer")
root.geometry("500x500")

# Filters & Prediction Frame
frame1 = tk.Frame(root, padx=10, pady=10)
frame1.pack(fill='x')
tk.Label(frame1, text="Latitude:").grid(row=0, column=0)
lat_entry = tk.Entry(frame1); lat_entry.grid(row=0, column=1)
tk.Label(frame1, text="Longitude:").grid(row=1, column=0)
lon_entry = tk.Entry(frame1); lon_entry.grid(row=1, column=1)
tk.Label(frame1, text="Year:").grid(row=2, column=0)
year_entry = tk.Entry(frame1); year_entry.grid(row=2, column=1)
tk.Label(frame1, text="Month:").grid(row=3, column=0)
month_entry = tk.Entry(frame1); month_entry.grid(row=3, column=1)
tk.Label(frame1, text="NO2 (µg/m³):").grid(row=4, column=0)
no2_entry = tk.Entry(frame1); no2_entry.grid(row=4, column=1)
tk.Button(frame1, text="Predict PM2.5", command=predict_pm25).grid(row=5, column=0, columnspan=2, pady=10)

# Plots Frame
frame2 = tk.Frame(root, padx=10, pady=10)
frame2.pack(fill='x')
tk.Button(frame2, text="Plot PM2.5 vs NO2", command=plot_pm25_vs_no2).pack(fill='x', pady=5)
tk.Button(frame2, text="Feature Importance", command=plot_feature_importance).pack(fill='x', pady=5)

# Metadata Frame
frame3 = tk.Frame(root, padx=10, pady=10)
frame3.pack(fill='x')
tk.Button(frame3, text="Show Metadata", command=show_metadata).pack(fill='x', pady=5)

# Run GUI
root.mainloop()
