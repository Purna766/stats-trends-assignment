"""
Statistics & Trends Assignment
Dataset: kc_house_data.csv (King County House Sales)

This script:
- Loads the dataset
- Cleans the data
- Computes 4 statistical moments (price)
- Produces 3 required plots:
    1) Histogram (Categorical requirement)
    2) Scatter + time trend (Relational requirement)
    3) Correlation heatmap (Statistical requirement)
- Saves plots to /outputs folder
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = Path("data/kc_house_data.csv")  
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)

# -----------------------------
# DATA CLEANING
# -----------------------------

# Remove duplicates
df = df.drop_duplicates()

# Convert date column properly
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Remove rows with missing values
df = df.dropna()

print("After cleaning:", df.shape)

# -----------------------------
# SELECT VARIABLES
# -----------------------------
numeric_col = "price"
second_numeric_col = "sqft_living"
time_col = "date"

# -----------------------------
# FOUR STATISTICAL MOMENTS (PRICE)
# -----------------------------
x = df[numeric_col]

mean_val = x.mean()
variance_val = x.var()
skew_val = x.skew()
kurt_val = x.kurt()

print("\n--- Statistical Moments for Price ---")
print("Mean:", round(mean_val, 2))
print("Variance:", round(variance_val, 2))
print("Skewness:", round(skew_val, 4))
print("Kurtosis:", round(kurt_val, 4))

# -----------------------------
# PLOT 1 — HISTOGRAM (Categorical)
# -----------------------------
plt.figure(figsize=(6,4))
plt.hist(x, bins=30)
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")

plt.savefig(OUTPUT_DIR / "plot1_histogram_price.png", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# PLOT 2 — RELATIONAL
# Scatter: Price vs Sqft Living
# -----------------------------
plt.figure(figsize=(6,4))
plt.scatter(df[second_numeric_col], df[numeric_col], alpha=0.5)
plt.title("Price vs Living Area")
plt.xlabel("Sqft Living")
plt.ylabel("Price")

plt.savefig(OUTPUT_DIR / "plot2_scatter_price_sqft.png", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# Optional Time Trend (Line Plot)
# -----------------------------
monthly_price = df.set_index(time_col).resample("M")["price"].mean()

plt.figure(figsize=(6,4))
plt.plot(monthly_price.index, monthly_price.values)
plt.title("Average House Price Over Time")
plt.xlabel("Time")
plt.ylabel("Average Price")

plt.savefig(OUTPUT_DIR / "plot2b_line_price_time.png", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# PLOT 3 — CORRELATION HEATMAP
# -----------------------------
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

plt.figure(figsize=(8,6))
plt.imshow(corr, aspect="auto")
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Correlation Heatmap of Housing Variables")

plt.savefig(OUTPUT_DIR / "plot3_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nAll plots saved in 'outputs' folder.")
