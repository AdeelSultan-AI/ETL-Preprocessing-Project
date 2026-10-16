import pandas as pd
# ==================================================
# STEP 1: EXTRACT DATA
# Load the retail store sales dataset.
# ==================================================
df = pd.read_csv("retail_store_sales.csv")
print("Original Data:")
print(df.head())
# ==================================================
# STEP 2: CLEAN COLUMN NAMES
# Issue: Column names may contain extra spaces.
# Handling: Remove extra spaces from column names.
# ==================================================
df.columns = df.columns.str.strip()
# ==================================================
# STEP 3: CHECK DATA QUALITY
# Check missing values, data types, and dataset size.
# ==================================================
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDataset Shape:")
print(df.shape)
# ==================================================
# STEP 4: REMOVE DUPLICATE RECORDS
# Issue: Duplicate rows may cause repeated records.
# Handling: Remove identical rows.
# ==================================================
print("\nDuplicate Records:", df.duplicated().sum())
df = df.drop_duplicates()
print("Shape After Removing Duplicates:", df.shape)
# ==================================================
# STEP 5: HANDLE MISSING QUANTITY
# Issue: Some Quantity values are missing.
# Handling: Fill missing values with the rounded mean.
# Convert Quantity into integer data type.
# ==================================================
df["Quantity"] = df["Quantity"].fillna(round(df["Quantity"].mean()))
df["Quantity"] = df["Quantity"].astype(int)
# ==================================================
# STEP 6: HANDLE MISSING PRICE PER UNIT
# Issue: Some unit prices are missing.
# Handling: Fill missing values with the median.
# ==================================================
df["Price Per Unit"] = df["Price Per Unit"].fillna(df["Price Per Unit"].median())
# ==================================================
# STEP 7: HANDLE MISSING ITEM VALUES
# Issue: Some item names are missing.
# Handling: Replace missing item names with "Unknown".
# ==================================================
df["Item"] = df["Item"].fillna("Unknown")
# ==================================================
# STEP 8: STANDARDIZE TEXT VALUES
# Issue: Text may contain extra spaces or inconsistent
# capitalization.
# Handling: Remove extra spaces and use title case.
# ==================================================
for col in ["Item", "Category", "Payment Method", "Location"]:
    df[col] = df[col].astype("string").str.strip().str.title()
# Replace missing categorical values with "Unknown".
df[["Category", "Payment Method", "Location"]] = (df[["Category", "Payment Method", "Location"]].fillna("Unknown"))
# ==================================================
# STEP 9: HANDLE MISSING DISCOUNT VALUES
# Issue: Some discount indicators are missing.
# Handling: Fill missing values with False.
# Assumption: Missing means no discount was applied.
# ==================================================
df["Discount Applied"] = (df["Discount Applied"].astype("string").str.strip().str.lower().map({"true": True,"false": False,"1": True,"0": False}))
df["Discount Applied"] = df["Discount Applied"].fillna(False)
# ==================================================
# STEP 10: CONVERT TRANSACTION DATE
# Issue: Dates may be stored as text.
# Handling: Convert text into datetime format.
# Invalid dates become missing values.
# ==================================================
df["Transaction Date"] = pd.to_datetime(df["Transaction Date"],errors="coerce")
# Remove records that have invalid or missing dates.
df = df.dropna(subset=["Transaction Date"])
# ==================================================
# STEP 11: CREATE DERIVED COLUMNS
# Create Year, Month, Day, and Weekday from the date.
# These columns help analyze sales by time.
# ==================================================
df["Year"] = df["Transaction Date"].dt.year
df["Month"] = df["Transaction Date"].dt.month
df["Day"] = df["Transaction Date"].dt.day
df["Weekday"] = df["Transaction Date"].dt.weekday
# ==================================================
# STEP 12: HANDLE MISSING TOTAL SPENT
# Issue: Some total spending values are missing.
# Handling: Fill missing values with the median.
# ==================================================
df["Total Spent"] = df["Total Spent"].fillna(df["Total Spent"].median())
# ==================================================
# STEP 13: CREATE CALCULATED COLUMNS
# Calculated Total = Quantity * Price Per Unit.
# Sales Difference compares recorded and calculated totals.
# ==================================================
df["Calculated Total"] = (df["Quantity"] * df["Price Per Unit"])
# ==================================================
# STEP 14: FILTER RECORDS
# Keep records with positive quantities and prices.
# This assumes zero and negative values are invalid
# for the ordinary sales analysis.
# ==================================================
df = df[(df["Quantity"] > 0) &(df["Price Per Unit"] > 0)].copy()
# ==================================================
# STEP 15: ENCODE CATEGORICAL COLUMNS
# Convert categorical values into numeric columns.
# One-hot encoding can help prepare data for ML.
# ==================================================
df = pd.get_dummies(
    df,
    columns=[
        "Category",
        "Payment Method",
        "Location",
        "Discount Applied"
    ],
    dtype=int
)
# ==================================================
# STEP 16: REMOVE UNNECESSARY COLUMNS
# Remove identifiers and the original date column.
# Date features have already been created.
# ==================================================
df = df.drop(
    columns=[
        "Transaction ID",
        "Customer ID",
        "Transaction Date"
    ],
    errors="ignore"
)
# ==================================================
# STEP 17: FINAL DATA QUALITY CHECK
# Check missing values, duplicates, data types, and shape.
# ==================================================
print("\nFinal Dataset:")
print(df.head())
print("\nFinal Data Types:")
print(df.dtypes)
print("\nRemaining Missing Values:")
print(df.isnull().sum())
print("\nRemaining Duplicate Records:", df.duplicated().sum())
df=df.drop_duplicates()
print("\nFinal Dataset Shape:")
print(df.shape)
# ==================================================
# STEP 18: LOAD CLEANED DATA
# Save the transformed dataset to a new CSV file.
# ==================================================
df.to_csv("cleaned_retail_store_saless.csv", index=False)
print("\nETL Process Completed Successfully!")
import os
path = os.path.join(os.path.expanduser("~"), "Downloads", "cleaned_retail_store_saless.csv")
df.to_csv(path, index=False)
print(f"Saved at: {path}")
