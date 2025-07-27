import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv('uber.csv')

'''# Show the first 5 rows
print(df.head())

# Display basic information about the dataset

# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:\n", df.head())

# Display summary statistics of the dataset
print(df.describe())
print(df.info())

# Check the shape (rows, columns)
print("Shape of dataset:", df.shape)

# Check column names and data types
print("\nColumn types:\n", df.dtypes)

# Get quick summary statistics
print("\nSummary statistics:\n", df.describe())

# Check missing values in each column
print(df.isnull().sum())
# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:\n", df.head())

'''
# Remove rows with any missing values
df_cleaned = df.dropna()

# Remove rows with any missing values
df_cleaned = df.dropna()

# Confirm shape after cleaning
print("New shape after removing nulls:", df_cleaned.shape)


# Save cleaned data
df_cleaned.to_csv('uber_cleaned.csv', index=False)

print("Cleaned data saved as 'uber_cleaned.csv'")


# Summary statistics
print("\n--- Descriptive Statistics ---")
print(df_cleaned.describe())






# Set seaborn style for plots
sns.set(style="whitegrid")

# === STEP 1: Load the Dataset ===
df = pd.read_csv('uber.csv')
print("First few rows of raw data:\n", df.head())
print("Original shape:", df.shape)
print("\nColumn Types:\n", df.dtypes)

# === STEP 2: Data Cleaning ===
# Remove missing values
df_cleaned = df.dropna()
print("New shape after removing nulls:", df_cleaned.shape)

# === STEP 3: Feature Engineering ===
# Convert pickup datetime to datetime object
df_cleaned['pickup_datetime'] = pd.to_datetime(df_cleaned['pickup_datetime'])

# Extract time features
df_cleaned['hour'] = df_cleaned['pickup_datetime'].dt.hour
df_cleaned['day'] = df_cleaned['pickup_datetime'].dt.day
df_cleaned['month'] = df_cleaned['pickup_datetime'].dt.month
df_cleaned['weekday'] = df_cleaned['pickup_datetime'].dt.dayofweek

# === STEP 4: Calculate Distance Using Haversine Formula ===
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

df_cleaned['distance'] = haversine_distance(
    df_cleaned['pickup_latitude'],
    df_cleaned['pickup_longitude'],
    df_cleaned['dropoff_latitude'],
    df_cleaned['dropoff_longitude']
)

# === STEP 5: Descriptive Statistics ===
print("\n--- Descriptive Statistics ---")
print(df_cleaned.describe())

# === STEP 6: Visualizations ===

# 6.1 Fare Amount Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['fare_amount'], bins=50, kde=True)
plt.title('Fare Amount Distribution')
plt.xlabel('Fare Amount ($)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 6.2 Fare Amount Boxplot (Outlier Detection)
plt.figure(figsize=(8, 4))
sns.boxplot(x=df_cleaned['fare_amount'])
plt.title('Fare Amount - Outlier Detection')
plt.tight_layout()
plt.show()

# 6.3 Fare vs Distance
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_cleaned, x='distance', y='fare_amount')
plt.title('Fare vs Distance')
plt.xlabel('Distance (km)')
plt.ylabel('Fare ($)')
plt.tight_layout()
plt.show()

# 6.4 Fare vs Hour of Day
fare_by_hour = df_cleaned.groupby('hour')['fare_amount'].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=fare_by_hour, x='hour', y='fare_amount', marker='o')
plt.title('Average Fare by Hour of Day')
plt.xlabel('Hour (0-23)')
plt.ylabel('Average Fare ($)')
plt.tight_layout()
plt.show()

# === STEP 7: Save Final Enhanced Dataset ===
df_cleaned.to_csv('uber_enhanced.csv', index=False)
print("Enhanced dataset saved as 'uber_enhanced.csv'")


