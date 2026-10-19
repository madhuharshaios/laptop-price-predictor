
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# File locations
data_path = Path("data/laptop_price.csv")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Load the dataset
df = pd.read_csv(data_path, encoding="latin1")

print("Original dataset shape:", df.shape)

# 1. Check duplicate records
print("\nDuplicate rows:", df.duplicated().sum())

# Remove exact duplicate rows
df = df.drop_duplicates().copy()

print("Shape after removing duplicates:", df.shape)

# 2. Convert RAM to numeric values
# Example: "8GB" becomes 8
df["Ram_GB"] = (
    df["Ram"]
    .str.replace("GB", "", regex=False)
    .astype(float)
)

# 3. Convert weight to numeric values
# Example: "1.37kg" becomes 1.37
df["Weight_kg"] = (
    df["Weight"]
    .str.replace("kg", "", regex=False)
    .astype(float)
)

# 4. Display descriptive statistics
print("\n--- PRICE STATISTICS (EUROS) ---")
print(df["Price_euros"].describe())

print("\n--- RAM VALUES ---")
print(sorted(df["Ram_GB"].unique()))

print("\n--- WEIGHT STATISTICS ---")
print(df["Weight_kg"].describe())

print("\n--- LAPTOPS BY COMPANY ---")
print(df["Company"].value_counts())

# 5. Plot price distribution
plt.figure(figsize=(9, 5))
plt.hist(df["Price_euros"], bins=30, edgecolor="black")
plt.title("Laptop Price Distribution")
plt.xlabel("Price (Euros)")
plt.ylabel("Number of Laptops")
plt.tight_layout()
plt.savefig(output_dir / "price_distribution.png")
plt.close()

# 6. Plot average price by brand
average_prices = (
    df.groupby("Company")["Price_euros"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(11, 6))
average_prices.plot(kind="bar")
plt.title("Average Laptop Price by Brand")
plt.xlabel("Brand")
plt.ylabel("Average Price (Euros)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(output_dir / "average_price_by_brand.png")
plt.close()

# Save cleaned data for the next step
df.to_csv(output_dir / "laptop_cleaned.csv", index=False)

print("\nEDA completed successfully!")
print("Charts and cleaned data saved in the outputs folder.")
