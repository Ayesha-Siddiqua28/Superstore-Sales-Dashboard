import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("cleaned_used_cars.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# -----------------------------
# Top 10 Car Brands
# -----------------------------
plt.figure(figsize=(10,6))
df["brand"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Car Brands")
plt.xlabel("Brand")
plt.ylabel("Number of Cars")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# Price Distribution
# -----------------------------
plt.figure(figsize=(8,5))
plt.hist(df["price"], bins=30)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# -----------------------------
# Fuel Type Distribution
# -----------------------------
plt.figure(figsize=(8,5))
df["fuel_type"].value_counts().plot(kind="bar")
plt.title("Fuel Type Distribution")
plt.xlabel("Fuel Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# Transmission Distribution
# -----------------------------
plt.figure(figsize=(8,5))
df["transmission"].value_counts().plot(kind="bar")
plt.title("Transmission Distribution")
plt.xlabel("Transmission")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# Mileage vs Price
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(df["milage"], df["price"], alpha=0.5)
plt.title("Mileage vs Price")
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

# -----------------------------
# Model Year vs Price
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(df["model_year"], df["price"], alpha=0.5)
plt.title("Model Year vs Price")
plt.xlabel("Model Year")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

print("\nEDA Completed Successfully!")