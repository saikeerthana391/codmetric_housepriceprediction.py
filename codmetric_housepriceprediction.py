# House Price Prediction - Codmetric Task 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ==========================
# Step 1: Load Dataset
# ==========================
# Example: California Housing Dataset from sklearn
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("Dataset shape:", df.shape)
print(df.head())

# ==========================
# Step 2: Data Preprocessing
# ==========================
# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Features and target
X = df.drop("MedHouseVal", axis=1)   # Features
y = df["MedHouseVal"]                # Target variable (House Price)

# Optional: Scale features for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# Step 3: Train-Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==========================
# Step 4: Train Model
# ==========================
model = LinearRegression()
model.fit(X_train, y_train)

# ==========================
# Step 5: Predictions
# ==========================
y_pred = model.predict(X_test)

# ==========================
# Step 6: Evaluation
# ==========================
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# ==========================
# Step 7: Visualization
# ==========================
plt.figure(figsize=(7, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
