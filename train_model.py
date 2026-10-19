

import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 1. Load the cleaned dataset
# --------------------------------------------------

data_path = Path("outputs/laptop_cleaned.csv")
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_path)

print("Dataset loaded:", df.shape)

# --------------------------------------------------
# 2. Select input features (X) and target (y)
# --------------------------------------------------

numeric_features = [
    "Inches",
    "Ram_GB",
    "Weight_kg"
]

categorical_features = [
    "Company",
    "TypeName",
    "ScreenResolution",
    "Cpu",
    "Memory",
    "Gpu",
    "OpSys"
]

feature_columns = numeric_features + categorical_features

X = df[feature_columns].copy()
y = df["Price_euros"].copy()

print("Input features:", X.shape[1])
print("Target variable: Price_euros")

# --------------------------------------------------
# 3. Split data into training and testing sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 4. Preprocess categorical features
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)

# --------------------------------------------------
# 5. Create the Random Forest model
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                min_samples_leaf=2,
                n_jobs=-1
            )
        )
    ]
)

# --------------------------------------------------
# 6. Train the model
# --------------------------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")

# --------------------------------------------------
# 7. Evaluate the model
# --------------------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\n--- MODEL EVALUATION ---")
print(f"MAE:  {mae:.2f} euros")
print(f"RMSE: {rmse:.2f} euros")
print(f"R2 Score: {r2:.4f}")

# --------------------------------------------------
# 8. Save the trained model
# --------------------------------------------------

model_path = model_dir / "laptop_price_model.joblib"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Location:", model_path)

# --------------------------------------------------
# 9. Compare a few predictions with actual prices
# --------------------------------------------------

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})

results["Absolute Error"] = (
    results["Actual Price"] - results["Predicted Price"]
).abs()

print("\n--- SAMPLE PREDICTIONS ---")
print(results.head(10).round(2).to_string(index=False))
