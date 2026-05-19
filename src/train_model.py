# ============================================================
# UK HOUSE PRICE PREDICTION PROJECT
# Training Script
# ============================================================

# Import libraries for data handling
import pandas as pd
import numpy as np

# Import libraries for visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Import machine learning tools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Import machine learning algorithms
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Import XGBoost
from xgboost import XGBRegressor

# Import joblib to save the trained model
import joblib


# ============================================================
# 1. LOAD DATASET
# ============================================================

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("data/UK_House_Price_Prediction_dataset_2015_to_2024.csv")

print("Dataset loaded successfully.")
print("Shape of dataset:", df.shape)

# Display first five rows
print(df.head())


# ============================================================
# 2. EXPLORE DATASET
# ============================================================

print("\nDataset Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nColumns:")
print(df.columns)


# ============================================================
# 3A. FEATURE ENGINEERING
# ============================================================

# Convert date column from text to datetime format
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Extract useful date features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter

# Extract postcode area from postcode
# Example: "SW1A 1AA" becomes "SW1A"
df["postcode_area"] = df["postcode"].astype(str).str.split().str[0]

# Drop columns that are too detailed or not useful for beginner model
df = df.drop(columns=["date", "postcode", "street", "locality"])

# Remove rows with missing values after feature engineering
df = df.dropna()

# ============================================================
# 3B. REMOVE OUTLIERS
# ============================================================

# Remove top 1% most expensive properties
# These are likely commercial properties or data errors
upper_limit = df["price"].quantile(0.99)
print(f"\n99th percentile price: £{upper_limit:,.2f}")

df = df[df["price"] <= upper_limit]
print(f"Shape after removing outliers: {df.shape}")
print(f"Max price after cleaning: £{df['price'].max():,.2f}")

print("\nDataset after feature engineering:")
print(df.head())


# ============================================================
# 4. VISUALISE DATASET
# ============================================================

# Price distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["price"], bins=50, kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/charts/price_distribution.png")
plt.close()

# Average price by property type
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="property_type", y="price")
plt.title("Average House Price by Property Type")
plt.xlabel("Property Type")
plt.ylabel("Average Price")
plt.tight_layout()
plt.savefig("outputs/charts/price_by_property_type.png")
plt.close()

print("\nVisualisations saved as images.")


# ============================================================
# 5. SELECT FEATURES AND TARGET
# ============================================================

# Target variable: what we want to predict
y = df["price"]

# Feature variables: information used to predict price
X = df.drop(columns=["price"])

print("\nFeatures used for prediction:")
print(X.columns)


# ============================================================
# 6. IDENTIFY CATEGORICAL AND NUMERICAL COLUMNS
# ============================================================

categorical_features = [
    "property_type",
    "new_build",
    "freehold",
    "town",
    "district",
    "county",
    "postcode_area"
]

numerical_features = [
    "year",
    "month",
    "quarter"
]


# ============================================================
# 7. PREPROCESSING
# ============================================================

# OneHotEncoder converts text categories into numerical columns
# handle_unknown="ignore" prevents errors when new unseen categories appear
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ============================================================
# 8. TRAIN TEST SPLIT
# ============================================================

# Split dataset into training and testing sets
# 80% is used for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])


# ============================================================
# 9. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    )
}

# ============================================================
# 10. TRAIN AND EVALUATE MODELS
# ============================================================

results = {}

best_model = None
best_model_name = None
best_rmse = float("inf")

for model_name, model in models.items():

    print(f"\nTraining model: {model_name}")

    # Create full pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train the model
    pipeline.fit(X_train, y_train)

    # Generate predictions
    predictions = pipeline.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Store result
    results[model_name] = rmse

    print(f"{model_name} RMSE: £{rmse:,.2f}")

    # Check if this is the best model so far
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = pipeline
        best_model_name = model_name


# ============================================================
# 10B. CROSS-VALIDATION
# ============================================================

print("\nCross-Validation Results (5-fold):")
print("=" * 60)

for model_name, model in models.items():

    pipeline_cv = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    cv_scores = cross_val_score(
        pipeline_cv,
        X,
        y,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    cv_rmse = -cv_scores
    print(f"{model_name}:")
    print(f"  Mean RMSE:  £{cv_rmse.mean():,.2f}")
    print(f"  Std RMSE:   £{cv_rmse.std():,.2f}")
    print(f"  Min RMSE:   £{cv_rmse.min():,.2f}")
    print(f"  Max RMSE:   £{cv_rmse.max():,.2f}")

# ============================================================
# 11. DISPLAY MODEL PERFORMANCE
# ============================================================

print("\nModel Performance Summary:")
for name, score in results.items():
    print(f"{name}: £{score:,.2f}")

print("\nBest Model:", best_model_name)
print("Best RMSE:", f"£{best_rmse:,.2f}")


# ============================================================
# 12. VISUALISE MODEL PERFORMANCE
# ============================================================

plt.figure(figsize=(10, 5))
plt.bar(results.keys(), results.values())
plt.title("Model Comparison using RMSE")
plt.xlabel("Machine Learning Model")
plt.ylabel("RMSE")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("outputs/charts/model_performance.png")
plt.close()

print("\nModel performance chart saved.")


# ============================================================
# 13. SAVE BEST MODEL
# ============================================================

joblib.dump(best_model, "models/best_house_price_model.pkl")

print("\nBest model saved as models/best_house_price_model.pkl")


# ============================================================
# 14. SAVE FEATURE OPTIONS FOR STREAMLIT APP
# ============================================================

# ============================================================
# 14. SAVE FEATURE OPTIONS FOR STREAMLIT APP
# ============================================================

feature_options = {
    "property_type": sorted(df["property_type"].unique().tolist()),
    "new_build": sorted(df["new_build"].unique().tolist()),
    "freehold": sorted(df["freehold"].unique().tolist()),
    "town": sorted(df["town"].unique().tolist()),
    "district": sorted(df["district"].unique().tolist()),
    "county": sorted(df["county"].unique().tolist()),
    "postcode_area": sorted(df["postcode_area"].unique().tolist()),
    "year": {
        "min": int(df["year"].min()),
        "max": int(df["year"].max()),
        "median": int(df["year"].median())
    },
    "month": {
        "min": int(df["month"].min()),
        "max": int(df["month"].max()),
        "median": int(df["month"].median())
    },
    "quarter": {
        "min": int(df["quarter"].min()),
        "max": int(df["quarter"].max()),
        "median": int(df["quarter"].median())
    }
}

joblib.dump(feature_options, "models/feature_options.pkl")

print("Feature options saved as models/feature_options.pkl")