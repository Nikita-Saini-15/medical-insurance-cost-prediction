"""
Medical Insurance Cost Prediction (Regression)
-----------------------------------------------
Dataset : insurance.csv (Kaggle: "Medical Cost Personal Dataset")
Target  : charges
Run     : python insurance_cost_prediction.py
Output  : prints results in the console and saves all plots in ./figures/
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
RANDOM_STATE = 42
os.makedirs("figures", exist_ok=True)


def save_fig(name):
    """Save the current figure and display it."""
    plt.tight_layout()
    plt.savefig(f"figures/{name}.png", dpi=150)
    plt.show()
    plt.close()


# ============================================================
# 2. LOAD DATASET
# ============================================================
df = pd.read_csv("insurance.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nInfo:")
df.info()
print("\nStatistical summary:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
# 3.1 Distribution of target variable
plt.figure(figsize=(7, 4))
sns.histplot(df["charges"], bins=40, kde=True, color="steelblue")
plt.title("Distribution of Insurance Charges")
plt.xlabel("Charges")
save_fig("01_charges_distribution")

# 3.2 Numerical feature distributions
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["age", "bmi", "children"]):
    sns.histplot(df[col], kde=(col != "children"), ax=ax, color="teal")
    ax.set_title(f"Distribution of {col}")
save_fig("02_numeric_distributions")

# 3.3 Categorical feature counts
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["sex", "smoker", "region"]):
    sns.countplot(x=col, data=df, ax=ax, palette="Set2")
    ax.set_title(f"Count of {col}")
    ax.tick_params(axis="x", rotation=15)
save_fig("03_categorical_counts")

# 3.4 Charges vs categorical features
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["smoker", "sex", "region"]):
    sns.boxplot(x=col, y="charges", data=df, ax=ax, palette="Set3")
    ax.set_title(f"Charges by {col}")
    ax.tick_params(axis="x", rotation=15)
save_fig("04_charges_vs_categorical")

# 3.5 Age and BMI vs charges (coloured by smoker)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(x="age", y="charges", hue="smoker", data=df, ax=axes[0], alpha=0.7)
axes[0].set_title("Age vs Charges")
sns.scatterplot(x="bmi", y="charges", hue="smoker", data=df, ax=axes[1], alpha=0.7)
axes[1].set_title("BMI vs Charges")
save_fig("05_age_bmi_vs_charges")

# 3.6 Correlation heatmap (after encoding)
df_enc = df.copy()
df_enc["sex"] = df_enc["sex"].map({"male": 0, "female": 1})
df_enc["smoker"] = df_enc["smoker"].map({"no": 0, "yes": 1})
plt.figure(figsize=(7, 5))
sns.heatmap(df_enc.drop(columns="region").corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
save_fig("06_correlation_heatmap")

# ============================================================
# 4. DATA PREPROCESSING
# ============================================================
df = df.drop_duplicates().reset_index(drop=True)

# One-hot encode categorical columns
X = pd.get_dummies(df.drop(columns="charges"), columns=["sex", "smoker", "region"],
                   drop_first=True)
y = df["charges"]

print("\n" + "=" * 60)
print("FEATURES AFTER ENCODING")
print("=" * 60)
print(list(X.columns))

# Train-test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print("\nTrain size:", X_train.shape, "| Test size:", X_test.shape)

# ============================================================
# 5. MODEL BUILDING AND EVALUATION
# ============================================================
models = {
    "Linear Regression": Pipeline([("scaler", StandardScaler()),
                                   ("model", LinearRegression())]),
    "Ridge Regression": Pipeline([("scaler", StandardScaler()),
                                  ("model", Ridge(alpha=1.0))]),
    "Lasso Regression": Pipeline([("scaler", StandardScaler()),
                                  ("model", Lasso(alpha=1.0))]),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=6,
                                           random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.1,
                                                   max_depth=3,
                                                   random_state=RANDOM_STATE),
}

results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()

    results.append({"Model": name, "MAE": mae, "RMSE": rmse,
                    "R2 (Test)": r2, "R2 (5-Fold CV)": cv_r2})

results_df = (pd.DataFrame(results)
              .sort_values("R2 (Test)", ascending=False)
              .reset_index(drop=True))

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df.round(4).to_string(index=False))
results_df.round(4).to_csv("figures/model_results.csv", index=False)

# Model comparison chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(x="R2 (Test)", y="Model", data=results_df, ax=axes[0], palette="viridis")
axes[0].set_title("R2 Score Comparison")
sns.barplot(x="RMSE", y="Model", data=results_df, ax=axes[1], palette="magma")
axes[1].set_title("RMSE Comparison")
save_fig("07_model_comparison")

# ============================================================
# 6. HYPERPARAMETER TUNING (Gradient Boosting)
# ============================================================
param_grid = {
    "n_estimators": [100, 150, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3, 4],
}
grid = GridSearchCV(GradientBoostingRegressor(random_state=RANDOM_STATE),
                    param_grid, cv=5, scoring="r2", n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)

print("\n" + "=" * 60)
print("TUNED GRADIENT BOOSTING")
print("=" * 60)
print("Best parameters :", grid.best_params_)
print("Best CV R2      :", round(grid.best_score_, 4))
print("Test MAE        :", round(mean_absolute_error(y_test, y_pred_best), 2))
print("Test RMSE       :", round(np.sqrt(mean_squared_error(y_test, y_pred_best)), 2))
print("Test R2         :", round(r2_score(y_test, y_pred_best), 4))

# ============================================================
# 7. RESULT VISUALISATION OF BEST MODEL
# ============================================================
# 7.1 Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_best, alpha=0.6, color="darkorange")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted (Tuned Gradient Boosting)")
save_fig("08_actual_vs_predicted")

# 7.2 Residual plot
residuals = y_test - y_pred_best
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].scatter(y_pred_best, residuals, alpha=0.6, color="purple")
axes[0].axhline(0, color="black", linestyle="--")
axes[0].set_xlabel("Predicted Charges")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Residuals vs Predicted")
sns.histplot(residuals, kde=True, ax=axes[1], color="green")
axes[1].set_title("Residual Distribution")
save_fig("09_residual_analysis")

# 7.3 Feature importance
importance = (pd.Series(best_model.feature_importances_, index=X.columns)
              .sort_values(ascending=True))
plt.figure(figsize=(7, 5))
importance.plot(kind="barh", color="steelblue")
plt.title("Feature Importance (Tuned Gradient Boosting)")
plt.xlabel("Importance")
save_fig("10_feature_importance")

print("\nFeature importance:\n", importance.sort_values(ascending=False).round(4))

# ============================================================
# 8. PREDICT COST FOR A NEW PERSON
# ============================================================
def predict_charges(age, sex, bmi, children, smoker, region):
    """Return predicted insurance charges for one person."""
    row = pd.DataFrame([{"age": age, "bmi": bmi, "children": children,
                         "sex_male": int(sex == "male"),
                         "smoker_yes": int(smoker == "yes"),
                         "region_northwest": int(region == "northwest"),
                         "region_southeast": int(region == "southeast"),
                         "region_southwest": int(region == "southwest")}])
    row = row[X.columns]
    return best_model.predict(row)[0]


print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)
print("Non-smoker, 30 yrs, BMI 25, 1 child, southwest : "
      f"{predict_charges(30, 'male', 25, 1, 'no', 'southwest'):.2f}")
print("Smoker,     30 yrs, BMI 25, 1 child, southwest : "
      f"{predict_charges(30, 'male', 25, 1, 'yes', 'southwest'):.2f}")
print("Smoker,     50 yrs, BMI 35, 2 child, southeast : "
      f"{predict_charges(50, 'female', 35, 2, 'yes', 'southeast'):.2f}")
