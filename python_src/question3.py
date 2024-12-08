# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)

# Basic dataset information
print("\nDataset Info:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

# Distribution of target variable
plt.figure(figsize=(8, 5))
sns.histplot(data["medv"], kde=True, bins=30, color="blue")
plt.title("Distribution of Target Variable (medv)")
plt.xlabel("medv")
plt.ylabel("Frequency")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.show()

# Split features and target variable
X = data.drop(columns=["medv"])  # Features
y = data["medv"]                 # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling for linear models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest Regression": RandomForestRegressor(random_state=42)
}

# Train and evaluate models
results = []
for name, model in models.items():
    if "Regression" in name and name != "Random Forest Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({"Model": name, "MAE": mae, "MSE": mse, "R²": r2})
    
    # Scatter plot for actual vs predicted
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.7, color="green")
    plt.title(f"Actual vs Predicted: {name}")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--")
    plt.show()

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print("\nModel Performance:")
print(results_df)

# Visualize model comparison
plt.figure(figsize=(10, 6))
results_df_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
sns.barplot(x="Metric", y="Value", hue="Model", data=results_df_melted, palette="viridis")
plt.title("Model Performance Comparison")
plt.show()

# Feature importance for Random Forest
rf_model = models["Random Forest Regression"]
feature_importance = rf_model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance (Random Forest):")
print(importance_df)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
plt.title("Feature Importance in Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()


