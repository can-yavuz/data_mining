import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load the dataset
data_path = '../datasets/heart.csv'
df = pd.read_csv(data_path)

# Basic information
print("\nDataset Information:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Target variable distribution
print("\nTarget Variable Distribution:")
print(df['target'].value_counts())

# Visualize target distribution
sns.countplot(x='target', data=df)
plt.title("Target Variable Distribution (0 = Healthy, 1 = Diseased)")
plt.show()

# Statistical summary of all columns
print("\nStatistical Summary:")
print(df.describe())

# Correlation matrix
correlation_matrix = df.corr()

# Visualize correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Examine correlation with target
target_corr = correlation_matrix["target"].sort_values(ascending=False)
print("\nCorrelation with Target:")
print(target_corr)

# Features to remove based on correlation analysis (example: low correlation features)
remove_features = ["trestbps", "chol", "fbs", "restecg"]

# Separate features and target variable
X = df.drop(columns=["target"] + remove_features)
y = df["target"]

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Evaluate performance
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree Model Performance:")
print(classification_report(y_test, y_pred_dt))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt="d", cmap="Blues")
plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# SVM Model
svm_model = SVC(random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Evaluate performance
y_pred_svm = svm_model.predict(X_test_scaled)
print("SVM Model Performance:")
print(classification_report(y_test, y_pred_svm))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt="d", cmap="Purples")
plt.title("SVM - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Function to calculate performance metrics
def calculate_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

# Decision Tree Metrics
dt_metrics = calculate_metrics(y_test, y_pred_dt)
print("Decision Tree Metrics:", dt_metrics)

# SVM Metrics
svm_metrics = calculate_metrics(y_test, y_pred_svm)
print("SVM Metrics:", svm_metrics)

# Comparison Table
comparison_df = pd.DataFrame({
    "Model": ["Decision Tree", "SVM"],
    "Accuracy": [dt_metrics[0], svm_metrics[0]],
    "Precision": [dt_metrics[1], svm_metrics[1]],
    "Recall": [dt_metrics[2], svm_metrics[2]],
    "F1 Score": [dt_metrics[3], svm_metrics[3]]
})

print("Model Performance Comparison:")
print(comparison_df)

# Visualize metrics
comparison_df.set_index("Model").plot(kind="bar", figsize=(10, 6))
plt.title("Model Performance Comparison")
plt.ylabel("Metric Value")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.show()
