# ===============================================
# Diabetes Prediction using Logistic Regression (Function-based)
# ===============================================

# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ===============================================
# Function: Load and Inspect Data
# ===============================================

filepath = r"C:\Users\adity\50_days_challenge\day_2\data\diabetes (4).csv"

def load_data(filepath):
    df = pd.read_csv(filepath)
    print("✅ Dataset Loaded Successfully!")
    print("\nFirst 5 rows:\n", df.head())
    print("\nShape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nMissing Values:\n", df.isnull().sum())
    return df


# ===============================================
# Function: Visualize Data
# ===============================================
def visualize_data(df):
    # Histograms
    df.hist(bins=20, figsize=(12, 8))
    plt.suptitle("Feature Distributions")
    plt.show()

    # Boxplot before outlier removal
    df.boxplot(figsize=(12, 8))
    plt.title("Boxplot Before Removing Outliers")
    plt.show()


# ===============================================
# Function: Remove Outliers using IQR
# ===============================================
def remove_outliers(df):
    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]
    print("✅ Outliers Removed Successfully!\n")
    return df


# ===============================================
# Function: Correlation Heatmap
# ===============================================
def show_correlation(df):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()


# ===============================================
# Function: Preprocess Data (scaling + split)
# ===============================================
def preprocess_data(df):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print("✅ Data Preprocessing Completed!\n")
    return x_train, x_test, y_train, y_test


# ===============================================
# Function: Train Model
# ===============================================
def train_model(x_train, y_train):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    print("✅ Model Trained Successfully!\n")
    return model


# ===============================================
# Function: Evaluate Model
# ===============================================
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"📊 Model Accuracy: {accuracy:.2f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# ===============================================
# Main Function
# ===============================================
def main():
    # Step 1: Load Data
    df = load_data(r"C:\Users\adity\50_days_challenge\day_2\data\diabetes (4).csv")

    # Step 2: Visualize Data
    visualize_data(df)

    # Step 3: Remove Outliers
    df = remove_outliers(df)

    # Step 4: Show Correlation
    show_correlation(df)

    # Step 5: Preprocess Data
    x_train, x_test, y_train, y_test = preprocess_data(df)

    # Step 6: Train Model
    model = train_model(x_train, y_train)

    # Step 7: Evaluate Model
    evaluate_model(model, x_test, y_test)


# ===============================================
# Entry Point
# ===============================================
if __name__ == "__main__":
    main()
