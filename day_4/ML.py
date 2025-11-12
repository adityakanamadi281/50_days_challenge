import os
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error



filepath = r"C:\Users\adity\50_days_challenge\day_4\data\Student_Performance.csv"


def load_data(filepath):
    df = pd.read_csv(filepath)
    print("dataset loaded successfully")
    print("first 5 rows: ", df.head())
    print("shape: ", df.shape)
    print("columns: ", df.columns.tolist())
    print("missing values : ", df.isnull().sum())
    print("Dataset Info : \n", df.info())
    return df

df= load_data(filepath)


def Preprocess_data(df):
    df["Performance Index"]= df["Performance Index"].astype(int)
    activities = pd.get_dummies(df["Extracurricular Activities"], drop_first=True)

    df = df.drop("Extracurricular Activities", axis=1)
    df = pd.concat([df, activities], axis=1)
    print("Data preprocessed successfully")
    print("Final columns: ", df.columns.tolist())
    return df
df = Preprocess_data(df)


def visualize_data(df, save_folder):
    print("\n Generating data visualizations...")
    os.makedirs(save_folder, exist_ok=True)

    # Histogram
    plt.figure(figsize=(12, 8))
    df.hist(bins=30, figsize=(12, 8), edgecolor="black")
    plt.suptitle("Feature Distributions", fontsize=16)
    hist_path = os.path.join(save_folder, "histogram.png")
    plt.savefig(hist_path, bbox_inches="tight")
    plt.close()

    # Boxplot
    plt.figure(figsize=(12, 8))
    df.plot(kind="box", figsize=(12, 8), title="Boxplot of Features")
    plt.savefig(os.path.join(save_folder, "boxplot.png"), bbox_inches="tight")
    plt.close()

    # Correlation Heatmap
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    heatmap_path = os.path.join(save_folder, "heatmap.png")
    plt.savefig(heatmap_path, bbox_inches="tight")
    plt.close()

filepath1 = r"C:\Users\adity\50_days_challenge\day_4\images"
save_folder = os.path.dirname(filepath1)

#df= visualize_data(df, save_folder)


def split_data(df):
    X = df.drop("Performance Index", axis=1)
    y = df["Performance Index"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n Data split completed!")
    print("Train shape:", x_train.shape)
    print("Test shape:", x_test.shape)

    return x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = split_data(df)


def train_model(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    print("\n Model Training completed")
    return model
model = train_model(x_train, y_train)



def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    score = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print("\n Model Evaluation:")
    print("R2 Score:", score)
    print("Mean Squared Error:", mse)
    return y_pred, score, mse
y_pred, score, mse = evaluate_model(model, x_test, y_test)




def save_model(model, save_dir, model_name="student_performance_model.pkl"):
    os.makedirs(save_dir, exist_ok=True)  
    model_path = os.path.join(save_dir, model_name)
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved successfully at: {model_path}")
    return model_path


# Define the folder to save the model
model_save_folder = r"C:\Users\adity\50_days_challenge\day_4\models"

# Save the trained model
save_model(model, model_save_folder)