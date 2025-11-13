path = r"C:\Users\adity\50_days_challenge\day_5\ML\data\iris.csv"


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



def load_data(path):
    df = pd.read_csv(path)
    return df



def visualize_data(df):
    print("Dataset Info:")
    print(df.info())
    print("\nMissing Values:\n", df.isnull().sum())

    df.hist(bins=30, figsize=(12, 8), edgecolor="black")
    plt.show()

    df.boxplot(figsize=(12, 8))
    plt.show()

    sns.pairplot(df)
    plt.show()

    print("\nStatistical Summary:\n")
    print(df.describe())



def split_features(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y



def train_model(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print("\nTraining Accuracy:", accuracy_score(y_train, y_pred_train))

    print("\nConfusion Matrix (Test Data):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)

    sns.heatmap(cm, annot=True,
                xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                yticklabels=['Setosa', 'Versicolor', 'Virginica'])
    plt.show()

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred_test))

    print("Test Accuracy:", accuracy_score(y_test, y_pred_test))



def main():
    # Load data
    path = r"C:\Users\adity\50_days_challenge\day_5\ML\data\iris.csv"
    df = load_data(path)
    print(df.head())

    # Visualize
    visualize_data(df)

    # Split features and labels
    X, y = split_features(df)

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = train_model(X_train, Y_train)

    # Evaluate
    evaluate_model(model, X_train, Y_train, X_test, Y_test)


main()


