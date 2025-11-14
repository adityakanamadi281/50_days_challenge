import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

def load_dataset():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    return df, iris


def visualize_data(df):
    df0 = df[df.target == 0]
    df1 = df[df.target == 1]
    df2 = df[df.target == 2]

    # Plot 1 - Sepal Length vs Width
    plt.figure(figsize=(6,4))
    plt.xlabel("sepal length (cm)")
    plt.ylabel("sepal width (cm)")
    plt.scatter(df0["sepal length (cm)"], df0["sepal width (cm)"], label="Class 0")
    plt.scatter(df1["sepal length (cm)"], df1["sepal width (cm)"], label="Class 1")
    plt.scatter(df2["sepal length (cm)"], df2["sepal width (cm)"], label="Class 2")
    plt.legend()
    plt.savefig("sepal_plot.png", dpi=300)
    plt.show()

    # Plot 2 - Petal Length vs Width
    plt.figure(figsize=(6,4))
    plt.xlabel("petal length (cm)")
    plt.ylabel("petal width (cm)")
    plt.scatter(df0["petal length (cm)"], df0["petal width (cm)"], label="Class 0")
    plt.scatter(df1["petal length (cm)"], df1["petal width (cm)"], label="Class 1")
    plt.scatter(df2["petal length (cm)"], df2["petal width (cm)"], label="Class 2")
    plt.legend()
    plt.savefig("petal_plot.png", dpi=300)
    plt.show()


def train_model(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    svm_model = SVC(kernel="linear", C=0.01)
    svm_model.fit(x_train, y_train)

    y_pred = svm_model.predict(x_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()

    return svm_model


def main():
    df, iris = load_dataset()
    print("Dataset Loaded Successfully!\n")
    print(df.head())

    print("\nVisualizing Data...")
    visualize_data(df)

    print("\nTraining SVM Model...")
    train_model(df)


if __name__ == "__main__":
    main()
