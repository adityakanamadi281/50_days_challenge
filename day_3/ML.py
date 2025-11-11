# toyota_mlr_model.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score


class ToyotaMLRModel:
    def __init__(self, file_path):
        """Initialize dataset and model parameters."""
        self.file_path = file_path
        self.df = None
        self.x_train = self.x_test = self.y_train = self.y_test = None
        self.model = LinearRegression()
        self.lasso = Lasso(alpha=50)
        self.ridge = Ridge(alpha=10)
        self.scaler = StandardScaler()

        # Get the directory of this script file
        self.save_path = os.path.dirname(os.path.abspath(__file__))
        print(f"📂 Images and results will be saved in: {self.save_path}")

    def load_data(self):
        """Load dataset with exception handling and drop Cylinders column."""
        try:
            self.df = pd.read_csv(self.file_path)
            print("✅ Data loaded successfully!")
            print(f"Initial shape: {self.df.shape}")

            # Drop 'Cylinders' if present
            if 'Cylinders' in self.df.columns:
                self.df.drop('Cylinders', axis=1, inplace=True)
                print("🧹 Dropped 'Cylinders' column successfully.")
            else:
                print("ℹ️ 'Cylinders' column not found (already removed or absent).")

            print(f"Updated shape: {self.df.shape}")
            print(self.df.head())

        except FileNotFoundError:
            print("❌ Error: File not found. Please check the file path.")
        except Exception as e:
            print(f"⚠️ Unexpected error while loading data: {e}")

    def preprocess_data(self):
        """Preprocess dataset (handle categorical data, scaling, etc.)."""
        try:
            print("\n🔹 Checking for missing values...")
            if self.df.isnull().sum().any():
                print("Filling missing values with column mean...")
                self.df.fillna(self.df.mean(), inplace=True)
            else:
                print("No missing values found.")

            # Label Encoding
            print("Encoding categorical features...")
            le = LabelEncoder()
            self.df['Fuel_Type'] = le.fit_transform(self.df['Fuel_Type'])

            # Convert Price to Euros
            conversion_rate = 0.011
            self.df["Price_Euro"] = self.df["Price"] * conversion_rate

            # Feature and Target split
            x = self.df.drop(["Price", "Price_Euro"], axis=1)
            y = self.df["Price_Euro"]

            # Scaling
            print("Standardizing numerical features...")
            x_scaled = self.scaler.fit_transform(x)

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                x_scaled, y, test_size=0.2, random_state=42
            )
            print("✅ Data preprocessing complete!")

        except Exception as e:
            print(f"⚠️ Error during preprocessing: {e}")

    def perform_eda(self):
        """Perform EDA (save plots to same folder as script)."""
        try:
            print("\n📊 Performing EDA...")

            # Histogram plot
            self.df.hist(bins=30, figsize=(12, 8))
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, "eda_histograms.png"))
            plt.close()

            # Boxplot
            self.df.plot(kind="box", figsize=(12, 8))
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, "eda_boxplots.png"))
            plt.close()

            # Correlation heatmap
            corr = self.df.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, cmap="coolwarm")
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, "correlation_heatmap.png"))
            plt.close()

            print("✅ EDA plots saved in the same folder as this script.")

        except Exception as e:
            print(f"⚠️ Error during EDA: {e}")

    def train_models(self):
        """Train Linear, Lasso, and Ridge regression models."""
        try:
            print("\n🤖 Training models...")

            # Linear Regression
            self.model.fit(self.x_train, self.y_train)
            y_pred_train = self.model.predict(self.x_train)
            y_pred_test = self.model.predict(self.x_test)

            train_score = r2_score(self.y_train, y_pred_train)
            test_score = r2_score(self.y_test, y_pred_test)

            print(f"Linear Regression R² (Train): {train_score:.3f}")
            print(f"Linear Regression R² (Test): {test_score:.3f}")

            # Lasso Regression
            self.lasso.fit(self.x_train, self.y_train)
            lasso_pred = self.lasso.predict(self.x_test)
            lasso_score = r2_score(self.y_test, lasso_pred)
            print(f"Lasso Regression R²: {lasso_score:.3f}")

            # Ridge Regression
            self.ridge.fit(self.x_train, self.y_train)
            ridge_pred = self.ridge.predict(self.x_test)
            ridge_score = r2_score(self.y_test, ridge_pred)
            print(f"Ridge Regression R²: {ridge_score:.3f}")

            # Plot Actual vs Predicted (Linear Regression)
            plt.scatter(self.y_test, y_pred_test, alpha=0.6, color='blue')
            plt.xlabel("Actual Price (€)")
            plt.ylabel("Predicted Price (€)")
            plt.title("Actual vs Predicted Prices (Linear Regression)")
            plt.grid(True)
            plt.savefig(os.path.join(self.save_path, "actual_vs_predicted.png"))
            plt.close()
            print("📊 Saved 'Actual vs Predicted' plot in the same folder as script.")

            return {
                "Linear_Train_Score": train_score,
                "Linear_Test_Score": test_score,
                "Lasso_Score": lasso_score,
                "Ridge_Score": ridge_score,
            }

        except Exception as e:
            print(f"⚠️ Error during model training: {e}")

    def save_results(self, results):
        """Save model results to a text file in same folder."""
        try:
            result_path = os.path.join(self.save_path, "model_results.txt")
            with open(result_path, "w") as f:
                f.write("Toyota Corolla MLR Model Results\n")
                f.write("=" * 40 + "\n")
                for key, value in results.items():
                    f.write(f"{key}: {value:.3f}\n")
            print(f"📁 Model results saved to '{result_path}'")

        except Exception as e:
            print(f"⚠️ Error saving results: {e}")


def main():
    """Main program execution."""
    file_path = r"C:\Users\adity\50_days_challenge\day_3\data\ToyotaCorolla - MLR.csv"  # Adjust this if needed

    model = ToyotaMLRModel(file_path)

    model.load_data()
    if model.df is not None:
        model.perform_eda()
        model.preprocess_data()
        results = model.train_models()
        if results:
            model.save_results(results)
            print("\n✅ All tasks completed successfully! Images and text saved beside this script.")
        else:
            print("❌ Model training failed.")
    else:
        print("❌ No dataset loaded.")


if __name__ == "__main__":
    main()
