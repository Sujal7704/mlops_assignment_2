import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_params():
    """Load params.yaml configuration"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params

def main():
    # Load dataset
    params = load_params()
    dataset_path = params["dataset"]
    df = pd.read_csv(dataset_path)

    # Create output folder
    os.makedirs("reports/eda", exist_ok=True)

    # 1. Dataset info
    with open("reports/eda/dataset_info.txt", "w") as f:
        f.write(f"Shape: {df.shape}\n\n")
        f.write("Data Types:\n")
        f.write(str(df.dtypes))
        f.write("\n\nMissing Values:\n")
        f.write(str(df.isnull().sum()))
        f.write("\n\nSummary Statistics:\n")
        f.write(str(df.describe()))

    # 2. Distribution plots for each feature
    for col in df.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"reports/eda/{col}_distribution.jpg")
        plt.close()

    # 3. Boxplots for outlier detection
    for col in df.columns:
        if df[col].dtype != "object":
            plt.figure(figsize=(6,4))
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot of {col}")
            plt.savefig(f"reports/eda/{col}_boxplot.jpg")
            plt.close()

    # 4. Correlation heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig("reports/eda/correlation_heatmap.jpg")
    plt.close()

    # 5. Pairplot (relationships between features)
    sns.pairplot(df, hue="Outcome", diag_kind="kde")
    plt.savefig("reports/eda/pairplot.jpg")
    plt.close()

    # 6. Countplot for target variable
    plt.figure(figsize=(6,4))
    sns.countplot(x="Outcome", data=df)
    plt.title("Class Distribution (Outcome)")
    plt.savefig("reports/eda/target_distribution.jpg")
    plt.close()

    print("✅ EDA completed. Plots and summary saved in reports/eda/")

if __name__ == "__main__":
    main()