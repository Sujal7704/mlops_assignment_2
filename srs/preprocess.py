import pandas as pd
import numpy as np
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_params():
    """Load params.yaml configuration"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params

def main():
    # Load parameters
    params = load_params()
    dataset_path = params["dataset"]
    test_size = params["training"]["test_size"]
    random_state = params["training"]["random_state"]
    scale = params["preprocessing"]["scale"]

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Features & target
    X = df.drop("Outcome", axis=1)   # 'Outcome' is target column in diabetes dataset
    y = df["Outcome"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Apply scaling if enabled
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert back to DataFrame for saving
        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)

    # Ensure processed folder exists
    os.makedirs("data/processed", exist_ok=True)

    # Save processed data
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("✅ Preprocessing complete. Files saved in data/processed/")

if __name__ == "__main__":
    main()