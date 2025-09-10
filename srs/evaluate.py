import os
import json
import joblib
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_params():
    """Load params.yaml configuration"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params

def evaluate_model(model, X_test, y_test):
    """Compute evaluation metrics"""
    y_pred = model.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
    }

def main():
    # Load params
    params = load_params()
    models_config = params["models"]

    # Load processed test data
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    # Create experiments folder
    os.makedirs("experiments", exist_ok=True)

    results = {}

    # Evaluate each model
    for model_name in models_config.keys():
        model_path = f"models/{model_name}.pkl"
        if os.path.exists(model_path):
            print(f"🔎 Evaluating {model_name}...")
            model = joblib.load(model_path)
            metrics = evaluate_model(model, X_test, y_test)
            results[model_name] = metrics
        else:
            print(f"⚠️ Skipping {model_name}, model file not found.")

    # Save results to JSON
    with open("experiments/results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Also save results to CSV
    df_results = pd.DataFrame(results).T
    df_results.to_csv("experiments/results.csv", index=True)

    print("✅ Evaluation complete. Results saved in experiments/")

if __name__ == "__main__":
    main()