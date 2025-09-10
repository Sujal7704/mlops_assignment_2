import os
import yaml
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def load_params():
    """Load params.yaml configuration"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params

# def get_model(name, params):
#     """Return model object based on name and params"""
#     if name == "LogisticRegression":
#         return LogisticRegression(**params)
#     elif name == "DecisionTreeClassifier":
#         return DecisionTreeClassifier(**params)
#     elif name == "RandomForestClassifier":
#         return RandomForestClassifier(**params)
#     elif name == "XGBClassifier":
#         return XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
#     elif name == "SVC":
#         return SVC(**params)
#     else:
#         raise ValueError(f"❌ Model {name} not supported")

def get_model(name, params):
    """Return model object based on name and params"""
    if name == "LogisticRegression":
        return LogisticRegression(**params)
    elif name == "DecisionTreeClassifier":
        return DecisionTreeClassifier(**params)
    elif name == "RandomForestClassifier":
        return RandomForestClassifier(**params)
    elif name == "XGBClassifier":
        # Prevent duplicate eval_metric issue
        return XGBClassifier(**params, use_label_encoder=False)
    elif name == "SVC":
        return SVC(**params)
    else:
        raise ValueError(f"❌ Model {name} not supported")


def main():
    # Load params
    params = load_params()
    models_config = params["models"]

    # Load processed data
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

    # Create models folder
    os.makedirs("models", exist_ok=True)

    # Train each model
    for model_name, model_params in models_config.items():
        print(f"🚀 Training {model_name}...")
        model = get_model(model_name, model_params)
        model.fit(X_train, y_train)

        # Save model
        model_path = f"models/{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"✅ {model_name} saved at {model_path}")

if __name__ == "__main__":
    main()