#!/usr/bin/env python3
"""
CSL236 - End-Term Project
Model Training Script with MLflow Experiment Tracking
Dataset: Global Cybersecurity Threats (2015-2024)
Target: Incident Severity Tier (Low, Medium, High)
"""

import argparse
import os
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_FEATURES = [
    "country",
    "attack_type",
    "target_industry",
    "attack_source",
    "security_vulnerability_type",
    "defense_mechanism_used",
]

NUMERICAL_FEATURES = [
    "year",
    "number_of_affected_users",
    "incident_resolution_time",
]


def load_and_preprocess_data(data_path: str):
    """
    Load raw dataset, standardize column names, and derive domain-consistent
    Incident Severity Tier (Low, Medium, High) using multi-factor breach impact scoring.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)
    rename_map = {
        "Country": "country",
        "Year": "year",
        "Attack Type": "attack_type",
        "Target Industry": "target_industry",
        "Financial Loss (in Million $)": "financial_loss",
        "Number of Affected Users": "number_of_affected_users",
        "Attack Source": "attack_source",
        "Security Vulnerability Type": "security_vulnerability_type",
        "Defense Mechanism Used": "defense_mechanism_used",
        "Incident Resolution Time (in Hours)": "incident_resolution_time",
    }
    df = df.rename(columns=rename_map)

    # Domain risk weights based on NIST SP 800-61 / CVSS 3.1 breach impact criteria
    actor_weights = {"Nation-state": 1.0, "Hacker Group": 0.7, "Insider": 0.6, "Unknown": 0.3}
    vuln_weights = {"Zero-day": 1.0, "Unpatched Software": 0.7, "Social Engineering": 0.5, "Weak Passwords": 0.3}

    norm_loss = df["financial_loss"] / 100.0
    norm_users = df["number_of_affected_users"] / 1000000.0
    norm_time = df["incident_resolution_time"] / 72.0
    norm_actor = df["attack_source"].map(actor_weights).fillna(0.5)
    norm_vuln = df["security_vulnerability_type"].map(vuln_weights).fillna(0.5)

    # Composite breach impact metric
    impact_score = (
        0.35 * norm_users +
        0.25 * norm_time +
        0.20 * norm_loss +
        0.10 * norm_actor +
        0.10 * norm_vuln
    )

    # Equal-frequency binning into 3 balanced tiers (1000 per class)
    df["severity_tier"] = pd.qcut(impact_score, q=3, labels=["Low", "Medium", "High"])

    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES].copy()
    y = df["severity_tier"].copy()

    return X, y, df


def build_pipeline(algorithm: str, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1, random_state: int = 42):
    """Construct full preprocessor + classifier pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
        ]
    )

    algo_lower = algorithm.lower().strip()
    if algo_lower in ["rf", "random_forest", "randomforest"]:
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
        )
    elif algo_lower in ["gb", "gbr", "gradient_boosting", "gradientboosting"]:
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
    elif algo_lower in ["dt", "decision_tree", "decisiontree"]:
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
        )
    elif algo_lower in ["lr", "logistic_regression", "logisticregression"]:
        clf = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from: rf, gb, dt, lr")

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Cybersecurity Incident Severity Classifier")
    parser.add_argument("--data_path", type=str, default="data/cybersecurity_threats.csv", help="Path to CSV dataset")
    parser.add_argument("--algorithm", type=str, default="rf", help="Algorithm: rf, gb, dt, lr")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of estimators (for tree ensembles)")
    parser.add_argument("--max_depth", type=int, default=6, help="Maximum tree depth")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate (for boosting)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Validation split proportion")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--save_model", action="store_true", default=True, help="Save trained model to model/model.joblib")
    parser.add_argument("--tracking_uri", type=str, default="sqlite:///mlflow.db", help="MLflow backend store URI")
    parser.add_argument("--experiment_name", type=str, default="Cybersecurity_Incident_Severity", help="MLflow experiment name")

    args = parser.parse_args()

    # Configure MLflow SQLite tracking
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    print(f"[*] Loading data from: {args.data_path}")
    X, y, df_raw = load_and_preprocess_data(args.data_path)
    print(f"[*] Dataset shape: {X.shape}, Class distribution:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    with mlflow.start_run(run_name=f"{args.algorithm.upper()}_depth{args.max_depth}_trees{args.n_estimators}"):
        # Log parameters
        mlflow.log_param("algorithm", args.algorithm)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("num_features", len(CATEGORICAL_FEATURES) + len(NUMERICAL_FEATURES))

        # Build and train model
        print(f"[*] Training {args.algorithm.upper()} model (n_estimators={args.n_estimators}, max_depth={args.max_depth})...")
        pipeline = build_pipeline(
            algorithm=args.algorithm,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=args.random_state,
        )
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_weighted", prec)
        mlflow.log_metric("recall_weighted", rec)
        mlflow.log_metric("f1_weighted", f1)

        print("\n=== Evaluation Results ===")
        print(f"Accuracy:           {acc:.4f}")
        print(f"Weighted Precision: {prec:.4f}")
        print(f"Weighted Recall:    {rec:.4f}")
        print(f"Weighted F1:        {f1:.4f}")

        # Model serialization
        if args.save_model:
            os.makedirs("model", exist_ok=True)
            model_path = os.path.join("model", "model.joblib")
            joblib.dump(pipeline, model_path)
            print(f"[+] Serialized model saved to: {model_path}")

            # Also save feature baseline metadata for drift checks
            meta_path = os.path.join("model", "baseline_metadata.json")
            baseline_meta = {
                "algorithm": args.algorithm,
                "classes": [str(c) for c in pipeline.classes_],
                "features": {
                    "categorical": {
                        col: [str(val) for val in X[col].unique()] for col in CATEGORICAL_FEATURES
                    },
                    "numerical": {
                        col: {
                            "mean": float(X[col].mean()),
                            "std": float(X[col].std()),
                            "min": float(X[col].min()),
                            "max": float(X[col].max()),
                            "q25": float(X[col].quantile(0.25)),
                            "median": float(X[col].median()),
                            "q75": float(X[col].quantile(0.75)),
                        }
                        for col in NUMERICAL_FEATURES
                    },
                },
                "metrics": {
                    "accuracy": float(acc),
                    "f1_weighted": float(f1),
                },
            }
            with open(meta_path, "w") as f:
                json.dump(baseline_meta, f, indent=2)
            print(f"[+] Baseline metadata saved to: {meta_path}")

        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        print("[+] Run successfully logged to MLflow.")


if __name__ == "__main__":
    main()
