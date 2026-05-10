"""Data loading and preprocessing utilities."""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os


def load_data(path: str) -> pd.DataFrame:
    """Load raw data from a CSV file."""
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and encode data for modeling."""
    df = df.drop_duplicates()
    df = df.dropna()

    # Drop non-feature columns
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    # Encode categoricals
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    df["country"] = df["country"].map({"France": 0, "Germany": 1, "Spain": 2})

    return df


def scale_features(X_train, X_test, save_path: str = "models/scaler.pkl"):
    """Fit a StandardScaler on train set, transform both splits, and save the scaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(scaler, save_path)
    print(f"Scaler saved to {save_path}")

    return X_train_scaled, X_test_scaled
