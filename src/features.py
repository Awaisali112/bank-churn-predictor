"""Feature engineering utilities."""

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features from cleaned data."""
    # Balance-to-salary ratio
    df["balance_salary_ratio"] = df["balance"] / (df["estimated_salary"] + 1)

    # Age group buckets
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 45, 60, 100],
        labels=[0, 1, 2, 3],
    ).astype(int)

    # Is zero balance
    df["zero_balance"] = (df["balance"] == 0).astype(int)

    return df
