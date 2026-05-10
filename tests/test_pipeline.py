"""
End-to-end tests for the bank churn predictor.
Run with:  uv run pytest tests/ -v
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pytest

# Make sure src/ is importable from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import load_data, preprocess, scale_features
from src.features import build_features
from src.train import split_data, select_best_model, save_model, plot_feature_importance, plot_model_comparison
from src.evaluate import evaluate

DATA_PATH   = "data/churn.csv"
MODEL_PATH  = "models/churn_model.pkl"
SCALER_PATH = "models/scaler.pkl"
TARGET      = "churn"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def raw_df():
    return load_data(DATA_PATH)

@pytest.fixture(scope="session")
def processed_df(raw_df):
    return preprocess(raw_df.copy())

@pytest.fixture(scope="session")
def featured_df(processed_df):
    return build_features(processed_df.copy())

@pytest.fixture(scope="session")
def splits(featured_df):
    X = featured_df.drop(columns=[TARGET])
    y = featured_df[TARGET]
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_s, X_test_s = scale_features(X_train, X_test, save_path="models/scaler.pkl")
    return X_train_s, X_test_s, y_train, y_test

@pytest.fixture(scope="session")
def best_model_and_scores(splits):
    X_train_s, _, y_train, _ = splits
    name, model, scores = select_best_model(X_train_s, y_train)
    return name, model, scores


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
class TestDataLoading:
    def test_loads_correct_shape(self, raw_df):
        assert raw_df.shape == (10000, 12), "Expected 10 000 rows and 12 columns"

    def test_expected_columns(self, raw_df):
        expected = {
            "customer_id", "credit_score", "country", "gender", "age",
            "tenure", "balance", "products_number", "credit_card",
            "active_member", "estimated_salary", "churn",
        }
        assert set(raw_df.columns) == expected

    def test_no_missing_values(self, raw_df):
        assert raw_df.isnull().sum().sum() == 0

    def test_no_duplicates(self, raw_df):
        assert raw_df.duplicated().sum() == 0

    def test_target_is_binary(self, raw_df):
        assert set(raw_df["churn"].unique()) == {0, 1}


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
class TestPreprocessing:
    def test_customer_id_dropped(self, processed_df):
        assert "customer_id" not in processed_df.columns

    def test_gender_encoded(self, processed_df):
        assert set(processed_df["gender"].unique()).issubset({0, 1})

    def test_country_encoded(self, processed_df):
        assert set(processed_df["country"].unique()).issubset({0, 1, 2})

    def test_no_nulls_after_preprocess(self, processed_df):
        assert processed_df.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
class TestFeatureEngineering:
    def test_new_features_exist(self, featured_df):
        for col in ("balance_salary_ratio", "age_group", "zero_balance"):
            assert col in featured_df.columns, f"Missing engineered feature: {col}"

    def test_zero_balance_flag(self, featured_df):
        mask = featured_df["balance"] == 0
        assert (featured_df.loc[mask, "zero_balance"] == 1).all()
        assert (featured_df.loc[~mask, "zero_balance"] == 0).all()

    def test_age_group_range(self, featured_df):
        assert featured_df["age_group"].between(0, 3).all()

    def test_balance_salary_ratio_non_negative(self, featured_df):
        assert (featured_df["balance_salary_ratio"] >= 0).all()


# ---------------------------------------------------------------------------
# 4. Model selection — best model is actually the best
# ---------------------------------------------------------------------------
class TestModelSelection:
    def test_four_candidates_evaluated(self, best_model_and_scores):
        _, _, scores = best_model_and_scores
        assert len(scores) == 4

    def test_best_model_has_highest_auc(self, best_model_and_scores):
        name, _, scores = best_model_and_scores
        assert scores[name] == max(scores.values()), \
            f"Saved model '{name}' is not the one with the highest CV AUC"

    def test_best_auc_above_threshold(self, best_model_and_scores):
        _, _, scores = best_model_and_scores
        best_auc = max(scores.values())
        assert best_auc >= 0.80, f"Best CV ROC-AUC {best_auc:.4f} is below 0.80"

    def test_model_can_predict(self, best_model_and_scores, splits):
        _, model, _ = best_model_and_scores
        X_train_s, X_test_s, _, _ = splits
        preds = model.predict(X_test_s)
        assert len(preds) == len(X_test_s)
        assert set(preds).issubset({0, 1})

    def test_model_has_predict_proba(self, best_model_and_scores, splits):
        _, model, _ = best_model_and_scores
        _, X_test_s, _, _ = splits
        proba = model.predict_proba(X_test_s)
        assert proba.shape == (len(X_test_s), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. Test-set performance
# ---------------------------------------------------------------------------
class TestTestSetPerformance:
    def test_roc_auc_above_threshold(self, best_model_and_scores, splits):
        from sklearn.metrics import roc_auc_score
        _, model, _ = best_model_and_scores
        _, X_test_s, _, y_test = splits
        auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
        assert auc >= 0.80, f"Test ROC-AUC {auc:.4f} is below 0.80"

    def test_recall_churn_class_above_threshold(self, best_model_and_scores, splits):
        from sklearn.metrics import recall_score
        _, model, _ = best_model_and_scores
        _, X_test_s, _, y_test = splits
        recall = recall_score(y_test, model.predict(X_test_s))
        assert recall >= 0.60, f"Churn recall {recall:.4f} is below 0.60"


# ---------------------------------------------------------------------------
# 6. Saved artifacts exist and are loadable
# ---------------------------------------------------------------------------
class TestArtifacts:
    def test_model_file_exists(self):
        assert os.path.exists(MODEL_PATH), "churn_model.pkl not found"

    def test_scaler_file_exists(self):
        assert os.path.exists(SCALER_PATH), "scaler.pkl not found"

    def test_model_loadable(self):
        import joblib
        model = joblib.load(MODEL_PATH)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_scaler_loadable(self):
        import joblib
        scaler = joblib.load(SCALER_PATH)
        assert hasattr(scaler, "transform")

    def test_output_plots_exist(self):
        for fname in ("confusion_matrix.png", "roc_curve.png",
                      "feature_importance.png", "model_comparison.png"):
            assert os.path.exists(os.path.join("outputs", fname)), \
                f"Missing output plot: {fname}"


# ---------------------------------------------------------------------------
# 7. FastAPI endpoint
# ---------------------------------------------------------------------------
class TestFastAPI:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from app.api import app
        # lifespan=True ensures the startup handler runs (loads model/scaler)
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_health_model_loaded(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is True

    def test_predict_low_risk(self, client):
        payload = {
            "credit_score": 800, "country": 0, "gender": 1,
            "age": 30, "tenure": 8, "balance": 0,
            "products_number": 2, "credit_card": 1,
            "active_member": 1, "estimated_salary": 120000,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert "churn" in body
        assert "churn_probability" in body
        assert "risk_level" in body
        assert body["churn"] in (0, 1)
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    def test_predict_high_risk_profile(self, client):
        """Older inactive German customer with high balance — typically high risk."""
        payload = {
            "credit_score": 400, "country": 1, "gender": 0,
            "age": 55, "tenure": 1, "balance": 150000,
            "products_number": 1, "credit_card": 0,
            "active_member": 0, "estimated_salary": 50000,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["churn_probability"] > 0.5, \
            f"Expected high-risk profile to score > 0.5, got {body['churn_probability']}"

    def test_predict_missing_optional_fields(self, client):
        """Engineered features are optional — API should compute them automatically."""
        payload = {
            "credit_score": 650, "country": 0, "gender": 1,
            "age": 35, "tenure": 5, "balance": 50000,
            "products_number": 1, "credit_card": 1,
            "active_member": 1, "estimated_salary": 60000,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200

    def test_predict_with_explicit_engineered_features(self, client):
        """Supplying engineered features explicitly should also work."""
        payload = {
            "credit_score": 650, "country": 0, "gender": 1,
            "age": 35, "tenure": 5, "balance": 50000,
            "products_number": 1, "credit_card": 1,
            "active_member": 1, "estimated_salary": 60000,
            "balance_salary_ratio": 0.833,
            "age_group": 1,
            "zero_balance": 0,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# 8. EDA notebook structure
# ---------------------------------------------------------------------------
class TestEDANotebook:
    def test_notebook_exists(self):
        assert os.path.exists("notebooks/01_eda.ipynb")

    def test_no_modeling_code(self):
        with open("notebooks/01_eda.ipynb", encoding="utf-8") as f:
            nb = json.load(f)
        modeling_keywords = [
            "train_test_split", "LogisticRegression", "RandomForest",
            "XGBClassifier", "GridSearchCV", "joblib.dump",
        ]
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            for kw in modeling_keywords:
                assert kw not in src, \
                    f"EDA notebook contains modeling code: '{kw}'"

    def test_has_markdown_headers(self):
        with open("notebooks/01_eda.ipynb", encoding="utf-8") as f:
            nb = json.load(f)
        md_cells = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
        assert len(md_cells) >= 4, "EDA notebook should have at least 4 markdown headers"
