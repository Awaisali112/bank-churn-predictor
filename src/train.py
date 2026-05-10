"""Model training pipeline with automatic best-model selection."""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------
def _candidate_models(scale_pos_weight: float, random_state: int) -> dict:
    """Return a dict of {name: unfitted model}."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
        ),
    }


# ---------------------------------------------------------------------------
# Select best model via cross-validation
# ---------------------------------------------------------------------------
def select_best_model(
    X_train,
    y_train,
    random_state: int = 42,
    cv_folds: int = 5,
) -> tuple:
    """
    Train all candidate models with stratified k-fold CV.
    Returns (best_model_name, best_fitted_model, scores_dict).
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    candidates = _candidate_models(scale_pos_weight, random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    scores = {}
    print(f"  Running {cv_folds}-fold CV on {len(candidates)} models...\n")

    for name, model in candidates.items():
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=-1,
        )
        mean_auc = cv_scores.mean()
        scores[name] = mean_auc
        print(f"  {name:<25}  CV ROC-AUC = {mean_auc:.4f}  (±{cv_scores.std():.4f})")

    best_name = max(scores, key=scores.get)
    print(f"\n  ✔ Best model: {best_name}  (ROC-AUC = {scores[best_name]:.4f})")

    # Refit the winner on the full training set
    best_model = candidates[best_name]
    best_model.fit(X_train, y_train)

    return best_name, best_model, scores


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_model(model, path: str = "models/churn_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"  Model saved → {path}")


# ---------------------------------------------------------------------------
# Feature importance (works for tree-based models; falls back gracefully)
# ---------------------------------------------------------------------------
def plot_feature_importance(model, feature_names, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # Try .feature_importances_ (tree models), then coef_ (linear models)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        label = "Importance (gain)"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        label = "|Coefficient|"
    else:
        print("  Skipping feature importance — model type not supported.")
        return

    indices      = np.argsort(importances)
    sorted_names = [feature_names[i] for i in indices]
    sorted_vals  = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(sorted_names, sorted_vals, color="steelblue")
    ax.set_xlabel(label)
    ax.set_title("Feature Importance")
    fig.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Feature importance plot saved → {path}")


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------
def plot_model_comparison(scores: dict, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)

    names  = list(scores.keys())
    values = list(scores.values())
    colors = ["steelblue" if v < max(values) else "darkorange" for v in values]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, values, color=colors)
    ax.set_ylabel("CV ROC-AUC")
    ax.set_title("Model Comparison (5-fold CV)")
    ax.set_ylim(min(values) - 0.05, 1.0)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    fig.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Model comparison chart saved → {path}")
