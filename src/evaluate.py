"""Model evaluation utilities."""

import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


def evaluate(model, X_test, y_test, output_dir: str = "outputs") -> dict:
    """Print metrics and save evaluation plots."""
    os.makedirs(output_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)

    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {auc:.4f}")

    # Confusion matrix
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    fig.savefig(os.path.join(output_dir, "roc_curve.png"), bbox_inches="tight")
    plt.close(fig)

    print(f"Plots saved to {output_dir}/")

    return {"classification_report": report, "roc_auc": auc}
