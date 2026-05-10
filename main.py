"""Entry point for the bank churn prediction pipeline."""

from src.data_preprocessing import load_data, preprocess, scale_features
from src.features import build_features
from src.train import split_data, select_best_model, save_model, plot_feature_importance, plot_model_comparison
from src.evaluate import evaluate

DATA_PATH = "data/churn.csv"
TARGET    = "churn"


def main():
    print("=== Bank Churn Predictor Pipeline ===\n")

    # 1. Load
    print("[1/5] Loading data...")
    df = load_data(DATA_PATH)
    print(f"      {len(df)} rows, {df.shape[1]} columns\n")

    # 2. Preprocess
    print("[2/5] Preprocessing...")
    df = preprocess(df)

    # 3. Feature engineering
    print("[3/5] Building features...")
    df = build_features(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    feature_names = list(X.columns)
    print(f"      Features: {feature_names}\n")

    # 4. Split → scale → model selection
    print("[4/5] Selecting best model...\n")
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    best_name, best_model, cv_scores = select_best_model(X_train_scaled, y_train)

    save_model(best_model)
    plot_model_comparison(cv_scores)

    # 5. Final evaluation on held-out test set
    print("\n[5/5] Evaluating best model on test set...")
    evaluate(best_model, X_test_scaled, y_test)
    plot_feature_importance(best_model, feature_names)

    print(f"\n=== Done — best model: {best_name} ===")


if __name__ == "__main__":
    main()
