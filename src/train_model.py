# src/train_model.py

from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline


def main():
    # ---------- PATHS ----------
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "loan_data.csv"
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # 👉 CHANGE THIS IF YOUR TARGET COLUMN NAME IS DIFFERENT
    target_col = "Loan_Status"

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Convert Y/N to 1/0
    df[target_col] = df[target_col].map({"Y": 1, "N": 0})

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ---------- SPLIT ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------- PREPROCESS ----------
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    print("Categorical Columns:", cat_cols)
    print("Numerical Columns:", num_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="mean"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ]
    )

    # ---------- MODELS ----------
    models = {
        "LogisticRegression": LogisticRegression(max_iter=300),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss"
        )
    }

    best_model = None
    best_score = -1
    best_name = ""

    # ---------- TRAIN + EVALUATE ----------
    for name, model in models.items():
        print(f"\nTraining: {name}")

        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        print(f"{name}: Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

        if f1 > best_score:
            best_score = f1
            best_model = pipe
            best_name = name

    # ---------- SAVE BEST MODEL ----------
    model_path = model_dir / "best_model.pkl"
    joblib.dump(best_model, model_path)

    print("\n==============================")
    print(f"BEST MODEL: {best_name}  (F1 = {best_score:.3f})")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
