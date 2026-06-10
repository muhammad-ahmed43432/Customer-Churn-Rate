import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(filepath):

    df = pd.read_csv(filepath)

    # =========================
    # CLEAN TARGET (IMPORTANT FIX)
    # =========================
    df["Churn Value"] = pd.to_numeric(df["Churn Value"], errors="coerce")
    df = df.dropna(subset=["Churn Value"])
    df["Churn Value"] = df["Churn Value"].astype(int)

    # =========================
    # CLEAN FEATURES
    # =========================
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)

    # Fix numeric column
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    df = df.fillna(0)

    # =========================
    # DROP LEAKAGE
    # =========================
    leakage_cols = ["Churn Label", "Churn Score", "Churn Reason", "CLTV"]
    df = df.drop(columns=leakage_cols)

    # =========================
    # SPLIT
    # =========================
    X = df.drop("Churn Value", axis=1)
    y = df["Churn Value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # COLUMN TYPES
    # =========================
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # =========================
    # PREPROCESSOR
    # =========================
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessor