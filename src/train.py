import os
import joblib

from preprocess import preprocess_data

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Telco_customer_churn.csv")
# =========================
# LOAD DATA
# =========================
X_train, X_test, y_train, y_test, preprocessor = preprocess_data(DATA_PATH)

# =========================
# MODEL (FIXED)
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight={0: 1, 1: 2}   # FIXED (NO 'balanced')
)

# =========================
# PIPELINE
# =========================
clf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# =========================
# TRAIN
# =========================
clf.fit(X_train, y_train)

# =========================
# PREDICT
# =========================
y_pred = clf.predict(X_test)

# =========================
# EVALUATION
# =========================
print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================
model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "model.pkl")
joblib.dump(clf, model_path)

print("\nModel saved successfully at:", model_path)