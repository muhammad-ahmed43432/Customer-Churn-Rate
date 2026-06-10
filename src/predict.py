import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "Telco_Customer_Churn.csv")

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)

# take valid row from dataset (SAFE METHOD)
sample = df.drop(columns=["Churn Value", "Churn Label", "Churn Score", "Churn Reason", "CLTV"]).iloc[[0]]

# modify values for testing
sample["Monthly Charges"] = 85.5
sample["Tenure Months"] = 5
sample["Contract"] = "Month-to-month"

# clean types
for col in sample.columns:
    sample[col] = sample[col].astype(str)

sample = sample.fillna(0)

prediction = model.predict(sample)
probability = model.predict_proba(sample)[:, 1]

print("Churn Prediction:", prediction[0])
print("Churn Probability:", round(probability[0], 4))