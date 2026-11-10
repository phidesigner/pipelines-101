# Step 4: the trap: save only the model, not the pipeline
from pathlib import Path

import joblib
import pandas as pd

path = Path("models/churn_pipeline.joblib")
if not path.is_file():
    raise FileNotFoundError(f"{path} not found: run step2_save_pipeline.py first")

pipe = joblib.load(path)
model_only = Path("models/model_only.joblib")  # new
joblib.dump(pipe.named_steps["model"], model_only)  # new: the last step alone, no fill, no scale

df = pd.read_csv("../data/customers.csv")
test = df.iloc[8:]
X_test, y_test = test[["usage_hours"]], test["churned"]

right = pipe.predict(X_test)
wrong = joblib.load(model_only).predict(X_test.to_numpy())  # new: raw hours
print("whole pipeline:", (right == y_test).sum(), "of 4 test rows right")
print("model alone:", (wrong == y_test).sum(), "of 4 test rows right")  # new
