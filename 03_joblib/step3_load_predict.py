# Step 3: a separate script loads the file and predicts the test rows, no training
from pathlib import Path

import joblib
import pandas as pd

LABEL = {1: "churn", 0: "stay"}

path = Path("models/churn_pipeline.joblib")
if not path.is_file():  # new: clear message instead of a traceback deep in joblib
    raise FileNotFoundError(f"{path} not found: run step2_save_pipeline.py first")

pipe = joblib.load(path)  # new: no fit anywhere in this file
print("loaded steps:", list(pipe.named_steps))

df = pd.read_csv("../data/customers.csv")
test = df.iloc[8:]  # new: only the rows the pipeline never saw
X_test, y_test = test[["usage_hours"]], test["churned"]

for hours, p, actual in zip(X_test["usage_hours"], pipe.predict(X_test), y_test):
    print(hours, "h ->", LABEL[p], "| actual:", LABEL[actual])
print("test accuracy:", pipe.score(X_test, y_test))
