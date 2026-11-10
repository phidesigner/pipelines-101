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

new = pd.DataFrame({"usage_hours": [8.0]})
right = pipe.predict(new)
wrong = joblib.load(model_only).predict(new.to_numpy())  # new: raw hours, nothing to prepare them
print("whole pipeline: 8.0 h ->", "churn" if right[0] == 1 else "stay")
print("model alone: 8.0 h ->", "churn" if wrong[0] == 1 else "stay")  # new
