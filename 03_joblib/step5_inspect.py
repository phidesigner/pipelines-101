# Step 5 (optional): look inside the saved file
from pathlib import Path

import joblib

path = Path("models/churn_pipeline.joblib")
if not path.is_file():
    raise FileNotFoundError(f"{path} not found: run step2_save_pipeline.py first")

pipe = joblib.load(path)
print("fill median:", pipe.named_steps["fill"].statistics_)  # new: learned in step 2
print("scale mean:", pipe.named_steps["scale"].mean_)  # new
print("scale std:", pipe.named_steps["scale"].scale_)  # new
print("model weight:", pipe.named_steps["model"].coef_)  # new
