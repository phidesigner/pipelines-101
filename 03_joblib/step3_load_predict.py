# Step 3: a separate script loads the file and predicts, no training
from pathlib import Path

import joblib
import pandas as pd

path = Path("models/churn_pipeline.joblib")
if not path.is_file():  # new: clear message instead of a traceback deep in joblib
    raise FileNotFoundError(f"{path} not found: run step2_save_pipeline.py first")

pipe = joblib.load(path)  # new: no fit anywhere in this file
print("loaded steps:", list(pipe.named_steps))

new = pd.DataFrame({"usage_hours": [8.0, 20.0, None]})  # new: one value is missing
for hours, pred in zip(new["usage_hours"], pipe.predict(new)):
    print(hours, "h ->", "churn" if pred == 1 else "stay")
