# Step 2: save the whole fitted pipeline to one file
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline  # new
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/customers.csv")
X = df[["usage_hours"]]
y = df["churned"]

pipe = Pipeline([  # new: the pipeline from 02_fit_and_predict
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("model", LogisticRegression()),
])
pipe.fit(X, y)

path = Path("models/churn_pipeline.joblib")  # new
path.parent.mkdir(exist_ok=True)
joblib.dump(pipe, path)  # new: three fitted steps, one file
print("saved:", path, path.stat().st_size, "bytes")
