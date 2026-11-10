# Step 2: fit the pipeline on training rows, save it to one file
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline  # new
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/customers.csv")
train = df.iloc[:8]  # new: training rows only; the test rows stay unseen
X_train, y_train = train[["usage_hours"]], train["churned"]

pipe = Pipeline([  # new: the pipeline from 02_fit_and_predict
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("model", LogisticRegression()),
])
pipe.fit(X_train, y_train)

path = Path("models/churn_pipeline.joblib")  # new
path.parent.mkdir(exist_ok=True)
joblib.dump(pipe, path)  # new: three fitted steps, one file
print("saved:", path, path.stat().st_size, "bytes")
