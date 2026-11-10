# Step 1: joblib basics: a fitted object -> a file -> the same object
from pathlib import Path

import joblib  # new
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv("../data/customers.csv")
train = df.iloc[:8]  # same split as before: fit on training rows only

imputer = SimpleImputer(strategy="median")
imputer.fit(train[["usage_hours"]])
print("before saving, median:", imputer.statistics_)

path = Path("models/imputer.joblib")  # new
path.parent.mkdir(exist_ok=True)  # new: create models/ if needed
joblib.dump(imputer, path)  # new: Python object -> file on disk
print("saved:", path, path.stat().st_size, "bytes")

loaded = joblib.load(path)  # new: file on disk -> Python object
print("after loading, median:", loaded.statistics_)  # new: the learned value survived
