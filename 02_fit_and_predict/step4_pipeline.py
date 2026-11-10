# Step 4: the same three steps, bundled in one Pipeline
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline  # new
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/customers.csv")
X = df[["usage_hours"]]
y = df["churned"]

pipe = Pipeline([  # new: one object, steps run in this order
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("model", LogisticRegression()),
])
pipe.fit(X, y)  # new: fit learns every step, once
print(pipe)

new = pd.DataFrame({"usage_hours": [8.0, None]})  # new: one customer has no hours
for hours, pred in zip(new["usage_hours"], pipe.predict(new)):  # new: predict replays every step
    print("pipeline:", hours, "h ->", "churn" if pred == 1 else "stay")
