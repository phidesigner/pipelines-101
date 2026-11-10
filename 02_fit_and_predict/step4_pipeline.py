# Step 4: the same three steps, bundled in one Pipeline
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline  # new
from sklearn.preprocessing import StandardScaler

LABEL = {1: "churn", 0: "stay"}

df = pd.read_csv("../data/customers.csv")
train = df.iloc[:8]
test = df.iloc[8:]
X_train, y_train = train[["usage_hours"]], train["churned"]
X_test, y_test = test[["usage_hours"]], test["churned"]

pipe = Pipeline([  # new: one object, steps run in this order
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("model", LogisticRegression()),
])
pipe.fit(X_train, y_train)  # new: fit learns every step once, training rows only
print(pipe)

pred = pipe.predict(X_test)  # new: predict replays every step on the test rows
for hours, p, actual in zip(X_test["usage_hours"], pred, y_test):
    print(hours, "h ->", LABEL[p], "| actual:", LABEL[actual])
print("test accuracy:", pipe.score(X_test, y_test))  # new: share of test rows right
