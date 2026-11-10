# Step 4: the same idea in scikit-learn: fit learns, transform applies
import pandas as pd
from sklearn.impute import SimpleImputer  # new

df = pd.read_csv("../data/customers.csv")
train = df.iloc[:8]
test = df.iloc[8:]

imputer = SimpleImputer(strategy="median")  # new: nothing learned yet
imputer.fit(train[["usage_hours"]])  # new: learn, on training rows only
print("learned median:", imputer.statistics_)  # new: attributes ending in _ appear after fit

new = pd.DataFrame({"usage_hours": [8.0, None]})  # new: one customer has no hours
print("filled:", imputer.transform(new).ravel())  # new: apply, no learning
