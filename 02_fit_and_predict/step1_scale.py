# Step 1: two stateful steps by hand: fill, then scale. Fit on training rows only
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler  # new

df = pd.read_csv("../data/customers.csv")
train = df.iloc[:8]  # same split as 01_stateless_stateful
test = df.iloc[8:]
X_train = train[["usage_hours"]]
X_test = test[["usage_hours"]]

imputer = SimpleImputer(strategy="median")
X_train_filled = imputer.fit_transform(X_train)  # fit, then transform, in one call
scaler = StandardScaler()  # new
X_train_scaled = scaler.fit_transform(X_train_filled)  # new: learns the mean and standard deviation

print("fill median:", imputer.statistics_)
print("scale mean:", scaler.mean_, "std:", scaler.scale_)  # new

X_test_scaled = scaler.transform(imputer.transform(X_test))  # new: transform only, same order
print("test hours:", X_test["usage_hours"].tolist())
print("test scaled:", X_test_scaled.ravel().round(2))
