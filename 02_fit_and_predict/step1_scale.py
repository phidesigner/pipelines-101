# Step 1: two stateful steps by hand: fill, then scale
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler  # new

df = pd.read_csv("../data/customers.csv")  # all 12 customers are training rows now
X = df[["usage_hours"]]

imputer = SimpleImputer(strategy="median")
X_filled = imputer.fit_transform(X)  # fit, then transform, in one call
scaler = StandardScaler()  # new
X_scaled = scaler.fit_transform(X_filled)  # new: learns the mean and standard deviation

print("fill median:", imputer.statistics_)
print("scale mean:", scaler.mean_, "std:", scaler.scale_)  # new

new = pd.DataFrame({"usage_hours": [8.0]})
scaled = scaler.transform(imputer.transform(new))  # new: same steps, same order
print("8.0 h scaled:", scaled.ravel())
