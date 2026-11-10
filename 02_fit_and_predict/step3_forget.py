# Step 3: predict, but forget the two transforms
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

LABEL = {1: "churn", 0: "stay"}

df = pd.read_csv("../data/customers.csv")
train = df.iloc[:8]
test = df.iloc[8:]
X_train, y_train = train[["usage_hours"]], train["churned"]
X_test, y_test = test[["usage_hours"]], test["churned"]

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
model = LogisticRegression()

X_train_filled = imputer.fit_transform(X_train)
X_train_scaled = scaler.fit_transform(X_train_filled)
model.fit(X_train_scaled, y_train)

right = model.predict(scaler.transform(imputer.transform(X_test)))
wrong = model.predict(X_test.to_numpy())  # new: raw hours straight into the model
print("hours | all three steps | forgot two steps | actual")  # new
for hours, r, w, actual in zip(X_test["usage_hours"], right, wrong, y_test):  # new
    print(hours, "|", LABEL[r], "|", LABEL[w], "|", LABEL[actual])
print("correct:", (right == y_test).sum(), "of 4 |", (wrong == y_test).sum(), "of 4")  # new
