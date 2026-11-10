# Step 2: add the model, the third step that learns
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression  # new
from sklearn.preprocessing import StandardScaler

LABEL = {1: "churn", 0: "stay"}  # new

df = pd.read_csv("../data/customers.csv")
train = df.iloc[:8]
test = df.iloc[8:]
X_train, y_train = train[["usage_hours"]], train["churned"]  # new: y, 1 = churned, 0 = stayed
X_test, y_test = test[["usage_hours"]], test["churned"]  # new

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
model = LogisticRegression()  # new

X_train_filled = imputer.fit_transform(X_train)
X_train_scaled = scaler.fit_transform(X_train_filled)
model.fit(X_train_scaled, y_train)  # new: learns the weights, training rows only

pred = model.predict(scaler.transform(imputer.transform(X_test)))  # new: fill, scale, predict
for hours, p, actual in zip(X_test["usage_hours"], pred, y_test):  # new
    print(hours, "h ->", LABEL[p], "| actual:", LABEL[actual])
