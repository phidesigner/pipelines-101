# Step 3: predict, but forget the two transforms
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/customers.csv")
X = df[["usage_hours"]]
y = df["churned"]

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
model = LogisticRegression()

X_filled = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_filled)
model.fit(X_scaled, y)

new = pd.DataFrame({"usage_hours": [8.0]})
pred = model.predict(scaler.transform(imputer.transform(new)))
print("all three steps: 8.0 h ->", "churn" if pred[0] == 1 else "stay")

wrong = model.predict(new.to_numpy())  # new: raw hours straight into the model
print("forgot two steps: 8.0 h ->", "churn" if wrong[0] == 1 else "stay")  # new
