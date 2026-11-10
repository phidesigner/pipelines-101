# Step 2: add the model, the third step that learns
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression  # new
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/customers.csv")
X = df[["usage_hours"]]
y = df["churned"]  # new: 1 = churned, 0 = stayed

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
model = LogisticRegression()  # new

X_filled = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_filled)
model.fit(X_scaled, y)  # new: learns the weights

new = pd.DataFrame({"usage_hours": [8.0]})
pred = model.predict(scaler.transform(imputer.transform(new)))  # new: fill, scale, predict
print("all three steps: 8.0 h ->", "churn" if pred[0] == 1 else "stay")
