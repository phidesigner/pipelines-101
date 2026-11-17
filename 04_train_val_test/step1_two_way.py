# Step 1: two splits, train and test. We pick the best of 50 models by its test score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(6)
X = rng.normal(size=(2000, 50))  # 2000 customers, 50 measurements of pure noise
y = rng.integers(0, 2, size=2000)  # churn is a coin flip: nothing to learn, truth is 50%

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)  # 80% train, 20% test

feature_sets = [np.random.default_rng(1000 + i).choice(50, 5, replace=False) for i in range(50)]
scores = []
for cols in feature_sets:  # train 50 models, each using a different set of 5 input features
    model = LogisticRegression().fit(X_train[:, cols], y_train)
    scores.append(model.score(X_test[:, cols], y_test))  # we look at the test set every time

print("rows: train", len(y_train), "| test", len(y_test))
print(f"best of 50 on the test set: {max(scores):.1%}  <- what we report")
print("truth: 50.0%, churn is a coin flip")
