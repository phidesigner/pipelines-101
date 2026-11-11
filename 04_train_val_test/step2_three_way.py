# Step 2: three splits. Choose on validation, look at the test set once
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(6)
X = rng.normal(size=(2000, 50))
y = rng.integers(0, 2, size=2000)

X_dev, X_test, y_dev, y_test = train_test_split(  # new: 15% test, locked away first
    X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(  # new: the team splits the rest
    X_dev, y_dev, test_size=0.15 / 0.85, random_state=42, stratify=y_dev)  # 70% train, 15% val

ideas = [np.random.default_rng(1000 + i).choice(50, 5, replace=False) for i in range(50)]
scores = []
for cols in ideas:
    model = LogisticRegression().fit(X_train[:, cols], y_train)
    scores.append(model.score(X_val[:, cols], y_val))  # new: choose on validation

best = ideas[int(np.argmax(scores))]  # new: the one model we ship
final = LogisticRegression().fit(X_train[:, best], y_train)
print("rows: train", len(y_train), "| val", len(y_val), "| test", len(y_test))
print(f"best of 50 on validation: {max(scores):.1%}  <- optimistic, we chose it")
print(f"test, used once: {final.score(X_test[:, best], y_test):.1%}  <- the honest number")  # new
