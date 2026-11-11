# Step 3: k-fold cross-validation instead of one validation split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

rng = np.random.default_rng(6)
X = rng.normal(size=(2000, 50))
y = rng.integers(0, 2, size=2000)

X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)  # the test set stays locked away

ideas = [np.random.default_rng(1000 + i).choice(50, 5, replace=False) for i in range(50)]
scores = []
for cols in ideas:
    cv = cross_val_score(LogisticRegression(), X_dev[:, cols], y_dev, cv=5)  # new: 5 folds
    scores.append(cv.mean())  # new: every dev row is validated once

best = ideas[int(np.argmax(scores))]
final = LogisticRegression().fit(X_dev[:, best], y_dev)  # new: refit on all dev rows
print("rows: dev", len(y_dev), "in 5 folds | test", len(y_test))
print(f"best of 50, 5-fold mean: {max(scores):.1%}  <- less noisy, still optimistic")
print(f"test, used once: {final.score(X_test[:, best], y_test):.1%}  <- the honest number")
