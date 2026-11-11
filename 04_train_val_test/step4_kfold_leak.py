# Step 4: k-fold done the wrong way: features chosen on all rows before the folds
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(6)
X = rng.normal(size=(2000, 50))
y = rng.integers(0, 2, size=2000)

X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)

picker = SelectKBest(k=5).fit(X_dev, y_dev)  # new: learns from every dev row, every fold
wrong = cross_val_score(LogisticRegression(), picker.transform(X_dev), y_dev, cv=5).mean()

pipe = make_pipeline(SelectKBest(k=5), LogisticRegression())  # new: picker inside the folds
right = cross_val_score(pipe, X_dev, y_dev, cv=5).mean()
test = pipe.fit(X_dev, y_dev).score(X_test, y_test)

print(f"5-fold, picker fitted before the folds: {wrong:.1%}  <- leak")
print(f"5-fold, picker inside a Pipeline: {right:.1%}")
print(f"test, used once: {test:.1%}")
