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

# SelectKBest is a feature selector: it scores each input column for its
# relationship with churn (y) and keeps the 5 highest-scoring columns
# fit learns which columns to keep; transform keeps only those columns
# WRONG: selecting features before the folds uses their validation labels too
feature_selector = SelectKBest(k=5).fit(X_dev, y_dev)  # new: sees all development rows
wrong = cross_val_score(LogisticRegression(), feature_selector.transform(X_dev), y_dev, cv=5).mean()

# RIGHT: the Pipeline fits feature selection on each fold's training rows only,
# then applies that selection to the validation rows before predicting churn
pipe = make_pipeline(SelectKBest(k=5), LogisticRegression())  # new: selection inside the folds
right = cross_val_score(pipe, X_dev, y_dev, cv=5).mean()
test = pipe.fit(X_dev, y_dev).score(X_test, y_test)

print(f"5-fold, feature selector fitted before the folds: {wrong:.1%}  <- leak")
print(f"5-fold, feature selector inside a Pipeline: {right:.1%}")
print(f"test, used once: {test:.1%}")
