# Step 3: learn before or after the split? Compare both
import pandas as pd  # new

df = pd.read_csv("../data/customers.csv")  # new: 12 customers
train = df.iloc[:8]  # new: first 8 rows to train
test = df.iloc[8:]  # new: last 4 rows to test, our "future" customers

median_all = df["usage_hours"].median()  # new: learned from all 12 rows
median_train = train["usage_hours"].median()  # new: learned from training rows only

print("rows: train", len(train), "| test", len(test))
print("learned from all 12 rows:", median_all, "h  <- the test rows leaked in")
print("learned from 8 training rows:", median_train, "h  <- correct")
