# 101 Stateless or stateful: what learns from data, and when

Run before slide "Stateless or stateful: split first". About 7 min

Right-click `101/01_stateless_stateful` in Explorer > **Open in Integrated Terminal**, then:

```text
conda activate pipeline-101
```

One example all the way: a customer used the product **8.0 hours**; customer 5 has no hours recorded.
Each step file is the previous one plus one small change, marked `# new`. Open it, point to the change, run it

## Step 1. A stateless rule

```text
python step1_stateless.py
```

Expect: `480 min -> 8.0 h`, `900 min -> 15.0 h`, `480 min -> 8.0 h`

Say: "We wrote the rule: divide by 60. It looks at no other row and stores nothing. 480 always gives 8.0, today, tomorrow, in training or in production. Stateless steps are safe before the split."

## Step 2. A stateful step: the value comes from the data

Change: a list of 8 customers with one gap; the median of the known hours fills it

```text
python step2_stateful.py
```

Expect: `learned median: 12.0`, then the list with `12.0` in place of `None`

Delete `22.5, ` and `30.0, ` from the list, save, run again. Expect: `learned median: 8.0`. Undo with `Ctrl+Z` and save

Say: "Nobody typed 12.0. The code learned it from these rows. Other rows, other number. That is what stateful means: the step has a memory, and the memory depends on the data it saw."

## Step 3. Learn before or after the split?

Open `../data/customers.csv`: 12 customers, customer 5 has an empty `usage_hours`

Change: read the file; the first 8 rows train, the last 4 test; learn the median both ways

```text
python step3_split_first.py
```

Expect: `rows: train 8 | test 4`, then `learned from all 12 rows: 9.5 h`, then `learned from 8 training rows: 12.0 h`

Say: "The test rows stand for future customers. If we learn the median before the split, they change it from 12.0 to 9.5. The model is then tested on data it already peeked at, and the score looks better than it will be in production. That is leakage. Split first, then learn."

## Step 4. The same idea in scikit-learn: `fit` and `transform`

Change: `SimpleImputer` replaces the hand-made median; `fit` on training rows, `transform` on new rows

```text
python step4_fit_transform.py
```

Expect: `learned median: [12.]`, then `filled: [ 8. 12.]`

Say: "`fit` learns and stores the median. `transform` applies it and learns nothing. 8.0 stays 8.0; the gap gets 12.0, the value learned on training rows only. Every stateful step in scikit-learn has these two calls."

Put `# ` in front of `imputer.fit(...)`, save, run again. Expect: a traceback ending in `AttributeError: 'SimpleImputer' object has no attribute 'statistics_'`. Remove the `# ` and save

Say: "Attributes that end in `_` exist only after `fit`. No fit, no memory."

## Link to the slides

| Here | On the slides |
| --- | --- |
| Step 1, divide by 60 | Stateless: rename, standardize formats, `clean_dataframe()` |
| Step 2, a learned median | Stateful: imputation learns fill values |
| Step 3, 9.5 versus 12.0 | Split contamination, one of the three leakage scenarios |
| Step 4, `fit` then `transform` | Fit on training rows only, replay at inference |

**Remember:** if a step learns from data, split first and fit it on training rows only
