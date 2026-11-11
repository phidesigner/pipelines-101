# 101: from one learned number to one saved pipeline

Tiny Python files, run before the pipeline slides of Sessions 9-11. They build, from zero, the demo on slide `s911-demo-pipeline`

A model learns churn from usage hours in `data/customers.csv`, 12 synthetic customers; customer 5 has no hours recorded. From `01_stateless_stateful` step 3 onwards the split never changes: the first 8 customers train, the last 4 test. Every `fit` sees training rows only
Each step adds one small change to the previous step, marked `# new`

| Run before | Folder | The example grows from |
| --- | --- | --- |
| Stateless or stateful: split first | `01_stateless_stateful` | A fixed rule > a learned median > learned before or after the split > `fit` and `transform` |
| One Pipeline, two calls | `02_fit_and_predict` | Fill and scale by hand > add the model > forget two steps (4 of 4 right becomes 2 of 4) > one `Pipeline` |
| Inside the .joblib file, then the demo | `03_joblib` | Save one object > save the pipeline > load it in a separate script and predict the test rows > the model-alone trap > look inside |
| Train, validation and test | `04_train_val_test` | Choose on the test set > choose on validation, test once > k-fold > k-fold with a step fitted outside the folds |

Each folder has a `STEPS.md` (what to run, what to expect, what to say)

## Setup, once

Right-click `toy_demo_pipeline` in VS Code Explorer > **Open in Integrated Terminal**, then:

```text
conda env create -f environment.yml
conda activate pipeline-101
```

If you already created `pipeline-101` for `Class exercises/pipeline_101`, reuse it: the packages are the same
