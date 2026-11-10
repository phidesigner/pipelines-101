# 101: from one learned number to one saved pipeline

Tiny Python files, run before the pipeline slides of Sessions 9-11. They build, from zero, the demo on slide `s911-demo-pipeline`

Every folder uses the same example as the best-practices 101: a customer used the product **8.0 hours**. This time a model learns churn from `data/customers.csv`, 12 synthetic customers; customer 5 has no hours recorded
Each step adds one small change to the previous step, marked `# new`

| Run before | Folder | The example grows from |
| --- | --- | --- |
| Stateless or stateful: split first | `01_stateless_stateful` | A fixed rule > a learned median > learned before or after the split > `fit` and `transform` |
| One Pipeline, two calls | `02_fit_and_predict` | Fill and scale by hand > add the model > forget two steps > one `Pipeline` |
| Inside the .joblib file, then the demo | `03_joblib` | Save one object > save the pipeline > load it in a separate script > the model-alone trap > look inside |

Each folder has a `STEPS.md` (what to run, what to expect, what to say)

## Setup, once

Right-click `toy_demo_pipeline` in VS Code Explorer > **Open in Integrated Terminal**, then:

```text
conda env create -f environment.yml
conda activate pipeline-101
```

If you already created `pipeline-101` for `Class exercises/pipeline_101`, reuse it: the packages are the same
