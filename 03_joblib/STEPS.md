# 101 joblib: one fitted pipeline, one file

Run before slides "Inside the .joblib file" and "Demo: one pipeline, one file". About 8 min

Right-click `101/03_joblib` in Explorer > **Open in Integrated Terminal**, then:

```text
conda activate pipeline-101
```

One example all the way: a customer used the product **8.0 hours**; the 12 customers are the training rows.
Each step file is the previous one plus one small change, marked `# new`. Open it, point to the change, run it

## Step 1. joblib basics: save one fitted object

Change: `joblib.dump` writes the fitted imputer to `models/imputer.joblib`; `joblib.load` reads it back

```text
python step1_dump_load.py
```

Expect: `before saving, median: [9.5]`, `saved: models/imputer.joblib 767 bytes`, `after loading, median: [9.5]` (the size may differ slightly)

Say: "`dump` turns a Python object into a file; `load` turns the file back into the object. What the object learned, 9.5, is inside the file. When the script ends, memory is gone; the file stays."

## Step 2. Save the whole pipeline

Change: the pipeline from `02_fit_and_predict`, saved to `models/churn_pipeline.joblib`

```text
python step2_save_pipeline.py
```

Expect: `saved: models/churn_pipeline.joblib 1905 bytes` (the size may differ slightly)

Say: "Same call, bigger object: three fitted steps in one file. `models/` is in `.gitignore`: artifacts are generated, not committed."

## Step 3. A separate script loads and predicts

Change: no scikit-learn import and no `fit`; `joblib.load(path)` returns the fitted pipeline

```text
python step3_load_predict.py
```

Expect: `loaded steps: ['fill', 'scale', 'model']`, then `8.0 h -> churn`, `20.0 h -> stay`, `nan h -> stay`

Say: "This is serving: load once, predict many times, never train. The missing value gets the training median, read from the file."

Now delete the `models` folder in Explorer and run step 3 again. Expect: a traceback ending in `FileNotFoundError: models/churn_pipeline.joblib not found: run step2_save_pipeline.py first`

Run `python step2_save_pipeline.py` again to restore the file

Say: "Fail fast, with a message that says how to fix it."

## Step 4. The trap: save only the model

Change: the last step alone goes to `models/model_only.joblib`, then predicts on raw hours

```text
python step4_model_alone.py
```

Expect: `whole pipeline: 8.0 h -> churn`, then `model alone: 8.0 h -> stay`

Say: "The model alone has no fill and no scale. Production would need to rebuild both by hand, and we saw in folder 02 what happens when we forget. Save the whole pipeline, never the model alone."

## Step 5 (optional). Look inside the file

Change: read what each step learned, by name, from the loaded pipeline

```text
python step5_inspect.py
```

Expect: `fill median: [9.5]`, `scale mean: [11.70833333]`, `scale std: [8.02199667]`, `model weight: [[-1.47349409]]`

Say: "These numbers were learned once, in step 2, and travel inside the file. The weight is negative: more hours, less churn. Two more rules: load only files you trust, because loading can run code, and load with the scikit-learn version you trained with, pinned in `environment.yml`."

## Link to the demo slide

| Here | On `s911-demo-pipeline` and in the golden repository |
| --- | --- |
| Folder 02, step 3 | `step1_by_hand.py`: `forgot two steps: 8.0 h -> stay` |
| Step 2, `joblib.dump` | `step3_save.py`; `save_model()` in `src/utils.py` |
| Step 3, `joblib.load` and the clear error | `step4_load.py`; `load_model()` in `src/utils.py`, then `run_inference()` in `src/infer.py` |
| Step 5, the learned values | Slide "Inside the .joblib file", `step5_inspect.py` |

**Remember:** fit the whole pipeline once, save it as one file, and load that same file wherever you predict
