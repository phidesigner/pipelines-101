# 101 Fit and predict: three steps by hand, then one Pipeline

Run before slide "One Pipeline, two calls". About 7 min

Right-click `101/02_fit_and_predict` in Explorer > **Open in Integrated Terminal**, then:

```text
conda activate pipeline-101
```

One example all the way: a customer used the product **8.0 hours**. From here on, all 12 customers are the training rows, as on the slides; new customers play the unseen data.
Each step file is the previous one plus one small change, marked `# new`. Open it, point to the change, run it

## Step 1. Two stateful steps by hand: fill, then scale

Change: `StandardScaler` after the imputer. `fit_transform` is `fit`, then `transform`, in one call

```text
python step1_scale.py
```

Expect: `fill median: [9.5]`, `scale mean: [11.70833333] std: [8.02199667]`, `8.0 h scaled: [-0.46227061]`

Say: "Two steps, two memories: the median, then the mean and standard deviation. To prepare a new customer we must replay both, in the same order, with the same fitted objects. 8.0 h becomes -0.46: a bit below the average customer."

## Step 2. Add the model

Change: the target `y`, a `LogisticRegression`, `model.fit`, then fill, scale and predict for 8.0 h

```text
python step2_model.py
```

Expect: `all three steps: 8.0 h -> churn`

Say: "The model is the third step that learns: its weights. To predict we wrote `model.predict(scaler.transform(imputer.transform(new)))`. Three objects to keep in step, by hand."

## Step 3. Forget two steps

Change: one more prediction, with the raw hours straight into the model

```text
python step3_forget.py
```

Expect: `all three steps: 8.0 h -> churn`, then `forgot two steps: 8.0 h -> stay`

Say: "Same customer, same model, opposite answer, and no error. The model expects -0.46, we gave it 8.0. This is training-serving skew: production prepares data differently from training."

## Step 4. One Pipeline

Change: `Pipeline([("fill", ...), ("scale", ...), ("model", ...)])`, then one `pipe.fit(X, y)` and one `pipe.predict(new)`

```text
python step4_pipeline.py
```

Expect: the printed pipeline with its three named steps, then `pipeline: 8.0 h -> churn` and `pipeline: nan h -> stay`

Say: "`fit` learns every step once, in order. `predict` replays the same steps with what they learned, then predicts. Raw data in, prediction out: nothing to forget. The missing customer gets the stored median, 9.5 h, so they stay."

## Link to the slides

| Here | On the slides |
| --- | --- |
| Step 1, `statistics_`, `mean_`, `scale_` | Lane `.fit`: learns the median 9.5 h, the mean and the deviation |
| Step 3, forgot two steps | Before, `step1_by_hand.py` on the demo slide |
| Step 4, `pipe.fit` and `pipe.predict` | Lanes `.fit` and `.predict`; `train_model()` in `src/train.py` |

**Remember:** `fit` learns every step once; `predict` replays them in the same order
