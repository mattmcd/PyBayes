# %%
import ydf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
# Doc example https://ydf.readthedocs.io/en/latest/#usage-example with minor edits to run in PyCharm cell mode

# %% Load dataset with Pandas
ds_path = "https://raw.githubusercontent.com/google/yggdrasil-decision-forests/main/yggdrasil_decision_forests/test_data/dataset/"
train_ds = pd.read_csv(ds_path + "adult_train.csv")
test_ds = pd.read_csv(ds_path + "adult_test.csv")

# %% Train a Gradient Boosted Trees model
model = ydf.GradientBoostedTreesLearner(label="income").train(train_ds)

# %% Look at a model (input features, training logs, structure, etc.)
print(model.describe())

# %% Evaluate a model (e.g. roc, accuracy, confusion matrix, confidence intervals)
evaluation = model.evaluate(test_ds)
print(evaluation.confusion_matrix)
# %% Generate predictions
model.predict(test_ds)

# %% Analyse a model (e.g. partial dependence plot, variable importance)
analysis = model.analyze(test_ds)
# %% Benchmark the inference speed of a model
benchmark = model.benchmark(test_ds)

# %% Save the model
model.save("/tmp/ydf_model_20250816")