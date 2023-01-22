# %% [markdown]
# Logistic Regression in SQL
#
# Author: Matt McDonnell  Date: 2023-01-22
#
# This script demonstrates the process of training a simple Scikit Learn pipeline
# consisting of Standard Scalar -> Logistic Regression on the Iris dataset reduced to a
# binary classification problem.
#
# The trained model parameters are then extracted and used to create an equivalent set of
# SQL statements for the model predictions.
#
# NB: the script is formatted for use with PyCharm Scientific Mode cell by cell evaluation, or
# using jupytext to evaluate as a Jupyter notebook

# %%
# Initial imports needed for model training
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
# %%
# Imports needed for SQL version
from sqlalchemy import create_engine, MetaData, select, func, Table, union_all, literal
# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# %%
# Load Iris data and turn into binary classification problem by ignoring one class
data = load_iris(as_frame=True)
ind = data.target != len(data.target_names) - 1
feature_data = data.data.loc[ind, :]
label_data = data.target.loc[ind]

# %%
# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    feature_data, label_data, test_size=0.2, random_state=42
)

# %%
# Define the model and fit
pipeline_steps = [StandardScaler(), LogisticRegression(solver='lbfgs')]
model = make_pipeline(*pipeline_steps)
model.fit(X_train, y_train)

# %%
# Examine model performance on training set
y_train_pred = model.predict(X_train)
print(confusion_matrix(y_train, y_train_pred))
print(f1_score(y_train, y_train_pred))
print(roc_auc_score(y_train, y_train_pred))

# %% 
# Examine model performance on test set
y_test_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(f1_score(y_test, y_test_pred))
print(roc_auc_score(y_test, y_test_pred))

# %% [markdown]
# We see that for this simple dataset we are able to achieve perfect class separation.
#
# Now look at converting this model to SQL

# %%
# Extract the model parameters into a Pandas DataFrame
df_m = pd.DataFrame.from_dict(
    {
        'ss_mean': model[0].mean_,
        'ss_scale': model[0].scale_,
        'lr_coef': model[1].coef_.ravel(),
    }
)
df_m.loc[:, 'lr_intercept'] = model[1].intercept_[0]
df_m.index = data.feature_names
df_m = df_m.reset_index().rename(columns={'index': 'feature'})
print(df_m)

# %%
# Create the connection to the database.
# Here we use an in-memory SQLite database.
engine = create_engine('sqlite:///:memory:')
metadata = MetaData()

# %%
# Insert the model parameters into the database
rows_inserted = df_m.to_sql('model', engine, if_exists='replace', index=False)

# %%
# Reflect the database model parameters as an SQLAlchemy Table and check contents
tb_m = Table('model', metadata, autoload_with=engine)
# Check that the model parameters in the database match those in the DataFrame
print(pd.read_sql(select(tb_m), engine))

# %%
# Add the feature data to the database
f_rows = X_train.to_sql('features', engine, index=True, if_exists='replace')
print(f_rows)

# %%
# Reflect the feature data as an SQLAlchemy Table and check contents
tb_f = Table('features', metadata, autoload_with=engine)
print(pd.read_sql(select(tb_f.c).limit(5), engine))

# %%
# Wide to tall conversion for features - equivalent to melt in pandas or unpivot in supported databases
tb_f_tall = (
    union_all(
        *[
            select([
                tb_f.c.index,
                literal(f).label('feature'),
                tb_f.c[f].label('feature_value')
            ])
            for f in data.feature_names
        ]
    )
).cte('features_tall')

# Check tall feature table
print(pd.read_sql(select(tb_f_tall).order_by('index').limit(10), engine))


# %%
# Model prediction definition using the SQLAlchemy columns

# Model parameters
mu = tb_m.c.ss_mean
s = tb_m.c.ss_scale
beta = tb_m.c.lr_coef
c = select(tb_m.c.lr_intercept).limit(1).scalar_subquery()

# Feature data
x = tb_f_tall.c.feature_value


def dot_product(a, b):
    return func.sum(a * b)


# Model prediction
z = dot_product((x - mu)/s, beta) + c
proba = 1/(1+func.exp(-z))

tb_p = select(
    [
        tb_f_tall.c.index,
        z.label('z'),
        proba.label('proba')
    ]
).select_from(
    tb_f_tall.join(tb_m, tb_f_tall.c.feature == tb_m.c.feature)
).group_by(
    tb_f_tall.c.index
).cte('prediction')

# %%
print('Model predictions from sklearn pipeline:')
print(model.predict_proba(X_train.sort_index().head()))

# %%
print('Model predictions from database:')
print(pd.read_sql(select(tb_p).order_by(tb_p.c.index).limit(5), engine))
