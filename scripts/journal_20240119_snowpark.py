from startup import np, pd, plt, sns
from sklearn.datasets import load_diabetes
from snowflake.snowpark import Session, functions as F, types as T, Window

# %%
# Create local Snowpark session

session = Session.builder.config('local_testing', True).create()


# %%
# Test dataset of diabetes data
df_X, df_y = load_diabetes(scaled=False, as_frame=True, return_X_y=True)

# %%
# Add a patient ID column and reorder so it is first column
features = df_X.columns.to_list()
df_X = df_X.assign(patient_id=lambda x: np.arange(len(x))).loc[:, ['patient_id'] + features]

# %%
# Loading from a dataframe directly results in a lot of warnings about using column keys as indexes in iloc
tb_f = session.create_dataframe(
    data=df_X.to_dict('records'))

# %%
tb_f.show()

# %%
tb_f.group_by('sex').agg(F.count('patient_id').as_('patients')).to_pandas()

