# %%
from startup import np, pd, plt, sns, sm, smf
import dowhy.api
# %%
hw_data = sm.datasets.get_rdataset('Davis', 'carData')
df = hw_data.data.query('height > 100 and weight < 110')

# %%
sns.violinplot(df, x='sex', y='weight')
plt.show()

# %%
sns.scatterplot(df.sort_values('sex'), x='height', y='weight', hue='sex', alpha=0.5)
plt.show()

# %%
# OLS fit
smf.ols('weight ~ height + C(sex)', df).fit().summary()


# %%
# https://www.pywhy.org/dowhy/v0.9/example_notebooks/lalonde_pandas_api.html
df_do = df.drop(columns=['repwt', 'repht']).causal.do(
    x='height', outcome='weight', common_causes='sex'.split('+'),
    variable_types={'height': 'c', 'weight': 'c', 'sex': 'd',}
)

# %%
smf.ols('weight ~ height + propensity_score', df_do).fit().summary()

# %%
sns.scatterplot(df_do.sort_values('sex'), x='height', y='propensity_score', hue='sex', alpha=0.5)
plt.show()

# %%
sns.displot(df_do.sort_values('sex'), x='propensity_score', hue='sex', alpha=0.5, kind='ecdf')
plt.show()


# %%
sns.scatterplot(df_do.sort_values('sex'), x='height', y='weight', hue='sex', alpha=0.5)
plt.show()

# %%
df.groupby('sex').weight.mean()

# %%
df_do.groupby('sex').weight.mean()

# %%
smf.wls(
    'weight ~ height + C(sex)',
    df_do,
    weights=1/df_do.propensity_score
).fit().summary()
