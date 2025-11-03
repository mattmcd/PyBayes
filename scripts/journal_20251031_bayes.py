# Bayes rule approach to estimating class probability

# %%
from startup import np, pd, plt, sns, sm, smf, os, Path
from collections import defaultdict
from scipy.stats import beta
import itertools
import ydf
from sklearn.manifold import TSNE
from journal_20251031_bayes_functions import BayesRule, marginalize
from journal_20251031_ydf_functions import ModelBasedAnalysis
# %%
# Census income dataset: https://archive.ics.uci.edu/dataset/2/adult
data_dir = Path.home() / 'Work' / 'Data' / 'uci_adult'
df_train = pd.read_csv(data_dir / "adult_train.csv")
df_test = pd.read_csv(data_dir / "adult_test.csv")

# %%
plot_dir = Path.home() / 'Work' / 'Projects'/ 'priv-obsidian-vault' / 'Work' / 'img'

# %%
categorical_vars = df_train.dtypes.pipe(lambda x: x[x == 'object']).index.tolist()
target = 'income'

# %%
# Analyst-style segmentation
df_segment = df_train.assign(
    high_income=lambda x: x[target] == '>50K'
).groupby(['occupation']).high_income.mean()

g = sns.barplot(df_segment.sort_values(ascending=False), orient='h')
plt.title('Probability( High Income | Occupation )')
plt.tight_layout()
plt.show()
g.figure.savefig(plot_dir / 'high_income_given_occupation_20251103.png')
# %%
# Use a RF model to get an idea of important variables

def eda_variables(df_train, df_test, target, top_n=None):
    model = ydf.RandomForestLearner(label=target).train(df_train)
    evaluation = model.evaluate(df_test)

    print(f'ROC AUC: {evaluation.characteristics[0].roc_auc:0.2f}')
    print(pd.DataFrame(model.variable_importances()['INV_MEAN_MIN_DEPTH']).head(top_n))
    df_p = df_test.sample(frac=0.2)
    manifold = TSNE(n_components=2).fit_transform(model.distance(df_p, df_p))
    sns.scatterplot(x=manifold[:, 0], y=manifold[:, 1], hue=df_p[target], alpha=0.2)
    plt.title(
        f'ROC AUC: {evaluation.characteristics[0].roc_auc:0.2f}'
    )

# %%
# Class version
# (2025-11-03: Code moved to journal_20251103_ydf_functions.py)

# %%
print('Using all features:')
all_features_analysis = ModelBasedAnalysis(df_train, df_test, target)
print(all_features_analysis)
print(all_features_analysis.variable_importance().head(10))
fig = all_features_analysis.plot()
plt.show()
fig.savefig(plot_dir / 'all_features_20251103.png')

# %%
print('Using only categorical features features:')
categorical_features_analysis = ModelBasedAnalysis(df_train.loc[:, categorical_vars], df_test, target)
print(categorical_features_analysis)
print(categorical_features_analysis.variable_importance().head(10))
fig = categorical_features_analysis.plot()
plt.show()
fig.savefig(plot_dir / 'categorical_features_20251103.png')

# %%
df_a = df_train.loc[:, categorical_vars].assign(
    label=lambda x: x[target] == '>50K'
).groupby(
    [c for c in categorical_vars if c != target]
).agg(
    trials=pd.NamedAgg(column='label', aggfunc='count'),
    successes=pd.NamedAgg(column='label', aggfunc='sum')
)

# %%
pop_agg_lookup = defaultdict(lambda: {'trials': 0, 'successes': 0})
for k, v in df_a.to_dict(orient='index').items():
    pop_agg_lookup[k]['trials'] += v['trials']
    pop_agg_lookup[k]['successes'] += v['successes']

# %%
class ProbabilityEstimator:
    def __init__(self, pop_lookup=None):
        self.pop_lookup = pop_lookup

    def __call__(self, x):
        seen = self.pop_lookup[x]
        return beta(
            seen['successes'] + 1,
            seen['trials'] - seen['successes'] + 1
        ).ppf([0.025, 0.975])

# %%
pe = ProbabilityEstimator(pop_agg_lookup)

# %%
print('Mean probability from training data:')
print(df_a.sort_values('successes').tail(1).assign(prob=lambda x: x.successes / x.trials))
print('95% confidence interval:')
print(pe(df_a.sort_values('successes').index[-1]))

# %%
# Bayes rule P(param|data) = P(data|param)P(param)/P(data)
# e.g. P(male| >50K) = P(>50K|male)P(male)/P(>50K)

p_male = df_a.xs('Male', level='sex').trials.sum()/df_a.trials.sum()
p_inc = df_a.successes.sum()/df_a.trials.sum()
p_inc_given_male = df_a.xs('Male', level='sex').successes.sum()/df_a.xs('Male', level='sex').trials.sum()
p_male_given_inc = p_inc_given_male*p_male/p_inc

# %%
print(f'P(male) = {p_male:.3f}')
print(f'P(>50K) = {p_inc:.3f}')
print(f'P(>50K|male) = {p_inc_given_male:.3f}')
print(f'P(male| >50K) = {p_male_given_inc:.3f}')

# %%
# Generalised Bayes rule functions
# (2025-11-03: Code moved to journal_20251103_bayes_functions.py)

# %%
pop_slice = {'occupation': ['Exec-managerial'] }
print(BayesRule(df_a, pop_slice, 'Income > 50K'))

# %%
pop_slice = {'sex': ['Male'], 'occupation': ['Exec-managerial'] }
print(BayesRule(df_a, pop_slice, 'Income > 50K'))

# %%
pop_slice = {'sex': ['Female'], 'relationship': ['Unmarried'] }
print(BayesRule(df_a, pop_slice, 'Income > 50K'))

# %%
pop_slice = {'sex': ['Male']}
print(BayesRule(df_a, pop_slice, 'Income > 50K'))

# %%
pop_slice = {'sex': ['Female']}
print(BayesRule(df_a, pop_slice, 'Income > 50K'))


# %%
# Next steps: can we use this to calculate marginal distributions?  e.g. marginal(['sex', 'occupation'])
# Easy way is to find all the values in the specified levels and loop, may be enough for a first pass.
# Vectorised would be nicer though :)

# (2025-11-03: Code moved to journal_20251103_bayes_functions.py)

# %%
groups = ['occupation']
df_m = marginalize(df_a, groups)
df_m.set_index(groups, inplace=True)

g = sns.barplot(df_m['prob'].sort_values(ascending=False), orient='h')
plt.title('Probability( Occupation | High Income )')
plt.tight_layout()
plt.show()
g.figure.savefig(plot_dir / 'occupation_given_high_income_20251103.png')

# %%
groups = ['sex', 'relationship']
groups = ['sex', 'race']
groups = ['education', 'occupation']
groups = ['education', 'workclass']
df_m = marginalize(df_a, groups)
df_m.set_index(groups, inplace=True)
print(df_m.sum())

# %%
fig, ax = plt.subplots(figsize=(10, 10))
g = sns.heatmap(df_m.prob.unstack().T, annot=True, cmap='viridis', fmt='.2f')
g.set_title('P(group| Income >50k)')
plt.tight_layout()
plt.show()

# %%
# Exclude the White & Male cohort because they're dominating the analysis
df_sp = df_a.loc[
    ~((df_a.index.get_level_values('sex') == 'Male') & (df_a.index.get_level_values('race') == 'White')), :
]

# %%
groups = ['sex', 'race']
df_m = marginalize(df_sp, groups)
df_m.set_index(groups, inplace=True)
print(df_m.sum())

# %%
g = sns.heatmap(df_m.prob.unstack().T, annot=True, cmap='viridis', fmt='.2f')
g.set_title('P(group| Income >50k)')
plt.tight_layout()
plt.show()