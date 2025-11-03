# Bayes rule approach to estimating class probability

# %%
from startup import np, pd, plt, sns, sm, smf, os, Path
from collections import defaultdict
from scipy.stats import beta
import itertools
import ydf
from sklearn.manifold import TSNE
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
class ModelBasedAnalysis:
    def __init__(self, df_train, df_test, target):
        self.df_train = df_train
        self.df_test = df_test
        self.target = target
        self.model =  ydf.RandomForestLearner(label=target).train(df_train)
        self.evaluation = self.model.evaluate(df_test)

    def variable_importance(self):
        return pd.DataFrame(self.model.variable_importances()['INV_MEAN_MIN_DEPTH'])

    def __repr__(self):
        out_str = f'ROC AUC: {self.evaluation.characteristics[0].roc_auc:0.2f}'
        return out_str

    def plot(self):
        df_p = self.df_test.sample(frac=0.2)
        manifold = TSNE(n_components=2).fit_transform(self.model.distance(df_p, df_p))
        sns.scatterplot(x=manifold[:, 0], y=manifold[:, 1], hue=df_p[target], alpha=0.2)
        plt.title(
            f'ROC AUC: {self.evaluation.characteristics[0].roc_auc:0.2f}'
        )


# %%
print('Using all features:')
eda_variables(df_train, df_test, target, 7)
plt.show()

print('Using only categorical features features:')
eda_variables(df_train.loc[:, categorical_vars], df_test, target, 7)
plt.show()
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
# Generalise to look up multiple categorical values
def extract_sub_population(df, sub_pop):
    level_names = df.index.names
    index_factory = defaultdict(lambda: slice(None))

    for k, v in sub_pop.items():
        index_factory[k] = v

    pop_indexer = tuple([index_factory[c] for c in level_names])

    try:
        return df.loc[pop_indexer, :]
    except KeyError:
        return pd.DataFrame({'trials': [0], 'successes': [0]},)

# %%
def bayes_rule(df, df_s):
    """Apply Bayes rule to estimate probability of target given sub population

    :param df: dataframe of full population with columns 'trials' and 'successes'
    :param df_s: dataframe of sub population
    :return:
    """
    # Alternatively, could use beta distributions for these to get confidence intervals
    p_sub_pop = df_s.trials.sum() / df.trials.sum()
    p_target = df.successes.sum() / df.trials.sum()
    p_target_given_sub_pop = df_s.successes.sum() / df_s.trials.sum()
    p_sub_pop_given_target = p_target_given_sub_pop * p_sub_pop / p_target
    # Or more simply: p_sub_pop_given_target = df_s.successes.sum()/df.successes.sum()
    # i.e. fraction of successes in sub-population as a fraction of successes in full population
    return p_target, p_sub_pop, p_target_given_sub_pop, p_sub_pop_given_target

# %%
def long_summary(pop_slice, target, p_target, p_sub_pop, p_target_given_sub_pop, p_sub_pop_given_target):
    """Summary of sub-population probabilities including level values in description"""
    slice_str = ', '.join({ k: f'{k}={" or ".join(v)}' for k, v in pop_slice.items()}.values())
    return f'{pop_slice}\n' \
           f'P({target}) = {p_target:.3f}\n' \
           f'P({slice_str}) = {p_sub_pop:.3f}\n' \
           f'P({target} | {slice_str}) = {p_target_given_sub_pop:.3f}\n' \
           f'P({slice_str} | {target}) = {p_sub_pop_given_target:.3f}\n' \
           f'Odds ratio = {p_sub_pop_given_target/(1 - p_sub_pop_given_target):.3f}:1'

def brief_summary(pop_slice, target, p_target, p_sub_pop, p_target_given_sub_pop, p_sub_pop_given_target):
    """Summary of sub-population probabilities with level names in description"""
    return f'{pop_slice}\n' \
           f'P({target}) = {p_target:.3f}\n' \
           f'P({", ".join(pop_slice.keys())}) = {p_sub_pop:.3f}\n' \
           f'P({target} | {", ".join(pop_slice.keys())}) = {p_target_given_sub_pop:.3f}\n' \
           f'P({", ".join(pop_slice.keys())} | {target}) = {p_sub_pop_given_target:.3f}\n' \
           f'Odds ratio = {p_sub_pop_given_target/(1 - p_sub_pop_given_target):.3f}:1'

# %%
class BayesRule:
    def __init__(self, df, sub_pop, target_label='target', summary_type='long'):
        self.df = df
        self.sub_pop = sub_pop
        self.target_label = target_label
        self.df_s = extract_sub_population(self.df, self.sub_pop)
        self.p_target, self.p_sub_pop, self.p_target_given_sub_pop, self.p_sub_pop_given_target = \
            bayes_rule(self.df, self.df_s)
        self.summary_fun = long_summary if summary_type == 'long' else brief_summary

    def __repr__(self):
        return self.summary_fun(
            self.sub_pop, self.target_label,
            self.p_target, self.p_sub_pop,
            self.p_target_given_sub_pop, self.p_sub_pop_given_target
        )

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

# %%
def marginalize(df, groups):
    sub_pops = [
        dict(zip(groups, g)) for g in
        itertools.product(*[df.index.unique(level=g).tolist() for g in groups])
    ]
    df_marginal_probs = pd.DataFrame(
        [sp | {'prob': BayesRule(df, sp).p_sub_pop_given_target} for sp in sub_pops]
    )
    return df_marginal_probs   #.fillna(0)  # leaving in NaN makes excluded groups more obvious

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