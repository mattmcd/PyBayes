# %%
# Reading the Getting Started guide for dowhy and taking notes
from torch.distributed.rpc.api import method_name

# %%
# Standard imports
from startup import np, pd, plt, sns

# %%
# https://stackoverflow.com/questions/79673823/dowhy-python-library-module-networkx-algorithms-has-no-attribute-d-separated
# change in networkx 3.*, incompatible with dowhy 0.12;
# remove this when dowhy>0.12 is available
import networkx as nx
nx.algorithms.d_separated = nx.algorithms.d_separation.is_d_separator
nx.d_separated = nx.algorithms.d_separation.is_d_separator

# %%
# DoWhy imports
import dowhy
import dowhy.datasets
from dowhy import CausalModel

# %%
# First example: effect estimation and four step process.  The setup is we have a response variable,
# several causes variables, and several instrumental variables.

# Generate some sample data
# Params (snipped):
#     :param beta: coefficient of the treatment(s) ('v?') in the generating equation of the outcome ('y').
#     :type beta: int or list/ndarray of length num_treatments of type int
#     :param num_common_causes: Number of variables affecting both the treatment and the outcome [w -> v; w -> y]
#     :type num_common_causes: int
#     :param num_samples: Number of records to generate
#     :type num_samples: int
#     :param num_instruments: Number of instrumental variables  [z -> v], defaults to 0
#     :type num_instruments: int
#     :param num_effect_modifiers: Number of effect modifiers, variables affecting only the outcome [x -> y], defaults to 0
#     :type num_effect_modifiers: int
#     :param num_treatments: Number of treatment variables [v]. By default inferred from the beta argument. When provided, beta is recycled to match num_treatments.
#     :type num_treatments : Union[None, int]
#     :param num_frontdoor_variables : Number of frontdoor mediating variables [v -> FD -> y], defaults to  0
#     :type num_frontdoor_variables: int

data = dowhy.datasets.linear_dataset(
    beta=10,  # This is the ground truth coefficient of the treatment that we want to estimate
    num_common_causes=5,  # Affect both treatment and outcome
    num_instruments=2,    # Affects treatment only
    num_samples=10000)

# %%
# The variable `data` is a dictionary with the following keys:
# - 'df': a pandas dataframe with the data
# - treatment_name (v0), outcome_name (y), common_causes_names (W0-W4), instrument_names (Z1, Z2)
# - also present but empty are effect_modifiers_names and frontdoor_variables_names
# - gml_graph which is the Graphical Causal Model
# - ate which is the average treatment effect

# %%
# Look at the data
print(data['df'].head())

# %%
# sns.pairplot(data['df'])
# plt.show()

# %%
# Naive estimate of effect
print(
    "E(y|v) - E(y|~v) = "
    f"{data['df'].groupby('v0').y.mean().pipe(lambda x: x[True] - x[False]):0.2f}"
)

print(f"Actual ATE = E(y_i|v_i) - E(y_i|~v_i): {data['ate']:0.2f}")

# %%
# Step 1: Create a causal model from the data and given graph.
model = CausalModel(
    data=data["df"].copy(),
    treatment=data["treatment_name"],
    outcome=data["outcome_name"],
    graph=data["gml_graph"])

# %%
# Step 2: Identify causal effect and return target estimands
identified_estimand = model.identify_effect()

# %%
identified_estimand.estimands


# %%
# Step 3: Estimate the target estimand using a statistical method.
method_names = [
    # "backdoor.propensity_score_matching", # Seems to modify df?
    "backdoor.propensity_score_stratification",
    "backdoor.propensity_score_weighting",
    "backdoor.linear_regression",
    # "backdoor.generalized_linear_model",
    "iv.instrumental_variable",
    # "iv.regression_discontinuity",
]
estimates = [
    model.estimate_effect(identified_estimand, method_name=method_name)
    for method_name in method_names
]
df_e = pd.DataFrame({'method': method_names, 'ATE': [e.value for e in estimates]})

print(df_e)

# %%
# Step 4: Refute the obtained estimate using multiple robustness checks.
refute_results = model.refute_estimate(
    identified_estimand, estimates[0], method_name="random_common_cause")