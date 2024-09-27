# %%
from mattmcd.journal_20240908_pymc import Analysis, PropertyBasedModel
from startup import np, pd, plt, sns, sm, smf
import pymc as pm
import arviz as az
# %%
analysis = Analysis()

# %%
model = analysis.call_rate()

# %%
with model:
    step = pm.Metropolis()
    trace = pm.sample(10000, tune=5000, step=step, return_inferencedata=True)

# %%
az.plot_trace(trace, compact=False)
plt.tight_layout()
plt.show()

# %%
gv = pm.model_to_graphviz(model, save='chat_rate.png')

# gv.view()

# %%
analysis2 = PropertyBasedModel()
obs2 = analysis2.obs  # Constructs the variables
model2 = analysis2.model

# %%
with model2:
    step = pm.Metropolis()
    trace = pm.sample(10000, tune=5000, step=step, return_inferencedata=True)

# %%
az.plot_trace(trace, compact=False)
plt.tight_layout()
plt.show()