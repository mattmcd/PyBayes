# %%
import jax.numpy as jnp
import jax.scipy as jsp
from startup import np, pd, plt, sns
import pymc as pm
import preliz as pz
import preliz.distributions as pzd
import arviz as az
import optax
import jax
from flax import nnx
from flax.nnx.nn import dtypes, initializers
import tensorflow_probability.substrates.jax as tfp
import cvxpy as cp
from lifelines import CoxPHFitter, CoxPHFitter, KaplanMeierFitter, ExponentialFitter
tfd = tfp.distributions
tfb = tfp.bijectors
# %%
t = np.linspace(0., 1., 1000)
x = 2.*t + 0.5

# %%
plt.plot(t, x)
plt.show()

# %%
# Fitting with Bayesian model
with pm.Model() as model:
    slope = pm.Uniform('slope', lower=0., upper=100.)
    intercept = pm.Uniform('intercept', lower=-1., upper=1.)
    x_model = pm.Normal('x', mu=t*slope + intercept, sigma=0.1, observed=x)
    idata = pm.sample()

# %%
az.plot_trace(idata)
plt.show()

# %%
az.plot_pair(idata, kind='kde', marginals=True)
plt.show()

# %%
# Curve fitting via optimisation
def pred(params, t):
    return params['slope']*t + params['intercept']

# %%
@jax.jit
def loss(params, t, y):
    return jnp.mean(optax.l2_loss(y, pred(params, t)))

# %%
params = {'slope': 0., 'intercept': 0.}
print(loss(params, t, x))

# %%
start_learning_rate = 1e-1
optimizer = optax.adam(start_learning_rate)
opt_state = optimizer.init(params)

# %%
for _ in range(1000):
  grads = jax.grad(loss)(params, t, x)
  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)

# %%
print(f'Loss: {loss(params, t, x):0.4f}, params: {params}')

# %%
# CVXPy
beta = cp.Variable(2)
A = np.hstack([np.ones_like(t[:, np.newaxis]), t[:, np.newaxis]])
b = x
objective = cp.Minimize(cp.sum_squares(A@beta - b))
constraints = None
prob = cp.Problem(objective, constraints)
result = prob.solve()
print(beta.value)

# %%
actual_lifetime = 10
dist = pzd.exponential.Exponential(1/actual_lifetime)

# %%
sns.displot(dist.rvs(10000), kind='ecdf')
plt.show()

# %%
dist.plot_cdf()
plt.tight_layout()
plt.show()
# %%
# Estimate with lifelines
n_sample = 10000
max_t = 30
df_s = pd.DataFrame({'LIFETIME': dist.rvs(n_sample)}).assign(
    DURATION=lambda x: np.where(x.LIFETIME < max_t, x.LIFETIME, max_t),
    OBSERVED=lambda x: np.where(x.DURATION < max_t, True, False),
)

# %%
kmf = KaplanMeierFitter()
kmf.fit(df_s.DURATION, df_s.OBSERVED)
ef = ExponentialFitter()
ef.fit(df_s.DURATION, df_s.OBSERVED)
kmf.plot_survival_function()
ax = ef.plot_survival_function()
plt.ylim(0, 1)
plt.title(f'Est. Lifetime: {ef.params_["lambda_"]:0.2f}  Actual lifetime:{actual_lifetime:0.2f}')
plt.show()
ef.print_summary()

# %%
class JaxExponentialFitter(nnx.Module):
    def __init__(self, rngs):
        self.dense = nnx.Linear(
            in_features=1, out_features=1, rngs=rngs,
            kernel_init=initializers.ones_init(),  # for pass through manual scan
        )
        self.dist = tfd.Exponential

    def __call__(self, x):
        return  self.dist(jnp.exp(self.dense(x)))

# %%
jef = JaxExponentialFitter(nnx.Rngs(params=0))


# %%

# %%
# ds = jef(jnp.ones((10, 1)))
# %%
# Manual scan over model parameter making use of initial bias = 0 and kernel =1
# so that we can scale the intercept term
res_obs = []
res_actual = []
log_rate = jnp.linspace(-4, -1, 1000)
for rate in log_rate:
    jef.dense.kernel.value = jnp.array([[rate]])
    ds = jef(jnp.ones_like(df_s.DURATION.values.reshape(-1, 1)))
    # Observed lifetimes with censoring
    y_obs = df_s.DURATION.values.reshape(-1, 1)
    # Actual lifetimes
    y_act = df_s.LIFETIME.values.reshape(-1, 1)
    res_obs.append(-ds.log_prob(y_obs).sum())
    res_actual.append(-ds.log_prob(y_act).sum())


res_obs = np.array(res_obs)
res_actual = np.array(res_actual)
plt.plot(log_rate, res_obs, label='Observed')
plt.plot(log_rate, res_actual, label='Actual')
plt.title(
    f'Best fit lifetime: {1/jnp.exp(log_rate[jnp.argmin(res_obs)]):0.2f}\n'
    f'Best fit actual lifetime: {1/jnp.exp(log_rate[jnp.argmin(res_actual)]):0.2f}\n'
)
plt.tight_layout()
plt.show()

# %%
# Manual scan over model parameter - vectorized version
log_rate = jnp.linspace(-4, -1, 1000)

def calc_losses(rate):
    jef.dense.kernel.value = jnp.array([[rate]])
    ds = jef(jnp.ones_like(df_s.DURATION.values.reshape(-1, 1)))
    return -ds.log_prob(df_s[['DURATION', 'LIFETIME']].values).sum(axis=0)

df_res = pd.DataFrame(
    jax.vmap(calc_losses)(log_rate), columns=['Observed', 'Actual']
).assign(rate=log_rate).set_index('rate')

df_res.plot()
plt.title(
    f'Best fit lifetime: {1/jnp.exp(log_rate[jnp.argmin(df_res.Observed.values)]):0.2f}\n'
    f'Best fit actual lifetime: {1/jnp.exp(log_rate[jnp.argmin(df_res.Actual.values)]):0.2f}\n'
)
plt.tight_layout()
plt.show()


# %%
# We see from the above parameter scan that only considering the observed decays
# underestimates the lifetime.

def calc_censored_losses(rate):
    jef.dense.kernel.value = jnp.array([[rate]])
    ds = jef(jnp.ones_like(df_s.DURATION.values.reshape(-1, 1)))
    t = df_s[['DURATION']].values
    obs = df_s[['OBSERVED']].values
    # If event is observed use p(T), if not use CCDF(T) a.k.a. S(T)
    ll = jnp.where(obs, ds.log_prob(t), ds.log_survival_function(t))
    return -ll.sum(axis=0)

# For plotting join with the uncensored fit to actual duration
df_res_cure = pd.DataFrame(
    jax.vmap(calc_censored_losses)(log_rate), columns=['Observed']
).assign(rate=log_rate).set_index('rate').join(df_res[['Actual']])

df_res_cure.plot()
plt.title(
    f'Best fit lifetime: {1/jnp.exp(log_rate[jnp.argmin(df_res_cure.Observed.values)]):0.2f}\n'
    f'Best fit actual lifetime: {1/jnp.exp(log_rate[jnp.argmin(df_res_cure.Actual.values)]):0.2f}\n'
)
plt.tight_layout()
plt.show()


# %%
def loss(network, y):
    return -network.log_prob(y)

# %%
# learning_rate = 0.005
# momentum = 0.9
#
# optimizer = nnx.Optimizer(jef, optax.adamw(learning_rate, momentum))
# metrics = nnx.MultiMetric(
#   accuracy=nnx.metrics.Accuracy(),
#   loss=loss,
# )

# %%
# Cure model
def cure_model(t_inf, params):
    pc_, lambda_, rho_ = params
    return tfd.Mixture(
        cat=tfd.Categorical(probs=[pc_, 1-pc_]),
        components=[
            tfd.Deterministic(loc=t_inf),  # Represent Cure fraction by spike at t_inf >> other times in problem
            tfd.Weibull(concentration=rho_, scale=lambda_),
        ]
    )

# %%
cure_dist = cure_model(10000, jnp.array([0.2, 10., 0.5]))

# %%
cure_dist.survival_function(500.)

# %%
times = jnp.linspace(0, 500, 1000)
sf = cure_dist.survival_function(times)

ax = sns.lineplot(x=times, y=sf, label='Survival function')
ax.set_ylim(0, 1)
ax.axhline(0.2, color='k', linestyle='--')
plt.show()