import pystan
import numpy as np

# Two compartment model from
# "Stan: A probabilistic programming language for
#  Bayesian inference and optimization" Gelman, Lee, Guo (2015)
# http://www.stat.columbia.edu/~gelman/research/published/stan_jebs_2.pdf

a = np.array([0.8, 1.0])
b = np.array([2, 0.1])
sigma = 0.2

x = np.arange(0, 1000, dtype='float')/100
N = len(x)

# The two compartment model we are attempting to fit
y_pred = a[0]*np.exp(-b[0]*x) + a[1]*np.exp(-b[1]*x)

# Include multiplicative noise
y = y_pred * np.exp(np.random.normal(0, sigma, N))

# Compile and sample model
fit = pystan.stan(file='two_compartment.stan',
                  data={'N': N, 'x': x, 'y': y},
                  iter=1000, chains=4)

# Plot parameter estimates of interest
fit.plot(pars=['a', 'b', 'sigma'])

# Print all parameter estimates (limitation of PyStan 2.0)
print(fit)

# Extension: Demonstrate reusing compiled model on new data
y_new = y_pred * np.exp(np.random.normal(0, sigma, N))
fit2 = pystan.stan(fit=fit, data={'N': N, 'x': x, 'y': y_new})
fit2.plot(pars=['a', 'b', 'sigma'])
print(fit2)
