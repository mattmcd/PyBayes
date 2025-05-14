import numpy as np
import pymc as pm
from sympy.abc import alpha


def observations():
    # Phone call counts from
    # https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/
    # blob/master/Chapter1_Introduction/Ch1_Introduction_TFP.ipynb
    data = [
        13, 24, 8, 24, 7, 35, 14, 11, 15, 11, 22, 22, 11, 57,
        11, 19, 29, 6, 19, 12, 22, 12, 18, 72, 32, 9, 7, 13,
        19, 23, 27, 20, 6, 17, 13, 10, 14, 6, 16, 15, 7, 2,
        15, 15, 19, 70, 49, 7, 53, 22, 21, 31, 19, 11, 18, 20,
        12, 35, 17, 23, 17, 4, 2, 31, 30, 13, 27, 0, 39, 37,
        5, 14, 13, 22,
    ]
    return data


class Analysis:
    def __init__(self):
        self.model = pm.Model()

    def call_rate(self) -> pm.Model :
        # Phone call counts from
        # https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/
        # blob/master/Chapter1_Introduction/Ch1_Introduction_TFP.ipynb

        n_obs = len(observations())

        with self.model:
            alpha = pm.Exponential('alpha', lam=0.1)
            rv_lambda_1 = pm.Poisson('rv_lambda_1', mu=alpha)
            rv_lambda_2 = pm.Poisson('rv_lambda_2', mu=alpha)
            tau = pm.DiscreteUniform('switch_time', lower=0, upper=n_obs-1)

        # with model:
            idx = np.arange(n_obs)
            rv_lambda = pm.math.switch(tau > idx, rv_lambda_1, rv_lambda_2)

        # with model:
            obs = pm.Poisson('obs', mu=rv_lambda, observed=observations())

        return self.model



class PropertyBasedModel:
    """Expose the variables as properties
    (half an hour later: this turns out to be a poor idea.
       1. Need to then call the output variable to construct the model
       2. Could have just set the model as an attribute in the model above
          then use model.rv_name to access the random variables (made this modification)
    Useful learning at least on what not to do
    )
    """
    def __init__(self):
        self.model = pm.Model()
        self.observations = observations()
        self._alpha = None
        self._lambda1 = None
        self._lambda2 = None
        self._tau = None
        self._lambda = None
        self._obs = None

    @property
    def n_obs(self):
        return len(self.observations)

    @property
    def alpha(self):
        if self._alpha is None:
            with self.model:
                self._alpha = pm.Exponential('alpha', lam=0.1)
        return self._alpha

    @property
    def lambda1(self):
        if self._lambda1 is None:
            with self.model:
                self._lambda1 = pm.Poisson('lambda1', mu=self.alpha)
        return self._lambda1

    @property
    def lambda2(self):
        if self._lambda2 is None:
            with self.model:
                self._lambda2 = pm.Poisson('lambda2', mu=self.alpha)
        return self._lambda2

    @property
    def tau(self):
        if self._tau is None:
            with self.model:
                self._tau = pm.DiscreteUniform('switch_time', lower=0, upper=self.n_obs-1)
        return self._tau

    @property
    def rv_lambda(self):
        if self._lambda is None:
            with self.model:
                idx = np.arange(self.n_obs)
                self._lambda = pm.math.switch(
                    self.tau > idx, self.lambda1, self.lambda2)
        return self._lambda

    @property
    def obs(self):
        if self._obs is None:
            with self.model:
                self._obs = pm.Poisson('obs', mu=self.rv_lambda, observed=self.observations)
        return self._obs
