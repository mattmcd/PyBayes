import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import seaborn as sns



class LinearModel:
    def __init__(self, params=None):
        if params is None:
            params = LinearModel._random_init()
        self.params = params

    def __call__(self, x):
        return x*self.params['w'] + self.params['b']

    @staticmethod
    def _random_init():
        rng = np.random.Generator(np.random.MT19937())
        return {'w': rng.uniform(-10, 10), 'b': rng.uniform(-10, 10)}


class NoisyMeasurement:
    def __init__(self, params=None, model=None):
        self.rng = np.random.Generator(np.random.MT19937())
        if params is None:
            params = {'loc': 0., 'scale': 2.}
        self._params = params
        self.model = model

    @property
    def params(self):
        return self.model.params

    def __call__(self, x):
        return self.model(x) + self.rng.normal(loc=self._params['loc'], scale=self._params['scale'], size=x.shape)


class LogisticModel:
    def __init__(self, params=None, model=None):
        if model is None:
            self.model = LinearModel(params)
        else:
            self.model = model

    @property
    def params(self):
        return self.model.params

    def __call__(self, x):
        return 1/(1+jnp.exp(-self.model(x)))


def mse(y_true, y_pred):
    return jnp.mean((y_true - y_pred)**2)
