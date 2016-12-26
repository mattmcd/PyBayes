import pystan
import numpy as np
import pandas as pd
import os
import pickle


class BayesModel(object):
    def __init__(self, name=''):
        self.name = name
        self.model_file = name + '.stan'
        self.pkl_file = name + '.pkl'

    @staticmethod
    def generate_data(**kwargs):
        pass

    def get_model(self):
        if os.path.isfile(self.pkl_file):
            # Reuse previously compiled model
            sm = pickle.load(open(self.pkl_file, 'rb'))
        else:
            # Compile and sample model
            sm = pystan.StanModel(file=self.model_file, model_name=self.name)
            with open(self.pkl_file, 'wb') as f:
                pickle.dump(sm, f)
        return sm

    def fit(self, data):
        # Sampling of parameters
        sm = self.get_model()
        fit = sm.sampling(data=data,
                          iter=2000, chains=4)
        return fit

    def optimize(self, data):
        # Point estimate of parameters
        sm = self.get_model()
        optim = sm.optimizing(data=data)
        return optim

    def vb(self, data):
        # Variational Bayes
        sm = self.get_model()
        try:
            res = sm.vb(data=data)
        except ValueError as e:
            print('Variational Bayes failed:' + e.message)
            return
        # Read generated samples
        # PyStan issue 163 should remove need for this
        out_file = res['args']['sample_file']
        df = pd.read_csv(out_file, skiprows=[0, 1, 2, 3, 5, 6])
        return df

    @staticmethod
    def report_fit(fit, pars=None):
        # Plot parameter estimates of interest
        fit.plot(pars=pars)

        # Print all parameter estimates (limitation of PyStan 2.0)
        print(fit)


# Two compartment model from
# "Stan: A probabilistic programming language for
#  Bayesian inference and optimization" Gelman, Lee, Guo (2015)
# http://www.stat.columbia.edu/~gelman/research/published/stan_jebs_2.pdf

class TwoCompartmentModel(BayesModel):
    def __init__(self):
        super(TwoCompartmentModel, self).__init__(name='two_compartment')

    @staticmethod
    def generate_data(**kwargs):
        a = np.array([0.8, 1.0])
        b = np.array([2, 0.1])
        sigma = 0.2

        x = np.arange(0, 1000, dtype='float')/100
        N = len(x)

        # The two compartment model we are attempting to fit
        y_pred = a[0]*np.exp(-b[0]*x) + a[1]*np.exp(-b[1]*x)

        # Include multiplicative noise
        y = y_pred * np.exp(np.random.normal(0, sigma, N))
        return {'N': N, 'x': x, 'y': y}


class BernoulliModel(BayesModel):
    def __init__(self):
        super(BernoulliModel, self).__init__('bernoulli')

    @staticmethod
    def generate_data(N=100, theta=0.5):
        y = np.random.binomial(1, theta, N)
        return {'N': N, 'y': y}


def get_model_params(name=None):
    if name == 'twocompartment':
        model = TwoCompartmentModel()
        data_args = {}
        report_args = {'pars': ['a', 'b', 'sigma']}
    elif name == 'bernoulli':
        model = BernoulliModel()
        data_args = {'N': 1000, 'theta': 0.3}
        report_args = {}
    else:
        raise ValueError('Unknown or missing model name')
    return model, data_args, report_args

if __name__ == '__main__':
    name = 'twocompartment'
    model, data_args, report_args = get_model_params(name=name)

    data = model.generate_data(**data_args)
    # Demonstrate sampling
    print('Sampling')
    fit = model.fit(data)
    model.report_fit(fit, **report_args)
    # Demonstrate optimizing for point estimate
    print('Optimizing')
    optim = model.optimize(data)
    print(optim)
    # Demonstrate using variational Bayes
    print('Variational Bayes')
    df = model.vb(data)
    print(df.describe())
