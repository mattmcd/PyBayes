import pystan
import numpy as np
import pandas as pd
import os
import pickle


# Two compartment model from
# "Stan: A probabilistic programming language for
#  Bayesian inference and optimization" Gelman, Lee, Guo (2015)
# http://www.stat.columbia.edu/~gelman/research/published/stan_jebs_2.pdf


class TwoCompartmentModel(object):
    def __init__(self):
        self.model_file = 'two_compartment.stan'
        self.pkl_file = 'two_compartment.pkl'

    @staticmethod
    def generate_data():
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

    def get_model(self):
        if os.path.isfile(self.pkl_file):
            # Reuse previously compiled model
            sm = pickle.load(open(self.pkl_file, 'rb'))
        else:
            # Compile and sample model
            sm = pystan.StanModel(file=self.model_file)
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
    def report_fit(fit):
        # Plot parameter estimates of interest
        fit.plot(pars=['a', 'b', 'sigma'])

        # Print all parameter estimates (limitation of PyStan 2.0)
        print(fit)


if __name__ == '__main__':
    model = TwoCompartmentModel()
    data = model.generate_data()
    # Demonstrate sampling
    print('Sampling')
    fit = model.fit(data)
    model.report_fit(fit)
    # Demonstrate optimizing for point estimate
    print('Optimizing')
    optim = model.optimize(data)
    print(optim)
    # Demonstrate using variational Bayes
    print('Variational Bayes')
    df = model.vb(data)
    print(df.describe())
