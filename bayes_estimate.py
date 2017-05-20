import pystan
import numpy as np
import pandas as pd
import os
import pickle
import argparse
import urllib2
import zipfile


lahman_file = 'baseballdatabank-2017.1.zip'


def download_lahman():
    # Download Lahman baseball statistics dataset for 2016
    # 2017-05019: latest update Feb 2017
    lahman_url = 'http://seanlahman.com/files/database/baseballdatabank-2017.1.zip'
    response = urllib2.urlopen(lahman_url)
    with open(lahman_file, 'wb') as f:
        f.write(response.read())


def get_lahman_data(name):
    """Read Lahman baseball data file into Pandas dataframe
    
    Args:
        name: name of file e.g. Batting, Pitching, Master

    Returns:
        dataframe with file contents
    """
    if not os.path.isfile(lahman_file):
        print('Downloading Lahman baseball statistics data')
        download_lahman()
    z = zipfile.ZipFile(lahman_file)
    df = pd.read_csv(z.extract('baseballdatabank-2017.1/core/{}.csv'.format(name)))
    return df


def get_processed_lahman():
    """Get processed Lahman batter statistics as used in 'Empirical Bayes' by David Robinson
    
    Returns:
        dataframe of career statistics for batters
    """
    df_b = get_lahman_data('Batting')
    df_p = get_lahman_data('Pitching')
    df_m = get_lahman_data('Master').set_index('playerID')

    # Career statistics for batters:
    # - At Bats > 0
    # - Not in the list of Pitchers
    # Total number of At Bats (AB) and Hits (H)

    df_c = (df_b[(df_b['AB'] > 0)
                 & ~df_b['playerID'].isin(df_p['playerID'])
            ]
            .groupby('playerID')[['H', 'AB']]
            .sum())
    df_c['average'] = df_c['H'] / df_c['AB']
    # Replace playerID with player name
    df_c = df_c.join((df_m['nameGiven'] + ' ' + df_m['nameLast']).to_frame(name='name')).reset_index()

    return df_c[['name', 'H', 'AB', 'average']].copy()


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
        # Point estimate of parameters using MLE
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
        df = pd.read_csv(out_file, comment='#')
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
# Model in two_compartment.stan

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


# Bernoulli model: given a list of {0,1} results estimate the probability of 1
# Think of this as a sequence of coin flips from a biased coin
# Model in bernoulli.stan

class BernoulliModel(BayesModel):
    def __init__(self):
        super(BernoulliModel, self).__init__('bernoulli')

    @staticmethod
    def generate_data(N=100, theta=0.5):
        y = np.random.binomial(1, theta, N)
        return {'N': N, 'y': y}


# Lahman baseball statistics model: estimate population batting average distribution
# Assume that each player's observed batting average is drawn from a Beta distribution
# Model in beta.stan

class LahmanModel(BayesModel):
    def __init__(self):
        super(LahmanModel, self).__init__('beta')

    @staticmethod
    def generate_data(**kwargs):
        # Only consider players with a reasonable history of > 500 At Bats
        df_c = get_processed_lahman().query('AB > 500')
        y = df_c['average'].values.tolist()
        N = len(y)
        return {'N': N, 'y': y}


def get_model_params(name=None):
    """Factory for Stan models and associated parameters for generating data and reporting
    
    Args:
        name: name of model - one of 'twocompartment', 'bernoulli' or 'lahman'

    Returns:
        model, data_args, report_args
    """
    if name == 'twocompartment':
        model = TwoCompartmentModel()
        data_args = {}
        report_args = {'pars': ['a', 'b', 'sigma']}
    elif name == 'bernoulli':
        model = BernoulliModel()
        data_args = {'N': 1000, 'theta': 0.3}
        report_args = {}
    elif name == 'lahman':
        model = LahmanModel()
        data_args = {}
        report_args = {}
    else:
        raise ValueError('Unknown or missing model name')
    return model, data_args, report_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bayesian estimation of model parameters')
    parser.add_argument('-m', '--model', dest='model', help='Name of model: twocompartment, bernoulli',
                        action='store', default='twocompartment')
    parser.add_argument('-a', '--analysis', dest='analysis',
                        help='Analysis to run: sample, mle, vb (variational Bayes).  Comma separated for multiple.',
                        action='store', default='mle')

    args = parser.parse_args()

    name = args.model
    analysis = args.analysis.split(',')
    model, data_args, report_args = get_model_params(name=name)

    data = model.generate_data(**data_args)
    if 'sample' in analysis:
        # Demonstrate sampling
        print('Sampling')
        fit = model.fit(data)
        model.report_fit(fit, **report_args)

    if 'mle' in analysis:
        # Demonstrate optimizing for point estimate
        print('Optimizing')
        optim = model.optimize(data)
        for k, v in optim.iteritems():
            print('{}: {}'.format(k, v))

    if 'vb' in analysis:
        # Demonstrate using variational Bayes
        print('Variational Bayes')
        df = model.vb(data)
        print(df.describe())
