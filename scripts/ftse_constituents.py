import requests
import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import yfinance as yf
# Analysis code that should be broken out into separate module
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import ipywidgets as widgets



INDEXES = {
    'FTSE 100': 'ftse-100',
    'FTSE 250': 'ftse-250',
    'FTSE 350':  'ftse-350',
    'FTSE All Share':  'ftse-all-share',
    'FTSE AIM UK 50 Index': 'ftse-aim-uk-50-index',
    'FTSE AIM UK 100 Index': 'ftse-aim-uk-100-index',
    'FTSE AIM All Share': 'ftse-aim-all-share',
}

ISHARES_US_SECTOR_ETFS = ['IU' + s for s in 'IT HC FS ES CD CS US IS MS CM'.split(' ')]

@dataclass
class Index:
    name: str
    code: str = field(init=False)
    constituents: pd.DataFrame = field(init=False)
    use_cache: bool = True
    cache_dir: str = os.path.expanduser('~/Work/Data/')

    def __post_init__(self):
        self.code = INDEXES[self.name]

    def _create_payload(self, *, page, size=20):
        tab_id = '1602cf04-c25b-4ea0-a9d6-64040d217877'
        constituent_params = f'indexname={self.code}&tab=table&page={page}&tabId={tab_id}'
        component_params = f'page={page}&size={size}&sort=marketcapitalization,desc'
        # noinspection PyUnresolvedReferences
        payload = {
            'path': 'ftse-constituents',
            'parameters': requests.utils.quote(constituent_params),
            'components': [
                {'componentId': 'block_content%3Aafe540a2-2a0c-46af-8497-407dc4c7fd71',
                 'parameters': component_params}
            ]
        }
        return payload

    def get_constituent_page(self, *, page):
        url = 'https://api.londonstockexchange.com/api/v1/components/refresh'
        resp = requests.post(url, json=self._create_payload(page=page))
        df = pd.DataFrame(resp.json()[0]['content'][0]['value']['content'])
        meta = resp.json()[0]['content'][0]['value']
        del meta['content']
        return df, meta

    @property
    def cache_file(self):
        return os.path.join(self.cache_dir, f'{self.code}.csv')

    def get(self):
        if self.use_cache and os.path.exists(self.cache_file):
            print(f'Reading from cache {self.cache_file}')
            df = pd.read_csv(self.cache_file)
        else:
            df_first, meta = self.get_constituent_page(page=0)
            total_pages = meta['totalPages']
            res = [df_first]
            for page in range(1, total_pages + 1):
                df_page, meta_page = self.get_constituent_page(page=page)
                res.append(df_page)
            df = pd.concat(res, axis=0).reset_index(drop=True)
            if self.use_cache:
                print(f'Saving to cache {self.cache_file}')
                df.to_csv(self.cache_file, index=False)

        self.constituents = df

    def yfinance_tickers(self):
        # Simple approach, currently fails for:
        #   BT.A.L -> BT-A.L
        # No data returned for:
        #   AAF.L, AUTO.L, AVST.L, BME.L, BT.A.L, CCH.L, EDV.L, GLEN.L, HLN.L, MNG.L, OCDO.L, PHNX.L, PSH.L
        def ticker_map(t):
            t += '.L'  # Yahoo finance marker for LSE
            t = t.replace('..', '.')
            t = t.replace('BT.A.L', 'BT-A.L')
            return t
        return [ticker_map(t) for t in self.constituents['tidm'].to_list()]


def reorder_instruments(df, keep=0.1):
    """Reorder instruments to make correlation matrix block diagonal
    Approach:
    Calculate correlation matrix
    Get lower triangle of correlations
    Threshold to keep only top and bottom 5% of correlations
    Use Reverse Cuthill-McKee algorithm to reduce bandwidth of sparsified correlation matrix

    :param df: dataframe of returnsns 'ticker' and 'date'
    :return: dataframe with columns 'ticker' and 'date'
    """
    corr_mat = df.corr().values
    pair_corrs = corr_mat[np.tri(len(df.columns), k=-1).astype(np.bool).T]
    low_thresh, high_thresh = np.quantile(pair_corrs, [keep/2, 1. - keep/2]).tolist()
    sp_corr_mat = corr_mat[::]  # Copy
    sp_corr_mat[(corr_mat > low_thresh) & (corr_mat < high_thresh)] = 0.
    g = csr_matrix(sp_corr_mat)
    ind = reverse_cuthill_mckee(g)
    # return ind, corr_mat, sp_corr_mat
    return df.columns[ind].to_list()


def extract_ticker(df_p, ticker):
    # Get adjusted close and volume date from downloaded price data
    return df_p.loc(axis=1)[
        ['Adj Close', 'Volume']
    ].swaplevel(0, 1, axis=1).loc(axis=1)[ticker].rename(
        columns={'Adj Close': 'close', 'Volume': 'volume'}
    )


def get_sector_history(sectors=None, **kwargs):
    sectors = sectors or [t + '.L' for t in ISHARES_US_SECTOR_ETFS]
    return yf.download(sectors, **kwargs)


def make_slider(start='2018', end='2023'):
    dates = pd.date_range(start, end, freq='D')
    # Slider definition
    # options = dates.to_list()  # First attempt: raw datetime object.  Works but can't see end date
    options = [(d.strftime('%Y%m%d'), d) for d in dates]  # Second attempt: (label, item) list with date format
    opt_index = (0, len(options)-1)
    slider = widgets.SelectionRangeSlider(
        index=opt_index,
        options=options,
        layout={'width': '500px'}
    )
    return slider


class TickerHistoryViewer:
    def __init__(self, df):
        self.df = df
        self.df_c = self.df['Adj Close'].pct_change().rolling('21D').corr().dropna().copy()

    def select_date_range(self, date_range):
        return self.df['Adj Close'].loc[
               date_range[0]:date_range[-1], :
               ]

    def plot(self, date_range):
        fig = plt.figure(figsize=(10, 6))
        gs = mpl.gridspec.GridSpec(nrows=2, ncols=3, figure=fig)
        ax = fig.add_subplot(gs[0:2, 0:2])
        date_range = date_range if (date_range is not None) else df_p.index
        df_r = self.select_date_range(date_range).pct_change().fillna(0)
        df_r.rename(
            columns={c: c.replace('.L', '') for c in df_r.columns},
            inplace=True
        )
        df_n = (1 + df_r).cumprod()
        df_n.plot(ax=ax)
        # Add correlation plot
        ax_c = fig.add_subplot(gs[2])
        sns.heatmap(df_r.corr(), ax=ax_c, cmap='coolwarm', center=0.)
        plt.tight_layout()

    def plot_corr(self, date_range=None, ref_ticker=None):
        if date_range is None:
            date_range = self.df_c.index.get_level_values(0)
        if ref_ticker is None:
            ref_ticker = self.df_c.columns[0]
        start = date_range[0]
        end = date_range[-1]
        _ = self.df_c.loc[start:end, :].xs(
            ref_ticker, level=1
        ).drop(columns=ref_ticker).plot(
            kind='hist', alpha=0.8, bins=20, subplots=True, grid=True,
            figsize=(6, 8), title=f'{ref_ticker} {start.strftime("%Y-%m-%d")} - {end.strftime("%Y-%m-%d")}'
        )

    def interactive(self, start=None, end=None):
        start = start or self.df.index[0]
        end = end or self.df.index[-1]
        return widgets.interact(self.plot, date_range=make_slider(start=start, end=end))

    def interactive_corr(self, start=None, end=None, ref_ticker=None):
        start = start or self.df_c.index.get_level_values(0)[0]
        end = end or self.df_c.index.get_level_values(0)[-1]
        date_slider = make_slider(start=start, end=end)
        tickers = self.df_c.columns.to_list()
        ref_ticker = ref_ticker or tickers[0]
        ticker_chooser = widgets.Dropdown(options=tickers, value=ref_ticker)
        return widgets.interact(self.plot_corr, date_range=date_slider, ref_ticker=ticker_chooser)


if __name__ == '__main__':
    index_name = 'FTSE 100'
    page = 1
    ftse_100 = Index(name=index_name)
    ftse_100.get()
    df_p = yf.download(ftse_100.yfinance_tickers(), period='3y', interval='1d')  # .loc[:, 'Adj Close']
