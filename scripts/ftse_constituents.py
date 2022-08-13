import requests
import json
import pandas as pd
from dataclasses import dataclass, field
import yfinance as yf


INDEXES = {
    'FTSE 100': 'ftse-100',
    'FTSE 250': 'ftse-250',
    'FTSE 350':  'ftse-350',
    'FTSE All Share':  'ftse-all-share',
    'FTSE AIM UK 50 Index': 'ftse-aim-uk-50-index',
    'FTSE AIM UK 100 Index': 'ftse-aim-uk-100-index',
    'FTSE AIM All Share': 'ftse-aim-all-share',
}


@dataclass
class Index:
    name: str
    code: str = field(init=False)
    constituents: pd.DataFrame = field(init=False)

    def __post_init__(self):
        self.code = INDEXES[self.name]

    def create_payload(self, *, page, size=20):
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
        resp = requests.post(url, json=self.create_payload(page=page))
        df = pd.DataFrame(resp.json()[0]['content'][0]['value']['content'])
        meta = resp.json()[0]['content'][0]['value']
        del meta['content']
        return df, meta

    def get(self):
        df_first, meta = self.get_constituent_page(page=0)
        total_pages = meta['totalPages']
        res = [df_first]
        for page in range(1, total_pages + 1):
            df_page, meta_page = self.get_constituent_page(page=page)
            res.append(df_page)
        df = pd.concat(res, axis=0).reset_index(drop=True)
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


if __name__ == '__main__':
    index_name = 'FTSE 100'
    page = 1
    ftse_100 = Index(name=index_name)
    ftse_100.get()
    df_p = yf.download(ftse_100.yfinance_tickers(), period='3y', interval='1d').loc[:, 'Close']
