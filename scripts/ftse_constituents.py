import requests
import json
import pandas as pd
from dataclasses import dataclass


INDEXES = {
    'FTSE 100': 'ftse-100',
    'FTSE 250': 'ftse-250',
    'FTSE 350':  'ftse-350',
    'FTSE All Share':  'ftse-all-share',
    'FTSE AIM UK 50 Index': 'ftse-aim-uk-50-index',
    'FTSE AIM UK 100 Index': 'ftse-aim-uk-100-index',
    'FTSE AIM All Share': 'ftse-aim-all-share',
}


def create_payload(*, index_name, page, size=20):
    tab_id = '1602cf04-c25b-4ea0-a9d6-64040d217877'
    constituent_params = f'indexname={index_name}&tab=table&page={page}&tabId={tab_id}'
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


def get_constituent_page(*, index_name, page):
    url = 'https://api.londonstockexchange.com/api/v1/components/refresh'
    resp = requests.post(url, json=create_payload(index_name=index_name, page=page))
    df = pd.DataFrame(resp.json()[0]['content'][0]['value']['content'])
    meta = resp.json()[0]['content'][0]['value']
    del meta['content']
    return df, meta


if __name__ == '__main__':
    index_name = 'ftse-100'
    page = 1
    df, meta = get_constituent_page(index_name=index_name, page=page)
    print(df.head())
