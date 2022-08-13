import yfinance as yf
import pandas as pd
import numpy as np
import networkx as nx
from dataclasses import dataclass


# Quick and dirty way of getting list of invalid fx pairs from all_pairs
# Better: do this dynamically and cache
# df = yf.download([p + '=X' for p in fx.pairs], start='2022-08-01', end='2022-08-01', group='ticker')
# df_n = df.loc[:, ('Close')]
# invalid_pairs = [p.replace('=X', '') for p in df_n.columns[df_n.isna().values[0]].to_list()]
# print(",".join(invalid_pairs))
invalid_pairs = (
        'AEDAED,AEDARS,AEDBRL,AEDCLP,AEDCNY,AEDCZK,AEDDKK,AEDHKD,AEDHUF,AEDIDR,AEDKRW,AEDMXN,AEDMYR,AEDNOK,AEDPLN,' +
        'AEDRUB,AEDSEK,AEDSGD,AEDTHB,AEDTRY,AEDTWD,ARSAED,ARSARS,ARSAUD,ARSCAD,ARSCHF,ARSCLP,ARSCNY,ARSCZK,ARSDKK,' +
        'ARSHKD,ARSHUF,ARSIDR,ARSINR,ARSKRW,ARSMXN,ARSMYR,ARSNOK,ARSNZD,ARSPLN,ARSRUB,ARSSEK,ARSSGD,ARSTHB,ARSTRY,' +
        'ARSTWD,ARSZAR,AUDAED,AUDAUD,AUDCLP,AUDRUB,AUDTRY,BRLAED,BRLBRL,BRLCAD,BRLCNY,BRLCZK,BRLDKK,BRLGBP,BRLHUF,' +
        'BRLIDR,BRLINR,BRLKRW,BRLMYR,BRLNOK,BRLNZD,BRLPLN,BRLRUB,BRLTHB,BRLTRY,BRLTWD,BRLZAR,CADCAD,CADCLP,CADCZK,' +
        'CADHUF,CADPLN,CADRUB,CADTRY,CHFCHF,CHFCLP,CHFRUB,CLPAED,CLPARS,CLPAUD,CLPBRL,CLPCAD,CLPCHF,CLPCLP,CLPCNY,' +
        'CLPCZK,CLPDKK,CLPEUR,CLPHKD,CLPHUF,CLPIDR,CLPINR,CLPJPY,CLPKRW,CLPMXN,CLPMYR,CLPNOK,CLPNZD,CLPPLN,CLPRUB,' +
        'CLPSEK,CLPSGD,CLPTHB,CLPTRY,CLPTWD,CLPZAR,CNYAED,CNYARS,CNYBRL,CNYCLP,CNYCNY,CNYCZK,CNYDKK,CNYHUF,CNYIDR,' +
        'CNYINR,CNYKRW,CNYMXN,CNYMYR,CNYNOK,CNYPLN,CNYRUB,CNYSEK,CNYSGD,CNYTHB,CNYTRY,CNYTWD,CZKAED,CZKARS,CZKAUD,' +
        'CZKBRL,CZKCAD,CZKCLP,CZKCNY,CZKCZK,CZKGBP,CZKHKD,CZKHUF,CZKIDR,CZKINR,CZKKRW,CZKMXN,CZKMYR,CZKNOK,CZKNZD,' +
        'CZKPLN,CZKRUB,CZKSGD,CZKTHB,CZKTRY,CZKTWD,CZKZAR,DKKAED,DKKARS,DKKAUD,DKKBRL,DKKCAD,DKKCLP,DKKCNY,DKKDKK,' +
        'DKKIDR,DKKKRW,DKKMXN,DKKMYR,DKKRUB,DKKTHB,DKKTRY,DKKTWD,EUREUR,GBPGBP,HKDAED,HKDARS,HKDCLP,HKDCZK,HKDHKD,' +
        'HKDHUF,HKDNOK,HKDRUB,HKDTRY,HUFAED,HUFARS,HUFAUD,HUFBRL,HUFCAD,HUFCLP,HUFCNY,HUFCZK,HUFGBP,HUFHKD,HUFHUF,' +
        'HUFIDR,HUFINR,HUFJPY,HUFKRW,HUFMXN,HUFMYR,HUFNOK,HUFNZD,HUFPLN,HUFRUB,HUFSEK,HUFSGD,HUFTHB,HUFTRY,HUFTWD,' +
        'HUFZAR,IDRAED,IDRARS,IDRAUD,IDRBRL,IDRCAD,IDRCHF,IDRCLP,IDRCZK,IDRDKK,IDREUR,IDRGBP,IDRHUF,IDRIDR,IDRMXN,' +
        'IDRNOK,IDRPLN,IDRRUB,IDRSEK,IDRSGD,IDRTRY,INRAED,INRARS,INRBRL,INRCLP,INRCZK,INRDKK,INRHUF,INRIDR,INRINR,' +
        'INRMXN,INRNOK,INRPLN,INRRUB,INRSEK,INRSGD,INRTRY,JPYCLP,JPYHUF,JPYJPY,JPYSGD,JPYTRY,KRWAED,KRWARS,KRWBRL,' +
        'KRWCLP,KRWCZK,KRWDKK,KRWHUF,KRWKRW,KRWMXN,KRWNOK,KRWPLN,KRWRUB,KRWTRY,MXNAED,MXNARS,MXNCLP,MXNCNY,MXNCZK,' +
        'MXNHUF,MXNIDR,MXNINR,MXNKRW,MXNMXN,MXNMYR,MXNNOK,MXNNZD,MXNPLN,MXNRUB,MXNSEK,MXNTHB,MXNTRY,MXNTWD,MYRAED,' +
        'MYRARS,MYRBRL,MYRCLP,MYRCZK,MYRDKK,MYRHUF,MYRMXN,MYRMYR,MYRNOK,MYRPLN,MYRRUB,MYRSEK,MYRSGD,MYRTRY,NOKARS,' +
        'NOKAUD,NOKBRL,NOKCAD,NOKCLP,NOKCNY,NOKCZK,NOKHKD,NOKHUF,NOKIDR,NOKINR,NOKKRW,NOKMXN,NOKMYR,NOKNOK,NOKNZD,' +
        'NOKPLN,NOKRUB,NOKSGD,NOKTHB,NOKTRY,NOKTWD,NOKZAR,NZDAED,NZDARS,NZDBRL,NZDCLP,NZDMXN,NZDNOK,NZDNZD,NZDRUB,' +
        'NZDTRY,PLNAED,PLNARS,PLNAUD,PLNBRL,PLNCAD,PLNCHF,PLNCLP,PLNCNY,PLNCZK,PLNGBP,PLNHKD,PLNIDR,PLNINR,PLNKRW,' +
        'PLNMXN,PLNMYR,PLNNOK,PLNNZD,PLNPLN,PLNRUB,PLNSGD,PLNTHB,PLNTRY,PLNTWD,PLNZAR,RUBAED,RUBARS,RUBAUD,RUBBRL,' +
        'RUBCAD,RUBCHF,RUBCLP,RUBCNY,RUBCZK,RUBDKK,RUBEUR,RUBHKD,RUBHUF,RUBIDR,RUBINR,RUBKRW,RUBMXN,RUBMYR,RUBNOK,' +
        'RUBNZD,RUBPLN,RUBRUB,RUBSEK,RUBSGD,RUBTHB,RUBTRY,RUBTWD,RUBZAR,SEKAED,SEKARS,SEKBRL,SEKCLP,SEKCNY,SEKHKD,' +
        'SEKHUF,SEKIDR,SEKKRW,SEKMXN,SEKMYR,SEKNZD,SEKPLN,SEKRUB,SEKSEK,SEKSGD,SEKTHB,SEKTRY,SEKTWD,SEKZAR,SGDCAD,' +
        'SGDCLP,SGDCZK,SGDHUF,SGDRUB,SGDSGD,THBAED,THBARS,THBBRL,THBCLP,THBCZK,THBDKK,THBHUF,THBMXN,THBNOK,THBPLN,' +
        'THBRUB,THBSEK,THBTHB,THBTRY,TRYAED,TRYARS,TRYAUD,TRYBRL,TRYCAD,TRYCLP,TRYCNY,TRYCZK,TRYEUR,TRYGBP,TRYHKD,' +
        'TRYHUF,TRYIDR,TRYINR,TRYKRW,TRYMXN,TRYMYR,TRYNOK,TRYNZD,TRYPLN,TRYRUB,TRYSEK,TRYTHB,TRYTRY,TRYTWD,TWDAED,' +
        'TWDARS,TWDBRL,TWDCLP,TWDCZK,TWDDKK,TWDEUR,TWDHUF,TWDMXN,TWDNOK,TWDPLN,TWDRUB,TWDTRY,TWDTWD,USDUSD,ZARARS,' +
        'ZARBRL,ZARCLP,ZARCZK,ZARHUF,ZARMXN,ZARPLN,ZARRUB,ZARTRY,ZARZAR'
)


@dataclass
class Pair:
    cur_from: str
    cur_to:  str

    def __repr__(self):
        return f"{self.cur_from}{self.cur_to}"


class ForexPairs:
    main = ['GBP', 'EUR', 'USD', 'AUD']
    majors = ['CAD', 'CNY', 'JPY', 'CHF', ]
    europe = ['CZK', 'DKK', 'HUF', 'NOK', 'PLN', 'RUB', 'SEK', 'TRY', ]
    americas = ['ARS', 'BRL', 'CLP', 'MXN']
    africa = ['ZAR', 'AED']
    asiapac = ['HKD', 'INR', 'IDR', 'MYR', 'NZD', 'SGD', 'KRW', 'TWD', 'THB']

    currencies = main + majors + europe + americas + africa + asiapac

    @property
    def pairs(self):
        all_pairs = [
            Pair(c1, c2)
            for c1 in self.currencies
            for c2 in self.currencies
        ]
        valid_pairs = [p for p in all_pairs if str(p) not in invalid_pairs]

        return valid_pairs

    def get_crosses(self, pairs=None, **kwargs):
        if pairs is None:
            pairs = self.pairs

        df = yf.download([str(p) + '=X' for p in pairs], **kwargs)
        df.columns = df.columns.set_levels(
            [c.replace('=X', '') for c in df.columns.levels[1].to_list()], level=1
        )
        df_n = df.loc[:, 'Close']
        df_n.columns = pd.MultiIndex.from_tuples([(c[:3], c[3:]) for c in df_n.columns])
        return df_n.copy()

    def calc_score_vect(self, df_n, date, currencies=None, scale=None):
        currencies = currencies or self.main
        A = df_n.loc[date, (currencies, currencies)].unstack().loc[currencies, currencies]

        # Improve numerical stability by scaling some currencies e.g. work with 100 JPY
        # This doesn't seem to help
        if scale is not None:
            for c, sc in scale:
                A.loc[c, :] *= sc
                A.loc[:, c] /= sc
        # For complete graphs this seems to give correct result
        # Y = np.log(A.fillna(1.)
        # s = np.exp(Y.sum(axis=1) / Y.shape[0])

        # Extract upper triangle of exchange rate matrix
        # Create list of these edges
        edges = []
        for i in range(len(A.columns)):
            for j in range(len(A.columns)):
                edges.append({'source': currencies[i], 'target': currencies[j], 'rate': A.values[i, j]})
        # Remove missing exchange rates
        df_e = pd.DataFrame(edges).dropna()
        G = nx.DiGraph(df_e)
        # grad operator i.e. takes vertices in and returns edges
        d_0 = nx.incidence_matrix(G, oriented=True).todense().T
        s = np.exp(np.linalg.pinv(d_0.T @ d_0) @ (-d_0.T @ np.log(df_e.rate)))
        # Add the score (a.k.a. value) to each node of Graph
        for i, n in enumerate(G.nodes):
            G.nodes[n]['score'] = s[i]
        # model_fx_rates = (s.reshape(-1, 1) / s.reshape(-1, 1).T)

        df_p = pd.DataFrame(
            [{'source': e1, 'target': e2,
              'actual': G.get_edge_data(e1, e2)['rate'],
              'predicted': G.nodes[e1]['score'] / G.nodes[e2]['score'],
              } for e1, e2 in G.edges]
        ).assign(diff=lambda x: x['actual'] - x['predicted'])

        return s, A, G, df_p

