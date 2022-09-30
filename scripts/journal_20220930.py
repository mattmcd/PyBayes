from itertools import chain
import re
import arxiv

graph_neural_network_papers_jraph_doc = """
https://arxiv.org/abs/1806.01261
https://arxiv.org/abs/1706.01427
https://arxiv.org/abs/1703.06114
https://arxiv.org/abs/1612.00593
https://arxiv.org/abs/1710.10903
https://arxiv.org/abs/1609.02907
https://arxiv.org/abs/1612.00222
"""

other_papers = """
https://arxiv.org/abs/1803.08475
https://arxiv.org/abs/1810.10136
https://arxiv.org/abs/2110.06357
https://arxiv.org/abs/2101.04562
https://arxiv.org/abs/1911.05467
https://arxiv.org/abs/1611.08097
https://arxiv.org/abs/1703.09307
https://arxiv.org/abs/2103.02559
https://arxiv.org/abs/2011.14999
https://arxiv.org/abs/2106.06020
https://arxiv.org/abs/2103.08057
https://arxiv.org/abs/2107.00630
https://arxiv.org/abs/2111.15161
https://arxiv.org/abs/2111.15323
"""

crypto_papers = """
https://arxiv.org/abs/1902.05164
https://arxiv.org/abs/2009.14021
https://arxiv.org/abs/2012.08040
https://arxiv.org/abs/2006.08806
https://arxiv.org/abs/1509.03264
https://arxiv.org/abs/2006.12388
https://arxiv.org/abs/2105.11053
https://arxiv.org/abs/2105.13822
https://arxiv.org/abs/2106.12033
"""

gnn_papers = """
https://arxiv.org/abs/1705.07664
https://arxiv.org/abs/1606.09375
https://arxiv.org/abs/2104.04883
https://arxiv.org/abs/2106.12575
"""

all_papers = [graph_neural_network_papers_jraph_doc, other_papers, crypto_papers, gnn_papers]


def print_papers():
    ap = '\n'.join(chain(all_papers))

    paper_ids = re.findall('(\d{4}\.\d{5})', ap)

    search = arxiv.Search(id_list=paper_ids)
    for paper in search.results():
        print(f'{paper.entry_id} {paper.title}')


if __name__ == '__main__':
    print_papers()
