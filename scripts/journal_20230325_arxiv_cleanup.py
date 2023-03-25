# %%
import os
import sys
import re
import arxiv
import glob
import pandas as pd
from pathlib import Path

# %%
downloads = Path(os.path.expanduser('~/Downloads'))

# %%
arxiv_files = [f for f in downloads.rglob('*.pdf') if re.search(r'/\d{4}\.\d{4,}', str(f))]

# %%
arxiv_ids = ['.'.join(re.split(r'\D', f.name)[:2]) for f in arxiv_files]

# %%
# Can then use f.stat() to get file details
df = pd.DataFrame(
    [
        {
            'file_size': s.st_size,
            'downloaded': pd.to_datetime(s.st_ctime, unit='s').floor('1min'),
            'last_accessed': pd.to_datetime(s.st_atime, unit='s').floor('1min')
        }
        for s in [f.stat() for f in arxiv_files]
    ],
    index=arxiv_ids
).sort_values('downloaded')
print(df.head())

# %%
arxiv_data = list(arxiv.Search(id_list=arxiv_ids).results())

# %%
# Preprint metadata
df_m = pd.DataFrame(
    [
        {
            'entry_id': r.entry_id,
            'title': r.title,
            'summary': r.summary.replace('\n', ' '),
            'authors': ', '.join([str(a) for a in r.authors])
        }
        for r in arxiv_data
    ],
    index=arxiv_ids
)
df_m.head()

# %%
df_a = df.join(df_m).sort_values('downloaded')
df_a.head()

# %%
df_a.to_html(
    'arxiv_links_20230325.html',
    render_links=True
)

# %%
df_a.to_csv('arxiv_links_20230325.csv')
