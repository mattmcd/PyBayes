import os
import numpy as np
import pandas as pd
from pathlib import Path


class EconReader:
    def __init__(self):
        self.data_dir = Path(os.path.expanduser('~/Work/Data/woolridge_7ed_datasets'))
        self._files = self._get_data_files()

    @property
    def files(self):
        return list(self._files.keys())

    def _get_data_files(self):
        dta_files = dict(
            sorted(
                [(f.stem.lower(), f) for f in self.data_dir.glob('*.[dD][tT][aA]')],
                key=lambda x: x[0]
            )
        )
        return dta_files

    def read(self, file_name):
        return pd.read_stata(self._files[file_name])
