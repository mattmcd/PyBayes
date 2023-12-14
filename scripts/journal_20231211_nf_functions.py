import numpy as np
import pandas as pd
import os
from dataclasses import dataclass


@dataclass
class MafData:
    data_dir: str = os.path.expanduser('~/Work/Data/maf/data')

    @property
    def power(self):
        data = np.load(os.path.join(self.data_dir, 'power', 'data.npy'))
        # See https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
        # NB: missing date and time columns in MAF download, last column is minutes since midnight
        col_names = (
            'Global_active_power;Global_reactive_power;Voltage;Global_intensity;' +
            'Sub_metering_1;Sub_metering_2;Sub_metering_3;time_mins'
        ).split(';')
        df = pd.DataFrame(data, columns=[c.lower() for c in col_names])
        return df


@dataclass
class MafDataOriginal:
    data_dir: str = os.path.expanduser('~/Work/Data/maf_original')

    @property
    def power(self):
        # From https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
        df = pd.read_csv(
            os.path.join(self.data_dir, 'household_power_consumption.txt'),
            sep=';', low_memory=False, na_values='?'
        )
        df.columns = [c.lower() for c in df.columns]
        df.loc[:, 'timestamp'] = pd.to_datetime(
            df.date + 'T' + df.time, format='%d/%m/%YT%H:%M:%S')
        df = df.drop(columns=['date', 'time']).dropna().set_index('timestamp').assign(
            sub_metering_other=lambda x:
            x.global_active_power * 1000 / 60
            - x.sub_metering_1 - x.sub_metering_2 - x.sub_metering_3
        ).rename(
            columns={
                'sub_metering_1': 'kitchen',
                'sub_metering_2': 'laundry',
                'sub_metering_3': 'heating',
                'sub_metering_other': 'other'
            }
        )
        return df
