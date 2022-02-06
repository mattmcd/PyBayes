import tensorflow as tf
import pandas as pd
import os
import logging
from dataclasses import dataclass


logger = logging.getLogger()


@dataclass
class DataReader:
    cache_loc: str = os.path.join(os.environ.get('MDA_DATA_DIR'), 'banknote.csv')
    data_url: str = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'

    def read_data(self, clean_cache=False):
        if clean_cache:
            logger.info('Removing cache file')
            os.remove(self.cache_loc)
        cache_exists = os.path.isfile(self.cache_loc)
        if cache_exists:
            logger.info(f'Using cached data {self.cache_loc}')
            df = pd.read_csv(self.cache_loc)
        else:
            logger.info(f'Using original data from {self.data_url}')
            df = pd.read_csv(self.data_url, names=['variance', 'skewness', 'curtosis', 'entropy', 'class'])
            df.to_csv(self.cache_loc, index=False)
        return df


class TrainDataReader(DataReader):
    frac: float = 0.8
    random_state: int = 42

    def read_data(self, clean_cache=False):
        df = super(TrainDataReader, self).read_data(clean_cache)
        # Sample reproducibly
        return df.sample(frac=self.frac, random_state=self.random_state)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        self.dense1 = tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(2,))

    def call(self, inputs):
        return self.dense1(inputs)


if __name__ == '__main__':
    train_reader = TrainDataReader()
    df = train_reader.read_data()
    model = MyModel()
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(
        df.loc[:, ['skewness', 'entropy']], df.loc[:, ['class']],
        epochs=20, batch_size=16, validation_split=0.2
    )
