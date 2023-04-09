import pandas as pd
import numpy as np
import unittest
from create_agg_by_minute import AggTsCals
import datetime

def create_df():
    # create a list of datetime objects
    start = datetime.datetime(2022, 1, 1, 0, 0)
    end = datetime.datetime(2022, 1, 31, 23, 59)
    step = datetime.timedelta(minutes=1)
    result = []
    while start <= end:
        result.append(start)
        start += step
    # create a dataframe with datetime column and a value column
    df = pd.DataFrame(result, columns=['datetime'])
    # add daily seasonality and make the values lower at midnight and higher at noon as a sin function and make data variance higher at midday
    df['value'] = np.sin(df['datetime'].dt.hour * 2 * np.pi / 24)
    # add a random noise to the values that is higher at midday
    df['value'] = df['value'] + np.random.normal(0, 0.1, size=len(df)) * np.sin(df['datetime'].dt.hour * 2 * np.pi / 24)
    # add a random trend changind every 40 days
    df['value'] = df['value'] + np.random.normal(0, 0.1, size=len(df)) * np.sin(df['datetime'].dt.day * 2 * np.pi / 40)
    # normalize the values to be between 0 and 1
    df['value'] = (df['value'] - df['value'].min()) / (df['value'].max() - df['value'].min())
    return df

df = create_df()


class TestTsClass(unittest.TestCase):
    def test_str_method(self):

        ts = pd.Series([1,2,3])
        exemplai = AggTsCals(ts)
        self.assertEqual(str(exemplai), '0    1\n1    2\n2    3\ndtype: int64')

    # test agg_by_day shape
    def test_agg_by_day(self):
        exemplai = AggTsCals(df)
        self.assertEqual(exemplai.agg_by_day().shape, (1440, 4))

    # test agg_by_day columns
    def test_agg_by_day_columns(self):
        exemplai = AggTsCals(df)
        self.assertEqual(list(exemplai.agg_by_day().columns), ['mean', 'std', 'lower_bound', 'upper_bound'])


if __name__ == '__main__':
    unittest.main()