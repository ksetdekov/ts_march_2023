# import seasonal_decompose
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import periodogram
# import shapiro from scipy.stats
from scipy.stats import shapiro
# use qqplot from statsmodels.graphics.gofplots
from statsmodels.graphics.gofplots import qqplot
#  import statsmodels.graphics.tsaplots.plot_acf and statsmodels.graphics.tsaplots.plot_pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose



class AggTsCals():
    def __init__(self, ts: pd.Series) -> None:
        """initialize

        Args:
            ts (pd.Series): временной ряд для анализа
        """
        self.ts = ts
        self.stl_decomposition_results = None

    def __str__(self) -> str:
        return str(self.ts)

    def agg_by_day(self):
        """group by minute and hour inside a day and calculate the aggregated (mean value, std, lower and upper bounds of a 99% quantile) values for each minute and hour
        """
        df = self.ts.copy()
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        df_agg = df.groupby(['hour', 'minute'])['value'].agg(['mean', 'std'])
        # estimate the quatiles of the empirical distribution of the mean values
        df_agg['lower_bound'] = df_agg['mean'] - 2.58 * df_agg['std']
        df_agg['upper_bound'] = df_agg['mean'] + 2.58 * df_agg['std']
        return df_agg

    def plot_results(self, df: pd.DataFrame, title: str, x_label: str, y_label: str, figsize: tuple = (15, 5)) -> None:
        """plot the results

        Args:
            df (pd.DataFrame): aggregated values for a given time window
            title (str): plot title
            x_label (str): x axis label
            y_label (str): y axis label
            figsize (tuple, optional): plot size. Defaults to (15, 5).
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # solve the error ValueError: setting an array element with a sequence.
        df['hour'] = df.index.get_level_values('hour')
        df['minute'] = df.index.get_level_values('minute')
        # create a time column that is a numeric value of minutes from the beginning of the day
        df['time'] = df['hour'] * 60 + df['minute']
        # plot the results
        sns.lineplot(data=df, x='time', y='mean', ax=ax, label='mean')
        sns.lineplot(data=df, x='time', y='lower_bound',
                     ax=ax, label='lower bound')
        sns.lineplot(data=df, x='time', y='upper_bound',
                     ax=ax, label='upper bound')
        # color the area between the lower and upper bounds with low transparency
        ax.fill_between(df['time'], df['lower_bound'],
                        df['upper_bound'], alpha=0.1)
        plt.show()

    # add a method that will test how stable the seasonality for a window of one day
    def test_seasonality_stability(self, df: pd.DataFrame, window_size: int, figsize: tuple = (15, 5)) -> None:
        """test how stable the seasonality for a window of one day

        Args:
            df (pd.DataFrame): aggregated values for a given time window
            window_size (int): window size in minutes
            figsize (tuple, optional): plot size. Defaults to (15, 5).
        """
        # create a time column that is a numeric value of minutes from the beginning of the day
        df['time'] = df.index.get_level_values(
            'hour') * 60 + df.index.get_level_values('minute')
        # create a column with the number of the window
        df['window'] = df['time'] // window_size
        # calculate the aggregated (mean value, std, lower and upper bounds of a 99% quantile) values for each window
        df_agg = df.groupby('window')[
            'mean', 'lower_bound', 'upper_bound'].agg(['mean', 'std'])
        # plot the results
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title('Aggregated values for a day')
        ax.set_xlabel('time (minutes from the beginning of the day)')
        ax.set_ylabel('value')
        # fix error ValueError: Data must be 1-dimensional
        df_agg = df_agg.reset_index()
        # plot the results
        sns.lineplot(data=df_agg, x='window', y='mean', ax=ax, label='mean')
        sns.lineplot(data=df_agg, x='window', y='lower_bound',
                     ax=ax, label='lower bound')
        sns.lineplot(data=df_agg, x='window', y='upper_bound',
                     ax=ax, label='upper bound')
        # color the area between the lower and upper bounds with low transparency
        ax.fill_between(df_agg['window'], df_agg['lower_bound'],
                        df_agg['upper_bound'], alpha=0.1)
        plt.show()

    # add a method that applies an stl decomposition with dayly, weekly and monthly seasonality to the time series that removes the trend from the time series
    def stl_decomposition(self, figsize: tuple = (15, 5)) -> None:
        """apply an stl decomposition with dayly, weekly and monthly seasonality to the time series that removes the trend from the time series

        Args:
            figsize (tuple, optional): plot size. Defaults to (15, 5).
        """
        # use self.ts
        # apply an stl decomposition with dayly, weekly and monthly seasonality to the time series that removes the trend from the time series
        result = seasonal_decompose(
            self.ts['value'], model='additive', period=1440*7)
        # plot the results on different subplots stacked vertically
        _, axes = plt.subplots(4, 1, figsize=figsize)
        result.observed.plot(ax=axes[0], legend=False)
        axes[0].set_ylabel('Observed')
        result.trend.plot(ax=axes[1], legend=False)
        axes[1].set_ylabel('Trend')
        result.seasonal.plot(ax=axes[2], legend=False)
        axes[2].set_ylabel('Seasonal')
        result.resid.plot(ax=axes[3], legend=False)
        axes[3].set_ylabel('Residual')
        plt.show()

        # return the results of the stl decomposition
        # write results into stl_decomposition_results attribute
        self.stl_decomposition_results = result
        return result

    # add a method to check pacf and acf plots of the residuals from self.stl_decomposition_results

    def check_residuals(self, figsize: tuple = (15, 5)) -> None:
        """check pacf and acf plots of the residuals from self.stl_decomposition_results

        Args:
            figsize (tuple, optional): plot size. Defaults to (15, 5).
        """
        # use self.stl_decomposition_results
        # plot the pacf and acf plots of the residuals
        # check if self.stl_decomposition_results is not None
        if self.stl_decomposition_results is None:
            raise ValueError(
                'self.stl_decomposition_results is None. Run self.stl_decomposition() first.')
        # drop na values
        decomposed_nona = self.stl_decomposition_results.resid.dropna()
        # plot the pacf and acf plots of the residuals
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        # use pacf and acf plots from statsmodels to plot the pacf and acf plots of the residuals
        plot_pacf(decomposed_nona, ax=axes[0])
        plot_acf(decomposed_nona, ax=axes[1])
        plt.show()

    # add a method to check if the residuals are normally distributed, using the shapiro test
    def check_residuals_normality(self) -> None:
        """check if the residuals are normally distributed, using the shapiro test

        """
        # use self.stl_decomposition_results
        # check if self.stl_decomposition_results is not None
        if self.stl_decomposition_results is None:
            raise ValueError(
                'self.stl_decomposition_results is None. Run self.stl_decomposition() first.')
        # drop na values
        decomposed_nona = self.stl_decomposition_results.resid.dropna()
        # check if the residuals are normally distributed, using the shapiro test
        # use the shapiro test from scipy.stats
        stat, p = shapiro(decomposed_nona)
        # print the results
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)')
        else:
            print('Sample does not look Gaussian (reject H0)')
        # plot the results

        sns.displot(decomposed_nona)
        plt.show()

        # plot qq plot
        # use qqplot from statsmodels.graphics.gofplots
        qqplot(decomposed_nona, line='s')
        plt.show()

    # add a class methord to ckeck the periodogram same as check_periodogram function
    @classmethod
    def check_periodogram(cls, df, title):
        """check periodogram of the residuals

        """
        f, ppx = periodogram(df.dropna())

        result_list = list(sorted(zip(f, ppx), key=lambda x: x[1]))[-50:]
        # plot the result_list with first term being inverse of the frequency and second term being the power
        plt.plot([1/x[0] for x in result_list], [x[1] for x in result_list])
        # add x and y labels
        # add plot title
        plt.title(title)
        plt.xlabel('Period')
        plt.ylabel('Power')
        plt.show()

    # add a method to check the periodogram of the residuals

    def check_periodogram_residuals(self):
        """check the periodogram of the residuals

        """
        # use self.stl_decomposition_results
        # check if self.stl_decomposition_results is not None
        if self.stl_decomposition_results is None:
            raise ValueError(
                'self.stl_decomposition_results is None. Run self.stl_decomposition() first.')
        # drop na values
        decomposed_nona = self.stl_decomposition_results.resid.dropna()
        # check the periodogram of the residuals
        self.check_periodogram(decomposed_nona, 'Periodogram of the residuals')

    # add a method to check the periodogram of the time series
    def check_periodogram_ts(self):
        """check the periodogram of the time series

        """
        # use self.ts
        # check the periodogram of the time series
        self.check_periodogram(
            self.ts['value'], 'Periodogram of the time series')
