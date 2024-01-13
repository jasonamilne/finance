from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
import datetime

"""
Code adopted from Algorithmic Trading with Python Quantitative Methods
and Strategy Development by Chris Nolan
---------------------------------------------------------------------
Skills: Quantitative Finance, Statistics, Mathematics, NumPy, Pandas,
Object Oriented Programming, Test Driven Development.
"""


class Performance:

    def base_return_series(self, prices: List[float]) -> List[float]:
        """
        Description: Stock price percent change over time with base python.

        Formula: r_t = (y_t - y_{t-1}) / y_{t-1} = y_t/ y_{t-1} - 1

        :param prices: stock market prices at a particular date and time.
        :return: a list of percentage returns for the stock.
        """
        return_series = [None]  # list object filled with None.

        # iterate over list of prices and calculate returns.
        for i in range(1, len(prices)):
            return_series.append((prices[i] / prices[i-1]) - 1)

        return return_series

    def return_series(self, series: pd.Series) -> pd.Series:
        """
        Description: Stock price percent change over time with Pandas.

        Formula: r_t = (y_t - y_{t-1}) / y_{t-1} = y_t/ y_{t-1} - 1

        :param series: stock market prices at a particular date and time.
        :return: a series of percentage returns for the stock.
        """
        return series / series.shift(1, axis=0) - 1

    def return_log_series(self, series: pd.Series) -> pd.Series:
        """
        Description: Log transform of stock percent change over time.

        Formula: r_t = log[(y_t - y_{t-1}) / y_{t-1}] = log[y_t/ y_{t-1} - 1]
        :param series:  stock market prices at a particular date and time.
        :return: a series of log of the percentage returns for the stock.
        """
        return pd.Series(np.log(series / series.shift(1, axis=0)))

    def get_dt_years(self, series: pd.Series) -> float:
        """
        Description:

        Formula:
        :param series:
        :return:
        """
        start_date = series.index[0]
        end_date = series.index[-1]
        dt_years = (end_date - start_date).days / 365.25
        return dt_years

    def get_entries_per_year(self,return_series: pd.Series, dt_years: float) -> float:
        return return_series.shape[0] / dt_years

    def annualized_volatility(self, return_series: pd.Series) -> float:
        """
        Description:

        Formula:
        :param return_series:
        :return:
        """
        dt_years = self.get_dt_years(return_series)
        entries_per_year = self.get_entries_per_year(return_series, dt_years)
        annualized_volatility = return_series.std() * np.sqrt(entries_per_year)
        return annualized_volatility

    def cagr(self, series: pd.Series) -> float:
        """
        Description:

        Formula:
        :param series:
        :return:
        """
        value_factor = series.iloc[-1] / series.iloc[0]
        dt_years = self.get_dt_years(series)
        return (value_factor ** (1/dt_years)) - 1

    def sharpe_ratio(self, price_series: pd.Series, benchmark_rate: float = 0) -> float:
        """
        Description:

        Formula:
        :param price_series:
        :param benchmark_rate:
        :return:
        """
        cagr = self.cagr(price_series)
        return_series = self.return_series(price_series)
        volatility = self.annualized_volatility(return_series)
        sharpe_ratio = (cagr - benchmark_rate) / volatility
        return sharpe_ratio

    def annualized_downside_deviation(self, return_series: pd.Series, benchmark_rate: float = 0) -> float:
        """
        Description:

        Formula:
        :param return_series:
        :param benchmark_rate:
        :return:
        """
        dt_years = self.get_dt_years(return_series)
        entries_per_year = self.get_entries_per_year(return_series, dt_years)

        adjusted_benchmark_rate = ((1+benchmark_rate) ** (1/entries_per_year)) - 1

        downside_series = adjusted_benchmark_rate - return_series
        downside_ss = (downside_series[downside_series > 0]**2).sum()
        denominator = return_series.shape[0] - 1
        downside_deviation = np.sqrt(downside_ss / denominator)

        return downside_deviation * np.sqrt(entries_per_year)

    def sortino_ratio(self, price_series: pd.Series, benchmark_rate: float=0):
        """
        Description:

        Formula:
        :param price_series:
        :param benchmark_rate:
        :return:
        """
        cagr = self.cagr(price_series)
        return_series = self.return_series(price_series)
        downside_deviation = self.annualized_downside_deviation(return_series)
        return (cagr - benchmark_rate) / downside_deviation