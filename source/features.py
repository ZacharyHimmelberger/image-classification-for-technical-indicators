"""
This script allows the user to add technical indicators to a dataset that contains open, high, low, and close values for firms. 

We have functions to compute the MACD, RSI, and BB buy indicators. 
"""

import numpy as np
import pandas as pd
import pandas_ta as ta


def build_indicator(df, indicator, window_size):
    """Computes a buy indicator.

    Args:
        df (pd.DataFrame): Dataframe that contains open, high, low, close prices.
        indicator (str: ["bbands", "macd", "rsi"]): Buy indicator to be calculated.
        window_size (int): Number of prior days to use in the calculation.

    Returns:
        pd.Series: [description]
    """
    tickers = df.index.unique('firm')
    return pd.concat(
        [
            df.query("firm == @ticker")
            .copy()
            .ta.__getattribute__(indicator)(length=window_size)
            for ticker in tickers
        ]
    )


def rsi_buy_indicator(rsi_col):
    """Adds a column that contains a 1 or 0 to indicate whether an Relative Strength Index (rsi) buy signal was reached

    Args:
        rsi_col (pd.Series): Series is expected to contain rsi signals

    Returns:
        pd.Series: A series of 1's and 0's to indicate whether a buy threshold was reached
    """
    temp_col = (rsi_col.fillna(1000) >= 30).astype(int).diff()
    new_col = np.where((temp_col == 1 & rsi_col.isnull()), 0, temp_col)
    pd.Series(new_col).replace(to_replace=[np.NaN, -1], value=0, inplace=True)

    return new_col


def bb_buy_indicator(bb_lower_band_col, close_col):
    """Adds a column that contains a 1 or 0 to indicate whether a Bollinger Bands (bbands) buy signal was reached

    Args:
        rsi_col (pd.Series): Series is expected to contain bband signals

    Returns:
        pd.Series: A series of 1's and 0's to indicate whether a buy threshold was reached
    """
    temp_col = (close_col < bb_lower_band_col.fillna(-1)).astype(int).diff()
    new_col = np.where((temp_col == 1 & close_col.isnull()), 0, temp_col)
    pd.Series(new_col).replace(to_replace=[np.NaN, -1], value=0, inplace=True)

    return new_col


def macd_buy_indicator(macd_signal_col, macd_col):
    """Adds a column that contains a 1 or 0 to indicate whether a Moving Average Convergence Divergence (macd) buy signal was reached

    Args:
        rsi_col (pd.Series): Series is expected to contain macd signals

    Returns:
        pd.Series: A series of 1's and 0's to indicate whether a buy threshold was reached
    """
    new_col = (macd_col > macd_signal_col).astype(int).diff()
    pd.Series(new_col).replace(to_replace=[np.NaN, -1], value=0, inplace=True)

    return new_col


def compute_buy_nobuy(df, fn):
    """
    TODO: refactor the three buy functions
    """
    df['temp_name'] = df.apply(fn)
    return df['temp_name']
