import numpy as np
import pandas as pd
from source import features


def test_rsi_buy_indicator_is_correct():
    test_rsi_col = pd.Series([np.NaN, 40.01, 20.02, 20.03, 40.04, 35.05, 20.06, 10.07, 100.08])

    expected_output = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1])
    actual_output = features.rsi_buy_indicator(test_rsi_col)

    np.testing.assert_array_equal(expected_output, actual_output)
