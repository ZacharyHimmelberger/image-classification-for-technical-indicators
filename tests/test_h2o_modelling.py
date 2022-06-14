import pandas as pd

from source import h2o_modelling
from h2o.exceptions import H2OTypeError

def test_parquet_to_h2o():
    actual_output = h2o_modelling.parquet_to_h2o('tests/parquet_files/no_buy.parquet.gzip', 'tests/parquet_files/buy.parquet.gzip')
    H2OTypeError(actual_output, 'H2OFrame')
    