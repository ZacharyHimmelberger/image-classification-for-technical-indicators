import pandas as pd
import pytest
import h2o
from h2o.exceptions import H2OTypeError

from source import h2o_modelling

@pytest.fixture(scope="session", autouse=True)
def init_h2o_cluster():
    init_h2o = h2o.init()
    yield init_h2o
    shut_h2o = h2o.shutdown()


def test_parquet_to_h2o(init_h2o_cluster):
    expected_output = pd.DataFrame({
    'label': [0, 0, 0, 1, 1, 1],
    'pixel_0': [253, 244, 253, 253, 244, 253],
    'pixel_1': [253, 229, 253, 253, 229, 253]
    })
    actual_output = h2o_modelling.parquet_to_h2o('tests/parquet_files/no_buy.parquet.gzip', 'tests/parquet_files/buy.parquet.gzip')
    
    H2OTypeError(actual_output, 'H2OFrame')
    pd.testing.assert_frame_equal(expected_output, actual_output.as_data_frame())
