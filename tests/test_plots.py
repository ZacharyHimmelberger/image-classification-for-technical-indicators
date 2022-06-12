from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from source import plots


def count_pixels(img_file):
    with Image.open(img_file).convert('L') as img:
        width, height = img.size
    
    return width * height


def test_check_signals():
    test_df = pd.DataFrame({'firm':['APPL','APPL','APPL','GOOG','GOOG','GOOG'], 'buy_signal':[1,1,0,0,0,0]})
    test_group_name = 'firm'
    test_col_name = 'buy_signal'
    test_value = 1
    test_n = 1

    expected_output = pd.DataFrame({'firm':['APPL','APPL','APPL'], 'buy_signal':[1,1,0]})
    actual_output = plots.check_signals(data=test_df, group_name=test_group_name, col_name=test_col_name, value=test_value, n=test_n)

    pd.testing.assert_frame_equal(expected_output, actual_output)


def test_sample_signals():
    test_df = pd.DataFrame({'buy_signal':[np.NaN, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]})
    test_col_name = 'buy_signal'
    test_value = 1.0
    test_n = 2
    test_new_col_name = 'sampled'

    expected_output = pd.DataFrame({'buy_signal':[np.NaN, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], 'sampled':[False, False, True, True, False, False, False]})
    actual_output = plots.sample_signals(test_df, test_col_name, test_value, test_n, test_new_col_name)
    
    pd.testing.assert_frame_equal(expected_output, actual_output)

def test_sample_signals_fast():
    test_df = pd.DataFrame({'buy_signal':[np.NaN, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]})
    test_col_name = 'buy_signal'
    test_value = 1.0
    test_n = 2
    test_new_col_name = 'sampled'

    expected_output = pd.DataFrame({'buy_signal':[np.NaN, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], 'sampled':[False, False, True, True, False, False, False]})
    actual_output = plots.sample_signals_fast(test_df, test_col_name, test_value, test_n, test_new_col_name)
    
    pd.testing.assert_frame_equal(expected_output, actual_output)

def test_sample_groups():
    test_df = pd.DataFrame({'firm':['AAPL','AAPL','AAPL','GOOG','GOOG','GOOG'], 'buy_signal':[1,0,0,1,0,0]})
    test_group_name = 'firm'
    test_col_name = 'buy_signal'
    test_value = 1
    test_n = 1
    test_new_col_name = 'sampled'

    expected_output = pd.DataFrame({'firm':['AAPL','AAPL','AAPL','GOOG','GOOG','GOOG'], 'buy_signal':[1,0,0,1,0,0], 'sampled':[True, False, False, True, False, False]})
    actual_output = plots.sample_groups(test_df, test_group_name, test_col_name, test_value, test_n, test_new_col_name)
    
    pd.testing.assert_frame_equal(expected_output, actual_output)


def test_flatten_image():
    imgpath = './tests/data/test_image.jpg'

    with Image.open(imgpath) as img:
        img_height = img.height
        img_width = img.width

        results = plots.flatten_image(imgpath)

        assert len(results) == img_height * img_width
        assert results.ndim == 1


def test_build_h2o_dataset():
    image_files = ["./tests/data/test_image.jpg",
                   "./tests/data/test_image.jpg"]
    labels = [1, 0]

    num_obs = len(image_files)
    num_features = count_pixels(image_files[0])

    results = plots.build_h2o_dataset(image_files, labels)
    assert isinstance(results, pd.DataFrame)
    assert results.shape[0] == num_obs
    assert results.columns[0] == "label"

    feature_columns = results.columns[1:]
    assert len(feature_columns) == num_features

def test_list_files():
    test_dir_path = "./tests/data/"

    expected_output = ["/workspaces/image-classification-for-technical-indicators/tests/data/test_image.jpg"]
    actual_output = plots.list_files(test_dir_path)

    assert expected_output == actual_output


def test_clear_files(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()

    p = d / "test.txt"
    p.write_text("testing...")

    list_of_files_pre = [f.absolute().as_posix() for f in d.glob("[!.]*") if f.is_file()]

    assert p.read_text() == "testing..."
    assert len(list(tmp_path.iterdir())) == 1 
    assert len(list_of_files_pre) == 1

    tmp_file_path = list_of_files_pre[0][:-8]
    plots.clear_files(tmp_file_path)
    list_of_files_post = [f.absolute().as_posix() for f in d.glob("[!.]*") if f.is_file()]

    assert len(list(d.iterdir())) ==  0
    assert len(list_of_files_post) == 0


@pytest.mark.slow
def test_build_h2o_del_dir(tmp_path):
    d = tmp_path / "sub"  
    d.mkdir()

    dir_path = "./tests/data/"
    save_path = f"{d}/test_df.parquet.gzip"

    plots.build_h2o_del_dir(dir_path, save_path, False)

    assert Path(f"{d}/test_df.parquet.gzip").is_file()
