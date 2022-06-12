"""
This script allows the user to randomly select a subset of buy indicators and non-buy indicators, 
create line and OHLC plots, and convert the images to an h2o dataset. 
"""

from pathlib import Path
import random
import re

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from PIL import Image


def check_signals(data, group_name, col_name, value, n):
    """Subsets a DataFrame to only include groups that have a minimum of n
        values in the specified column.

    Args:
        data (pd.DataFrame): A DataFrame to subset. 
        group_name (str): A grouping column in the DataFrame.
        col_name (str): The column in the DataFrame that should be checked.
        value (any): The value to match.
        n (int): The minimum number of times the value can appear in the specified column.

    Returns:
        pd.DataFrame: A subsetted pandas DataFrame. 
    """
    return data.groupby(group_name).filter(lambda x: x[x[col_name] == value].shape[0] >= n)


def sample_signals(data, col_name, value, n, new_col_name, random_state=None):
    """Adds a column to a pandas DataFrame that contains a True value for randomly sampled
        values in a specified column.

    Args:
        data (pd.DataFrame): A DataFrame to sample from.
        col_name (str): The column in the DataFrame that should be sampled.
        value ([type]): The value to sample from. 
        n ([type]): The minimum number of times the value can appear in the specified column.
        new_col_name (str): The name of the newly created column.
        random_state (int, optional): The seed for the random number generator. Defaults to None.

    Returns:
        pd.DataFrame: A pandas DataFrame that contains an added column of boolean values.
    """
    data['row_num'] = list(range(0, len(data.index)))
    potential_choices = list(data['row_num'][data[col_name] == value])

    random.seed(random_state)
    sampled_choices = random.sample(potential_choices, k=n)
    sampled_list = [True if i in sampled_choices else False for i in range(0, len(data.index))]
    data[new_col_name] = sampled_list
    data.drop('row_num', axis=1, inplace=True)

    return data

def sample_signals_fast(data, col_name, value, n, new_col_name, random_state=None):
    """Adds a column to a pandas DataFrame that contains a True value for randomly sampled
        values in a specified column.

    Args:
        data (pd.DataFrame): A DataFrame to sample from.
        col_name (str): The column in the DataFrame that should be sampled.
        value ([type]): The value to sample from. 
        n ([type]): The minimum number of times the value can appear in the specified column.
        new_col_name (str): The name of the newly created column.
        random_state (int, optional): The seed for the random number generator. Defaults to None.

    Returns:
        pd.DataFrame: A pandas DataFrame that contains an added column of boolean values.
    """
    # figure out which rows to sample from
    population_idx = data[col_name] == value
    population_df = data[population_idx]
    
    sample_df = population_df.sample(n = n, random_state = random_state)

    # build the column to indicate which rows has been sampled
    data["sampled"] = False
    data.loc[sample_df.index, new_col_name] = True

    return data

def sample_groups(data, group_name, col_name, value, n, new_col_name, random_state=None):
    """Applies a a pandas groupby() function to the sample_signals() function.  

    Args:
        data (pd.DataFrame): A DataFrame to sample from.
        group_name (str): A grouping column in the DataFrame.
        col_name (str): The column in the DataFrame that should be sampled.
        value ([type]): The value to sample from. 
        n ([type]): The minimum number of times the value can appear in the specified column.
        new_col_name (str): The name of the newly created column.
        random_state (int, optional): The seed for the random number generator. Defaults to None.

    Returns:
        pd.DataFrame: A pandas DataFrame that contains an added column of boolean values.
    """
    return data.groupby(group_name).apply(sample_signals, col_name, value, n, new_col_name, random_state)


def create_line_plot(x, y):
    """Creates a line plot object for prices over a number of dates. 

    Args:
        x (ArrayLike): ArrayLike object passed to the x-axis.
        y (ArrayLike): ArrayLike object passed to the y-axis.

    Returns:
        List[Line2D]: A matplotlib line plot object.
    """
    fig, ax = plt.subplots(figsize=(.3, .3))
    ax.scatter(x, y, marker='.')
    ax.axis('off')
    ax.plot(x, y)

    return fig


def save_plot(fig, savepath):
    """Given a matplotlib plot object, saves the object to a file with a specified name and closes the plot object.

    Args:
        fig (any): A matplotlib plot object to be saved.
        savepath (str): A filepath to specify where the plot will be saved.
    """
    fig.savefig(savepath, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def create_candlestick_plot(data, file):
    """Creates a candlestick plot given open, high, low, close data and returns an image file stored in a 
    specified filepath.

    Args:
        data (pd.DataFrame): A pandas DataFrame that contains columns called "open", "high", "low", and "close". 
        file (str): A filepath for the plot to be saved. 

    Returns:
        image file: An image file stored in a specified filepath.
    """
    mc = mpf.make_marketcolors(up='white', down='grey')
    s = mpf.make_mpf_style(marketcolors=mc)
    fig = mpf.plot(data, type='candle', axisoff=True,
                   style=s, figsize=(.3, .3), savefig=file)

    return fig


def flatten_image(image_path):
    """
    Converts the image file at image_path to a 1-D numpy array

    The array is a grayscale version of the image
    """
    with Image.open(image_path).convert('L') as img:
        return np.array(img.getdata())


def build_h2o_dataset(image_files, labels):
    """
    Returns a DataFrame where the first column is the label and
    subsequent colums are the elements of the flattened image
    """
    imgs = [flatten_image(f) for f in image_files]
    imgs = np.row_stack(imgs)
    _, num_features = imgs.shape

    labels = np.array(labels).astype('int')

    results = np.column_stack([labels, imgs])
    column_names = ["label"] + [f"pixel_{i}" for i in range(num_features)]

    return pd.DataFrame(results, columns=column_names)
    

def plot_sampled(data, sampled_col, indicator, signal, window_size, dir_path, plot_var, plot_type):
    """Given a pandas DataFrame where "time" is a MultiIndex and a specified column, creates a line plot and candlestick
    plot for a specified window size. The plots are then saved to a filepath and the plot
    objects are closed.

    Args:
        data (pd.DataFrame): A pandas DataFrame containing the data to be plotted.
        sampled_col (str): A column of boolean values in the pandas DataFrame, where only True values will be plotted.
        indicator (str): The indicator being plotted, to be used in the naming of the file.
        signal (str): The signal to be used when naming the plot: "buy"  or "nobuy".
        window_size (int): The number of prior rows to be used in the pd.DataFrame.rolling method.
        dir_path (str): The directory to save the files.
        plot_var (str): The column name of the values to be plotted.
        plot_type (str): The type of plot to make: "line" or "candle".
    """
    for idx, window in enumerate(data.rolling(window=window_size)):
        if data[sampled_col][idx] == True: 
            filename = f'{dir_path}/{window.index.get_level_values("firm")[0]}_{indicator}_{signal}_{window.index.get_level_values("time")[0].strftime("%Y-%m-%d")}'

            if plot_type == "line":
                line_plot = create_line_plot(window.index.get_level_values("time"), window[plot_var])
                save_plot(line_plot, f'{filename}_line.jpg')

            elif plot_type == "candle":
                candle_plot = create_candlestick_plot(window.reset_index(level="firm", drop=True), f'{filename}_candle.jpg')
                plt.close(candle_plot)


def list_files(dir_path):
    """Creates a list of files that do not begin with a period.

    Args:
        dir_path (str): The directory to search for files. 
    """
    p = Path(dir_path)

    return [f.absolute().as_posix() for f in p.glob("[!.]*") if f.is_file()]


def clear_files(dir_path):
    """Removes all files from the directory.

    Args:
        dir_path (str): The directory to delete the files.
    """
    p = Path(dir_path)
    [f.unlink() for f in p.glob("[!.]*") if f.is_file()] 


def build_h2o_del_dir(dir_path, save_path, clear_dir=False):
    """Builds an h2o DataFrame and removes all files from the directory.

    Args:
        dir_path (str): The directory to delete the files.
        save_path (str): The directory to save the files.
        clear_dir (bool): Should all files be removed from the dir_path after saving DataFrame? Defaults to False.
    """
    list_of_files = list_files(dir_path)

    if re.search("nobuy", list_of_files[0]):
        signal = 0
    else:
        signal = 1

    h2o_df = build_h2o_dataset(list_of_files, [signal]*len(list_of_files))
    h2o_df["name"] = list_of_files

    h2o_df.to_parquet(save_path)

    if clear_dir:
        clear_files(dir_path) 
