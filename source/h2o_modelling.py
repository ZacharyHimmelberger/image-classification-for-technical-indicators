"""
This script allows the user to build models using h2o. 

We have functions to read and concatenate parquet files, convert pandas dataframes to h2o dataframes, 
select an outcome variable to classify, and train the model using h2o.
"""

import h2o
from h2o.automl import H2OAutoML
import pandas as pd


def parquet_to_h2o(*args):
    """Reads multiple parquet files and converts them to a single h2o dataframe.

    Args:
        *args (str): Parquet files to be read.

    Returns:
        h2o_frame (h2o.frame.H2OFrame): An h2o dataframe containing each of the parquet files.
    """
    h2o.init()

    list_of_dfs = []

    for file in args:
        list_of_dfs.append(pd.read_parquet(file))

    dfs = pd.concat(list_of_dfs, ignore_index=True)
    h2o_frame = h2o.H2OFrame(dfs)

    return h2o_frame


def prepare_h2o_df(df, outcome, to_factor=True):
    """Converts an h2o dataframe with a specific outcome variable to an outcome and set of predictors.
        The outcome variable is converted to a factor by default. 

    Args:
        df (h2o.frame.H2OFrame): An h2o dataframe.
        outcome (str): The name of the outcome variable.
        to_factor (logical): A logical indicating whether the outcome variable should be transformed to a factor. 
                            Defaults to True. 

    Returns:
        tuple: A tuple with two elements. The first is the name of the outcome variable and the second is a list of 
        predictor variables.
    """
    h2o.init()

    if to_factor:
        df[outcome].asfactor()
    
    y = outcome
    x = df.columns.remove(y)
    
    return (y, x)


def train_and_save(df, outcome, predictors, save_path, max_models=100, max_runtime_min=5):
    """Trains and saves an h2o model using H2OAutoML.
    
    Args:
        df (h2o.frame.H2OFrame): The data used to train the model.
        outcome (str): Name of the outcome variable. 
        predictors (str): List of predictor variables.
        save_path (str): File path to save the best model.
        max_models (int): Maximum number of models to create. Defaults to 100.
        max_runtime_secs (int): Maximum run time in minutes. Defaults to 5.
    """
    h2o.init()

    aml = H2OAutoML(max_models=max_models, max_runtime_secs=60*max_runtime_min)
    aml.train(x=predictors, y=outcome, training_frame=df)

    SAVE_PATH = save_path
    h2o.save_model(aml.leader, path=SAVE_PATH)

    lb = aml.leaderboard

    return (lb)
