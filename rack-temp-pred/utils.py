"""
Portions of this page are reproduced from work created and shared by Google and used
according to terms described in the Creative Commons 4.0 Attribution License.
"""
from math import sqrt

import pandas as pd
import tensorflow as tf
from scipy.signal import savgol_filter

from window import WindowGenerator



def plot_losses(history):
    """ Plot loss and RMSE with given history. """

    history.history['loss'] = list(map(inverse_transform_loss, history.history['loss']))
    history.history['val_loss'] = list(map(inverse_transform_loss, history.history['val_loss']))
    history.history['root_mean_squared_error'] = list(map(inverse_transform_rmse, history.history['root_mean_squared_error']))
    history.history['val_root_mean_squared_error'] = list(map(inverse_transform_rmse, history.history['val_root_mean_squared_error']))

    losses = pd.DataFrame(history.history)


    # losses[['loss', 'val_loss']].plot(figsize=(10,8), xlabel="Epoch", ylabel="Loss", title="Model loss")
    losses[['root_mean_squared_error', 'val_root_mean_squared_error']].plot(figsize=(10,8), xlabel="Epoch", ylabel="RMSE", title="RMSE")

def train_test_val_split(df, test_size=0.1, val_size=0.2):
    """ Split dataset into train, test and validation.

    Returns:
    - train_df
    - test_df
    - val_df
    """

    n = len(df)
    train_limit = 1 - test_size - val_size
    val_limit = 1 - test_size

    train_df = df[0:int(n*train_limit)]
    val_df = df[int(n*train_limit):int(n*val_limit)]
    test_df = df[int(n*val_limit):]

    # num_features = df.shape[1]
    # print(num_features)

    return train_df, test_df, val_df

def normalize_datasets(dataset:pd.DataFrame, outlet_temp_cols:list[str], inlet_temp_cols:list[str]):
    """ Normalize datasets. """

    # normalize fan speed
    dataset['inlet_fan_speed'] = dataset['inlet_fan_speed'] / 100
    dataset['outlet_fan_speed'] = dataset['outlet_fan_speed'] / 100

    # normalize temperatures based on ASHRAE recommendations
    min_temp = dataset[inlet_temp_cols].min().min()
    max_temp = dataset[inlet_temp_cols].max().max()
    dataset[inlet_temp_cols] = (dataset[inlet_temp_cols] - min_temp) / (max_temp - min_temp)
    dataset[outlet_temp_cols] = (dataset[outlet_temp_cols] - min_temp) / (max_temp - min_temp)

    # apply Savitsky-Golay filter
    dataset[outlet_temp_cols] = savgol_filter(dataset[outlet_temp_cols], window_length=7, polyorder=3, mode="nearest", axis=0)
    dataset[inlet_temp_cols] = savgol_filter(dataset[inlet_temp_cols], window_length=7, polyorder=3, mode="nearest", axis=0)

    return dataset 

def inverse_transform_loss(x, _min=9.08, _max=33.12):
    """ Inverse transform loss Min-Max Scaler. """

    return (x * (_max - _min) + _min)

def inverse_transform_rmse(x, _min=9.08, _max=33.12):
    """ Inverse transform RMSE Min-Max Scaler. """

    return sqrt((x**2) * (_max - _min) + _min)

def train_case(case_df, model, label_cols, in_width, out_steps, max_epochs, batch_size=32):
    """ Train model for given case df. Currently uses 20% of data for validation and 80% for training. """
    
    train_df, test_df, val_df = train_test_val_split(case_df, test_size=0, val_size=0.2)

    window = WindowGenerator(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        input_width=in_width,
        label_width=out_steps,
        shift=out_steps,
        label_columns=label_cols,
        batch_size=batch_size
    )

    history = model.fit(
        window.train,
        epochs=max_epochs,
        validation_data=window.val,
        verbose=0
        # callbacks=[early_stopping]
    )

    return window, history

