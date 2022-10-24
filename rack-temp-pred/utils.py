"""
Portions of this page are reproduced from work created and shared by Google and used
according to terms described in the Creative Commons 4.0 Attribution License.
"""
from math import sqrt

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def plot_losses(history):
    """ Plot loss and RMSE with given history. """

    history.history['loss'] = list(map(inverse_transform, history.history['loss']))
    history.history['val_loss'] = list(map(inverse_transform, history.history['val_loss']))
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
    dataset[outlet_temp_cols] = savgol_filter(dataset[outlet_temp_cols], window_length=9, polyorder=3, mode="nearest", axis=0)
    dataset[inlet_temp_cols] = savgol_filter(dataset[inlet_temp_cols], window_length=9, polyorder=3, mode="nearest", axis=0)

    return dataset 

def inverse_transform(x, _min=9.08, _max=33.12):
    """ Inverse transform Min-Max Scaler. """

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


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df: pd.DataFrame, val_df: pd.DataFrame=None, test_df: pd.DataFrame=None,
                 label_columns=None, batch_size=32):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                    for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='outlet_70', max_subplots=3, filename=None):
        inputs, labels = self.example

        fig = plt.figure(figsize=(9, 6), dpi=300)

        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [°C]')
            plt.plot(self.input_indices, inverse_transform(inputs[n, :, plot_col_index]),
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(
                    plot_col, None)
                print("label_col_index", label_col_index)

            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, inverse_transform(labels[n, :, label_col_index]),
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model.predict(inputs)
                
                plt.scatter(self.label_indices, inverse_transform(predictions[n, :, label_col_index]),
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [s]')
        plt.tight_layout()

        if filename != None:
            plt.savefig(filename, dpi=300)


    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size,)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

