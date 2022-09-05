"""
Portions of this page are reproduced from work created and shared by Google and used
according to terms described in the Creative Commons 4.0 Attribution License.
"""
import pandas as pd


def plot_losses(history):
    """ Plot loss and RMSE with given history. """

    losses = pd.DataFrame(history.history)

    losses[['loss', 'val_loss']].plot(figsize=(10,8), xlabel="Epoch", ylabel="Loss", title="Model loss")
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

def normalize_datasets(train_df, test_df, val_df):
    """ Normalize datasets. """

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, test_df, val_df