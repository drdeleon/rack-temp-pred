"""
Portions of this page are reproduced from work created and shared by Google and used
according to terms described in the Creative Commons 4.0 Attribution License.
"""

import tensorflow as tf


def compile_and_fit(model, window, patience=2, max_epochs=20):
    """ Training procedure. """

    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
    #                                                     patience=patience,
    #                                                     mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.RootMeanSquaredError()])

    history = model.fit(
        window.train, epochs=max_epochs,
        validation_data=window.val,
        # callbacks=[early_stopping]
    )

    return history

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

    train_df = df[0:int(n*train_limit)]                 # 70% for training
    val_df = df[int(n*train_limit):int(n*val_limit)]    # 20% for validation
    test_df = df[int(n*val_limit):]                     # 10% for testing

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