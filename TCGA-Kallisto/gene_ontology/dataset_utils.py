import pandas as pd
import numpy as np
import random


def oversample(df, x=2):
    """
    Oversamples the minority class (examples with no drug response label).
    :param df: Input dataframe
    :param x: Factor to oversample by
    :return: Oversampled dataframe
    """
    new_df = pd.DataFrame(columns=df.columns)
    i, j = 0, 0
    while i < df.shape[0]:
        new_df.loc[j, :] = df.iloc[i, :]
        j += 1
        if not df.iloc[i, -1]:
            for _ in range(x-1):
                new_df.loc[j, :] = df.iloc[i, :]
                j += 1
        i += 1
    return new_df


def divide_data(filename, numpy=False, seed=17):
    """
    Divides the data in the specified csv file into 80% training, 10% validation, and 10% testing.
    :param filename: name of csv file.
    :param numpy: whether the ouput should be returned as a numpy array.
    :param seed: seed for random to replicate results
    :return: train, validation, and testing dataframes / numpy arrays
    """
    random.seed(seed)
    df = pd.read_csv(filename)
    df = df.fillna(0)
    # divides data into 80% training, 10% validation, 10% testing
    n_examples = df.shape[0]
    idx = [i for i in range(n_examples)]
    random.shuffle(idx)

    train = df.iloc[idx[:int(0.8 * n_examples)], :]
    val = df.iloc[idx[int(0.8 * n_examples):int(0.9 * n_examples)], :]
    test = df.iloc[idx[int(0.9 * n_examples):], :]

    if numpy:
        return np.array(train), np.array(val), np.array(test)

    return train, val, test
