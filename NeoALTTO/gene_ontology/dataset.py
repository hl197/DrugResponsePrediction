import pandas as pd
import numpy as np
import random


def oversample(df, x=2):
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


def divide_data(csv, numpy=False, seed=17):
    random.seed(seed)
    df = pd.read_csv(csv)
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
