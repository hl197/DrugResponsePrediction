import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from dataset import divide_data
np.random.seed(7)

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers
tf.set_random_seed(1)


# Hyper-parameters
EPOCH = 500
BATCH_SIZE = 64
LR = 0.001
K = 5


def make_base_model(n_input, n_hidden):
    """
    Creates the keras neural network.
    Architecture: genes -> gene sets -> hidden layer -> output; drugs -> hidden layer -> output
    :param n_input: number of genes
    :param n_hidden: number of gene sets
    :return: keras neural net
    """
    inputs = Input(shape=(n_input,), name='gene_input')
    pathways = Dense(
        n_hidden,
        activation='relu',
        kernel_initializer='glorot_normal',
        kernel_regularizer=regularizers.l1(0.01)
        )(inputs)
    pathways = Dropout(0.25, seed=1)(pathways)

    drug_input = Input(shape=(4,), name='drug_input')
    x = keras.layers.concatenate([pathways, drug_input])
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25, seed=1)(x)

    predictions = Dense(1, activation='sigmoid',
                        kernel_initializer='glorot_normal')(x)

    model = Model(inputs=[inputs, drug_input], outputs=predictions)

    # Configure a model for categorical classification.
    model.compile(optimizer=tf.train.AdamOptimizer(LR),
                  loss=keras.losses.binary_crossentropy,
                  metrics=[keras.metrics.binary_accuracy])
    return model


class ZeroWeights(keras.callbacks.Callback):
    """
    Callback that zeros out weights between first and second layer for genes that are not part of the specific gene set.
    """
    def __init__(self, t):
        super(ZeroWeights, self).__init__()
        self.t = t

    def on_train_begin(self, logs=None):
        self.zero_weights()

    def on_batch_end(self, batch, logs=None):
        self.zero_weights()

    def zero_weights(self):
        w, b = self.model.layers[1].get_weights()
        self.model.layers[1].set_weights([w * self.t, b])


def get_edges():
    """
    Reads in the adjacency matrix between genes and gene sets.
    :return: adjacency matrix
    """
    df = pd.read_csv("connections.csv", index_col=0)
    return df.values


acc = []
for i in range(5):
    train, val, test = divide_data('../rnaseq_scaled_all_drug.csv', numpy=True, seed=i)

    train_data, train_drug, train_labels = train[:, 1:-5], train[:, -5:-1], train[:, -1]
    val_data, val_drug, val_labels = val[:, 1:-5], val[:, -5:-1], val[:, -1]
    test_data, test_drug, test_labels = test[:, 1:-5], test[:, -5:-1], test[:, -1]

    majority = sum(test_labels) / len(test_labels)

    t = get_edges()
    print(train_data.shape)

    callbacks = [
        # Interrupt training if `val_acc` stops improving for over K epochs
        keras.callbacks.EarlyStopping(patience=K, monitor='val_binary_accuracy'),
        ZeroWeights(t),
    ]
    model = make_base_model(t.shape[0], t.shape[1])
    model.fit({'gene_input': train_data, 'drug_input': train_drug}, train_labels, epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0,
              validation_data=([val_data, val_drug], val_labels))

    y_pred = model.predict([test_data, test_drug]).flatten()
    test_loss, test_acc = model.evaluate([test_data, test_drug], test_labels, verbose=0)

    pred = y_pred > 0.5
    truth = test_labels > 0.5
    pathways_acc = float((pred == truth).sum()) / float(len(test_labels))
    acc.append(pathways_acc)
    print(y_pred)
    print(truth)
    # print("auc: ", roc_auc_score(test_labels.astype(np.float32), pred))

print("accuracy: ", np.mean(acc), "standard dev: ", np.std(acc))
