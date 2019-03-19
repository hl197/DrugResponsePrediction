import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
import dataset_utils
from scipy import interp
from collections import defaultdict
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
BATCH_SIZE = 24
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
        # kernel_regularizer=regularizers.l1(0.0012) #0.001
        )(inputs)

    drug_input = Input(shape=(2,), name='drug_input')
    x = keras.layers.concatenate([pathways, drug_input])

    predictions = Dense(1, activation='sigmoid', kernel_initializer='glorot_normal')(x)

    model = Model(inputs=[inputs, drug_input], outputs=predictions)

    # Configure a model for categorical classification.
    model.compile(optimizer=tf.train.AdamOptimizer(LR),
                  loss=keras.losses.binary_crossentropy,
                  metrics=[keras.metrics.binary_accuracy])
    return model


# Zero out weights that do not correspond to relation
class ZeroWeights(keras.callbacks.Callback):
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


def get_edges(top_sets=None):
    """
    Reads in the adjacency matrix between genes and gene sets.
    :return: adjacency matrix
    """
    if top_sets:
        df = dataset_utils.top_gene_set_connections(top_sets)
    else:
        df = pd.read_csv("connections_1.csv", index_col=0)
    return df.values


def run():
    gene_set_mapping = {}
    with open("gene_set_mapping.txt", "r") as f:
        for line in f:
            tokens = line.split()
            gene_set_mapping[int(tokens[0])] = tokens[1]

    all_auc = defaultdict(list)
    tprs = defaultdict(list)
    aucs = defaultdict(list)
    acc = defaultdict(float)
    pos = ['c', 'l', 't', 'cl']
    mean_fpr = np.linspace(0, 1, 100)

    for j in range(1):
        c_train_sets, c_test_sets, c_val_sets = dataset_utils.kfold_train_test_sets('../c_rnaseq_scaled_symbols.csv', seed=j*j)
        l_train_sets, l_test_sets, l_val_sets = dataset_utils.kfold_train_test_sets('../l_rnaseq_scaled_symbols.csv', seed=j*j)
        t_train_sets, t_test_sets, t_val_sets = dataset_utils.kfold_train_test_sets('../t_rnaseq_scaled_symbols.csv', seed=j*j)

        cl_test = pd.read_csv('../cell_lines_scaled_symbols.csv', index_col=0)
        cl_test = cl_test.fillna(0)
        cl_test = np.array(cl_test)

        cl_test_data, cl_test_drug, cl_test_labels = cl_test[:, :-3], cl_test[:, -3:-1], cl_test[:, -1]

        train_sets = []
        val_sets = []
        for i in range(5):
            train_sets.append(pd.concat([c_train_sets[i], l_train_sets[i], t_train_sets[i]]))
            val_sets.append(pd.concat([c_val_sets[i], l_val_sets[i], t_val_sets[i]]))

        t = get_edges("noreg_top_sets.txt")

        for k in range(5):
            train, val, c_test, l_test, t_test = train_sets[k], val_sets[k], c_test_sets[k], l_test_sets[k], t_test_sets[k]
            train, val, c_test, l_test, t_test = np.array(train), np.array(val), np.array(c_test), np.array(l_test), np.array(t_test)

            # using numpy arrays
            train_data, train_drug, train_labels = train[:, :-3], train[:, -3:-1], train[:, -1]
            val_data, val_drug, val_labels = val[:, :-3], val[:, -3:-1], val[:, -1]
            c_test_data, c_test_drug, c_test_labels = c_test[:, :-3], c_test[:, -3:-1], c_test[:, -1]
            l_test_data, l_test_drug, l_test_labels = l_test[:, :-3], l_test[:, -3:-1], l_test[:, -1]
            t_test_data, t_test_drug, t_test_labels = t_test[:, :-3], t_test[:, -3:-1], t_test[:, -1]

            callbacks = [
                # Interrupt training if `val_acc` stops improving for over K epochs
                keras.callbacks.EarlyStopping(patience=K, monitor='val_binary_accuracy'),
                ZeroWeights(t),
            ]
            model = make_base_model(t.shape[0], t.shape[1])
            model.fit({'gene_input': train_data, 'drug_input': train_drug}, train_labels, epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0,
                      validation_data=([val_data, val_drug], val_labels))

            test_data = [c_test_data, l_test_data, t_test_data, cl_test_data]
            test_drug = [c_test_drug, l_test_drug, t_test_drug, cl_test_drug]
            test_labels = [c_test_labels, l_test_labels, t_test_labels, cl_test_labels]

            for i in range(4):
                y_pred = model.predict([test_data[i], test_drug[i]]).flatten()

                pred = y_pred > 0.5
                truth = test_labels[i] > 0.5
                all_auc[pos[i]].append(roc_auc_score(test_labels[i].astype(np.float32), pred))

                fpr, tpr, thresholds = roc_curve(truth, y_pred)
                tprs[pos[i]].append(interp(mean_fpr, fpr, tpr))
                tprs[pos[i]][-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs[pos[i]].append(roc_auc)
                acc[pos[i]] += accuracy_score(truth, pred)

    with open("noreg_top_multi_trast_out_NN.txt", "w") as f:
        f.write("tprs\n")
        for arr in tprs['t']:
            for val in arr:
                f.write(str(val) + "\t")
            f.write("\n")
        f.write("aucs\n")
        for val in aucs['t']:
            f.write(str(val) + "\t")
        f.write("\n")

    with open("noreg_top_multi_combo_out_NN.txt", "w") as f:
        f.write("tprs\n")
        for arr in tprs['c']:
            for val in arr:
                f.write(str(val) + "\t")
            f.write("\n")
        f.write("aucs\n")
        for val in aucs['c']:
            f.write(str(val) + "\t")
        f.write("\n")

    with open("noreg_top_multi_lap_out_NN.txt", "w") as f:
        f.write("tprs\n")
        for arr in tprs['l']:
            for val in arr:
                f.write(str(val) + "\t")
            f.write("\n")
        f.write("aucs\n")
        for val in aucs['l']:
            f.write(str(val) + "\t")
        f.write("\n")

    with open("noreg_top_multi_cl_lap_NN.txt", "w") as f:
        f.write("tprs\n")
        for arr in tprs['cl']:
            for val in arr:
                f.write(str(val) + "\t")
            f.write("\n")
        f.write("aucs\n")
        for val in aucs['cl']:
            f.write(str(val) + "\t")
        f.write("\n")

    for i in range(4):
        print(pos[i] + " accuracy:")
        print(acc[pos[i]]/15.)

run()