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
        kernel_regularizer=regularizers.l1(0.0012) #0.001
        )(inputs)
    pathways = Dropout(0.1, seed=1)(pathways)

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


def get_edges():
    """
    Reads in the adjacency matrix between genes and gene sets.
    :return: adjacency matrix
    """
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
    top_sets = defaultdict(float)
    avg_act = defaultdict(float)
    pos = ['cl', 'l']
    mean_fpr = np.linspace(0, 1, 100)

    for j in range(3):
        train_sets, test_sets, val_sets = dataset_utils.kfold_train_test_sets('../cell_lines_scaled_symbols.csv', seed=j)
        clin_test = pd.read_csv('../l_rnaseq_scaled_symbols.csv', index_col=0)
        clin_test = clin_test.fillna(0)
        clin_test = np.array(clin_test)

        clin_test_data, clin_test_drug, clin_test_labels = clin_test[:, :-3], clin_test[:, -3:-1], clin_test[:, -1]

        for k in range(5):
            train, val, test = train_sets[k], val_sets[k], test_sets[k]
            train, val, test = np.array(train), np.array(val), np.array(test)

            # using numpy arrays
            train_data, train_drug, train_labels = train[:, :-3], train[:, -3:-1], train[:, -1]
            val_data, val_drug, val_labels = val[:, :-3], val[:, -3:-1], val[:, -1]
            test_data, test_drug, test_labels = test[:, :-3], test[:, -3:-1], test[:, -1]

            t = get_edges()

            callbacks = [
                # Interrupt training if `val_acc` stops improving for over K epochs
                keras.callbacks.EarlyStopping(patience=K, monitor='val_binary_accuracy'),
                ZeroWeights(t),
            ]
            model = make_base_model(t.shape[0], t.shape[1])
            model.fit({'gene_input': train_data, 'drug_input': train_drug}, train_labels, epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0,
                      validation_data=([val_data, val_drug], val_labels))

            test_data = [test_data, clin_test_data]
            test_drug = [test_drug, clin_test_drug]
            test_labels = [test_labels, clin_test_labels]

            for i in range(2):
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

            w, _ = model.layers[5].get_weights()
            acts = []
            for l in range(w.shape[0]-2):
                activation = sum(map(abs, w[l]))
                acts.append((l, activation))
            acts.sort(key=lambda x:-x[1])
            for l in range(10):
                gene_set = gene_set_mapping[acts[l][0]]
                top_sets[gene_set] += 1
            for g in acts:
                activation = g[1]
                gene_set = gene_set_mapping[g[0]]
                avg_act[gene_set] += activation

    with open("cell_line_weights.txt", "w") as f:
        for gene_set in avg_act:
            f.write("{}\t{}\t{}\n".format(gene_set, str(avg_act[gene_set]/15.), str(top_sets[gene_set]/15.)))

    with open("cell_line_out_NN.txt", "w") as f:
        f.write("tprs\n")
        for arr in tprs['cl']:
            for val in arr:
                f.write(str(val) + "\t")
            f.write("\n")
        f.write("aucs\n")
        for val in aucs['cl']:
            f.write(str(val) + "\t")
        f.write("\n")

    with open("cl_clin_out_NN.txt", "w") as f:
        f.write("tprs\n")
        for arr in tprs['l']:
            for val in arr:
                f.write(str(val) + "\t")
            f.write("\n")
        f.write("aucs\n")
        for val in aucs['l']:
            f.write(str(val) + "\t")
        f.write("\n")


run()
