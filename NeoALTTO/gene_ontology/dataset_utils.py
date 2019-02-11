import pandas as pd
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import auc, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.ensemble import VotingClassifier


def oversample(df, label=0, x=2):
    """
    Oversamples the specified label.
    :param df: Input dataframe
    :param label: The label of samples to oversample
    :param x: Factor to oversample by
    :return: Oversampled dataframe
    """
    new_df = pd.DataFrame(columns=df.columns)
    i, j = 0, 0
    while i < df.shape[0]:
        new_df.loc[j, :] = df.iloc[i, :]
        j += 1
        if df.iloc[i, -1] == label:
            for _ in range(x-1):
                new_df.loc[j, :] = df.iloc[i, :]
                j += 1
        i += 1
    return np.array(new_df)


def divide_data(filename, numpy=False, seed=17):
    """
    Divides the data in the specified csv file into 80% training, 10% validation, and 10% testing.
    :param filename: name of csv file.
    :param numpy: whether the ouput should be returned as a numpy array.
    :param seed: seed for random to replicate results
    :return: train, validation, and testing dataframes / numpy arrays
    """
    random.seed(seed)
    df = pd.read_csv(filename, index_col=0)
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


def kfold_train_test_sets(filename):
    train_sets = []
    test_sets = []

    df = pd.read_csv(filename, index_col=0)
    df = df.fillna(0)
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    for train_index, test_index in kf.split(df):
        train_sets.append(df.iloc[train_index, :])
        test_sets.append(df.iloc[test_index, :])
    return train_sets, test_sets


def condense_c6():
    """
    Condenses the C6 gene sets by combining up-regulated and down-regulated sets into one.
    Writes the new sets into c6_condensed.gmt
    :return: None
    """
    mapping = defaultdict(set)
    with open("../../c6.gmt", "r") as f:
        for line in f:
            tokens = line.strip().split()
            name = tokens[0]
            ch = '.'
            while name.endswith("UP") or name.endswith("DN"):
                new_name = ch.join(name.split(ch)[:-1])
                if new_name != '':
                    name = new_name
                if ch == '.':
                    ch = '_'
                else:
                    ch = '.'
            for gene in tokens[2:]:
                mapping[name].add(gene)
    with open("../../c6_condensed.gmt", 'w') as f:
        for gene_set in mapping:
            out = gene_set
            for gene in mapping[gene_set]:
                out += '\t' + gene
            out += '\n'
            f.write(out)


def make_gene_sets():
    """
    Writes the gene symbols in each gene set into its own separate file in the 'gene_sets' directory.
    :return: None
    """
    mapping = defaultdict(set)
    with open("../../c6_condensed.gmt", "r") as f:
        for gene_set in f:
            tokens = gene_set.strip().split()
            for gene in tokens[1:]:
                mapping[tokens[0]].add(gene)

    for gene_set in mapping:
        genes = mapping[gene_set]
        with open("gene_sets/" + gene_set + ".txt", "w") as f_out:
            with open("../NeoALTTO_genes.txt", "r") as f_in:
                for line in f_in:
                    if line.strip() in genes:
                        f_out.write(line)


def plot_roc(tprs, aucs, title='Receiver Operating Characteristic'):
    """
    Plots the average ROC curve with standard deviation from the given true positive rates and area under curves.
    :param tprs: List of lists of true positive curves.
    :param aucs: List of area under curve values.
    :param title: Title of figure.
    :return: None.
    """
    mean_fpr = np.linspace(0, 1, 100)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

    return mean_auc


def plot_confusion_matrix(clf, X_test, y_test, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    :param clf: The classifier.
    :param X_test: Input test set.
    :param y_test: True test labels.
    :param classes: List of names of classes.
    :param title: Title of figure.
    :param cmap: Color map for figure.
    :return: None.
    """
    y_pred = clf.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    plt.figure()

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.show()


def get_genes(filename):
    """
    Returns a list of all the gene symbols in the specified gene set.
    :param filename: The specified gene set file name.
    :return: List of gene symbols.
    """
    genes = set()
    with open('gene_sets/' + filename, 'r') as f:
        for gene in f:
            genes.add(gene.strip())
    return list(genes)


def plot_roc_from_file(filename, title='Receiver Operating Characteristic'):
    """
    Plots the average ROC curve from a file with the true positive rates and area under curve values.
    :param filename: File with true positive rates and area under curve values.
    :param title: Title of figure.
    :return: None.
    """
    tprs = []
    aucs = []
    with open(filename, "r") as f:
        read_tprs = True
        f.readline()  # tprs header
        for line in f:
            if line.strip() == "aucs":
                read_tprs = False
            else:
                if read_tprs:
                    arr = line.strip().split()
                    tprs.append(np.array(arr).astype(np.float))
                else:
                    auc_vals = np.array(line.strip().split()).astype(np.float)
                    aucs = auc_vals

    plot_roc(tprs, aucs, title)


def split_dataset(filename):
    """
    Split the NeoALTTO dataset by drug (Lapatinib, Trastuzumab, Combo)
    :param filename: csv file name.
    :return: None.
    """
    df = pd.read_csv(filename, index_col=0)
    df_l = df.loc[(df['Lapatinib'] == 1) & df['Trastuzumab'] == 0]
    df_t = df.loc[(df['Lapatinib'] == 0) & df['Trastuzumab'] == 1]
    df_c = df.loc[(df['Lapatinib'] == 1) & df['Trastuzumab'] == 1]
    df_l.to_csv('l_' + filename)
    df_t.to_csv('t_' + filename)
    df_c.to_csv('c_' + filename)


def get_gene_sets_union(gene_sets_perf, K):
    """
    Gets the union of the specified gene sets.
    :param gene_sets_perf: List of tuples of the form (<gene set name>, <performance>).
    :param K: Number of top gene sets to include.
    :return: Set of genes in the union of the gene sets.
    """
    top_union = set()
    for i in range(K):
        gene_set = gene_sets_perf[i][0].split()[0]
        genes = get_genes(gene_set + ".txt")
        for gene in genes:
            top_union.add(gene)
    return top_union

