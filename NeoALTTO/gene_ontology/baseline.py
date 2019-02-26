from dataset_utils import *
from collections import defaultdict
import pandas as pd
import numpy as np


# split_dataset("rnaseq_scaled_symbols.csv")
#
# gene_set_rankings("../c_rnaseq_scaled_symbols.csv", "combo_top_sets.txt")
# gene_set_rankings("../l_rnaseq_scaled_symbols.csv", "lapatinib_top_sets.txt")
# gene_set_rankings("../t_rnaseq_scaled_symbols.csv", "trastuzumab_top_sets.txt")


gene_set_auc = {'c': defaultdict(float), 'l': defaultdict(float), 't': defaultdict(float)}
gene_set_top = {'c': defaultdict(int), 'l': defaultdict(int), 't': defaultdict(int)}
pathways = sorted(os.listdir('gene_sets'))
for r in range(100):
    c_svm_perf = []
    l_svm_perf = []
    t_svm_perf = []

    c_train_sets, c_test_sets = kfold_train_test_sets('../c_rnaseq_scaled_symbols.csv')
    l_train_sets, l_test_sets = kfold_train_test_sets('../l_rnaseq_scaled_symbols.csv')
    t_train_sets, t_test_sets = kfold_train_test_sets('../t_rnaseq_scaled_symbols.csv')

    train_sets = []
    for i in range(5):
        train_sets.append(pd.concat([c_train_sets[i], l_train_sets[i], t_train_sets[i]]))

    for p in pathways:
        genes = get_genes(p)
        name = '.'.join(p.split('.')[:-1])

        mean_auc = avg_auc(train_sets, c_test_sets, genes)
        c_svm_perf.append((name, mean_auc))

        mean_auc = avg_auc(train_sets, l_test_sets, genes)
        l_svm_perf.append((name, mean_auc))

        mean_auc = avg_auc(train_sets, t_test_sets, genes)
        t_svm_perf.append((name, mean_auc))
    c_svm_perf.sort(key=lambda x: x[1], reverse=True)
    l_svm_perf.sort(key=lambda x: x[1], reverse=True)
    t_svm_perf.sort(key=lambda x: x[1], reverse=True)

    for i in range(len(c_svm_perf)):
        name, perf = c_svm_perf[i][0], c_svm_perf[i][1]
        if i < 10:
            gene_set_top['c'][name] += 1
        gene_set_auc['c'][name] += perf

    for i in range(len(l_svm_perf)):
        name, perf = l_svm_perf[i][0], l_svm_perf[i][1]
        if i < 10:
            gene_set_top['l'][name] += 1
        gene_set_auc['l'][name] += perf

    for i in range(len(t_svm_perf)):
        name, perf = t_svm_perf[i][0], t_svm_perf[i][1]
        if i < 10:
            gene_set_top['t'][name] += 1
        gene_set_auc['t'][name] += perf

with open("multi_combo_top_sets.txt", "w") as f:
    for gene_set in gene_set_auc['c']:
        f.write("{}\t{}\t{}\n".format(gene_set, gene_set_auc['c'][gene_set], gene_set_top['c'][gene_set]/100.))

with open("multi_lap_top_sets.txt", "w") as f:
    for gene_set in gene_set_auc['l']:
        f.write("{}\t{}\t{}\n".format(gene_set, gene_set_auc['l'][gene_set], gene_set_top['l'][gene_set]/100.))

with open("multi_trast_top_sets.txt", "w") as f:
    for gene_set in gene_set_auc['t']:
        f.write("{}\t{}\t{}\n".format(gene_set, gene_set_auc['t'][gene_set], gene_set_top['t'][gene_set]/100.))
