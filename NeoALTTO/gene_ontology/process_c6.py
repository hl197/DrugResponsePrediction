import pandas as pd
import numpy as np
from collections import defaultdict


symbol_to_set = defaultdict(list)
set_to_idx = {}
all_symbols = set()
set_count = 0
# parse and map gene symbols to gene_sets
with open("../../c6_condensed.gmt", "r") as f:
    for line in f:
        tokens = line.strip().split()
        gene_set = tokens[0]
        set_to_idx[gene_set] = set_count
        set_count += 1
        for symbol in tokens[1:]:
            all_symbols.add(symbol)
            symbol_to_set[symbol].append(gene_set)

symbols_to_idx = defaultdict(list)
symbols_count = 0
with open("../NeoALTTO_genes.txt", "r") as f:
    for symbol in f:
        symbols_to_idx[symbol.strip()].append(symbols_count)
        symbols_count += 1


def clean(gene):
    """
    Removes trailing numbers from the gene symbol (caused by making the gene symbols unique in load_TCGA.R
    For example: 'gene_a.1' -> 'gene_a'
    :param gene: full gene name
    :return: cleaned gene name
    """
    if '.' in gene:
        tokens = gene.split('.')
        if tokens[-1].isdigit():
            cleaned = '.'.join(tokens[:-1])
            return cleaned
    return gene


# create adjacency matrix of gene symbols to gene sets (allow each gene set to have n nodes)
n = 1
cols = [val for val in list(set_to_idx.keys()) for _ in range(n)]
df = pd.DataFrame(0, index=np.arange(symbols_count), columns=cols)
print(df.shape)
for gene in symbols_to_idx:
    cleaned = clean(gene)
    sym_indexes = symbols_to_idx[gene]
    for sym_idx in sym_indexes:
        for gene_set in symbol_to_set[cleaned]:
            set_idx = set_to_idx[gene_set] * n
            for i in range(n):
                df.iloc[sym_idx, set_idx+i] = 1
df.to_csv("connections_{}.csv".format(n))

# print out gene set mapping
gene_set_mapping = [(gene_set, idx) for gene_set, idx in set_to_idx.items()]
gene_set_mapping.sort(key=lambda x: x[1])
with open("gene_set_mapping.txt", "w") as f:
    for gene_set, idx in gene_set_mapping:
        f.write("{}\t{}\n".format(str(idx), gene_set))
