from collections import defaultdict
import csv
import pandas as pd
import numpy as np


class Graph(object):
    """ Graph data structure, undirected by default. """

    def __init__(self, connections, directed=False):
        self._graph = defaultdict(set)
        self._directed = directed
        self.add_connections(connections)

    def __contains__(self, item):
        return item in self._graph

    def add_connections(self, connections):
        """ Add connections (list of tuple pairs) to graph """

        for node1, node2 in connections:
            self.add(node1, node2)

    def add(self, node1, node2):
        """ Add connection between node1 and node2 """

        self._graph[node1].add(node2)
        if not self._directed:
            self._graph[node2].add(node1)
        else:
            self._graph[node2] = set()

    def remove(self, node):
        """ Remove all references to node """

        for n, cxns in self._graph.items():
            try:
                cxns.remove(node)
            except KeyError:
                pass
        try:
            del self._graph[node]
        except KeyError:
            pass

    def is_connected(self, node1, node2):
        """ Is node1 directly connected to node2 """

        return node1 in self._graph and node2 in self._graph[node1]

    def get_connections(self, node):
        if node not in self._graph:
            return []
        return list(self._graph[node])

    def find_path(self, node1, node2, path=[]):
        """ Find any path between node1 and node2 (may not be shortest) """

        path = path + [node1]
        if node1 == node2:
            return path
        if node1 not in self._graph:
            return None
        for node in self._graph[node1]:
            if node not in path:
                new_path = self.find_path(node, node2, path)
                if new_path:
                    return new_path
        return None

    def top(self):
        top = []
        for node in self._graph:
            if self.is_top(node):
                top.append(node)
        return top

    def is_top(self, node):
        if node not in self._graph:
            return True
        return len(self._graph[node]) == 0

    def height(self, node):
        if node not in self._graph:
            return 0
        if self.is_top(node):
            return 1
        h = float("inf")
        for p in self._graph[node]:
            h = min(h, self.height(p))
        return h + 1

    def find_roots(self, node):
        if self.is_top(node):
            return [node]
        else:
            roots = []
            for p in self._graph[node]:
                roots.extend(self.find_roots(p))
            return list(set(roots))

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))


GO_relationships = defaultdict(list)
GO_defs = {}
graph = Graph([], True)
with open('go-basic.obo', 'r') as f:
    line = f.readline()
    while line:
        if '[Term]' in line:
            id_line = f.readline()
            id = id_line.split()[1]

            next_line = f.readline()
            if next_line.startswith('name'):
                name = " ".join(next_line.split()[1:])
                GO_defs[id] = name
            empty = False
            while not next_line.startswith("is_a") and not next_line.startswith("part_of"):
                if next_line == "\n":
                    empty = True
                    break
                next_line = f.readline()
            if empty:
                line = f.readline()
                continue
            parent = [next_line.split()[1]]
            next_line = f.readline()
            while next_line.startswith("is_a") or next_line.startswith("part_of"):
                parent.append(next_line.split()[1])
                next_line = f.readline()
            for p in parent:
                graph.add(id, p)
        line = f.readline()

genes = defaultdict(list)

with open("GO_mapping.txt", "r") as f:
    f.readline()
    for line in f:
        gene_id, GO = line.strip().split(",")
        genes[gene_id].append(GO)


i = 0
gene_to_roots = {}

gene_i = 0
root_i = 0
gene_idx = {}
root_idx = {}

with open('included_genes.csv', 'r') as f:
    all_genes = f.readline().strip().split(',')
    for gene in all_genes:
        gene_id = gene.split(".")[0]
        if gene_id in genes:
            gene_idx[gene_id] = gene_i
            gene_i += 1
            roots = set()
            for p in genes[gene_id]:
                if p in graph:
                    all_roots = graph.find_roots(p)
                    for a in all_roots:
                        roots.add(a)
                        if a not in root_idx:
                            root_idx[a] = root_i
                            root_i += 1
            roots = list(roots)
            gene_to_roots[gene_id] = roots
        i += 1
print("gene_i: ", gene_i)
print("root_i: ", root_i)
# with open("included_genes.csv", "w") as f:
#     f.write(",".join(idx) + "\n")
# print(len(idx))

# np.random.seed(2)

def get_tree():
    tree = Graph([], directed=True)
    for gene in gene_idx.keys():
        if gene in genes:
            connections = genes[gene]
            i = np.random.randint(0, len(connections))
            cur = connections[i]
            if cur not in graph:
                for j in range(len(connections)):
                    if connections[j] in graph:
                        cur = connections[j]
                        break
                if cur not in graph:
                    tree._graph[gene] = set()
                    continue
            tree.add(gene, cur)
            while not graph.is_top(cur):
                connections = graph.get_connections(cur)
                i = np.random.randint(0, len(connections))
                tree.add(cur, connections[i])
                cur = connections[i]
    return tree


def make_tree(t):
    tree = get_tree()
    h_i = [0, 0, 0, 0, 0]
    l_i = [{}, {}, {}, {}, {}]

    for node in tree._graph:
        h = tree.height(node)
        if node.startswith("ENSG"):
            for i in range(h, 4):
                l_i[i][node] = h_i[i]
                h_i[i] += 1
            l_i[4][node] = gene_idx[node]
        if h < 5:
            l_i[h-1][node] = h_i[h-1]
            h_i[h-1] += 1

    for i in range(4):
        df1 = pd.DataFrame(0, index=np.arange(len(l_i[i+1])), columns=list(l_i[i].keys()))
        for node in l_i[i+1].keys():
            connections = tree._graph[node]
            connections.add(node)
            node_idx = l_i[i+1][node]
            for c in connections:
                if c in l_i[i]:
                    c_idx = l_i[i][c]
                    df1.iat[node_idx, c_idx] = 1
        df1.to_csv("tree_" + str(t) + "/edges_" + str(i) + ".csv", header=False, index=False)


def reverse_graph(g):
    new_g = Graph([], directed=True)
    for node in g._graph:
        connections = g._graph[node]
        for c in connections:
            new_g.add(c, node)
    return new_g


# for t in range(5):
#     make_tree(t)

# df2 = pd.DataFrame(0, index=np.arange(len(l_i[2])), columns=list(l_i[1].keys()))
# for node in l_i[2].keys():
#     connections = tree._graph[node]
#     node_idx = l_i[2][node]
#     for c in connections:
#         c_idx = l_i[1][c]
#         df1.at[node_idx, c_idx] = 1


# df = pd.DataFrame(0, index=np.arange(len(root_idx)), columns=list(gene_idx.keys()))
# df = [["0" for _ in range(len(gene_idx))] for _ in range(len(root_idx))]
#
# for gene in gene_idx.keys():
#     gene_i = gene_idx[gene]
#     if gene_i >= 100:
#         continue
#     roots = gene_to_roots[gene]
#     for r in roots:
#         if r in root_idx:
#             r_i = root_idx[r]
#             # df.at[r_i, gene_i] = 1
#             df[r_i][gene_i] = "1"
#
# with open("graph_edges.csv", "w") as f:
#     for row in df:
#         f.write(",".join(row) + "\n")

# df.to_csv("graph_edges.csv")

def make_pathways():
    g = graph
    for gene in gene_idx.keys():
        if gene in genes:
            connections = genes[gene]
            for j in range(len(connections)):
                if connections[j] in graph:
                    cur = connections[j]
                    g.add(gene, cur)
    pathways = defaultdict(list)
    for node in g._graph:
        if node.startswith("ENSG"):
            for top in g.find_roots(node):
                pathways[top].append(node)
    return pathways


pathways = make_pathways()
print("num pathways: ", len(pathways))
count = 0
rm = []
for k, v in pathways.items():
    if len(v) > 25:
        count += 1
    else:
        rm.append(k)

for k in rm:
    pathways.pop(k)

count = 0
mapping = {}
for p, l in pathways.items():
    df = pd.DataFrame(0, index=np.arange(len(gene_idx)), columns=list(l))
    for i, node in enumerate(l):
        node_idx = gene_idx[node]
        df.iat[node_idx, i] = 1
    df.to_csv("pathways/" + str(count) + ".csv", header=False, index=False)
    mapping[count] = GO_defs[p]
    count += 1

print(count)

with open("pathways_mapping.txt", "w") as f:
    for i in range(count):
        f.write(str(i) + mapping[i] + "\n")

# p_g = reverse_graph(graph)
# p = list(pathways.keys())
#
#
# def get_p_graph(pathway):
#     pathway_graph = Graph([], directed=True)
#
#     def explore(cur_node):
#         connections = p_g._graph[cur_node]
#         for c in connections:
#             pathway_graph.add(c, cur_node)
#             explore(c)
#
#     explore(pathway)
#     return pathway_graph
#
#
# for i in range(len(p)):
#     x = get_p_graph(p[i])
#
#     max_h = 0
#     for node in x._graph:
#         max_h = max(x.height(node), max_h)
#     print(max_h)