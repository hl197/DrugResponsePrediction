from collections import defaultdict
import csv
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

keep_genes = []
with open('unprocessed_genes.txt', 'r') as f:
    for gene in f:
        gene = gene.strip()
        gene_id = gene.split(".")[0]
        if gene_id in genes:
            keep_genes.append(gene)
print("Kept {} genes.".format(len(keep_genes)))

with open('included_genes.csv', 'w') as f:
    for gene in keep_genes:
        f.write(gene + ",")
    f.write("\n")