import numpy as np

def get_shortest_paths(src, dest, E):
    from all_shortest_paths import add_edge
    from all_shortest_paths import get_all_shortest_paths
    # Number of vertices
    n = 8

    # array of vectors is used to store the graph
    # in the form of an adjacency list
    adj = [[] for _ in range(n)]

    v1 = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H" }
    v2 = {"A": 0, "B": 1,  "C": 2,  "D": 3,  "E": 4,  "F": 5,  "G": 6,  "H": 7}
    # Given Graph
    for e in E:
        add_edge(adj, v2[e[0]], v2[e[1]])

    # Given source and destination
    src = v2[src]
    dest = v2[dest]

    # Function Call
    paths = get_all_shortest_paths(adj, n, src, dest)
    paths_decoded = []

    for path in paths:
        path = list(reversed(path))
        decoded = []
        for u in path:
            decoded.append(v1[u])
        paths_decoded.append(decoded)
    return paths_decoded


def cnt_paths_between(i, j, E, over_edge=None):
    x = get_shortest_paths(i, j, E)
    cnt = len(x)
    cnt_over = 0
    if cnt_over is not None:
        for p in x:
            if over_edge[0] in p and over_edge[1] in p:
                # if one after each other... i.e. edge
                idx_a = p.index(over_edge[0])
                idx_b = p.index(over_edge[1])
                if abs(idx_a - idx_b) == 1:
                    cnt_over += 1
    return cnt, cnt_over


def compute_edge_betweeness(V, E):
    values = []
    for idx, edge in enumerate(E):
        eb_new = 0
        for idx_i, i in enumerate(V):
            for idx_j in range(idx_i + 1, len(V)):
                j = V[idx_j]
                if i != j:
                    # todo -- can edge contain i, j ?
                    cnt, cnt_between = cnt_paths_between(i, j, E, over_edge=edge)
                    if cnt == 0 or cnt_between == 0:
                        continue
                    eb_new += cnt_between/cnt
        values.append([eb_new, edge])
    return values


def community_detection(vertices, graph_edges):
    for iteration in range(4):
        res = compute_edge_betweeness(vertices, graph_edges)
        print("Iteration={}".format(iteration))
        max_v, max_idx = -1, -1
        for idx, [v, e] in enumerate(res):
            print("{}--{} : {}".format(e[0], e[1], v))
            if v > max_v:
                max_v, max_idx = v, idx
        # remove edge...
        del graph_edges[max_idx]

def get_neigbors_set(i, E):
    i_n = []
    for e in E:
        if i == e[0]:
            i_n.append(e[1])
        elif i == e[1]:
            i_n.append(e[0])
    i_n = set(i_n)
    return i_n

def jacard_based(v1, v2, V, E):
    n1 = get_neigbors_set(v1, E)
    n2 = get_neigbors_set(v2, E)
    i_len = len(n1.intersection(n2))
    u_len = len(n1.union(n2))
    val = 0
    if i_len != 0:
        val = i_len/u_len
    return val, n1.intersection(n2), n1.union(n2)

def adamic_based(v1, v2, V, E):
    n1 = get_neigbors_set(v1, E)
    n2 = get_neigbors_set(v2, E)
    val = 0
    for n in n1.intersection(n2):
        val += 1/np.log(len(get_neigbors_set(n, E)))
    return val, n1.intersection(n2), n1.union(n2)

if __name__ == '__main__':
    vertices = ["A", "B", "C", "D", "E", "F", "G", "H"]
    graph_edges = [
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
        ("C", "D"),
        ("C", "E"),
        ("C", "F"),
        ("F", "G"),
        ("E", "G"),
        ("D", "E"),
        ("E", "H")
    ]
    for idx_i, i in enumerate(vertices):
        for idx_j in range(idx_i + 1, len(vertices)):
            j = vertices[idx_j]
            if (i, j) not in graph_edges:
                v, inter, uni = adamic_based(i, j, vertices, graph_edges)
                if v != 0:
                    print("{}--{}\t{:.1f}".format(i, j, v), "\t", inter, "\t", uni)