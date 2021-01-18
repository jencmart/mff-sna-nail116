import matplotlib.pyplot as plt
import networkx as nx
import csv
# from scipy.interpolate import interp1d
import numpy as np
import pickle
import time
import community as community_louvain
try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydot
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("This example needs Graphviz and either "
                          "PyGraphviz or pydot")

def load_graph(fname="edges.csv"):
    G = nx.Graph()
    with open(fname, newline='') as csvfile:
        rdr = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in rdr:
            G.add_node(row[0], bipartite=0)
            G.add_node(row[1], bipartite=1)
            G.add_edge(row[0], row[1])
    return G


def plot_degree_dist(G, nodes):
    degrees = [G.degree(n) for n in nodes]
    degrees = [x for x in degrees]
    # print(sorted(list(set(degrees)), reverse=True))
    n, x, _ = plt.hist(degrees, bins=10, log=True, label="dfd", histtype='bar')
    plt.title('Degree distribution between Heroes')
    plt.xlabel('Number of heroes')
    plt.ylabel('Number of connections')
    plt.show()


def create_weighted_hero_network(G, X):
    C = nx.Graph()
    C.add_nodes_from(X)
    LEN = len(X)
    print(LEN)

    start_time = time.time()
    for idx_a in range(len(X)):
        if (idx_a + 1) % (LEN//100) == 0:
            end_time = time.time()
            print("{:.2f} : ".format((idx_a + 1) / LEN), "{:.2f}m".format((end_time - start_time) / 60))
            start_time = end_time
        for idx_b in range(idx_a + 1, len(X)):
            a = X[idx_a]
            b = X[idx_b]
            c = 0
            for _ in nx.common_neighbors(G, a, b):
                c += 1
            if c >= 1:
                C.add_edge(a, b, wight=c)
    return C


def compute_all_paths_lens(G, fname):
    # takes cca 55-60min
    print("computing paths")
    sp1 = nx.algorithms.shortest_paths.all_pairs_shortest_path_length(G)
    sp = {}
    idx = 0
    start_time = time.time()
    for a in sp1:
        idx += 1
        if idx % 640 == 0:
            end_time = time.time()
            print("{:.2f} : ".format(idx / 6400), "{:.2f}m".format((end_time - start_time) / 60))
            start_time = end_time
        k = a[0]
        v = a[1]
        sp[k] = v
    print("saving pcikle")
    with open(fname, 'wb') as f:
        pickle.dump(sp, f)
    return sp

def get_largest_component(G):
    comp = nx.connected_components(G)
    max_comp, max_comp_size = None, 0
    for c in comp:
        if len(c) > max_comp_size:
            max_comp_size = len(c)
            max_comp = c

    G = G.subgraph(max_comp)
    return G

def create_and_save_graph(fname, fname_paths):
    # 1. Load the bipartite network
    G = load_graph()
    heroes = [x for x, y in G.nodes(data=True) if y['bipartite'] == 0]
    comics = [x for x, y in G.nodes(data=True) if y['bipartite'] == 1]
    # 2. Degree distribution
    # plot_degree_dist(G, comics)

    # 3. Components
    # comp_sizes = [len(c) for c in sorted(comp, key=len, reverse=True)]
    # print(comp_sizes)
    # for idx, c in enumerate(comp):
    #     if idx > 0:
    #         print("Len: {}".format(len(c)), " ", c)


    # 4. consider only largest component
    G = get_largest_component(G)
    heroes = [x for x, y in G.nodes(data=True) if y['bipartite'] == 0]
    # comics = [x for x, y in G.nodes(data=True) if y['bipartite'] == 1]

    # NOW WE WILL CREATE WEIGHTED NETWORK OF HEROES
    G = create_weighted_hero_network(G, heroes)
    nx.write_gpickle(G, fname)
    # 5. Diameter and radius
    sp = compute_all_paths_lens(G, fname_paths)
    print("computing ecc")
    e = nx.eccentricity(G, sp=sp)
    print("computing diam")
    diam = nx.diameter(G, e=e, usebounds=True)
    print(diam)
    radius = nx.radius(G, e=e, usebounds=True)
    print(radius)

    pass


def simple_Louvain(G, min_node_size, max_node_size):
    """ Louvain method github basic example"""
    partition = community_louvain.best_partition(G)


    max_k_w = []
    for com in set(partition.values()):
        list_nodes = [nodes for nodes in partition.keys()
                      if partition[nodes] == com]
        max_k_w = max_k_w + [list_nodes]

    node_mapping = {}
    map_v = 0
    for node in G.nodes():
        node_mapping[node] = map_v
        map_v += 1

    community_num_group = len(max_k_w)
    color_list_community = [[] for i in range(len(G.nodes()))]
    node_sizes = [[] for i in range(len(G.nodes()))]

    # color
    weights = [ sum([w['wight'] for x, y, w in G.edges(node, data=True)]) for node in G.nodes ]
    def renormalize(l, min_v, max_v):
        list_min = min(l)
        list_max = max(l)

        l = [min_v + (max_v-min_v)*(v-list_min)/(list_max-list_min) for v in l ]

        return l
    weights = renormalize(weights, min_node_size, max_node_size)
    node_sizes = weights
    for i in G.nodes():
        for j in range(community_num_group):
            if i in max_k_w[j]:
                color_list_community[node_mapping[i]] = community_num_group - j

    for idxx, i in enumerate(max_k_w):
        print("C:{} size:{}".format(idxx+1, len(i)))
    return color_list_community, node_sizes, community_num_group, max_k_w

# all_pairs_shortest_path_length(G)
# length[1] ---> {0: 1, 1: 0, 2: 1, 3: 2, 4: 3}


def prepare_smaller_graph(G, top_x_nodes, edge_weight_limit, only_biggest_component):

    ee = [(x, y, w) for x, y, w in G.edges(data=True) if w['wight'] < edge_weight_limit]
    G.remove_edges_from(ee)
    G.remove_nodes_from(list(nx.isolates(G)))

    # only 1 connected component...
    if only_biggest_component:
        G = get_largest_component(G)
    G = nx.Graph(G)

    sum_w = sorted([sum([w['wight'] for x, y, w in G.edges(node, data=True)]) for node in G.nodes], reverse=True)
    top_n_nodes = sum_w[top_x_nodes]
    rr = [node for node in G.nodes if sum([w['wight'] for x, y, w in G.edges(node, data=True)]) < top_n_nodes]
    G.remove_nodes_from(rr)
    G.remove_nodes_from(list(nx.isolates(G)))

    if only_biggest_component:
        G = get_largest_component(G)

    G = nx.Graph(G)
    print("Reduced graph |V|={}".format(len(G.nodes)))
    return G

def load_my_graph(fname):
    G = nx.read_gpickle(fname)
    mapping = {'b': 'c'}
    for node in G.nodes():
        if '/' in node:
            mapping[node] = (node.split("/")[0]).strip()
        elif '[' in node:
            mapping[node] = (node.split("[")[0]).strip()
    G = nx.relabel_nodes(G, mapping)
    return G

def recalculate_label_pos(G, pos):
    # Please note, the code below uses the original idea of re-calculating a dictionary of adjusted label positions per node.
    label_ratio = 0.001
    pos_labels = {}
    # For each node in the Graph
    for aNode in G.nodes():
        # Get the node's position from the layout
        x, y = pos[aNode]
        # Get the node's neighbourhood
        N = G[aNode]
        # Find the centroid of the neighbourhood. The centroid is the average of the Neighbourhood's node's x and y coordinates respectively.
        # Please note: This could be optimised further
        cx = sum(map(lambda x: pos[x][0], N)) / len(pos)
        cy = sum(map(lambda x: pos[x][1], N)) / len(pos)
        # Get the centroid's 'direction' or 'slope'. That is, the direction TOWARDS the centroid FROM aNode.
        slopeY = (y - cy)
        slopeX = (x - cx)
        # Position the label at some distance along this line. Here, the label is positioned at about 1/8th of the distance.
        pos_labels[aNode] = (x + slopeX * label_ratio, y + slopeY * label_ratio)

    return pos_labels


def plot_marvel_graph(G, node_sizes, node_colors, font_size, vmax=None, TITLE=None, cmap='jet', edge_colors=None, edge_weights=None):
    pos = graphviz_layout(G)
    fig = plt.figure(figsize=(40, 30))


    im = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=cmap, vmin=0, vmax=vmax)

    # edges
    width=1
    if edge_weights:
        width = edge_weights
    edge_color = 'k'
    if edge_colors:
        edge_color = edge_colors
    nx.draw_networkx_edges(G, pos, width=width, edge_color=edge_color)


    # labels
    label_pos = recalculate_label_pos(G, pos)
    # nx.draw_networkx_labels(G, label_pos, font_size=font_size, font_color="black", font_weight="bold")


    list_min = min(node_sizes)
    list_max = max(node_sizes)
    if list_max != list_min:
        fsizes = [font_size + 5*(v-list_min)/(list_max-list_min) for v in node_sizes ]
    else:
        fsizes = [font_size for v in node_sizes ]
    idx = 0
    for node, (x, y) in pos.items():
        plt.text(x, y, node, fontsize=fsizes[idx], fontweight="bold", ha='center', va='center')
        idx += 1

    # resize graph
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(im)
    l, r = plt.xlim()
    plt.xlim(l - 25, r + 25)
    plt.title(TITLE, fontsize=50)
    plt.show(block=False)


def plot_centralities(G, centrality, max_size, min_size, font_size):
    if centrality == "Eigenvector":
        Cb = nx.eigenvector_centrality(G)
    elif centrality == "Betweenness":
        Cb = nx.betweenness_centrality(G)
    elif centrality == "Closeness":
        Cb = nx.closeness_centrality(G)
    else:
        raise NotImplementedError("Unknown centrality: {}".format(centrality))

    min_v, max_v = None, None
    for k, v in Cb.items():
        if min_v is None or v < min_v:
            min_v = v
        if max_v is None or v > max_v:
            max_v = v

    node_mapping = {}
    map_v = 0
    for node in G.nodes():
        node_mapping[node] = map_v
        map_v += 1

    node_colors = [[] for i in range(len(G.nodes()))]
    # node_sizes = [[] for i in range(len(G.nodes()))]
    edge_sizes = [[] for i in range(len(G.nodes()))]

    node_sizes = []
    for i in G.nodes():
        node_colors[node_mapping[i]] = (Cb[i] - min_v) / (max_v - min_v)
        node_sizes.append(node_colors[node_mapping[i]] * (max_size - min_size) + min_size)

    vmax = 1
    plot_marvel_graph(G, node_sizes, node_colors, vmax,
                      TITLE="{} Centrality".format(centrality), cmap="cool")

def edge_prediction(G, top):
    preds = nx.adamic_adar_index(G)

    result = {}

    for u, v, p in preds:
        result[(u, v)] = p
        # print(f"({u}, {v}) -> {p:.8f}")
    result = {k: v for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)}


    extra_edges = []
    extra_edges_weights = []
    idxx=0
    for k, v in result.items():
        print(k, ": ", v)
        idxx +=1
        extra_edges.append(k)
        extra_edges_weights.append(v)
        if idxx == top:
            break

    return extra_edges, extra_edges_weights

def plot_edge_predtion(G, min_size, max_size, font_size):
    extra_edges, extra_weights = edge_prediction(G, top=10)
    # G = nx.Graph(G)
    # print(extra_edges)
    G.add_edges_from(extra_edges)
    mmax = max(extra_weights)
    mmin = min(extra_weights)
    nodes_of_extra_edges = []

    denom = mmax - mmin
    edges = G.edges()
    edge_colors = []
    edge_weights = []
    cmap = plt.cm.inferno
    for u, v in edges:
        vval = 1
        ec = 0
        if (u, v) in extra_edges:
            vval = 5 + 10 * ((extra_weights[extra_edges.index((u, v))] - mmin) / denom)
            ec = 0.5 + 0.5 * ((extra_weights[extra_edges.index((u, v))] - mmin) / denom)
            nodes_of_extra_edges.append(u)
            nodes_of_extra_edges.append(v)
        elif (v, u) in extra_edges:
            vval = 5 + 10 * ((extra_weights[extra_edges.index((v, u))] - mmin) / denom)
            ec = 0.5 + 0.5 * ((extra_weights[extra_edges.index((v, u))] - mmin) / denom)
            nodes_of_extra_edges.append(u)
            nodes_of_extra_edges.append(v)
        edge_weights.append(vval)
        edge_colors.append(cmap([ec])[0])
    node_mapping = {}
    map_v = 0
    for node in G.nodes():
        node_mapping[node] = map_v
        map_v += 1

    node_colors = [[] for i in range(len(G.nodes()))]
    node_sizes = []
    for i in G.nodes():
        vv = 0.3 if i not in nodes_of_extra_edges else 0.8
        ss = min_size*8 if i not in nodes_of_extra_edges else min_size*20
        node_colors[node_mapping[i]] = vv
        node_sizes.append(ss)


    plot_marvel_graph(G, node_sizes, node_colors, font_size, 1,  TITLE="Edge prediction", cmap="inferno", edge_colors=edge_colors, edge_weights=edge_weights)


def analayze_graph(fname, fname_paths):
    G = load_my_graph(fname)
    G = prepare_smaller_graph(G, edge_weight_limit=98, top_x_nodes=100, only_biggest_component=True)
    min_size = 150
    max_size = 4000
    font_size = 9


    # Analyze communities
    # node_colors, node_sizes, community_num_group, max_k_w = simple_Louvain(G, min_size, max_size)
    # plot_marvel_graph(G, node_sizes, node_colors, font_size, community_num_group, TITLE="Comunities")

    # Centrality measures
    # plot_centralities(G, "Eigenvector", max_size, min_size, font_size)

    # Edge prediction
    plot_edge_predtion(G, min_size, max_size, font_size)


def print_top_degrees(G, nodes):
    degrees = {n: G.degree(n) for n in nodes}
    degrees = {k: v for k, v in sorted(degrees.items(), key=lambda item: item[1], reverse=True)}
    idx = 0
    for k, v in degrees.items():
        print("{} :  {}".format(k,v))
        idx+=1
        if idx == 10:
            break

def print_info_about_dataset():
    G = load_graph()
    heroes = [x for x, y in G.nodes(data=True) if y['bipartite'] == 0]
    comics = [x for x, y in G.nodes(data=True) if y['bipartite'] == 1]
    print_top_degrees(G, heroes)
    print("-----------------------")
    print_top_degrees(G, comics)



if __name__ == '__main__':
    graph_file = 'heroes.pickle'
    paths_file = "heroes-sp.picke"
    #create_and_save_graph(fname=graph_file, fname_paths = paths_file)
    # analayze_graph(fname=graph_file, fname_paths = paths_file)
    print_info_about_dataset()