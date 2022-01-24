import networkx as nx
import numpy as np
import math
from scipy.sparse import coo_matrix

def l2(a,b):
    '''
    l2 norm between any two nodes
    :param a: numpy array, 1*n vector
    :param b: numpy array, 1*n vector(the same dim with a)
    :return: distance between a and b
    '''
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx*dx + dy*dy)


def graph_gen(nodes, max_dis=18, reduce_prob=0.4):
    N = nodes.shape[0]
    #Initialize the graph
    G=nx.Graph()
    G.add_nodes_from(np.arange(0,N).tolist())
    D=[]
    for i in range(N):
        for j in range(i+1,N):
            dis = l2(nodes[i],nodes[j])
            G.add_edge(i, j, weight=dis)

    T = nx.minimum_spanning_tree(G)
    TG = T.copy()
    for i in range(N):
        for j in range(i + 1, N):
            dis = l2(nodes[i], nodes[j])
            if dis < max_dis and np.random.random() > reduce_prob:
                TG.add_edge(i, j, weight=dis)
        D.append(TG.degree(i))
    return G, T, TG, D


def graph_generation1(facilities, clients, max_dis=18, reduce_prob=0.4):
    '''
    Given position of the facilities and clients,
    generate a graph that connects all the nodes but
    not fully connected
    Args:
        facilities: position array of facilities n*2
        clinets:   position array of clients
    Return:
        G: fully connected graph
        T: minimum spanning tree of the graph
        TG: Minimum spanning tree with some random edges
    '''

    fac_num = facilities.shape[0]
    cli_num = clients.shape[0]

    # combine nodes
    nodes = np.concatenate((facilities, clients), axis=0)

    # Initialize the graph
    G = nx.Graph()
    N = cli_num + fac_num
    # add nodes
    G.add_nodes_from(np.arange(0, N).tolist())

    # suppose bidirectional edges
    for i in range(fac_num):
        for j in range(fac_num, N):
            dis = l2(nodes[i], nodes[j])
            G.add_edge(i, j, weight=dis)

    # # Minimum spanning tree
    # T = nx.minimum_spanning_tree(G)
    # # More connection tree
    # TG = T.copy()
    # # suppose bidirectional edges
    # for i in range(N):
    #     for j in range(i + 1, N):
    #         dis = l2(nodes[i], nodes[j])
    #         if dis < max_dis and np.random.random() > reduce_prob:
    #             TG.add_edge(i, j, weight=dis)

    return G


def graph_generation(facilities, clients, max_dis=18, reduce_prob=0.4):
    '''
    Given position of the facilities and clients,
    generate a graph that connects all the nodes but
    not fully connected
    Args:
        facilities: position array of facilities n*2
        clinets:   position array of clients
    Return:
        G: fully connected graph
        T: minimum spanning tree of the graph
        TG: Minimum spanning tree with some random edges
    '''

    fac_num = facilities.shape[0]
    cli_num = clients.shape[0]

    # combine nodes
    nodes = np.concatenate((facilities, clients), axis=0)

    # Initialize the graph
    G = nx.Graph()
    N = fac_num + cli_num
    # add nodes
    G.add_nodes_from(np.arange(0, N).tolist())

    # suppose bidirectional edges
    for i in range(N):
        for j in range(i + 1, N):
            dis = l2(nodes[i], nodes[j])
            G.add_edge(i, j, weight=dis)

    # Minimum spanning tree
    T = nx.minimum_spanning_tree(G)
    # More connection tree
    TG = T.copy()
    # suppose bidirectional edges
    for i in range(N):
        for j in range(i + 1, N):
            dis = l2(nodes[i], nodes[j])
            if dis < max_dis and np.random.random() > reduce_prob:
                TG.add_edge(i, j, weight=dis)

    return G, T, TG

def save_graph(G):
    '''
    Save a graph to an adjacent matrix
    store the sparse matrix as coo format and
    put it in a dict
    Args:
        G: graph in networkx
        fac_num: number of facilities
        cli_num: number of clients
    Return:
        g_dict: a dict that stores the sparse matrix
    '''

    # Convert to adjacent matrix
    A = nx.adjacency_matrix(G)
    Ac = coo_matrix(A)
    g_dict = {}
    g_dict['data'] = Ac.data.tolist()
    g_dict['row'] = Ac.row.tolist()
    g_dict['col'] = Ac.col.tolist()

    return g_dict
