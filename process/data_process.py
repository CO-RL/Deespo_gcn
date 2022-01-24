from gurobipy import *
import math
import numpy as np
import networkx as nx
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

def load_adj(g_dict,N):
    '''
    load graph from adjacent matrix
    Args:
        g_dict: a dict that stores the sparse matrix
        N: the number of nodes
    Return:
        G: graph represented by Networkx
    '''
    row  = np.array(g_dict['row'])
    col  = np.array(g_dict['col'])
    data = np.array(g_dict['data'])
    A = coo_matrix((data, (row, col)), shape=(N, N)).toarray()
    G = nx.Graph(A)

    return G

def graph_dis(G,i,j):
    '''
    Graph distance between any two point
    Args:
        G: graph we generated
        c_indx: the index of client
        f_indx: the index of facility
        numFacilities: the number of all the facilities
    Return:
        distance between a and b with dijkstra algorithm
    '''
    distance=nx.dijkstra_path_length(G, source = i, target = j)
    return distance

def gurobi_solver_p_median(data, PN):
    """
    Exact solver for facility location
    Args:
    data.keys():
    Nodes: [N,2] list, e.g. [[0,0],[0,1],[0,1],
                                 [1,0],[1,1],[1,2],
                                 [2,0],[2,1],[2,2]]
    charge: [N,1] list, e.g. [3,2,3,1,3,3,4,3,2]
    alpha: const, cost per mile
    Return:
    x: [N] binary array
    y: [N] scalar array
    d: [N,N] distance array
    """
    # Problem data
    nodes = data['nodes']
    PN = PN
    N = len(nodes)

    G = load_adj(data['graph_dict'], N)

    model = Model('p-median')
    model.setParam('OutputFlag', False)
    # Add variables
    x = {}
    y = {}
    d = {}
    for i in range(N):
        x[i] = model.addVar(vtype="B", name="x(%s)"%i)
        for j in range(N):
            y[i, j] = model.addVar(vtype="B", name="y(%s, %s)"%(i,j))

    for i in range(N):
        for j in range(i+1, N):
            d[(i,j)] = graph_dis(G,i,j)
    for j in range(N):
        for i in range(j+1, N):
            d[(i,j)] = d[(j,i)]
    for i in range(N):
        d[(i,i)] = 0

    model.update()
    # Add constraints
    for j in range(N):
        model.addConstr(quicksum(y[i,j] for i in range(N)) == 1)
        for i in range(N):
            model.addConstr(y[i,j] <= x[i])
    model.addConstr(quicksum(x[i] for i in range(N)) == PN)

    model.setObjective(quicksum(d[(i, j)] * y[i, j] for i in range(N) for j in range(N)))

    model.optimize()

    # return a stardard result list
    x_result = []
    for j in range(N):
        x_result.append(x[j].X)
    y_result = []
    for i in range(N):
        for j in range(N):
            if y[(j, i)].X == 1:
                y_result.append(j)
                continue
    d_results = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            d_results[i, j] = d[(i, j)]
    return x_result, y_result, d_results

def gurobi_solver_p_center(data, PN):
    """
    Exact solver for facility location
    Args:
    data.keys():
    Nodes: [N,2] list, e.g. [[0,0],[0,1],[0,1],
                                 [1,0],[1,1],[1,2],
                                 [2,0],[2,1],[2,2]]
    charge: [N,1] list, e.g. [3,2,3,1,3,3,4,3,2]
    alpha: const, cost per mile
    Return:
    x: [N] binary array
    y: [N] scalar array
    d: [N,N] distance array
    """
    # Problem data
    nodes = data['nodes']
    PN = PN
    N = len(nodes)

    G = load_adj(data['graph_dict'], N)

    model = Model('p-center')
    # model.setParam('OutputFlag', True)
    model.setParam('OutputFlag', False)
    # Add variables
    x = {}
    y = {}
    d = {}
    z = model.addVar()
    for i in range(N):
        x[i] = model.addVar(vtype="B", name="x(%s)"%i)
        for j in range(N):
            y[i, j] = model.addVar(vtype="B", name="y(%s, %s)"%(i,j))

    for i in range(N):
        for j in range(i+1, N):
            d[(i,j)] = graph_dis(G,i,j)
    for j in range(N):
        for i in range(j+1, N):
            d[(i,j)] = d[(j,i)]
    for i in range(N):
        d[(i,i)] = 0

    model.update()
    # Add constraints
    model.addConstr(quicksum(x[i] for i in range(N)) == PN)
    for i in range(N):
        model.addConstr(quicksum(y[i,j] for j in range(N)) == 1)
        for j in range(N):
            model.addConstr(y[i,j] <= x[j])
    for i in range(N):
        model.addConstr(quicksum(d[(i, j)] * y[i, j] for j in range(N))<= z)

    model.setObjective(z)

    model.optimize()

    # return a stardard result list
    x_result = []
    for j in range(N):
        x_result.append(x[j].X)
    y_result = []
    for i in range(N):
        for j in range(N):
            if y[(i, j)].X == 1:
                y_result.append(j)
                continue
    d_results = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            d_results[i, j] = d[(i, j)]
    return x_result, y_result, d_results

def gurobi_solver_MCLP(data, PN, radius):
    # Problem data
    clients = data['clients']
    facilities = data['facilities']
    N = len(clients)
    M = len(facilities)
    PN = PN
    demand = data['charge']
    G = load_adj(data['graph_dict'], N+M)

    model = Model('MCLP')
    model.setParam('OutputFlag', False)
    model.setParam('MIPFocus', 2)
    # Add variables
    client_var = {}
    serv_var = {}
    d = {}
    A = {}
    for i in range(M):
        for j in range(N):
            d[(i,j)] = graph_dis(G,i,j+M)
            if d[(i, j)] <= radius:
                A[(i, j)] = 1
            else:
                A[(i, j)] = 0
    # Add Client Decision Variables and Service Decision Variables
    for j in range(N):
        client_var[j] = model.addVar(vtype="B", name="y(%s)"%j)
        for i in range(M):
            serv_var[i] = model.addVar(vtype="B", name="x(%s)"%i)
    # Update Model Variables
    model.update()
    #     Set Objective Function
    model.setObjective(quicksum(demand[j] * client_var[j] for j in range(N)), GRB.MAXIMIZE)
    #     Add Constraints
    # Add Coverage Constraints
    for j in range(N):
        model.addConstr(quicksum(A[(i,j)]*serv_var[i] for i in range(M)) - client_var[j] >= 0,
                        'Coverage_Constraint_%d' % j)

    # Add Facility Constraint
    model.addConstr(quicksum(serv_var[i] for i in range(M)) == PN,
                "Facility_Constraint")

    model.optimize()

    # return a stardard result list
    x_result = []
    for i in range(M):
        x_result.append(serv_var[i].X)
    y_result = []
    for j in range(N):
        y_result.append(client_var[i].X)
    d_results = np.zeros((M, N))
    for j in range(M):
        for i in range(N):
            d_results[j, i] = d[(j, i)]
    return x_result, y_result, d_results

def gurobi_solver_LSCP(data, radius):
    # Problem data
    clients = data['clients']
    facilities = data['facilities']
    N = len(clients)
    M = len(facilities)
    # demand = data['charge']
    G = load_adj(data['graph_dict'], N+M)

    model = Model('LSCP')
    model.setParam('OutputFlag', False)
    model.setParam('MIPFocus', 2)
    # Add variables
    client_var = {}
    serv_var = {}
    d = {}
    A = {}
    for i in range(M):
        for j in range(N):
            d[(i,j)] = graph_dis(G,i,j+M)
            if d[(i, j)] <= radius:
                A[(i, j)] = 1
            else:
                A[(i, j)] = 0
    # Add Client Decision Variables and Service Decision Variables
    # for j in range(N):
    #     client_var[j] = model.addVar(vtype="B", name="y(%s)"%j)
    for i in range(M):
        serv_var[i] = model.addVar(vtype="B", name="x(%s)"%i)
    # Update Model Variables
    model.update()
    #     Set Objective Function
    model.setObjective(quicksum(serv_var[i] for i in range(M)), GRB.MINIMIZE)
    #     Add Constraints
    # Add Coverage Constraints
    for j in range(N):
        model.addConstr(quicksum(A[(i,j)]*serv_var[i] for i in range(M)) - 1 >= 0,
                        'Coverage_Constraint_%d' % j)

    # # Add Facility Constraint
    # model.addConstr(quicksum(serv_var[i] for i in range(M)) <= PN,
    #             "Facility_Constraint")

    model.optimize()

    # return a stardard result list
    x_result = []
    for i in range(M):
        x_result.append(serv_var[i].X)
    y_result = np.zeros(150)
    # for j in range(N):
    #     y_result.append(client_var[i].X)
    d_results = np.zeros((M, N))
    for j in range(M):
        for i in range(N):
            d_results[j, i] = d[(j, i)]
    return x_result, y_result, d_results
