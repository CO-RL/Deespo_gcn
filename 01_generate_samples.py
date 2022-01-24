import os
import argparse
import json
import numpy as np
from process.graph_process import graph_gen, save_graph, graph_generation
from process.data_process import gurobi_solver_p_median, gurobi_solver_p_center, gurobi_solver_MCLP, gurobi_solver_LSCP
import datetime
import time

def load_config(config_path):
    assert(os.path.exists(config_path))
    cfg = json.load(open(config_path, 'r'))
    return cfg

def generate_p_meidan(cfg, p):
    """
    Generate random samples
    Args:
    config: configuration
    config.keys():
        total_nodes: total nodes of the graph, must equal GCN input dim
        world_size: Euclidean world grid size
        random_seed: random seed for generating a batch of training data
        sample_num: number of samples to generate
        travel_cost: the cost per euclidean distance from nodes to nodes
    Return:
    data: list of dictionaries
    data[i].keys():
        nodes: [N,2] list
        alpha: scalar
        D: [N] list, the degree matrix of graph
        x: [N] binary array, 1 if the nodes is the median
        y: [N] scalar array, y[i] is the connected nodes index
        d: [N,n] distance array, d[i,j] is the distance from i to j
    """
    np.random.seed(cfg['random_seed'])
    data = []
    for s in range(cfg['sample_num']):
        node_num = cfg['total_nodes']
        nodes = list(map(list,
                              list(zip(np.random.rand(node_num) * cfg['world_size'][0],
                                       np.random.rand(node_num) * cfg['world_size'][1]))))
        alpha = cfg['travel_cost']
        _, _, gen_graph, D = graph_gen(np.array(nodes))
        graph_dict = save_graph(gen_graph)

        data.append({
            'nodes': nodes,
            'alpha': alpha,
            'graph_dict': graph_dict,
            'D': D
        })
        x, y, d = gurobi_solver_p_median(data[s], p)
        data[s]['x'] = x
        data[s]['y'] = y
        data[s]['d'] = d.tolist()

    return data

def generate_p_center(cfg, p):
    """
    Generate random samples
    Args:
    config: configuration
    config.keys():
        total_nodes: total nodes of the graph, must equal GCN input dim
        world_size: Euclidean world grid size
        random_seed: random seed for generating a batch of training data
        sample_num: number of samples to generate
        travel_cost: the cost per euclidean distance from nodes to nodes
    Return:
    data: list of dictionaries
    data[i].keys():
        nodes: [N,2] list
        alpha: scalar
        D: [N] list, the degree matrix of graph
        x: [N] binary array, 1 if the nodes is the median
        y: [N] scalar array, y[i] is the connected nodes index
        d: [N,n] distance array, d[i,j] is the distance from i to j
    """
    np.random.seed(cfg['random_seed'])
    data = []
    for s in range(cfg['sample_num']):
        node_num = cfg['total_nodes']
        nodes = list(map(list,
                              list(zip(np.random.rand(node_num) * cfg['world_size'][0],
                                       np.random.rand(node_num) * cfg['world_size'][1]))))
        alpha = cfg['travel_cost']
        _, _, gen_graph, D = graph_gen(np.array(nodes))
        graph_dict = save_graph(gen_graph)

        data.append({
            'nodes': nodes,
            'alpha': alpha,
            'graph_dict': graph_dict,
            'D': D
        })
        x, y, d = gurobi_solver_p_center(data[s], p)
        data[s]['x'] = x
        data[s]['y'] = y
        data[s]['d'] = d.tolist()

    return data

def generate_MCLP(cfg, p, radius):
    """
    Generate random samples
    Args:
    config: configuration
    config.keys():
        total_nodes: total nodes of the graph, must equal GCN input dim
        world_size: Euclidean world grid size
        random_seed: random seed for generating a batch of training data
        sample_num: number of samples to generate
        travel_cost: the cost per euclidean distance from nodes to nodes
    Return:
    data: list of dictionaries
    data[i].keys():
        nodes: [N,2] list
        demand: [N] list
        alpha: scalar
        x: [N] binary array, 1 if the nodes is the median
        y: [N] scalar array, y[i] is the connected nodes index
        d: [N,n] distance array, d[i,j] is the distance from i to j
    """
    np.random.seed(cfg['random_seed'])
    data = []
    for s in range(cfg['sample_num']):
        f_num = np.random.randint(cfg['facility_num'][0],
                                  cfg['facility_num'][1])
        facilities = list(map(list,
                              list(zip(np.random.rand(f_num) * cfg['world_size'][0],
                                       np.random.rand(f_num) * cfg['world_size'][1]))))
        c_num = cfg['total_nodes'] - f_num
        clients = list(map(list,
                           list(zip(np.random.rand(c_num)*cfg['world_size'][0],
                                    np.random.rand(c_num)*cfg['world_size'][1]))))
        demand = np.random.randint(cfg['facility_cost'][0],
                                   cfg['facility_cost'][1], c_num).tolist()

        alpha = cfg['travel_cost']

        _, _, gen_graph = graph_generation(np.array(facilities), np.array(clients))
        graph_dict = save_graph(gen_graph)

        data.append({
            'clients': clients,
            'facilities': facilities,
            'charge': demand,
            'alpha': alpha,
            'graph_dict': graph_dict
        })
        x, y, d = gurobi_solver_MCLP(data[s], p, radius)
        data[s]['x'] = x
        data[s]['y'] = y
        data[s]['d'] = d.tolist()

    return data

def generate_LSCP(cfg, radius):
    """
    Generate random samples
    Args:
    config: configuration
    config.keys():
        total_nodes: total nodes of the graph, must equal GCN input dim
        world_size: Euclidean world grid size
        random_seed: random seed for generating a batch of training data
        sample_num: number of samples to generate
        travel_cost: the cost per euclidean distance from nodes to nodes
    Return:
    data: list of dictionaries
    data[i].keys():
        nodes: [N,2] list
        charge: [N] list
        alpha: scalar
        x: [N] binary array, 1 if the nodes is the median
        y: [N] scalar array, y[i] is the connected nodes index
        d: [N,n] distance array, d[i,j] is the distance from i to j
    """
    np.random.seed(cfg['random_seed'])
    data = []
    for s in range(cfg['sample_num']):
        f_num = np.random.randint(cfg['facility_num'][0],
                                  cfg['facility_num'][1])
        facilities = list(map(list,
                              list(zip(np.random.rand(f_num) * cfg['world_size'][0],
                                       np.random.rand(f_num) * cfg['world_size'][1]))))
        c_num = cfg['total_nodes'] - f_num
        clients = list(map(list,
                           list(zip(np.random.rand(c_num)*cfg['world_size'][0],
                                    np.random.rand(c_num)*cfg['world_size'][1]))))
        demand = list(np.zeros(c_num))

        alpha = cfg['travel_cost']

        _,_,gen_graph = graph_generation(np.array(facilities), np.array(clients))
        graph_dict = save_graph(gen_graph)

        data.append({
            'clients': clients,
            'facilities': facilities,
            'charge': demand,
            'alpha': alpha,
            'graph_dict': graph_dict
        })
        x, y, d = gurobi_solver_LSCP(data[s], radius)
        data[s]['x'] = x
        data[s]['y'] = y.tolist()
        data[s]['d'] = d.tolist()

    return data

def savedata(data, cfg, name=None, data_dir= './dataset'):
    """
    Data will be saved in json format
    save the configuration and data both
    in the json file.

    Args:
        data: data generated with optimal solutions
        cfg: configuration loaded from file
    Return:
        file_name: absolute path for the data file
    """

    s_data = {'cfg': cfg, 'data': data}
    if not name:
        now = datetime.datetime.now()
        name = 'dataset-%02d-%02d-%2d-%2d-%2d.json' % (now.day, now.month, now.hour, now.minute, now.second)

    if os.path.exists(data_dir):
        file_name = os.path.join(data_dir, name)
    else:
        os.makedirs(data_dir)

    file_name = os.path.join(data_dir, name)

    with open(file_name, 'w') as fp:
        fp.write(json.dumps(s_data, indent=3))
    return file_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data generator')
    parser.add_argument('--config', dest='config', default='config.json',
                        help='hyperparameters')

    parser.add_argument(
        'problem',
        help='instance type to process.',
        choices=['p-median', 'p-center', 'LSCP', 'MCLP'],
    )
    args = parser.parse_args()
    cfg = load_config(args.config)['data']
    sample_nodes = cfg['sample_num']

    if args.problem == 'p-median':
        num_nodes = cfg['total_nodes']
        num_median = 10
        s = 10
        print('Starting generate p-median samples...')
        start = time.time()
        for i in range(s):
            data = generate_p_meidan(cfg, num_median)
            file_name = savedata(data, cfg, name=f'dataset_0{i}.json', data_dir=f'./dataset/p-median/synthetic/{num_nodes}_{num_median}')
            print("%d/%d of data have generated..."%(i+1,s))
        end = time.time()
        print("done")
        print("The total time consumed to generate %d p-median samples(n=%d, k=%d) is: %d"
              %(s*sample_nodes, num_nodes, num_median, end-start))
    elif args.problem == 'p-center':
        num_nodes = cfg['total_nodes']
        num_center = 10
        s = 10
        print('Starting generate p-center samples...')
        start = time.time()
        for i in range(s):
            data = generate_p_center(cfg, num_center)
            file_name = savedata(data, cfg, name=f'dataset_0{i}.json',
                                 data_dir=f'./dataset/p-center/synthetic/{num_nodes}_{num_center}')
            print("%d/%d of data have generated..." %(i+1,s))
        end = time.time()
        print("done")
        print("The total time consumed to generate %d p-center samples(n=%d, k=%d) is: %f seconds"
              %(s*sample_nodes, num_nodes, num_center, end-start))
    elif args.problem == 'LSCP':
        num_nodes = cfg['total_nodes']
        radius = 60
        s = 10
        print('Starting generate LSCP samples...')
        start = time.time()
        for i in range(s):
            data = generate_LSCP(cfg, radius)
            file_name = savedata(data, cfg, name=f'dataset_0{i}.json',
                                 data_dir=f'./dataset/{args.problem}/synthetic/{num_nodes}_{radius}')
            print("%d/%d of data have generated..." % (i + 1, s))
        end = time.time()-start
        print("done")
        print("The total time consumed to generate %d LSCP samples(n=%d, radius=%d) is: %f"
              % (s * sample_nodes, num_nodes, radius, end))
    elif args.problem == 'MCLP':
        num_nodes = cfg['total_nodes']
        p = 10
        radius = 10
        s = 10
        print('Starting generate MCLP samples...')
        start = time.time()
        for i in range(s):
            data = generate_MCLP(cfg, p, radius)
            file_name = savedata(data, cfg, name=f'dataset_0{i}.json',
                                 data_dir=f'./dataset/{args.problem}/synthetic/{num_nodes}_{p}_{radius}')
            print("%d/%d of data have generated..."%(i+1,s))
        end = time.time()-start
        print("done")
        print("The total time consumed to generate %d p-median samples(n=%d, k=%d, radius=%d) is: %f"
              %(s*sample_nodes, num_nodes, p, radius, end))
    else:
        raise NotImplementedError