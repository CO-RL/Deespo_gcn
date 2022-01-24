import os
import numpy as np
import json
import argparse
from process.spo_dataset import load_data

def load_config(config_path):
    assert(os.path.exists(config_path))
    cfg = json.load(open(config_path, 'r'))
    assert isinstance(cfg, object)
    return cfg

def eval_result(data_node, problem, idx):

    gd_mask = np.array(data_node['x'])
    pd_mask = np.array(data_node['predict_x'])

    if problem == 'p-median':
        node_num = len(data_node['nodes'])
        dist_overall = data_node['d']
        gd_dis_ma = np.ma.masked_array(dist_overall, \
                                       mask=np.broadcast_to(1 - gd_mask, (node_num, node_num)))

        pd_dis_ma = np.ma.masked_array(dist_overall, \
                                       mask=np.broadcast_to(1 - pd_mask, (node_num, node_num)))

        gd_dis = gd_dis_ma.min(axis=1)
        pd_dis = pd_dis_ma.min(axis=1)

        cost_overall = np.zeros(node_num)
        c1 = np.ma.masked_array(cost_overall, mask=1 - gd_mask)
        gd_cost = gd_dis.data.sum() + c1.sum()
        c2 = np.ma.masked_array(cost_overall, mask=1 - pd_mask)
        if c2.all() is np.ma.masked:
            print("%03d: Not feasible" % (idx))
            pd_cost = 1e5
        else:
            pd_cost = pd_dis.data.sum() + c2.sum()
            print("%03d: The optimal cost is %0.2f, the predicted cost is %0.2f , ratio is %0.2f" \
                  % (idx, gd_cost, pd_cost, pd_cost / gd_cost))
    elif problem == 'p-center':
        node_num = len(data_node['nodes'])
        dist_overall = data_node['d']
        gd_dis_ma = np.ma.masked_array(dist_overall, \
                                       mask=np.broadcast_to(1 - gd_mask, (node_num, node_num)))

        pd_dis_ma = np.ma.masked_array(dist_overall, \
                                       mask=np.broadcast_to(1 - pd_mask, (node_num, node_num)))

        gd_dis = gd_dis_ma.min(axis=1)
        pd_dis = pd_dis_ma.min(axis=1)

        cost_overall = np.zeros(node_num)
        c1 = np.ma.masked_array(cost_overall, mask=1 - gd_mask)
        gd_cost = gd_dis.data.max() + c1.sum()
        c2 = np.ma.masked_array(cost_overall, mask=1 - pd_mask)
        if c2.all() is np.ma.masked:
            print("%03d: Not feasible" % (idx))
            pd_cost = 1e5
        else:
            pd_cost = pd_dis.data.max() + c2.sum()
            print("%03d: The optimal cost is %0.2f, the predicted cost is %0.2f, ratio is %0.2f, " \
                  % (idx, gd_cost, pd_cost, pd_cost / gd_cost))
    elif problem == 'MCLP':
        radius = 20
        clients_num = len(data_node['clients'])
        facilities_num = len(data_node['facilities'])
        dist_overall = data_node['d']

        gd_dis_ma = np.ma.masked_array(dist_overall, \
                                       mask=np.broadcast_to((1 - gd_mask).reshape(-1, 1),
                                                            (facilities_num, clients_num)))

        pd_dis_ma = np.ma.masked_array(dist_overall, \
                                       mask=np.broadcast_to((1 - pd_mask).reshape(-1, 1),
                                                            (facilities_num, clients_num)))

        gd_dis = gd_dis_ma.min(axis=0)
        pd_dis = pd_dis_ma.min(axis=0)

        gd_cost = np.sum(gd_dis <= radius)
        pd_cost = np.sum(pd_dis <= radius)
        print("%03d: The optimal number of clients covered is %0.2f, the predicted number of clients covered is"
              " %0.2f , ratio is %0.2f, the total clients is %0.2f"
              % (idx, gd_cost, pd_cost, pd_cost / gd_cost, clients_num))
    elif problem == 'LSCP':
        gd_cost = gd_mask.sum()
        pd_cost = pd_mask.sum()
        print("%03d: The optimal number of facilities to open is %0.2f, the predicted number of facilities to open is"
              " %0.2f , ratio is %0.2f" % (idx, gd_cost, pd_cost, pd_cost / gd_cost))

    return([gd_cost, pd_cost, pd_cost/gd_cost])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate result')
    parser.add_argument(
        'problem',
        help='instance type to process.',
        choices=['p-median', 'p-center', 'LSCP', 'MCLP'],
    )
    parser.add_argument(
        '-m', '--model',
        help='GCNN model to be trained.',
        type=str,
        default='gcn_5layers',
        choices=['gcn_3layers', 'gcn_4layers', 'gcn_6layers', 'gcn_7layers']
    )


    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_data = cfg['data']
    data_save = []

    if args.problem == 'p-median':
        num_nodes = cfg_data['total_nodes']
        num_medians = 10

        file_path =  f'./dataset/{args.problem}/output/{args.model}/{num_nodes}_{num_medians}'
        eval_file = file_path + '/' + 'dataset_output.json'
        cfg, data = load_data(eval_file)

        N = 2 * cfg['sample_num']
        for i in range(N):
            s = eval_result(data[i], args.problem, i)
            data_save.append(s)

        file_name = file_path + '/' + 'dataset_result.json'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        with open(file_name, 'w') as fp:
            fp.write(json.dumps(str(data_save), indent=3))

        data_np = np.array(data_save)
        a = data_np[:, 2]
        a = a[a < 50]
        print(a.mean())
        print(a.shape[0])

    elif args.problem == 'p-center':
        num_nodes = cfg_data['total_nodes']
        num_centers = 10

        file_path = f'./dataset/{args.problem}/output/{args.model}/{num_nodes}_{num_centers}'
        eval_file = file_path + '/' + 'dataset_output.json'
        cfg, data = load_data(eval_file)

        N = 2 * cfg['sample_num']
        for i in range(N):
            s = eval_result(data[i], args.problem, i)
            data_save.append(s)

        file_name = file_path + '/' + 'dataset_result.json'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        with open(file_name, 'w') as fp:
            fp.write(json.dumps(str(data_save), indent=3))

        data_np = np.array(data_save)
        a = data_np[:, 2]
        a = a[a < 50]
        print(a.mean())
        print(a.shape[0])

    elif args.problem == 'MCLP':
        num_nodes = cfg_data['total_nodes']
        p = 20
        radius = 15

        file_path = f'./dataset/{args.problem}/output/{args.model}/{num_nodes}_{p}_{radius}'
        eval_file = file_path + '/' + 'dataset_output.json'
        cfg, data = load_data(eval_file)

        N = 2 * cfg['sample_num']
        for i in range(N):
            s = eval_result(data[i], args.problem, i)
            data_save.append(s)

        file_name = file_path + '/' + 'dataset_result.json'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        with open(file_name, 'w') as fp:
            fp.write(json.dumps(str(data_save), indent=3))

        data_np = np.array(data_save)
        a = data_np[:, 2]
        a = a[a < 50]
        print(a.mean())
        print(a.shape[0])

    elif args.problem == 'LSCP':
        num_nodes = cfg_data['total_nodes']
        radius = 60

        file_path = f'./dataset/{args.problem}/output/{args.model}/{num_nodes}_{radius}'
        eval_file = file_path + '/' + 'dataset_output.json'
        cfg, data = load_data(eval_file)

        N = 2 * cfg['sample_num']
        for i in range(N):
            s = eval_result(data[i], args.problem, i)
            data_save.append(s)

        file_name = file_path + '/' + 'dataset_result.json'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        with open(file_name, 'w') as fp:
            fp.write(json.dumps(str(data_save), indent=3))

        data_np = np.array(data_save)
        a = data_np[:, 2]
        a = a[a < 50]
        print(a.mean())
        print(a.shape[0])

