import os
from torch.utils.data import DataLoader
from process.spo_dataset import SpoDataset
from process.spo2_dataset import Spo2Dataset
import argparse
import json
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import sys


def load_config(config_path):
	assert(os.path.exists(config_path))
	cfg = json.load(open(config_path, 'r'))
	return cfg

def build_data_loader(problem, dataset_path):
    """
    Build dataloader
    Args:
    Return: dataloader for training, validation
    """
    dataset_list = []
    for r, d, f in os.walk(dataset_path):
        for file in f:
            if '.json' in file:
                dataset_list.append(os.path.join(r, file))

    test_dataset_path = [dataset_list[8], dataset_list[9]]

    if problem == 'p-median':
        test_dataset = SpoDataset(test_dataset_path)
    elif problem == 'p-center':
        test_dataset = SpoDataset(test_dataset_path)
    elif problem == 'MCLP':
        test_dataset = Spo2Dataset(test_dataset_path)
    elif problem == 'LSCP':
        test_dataset = Spo2Dataset(test_dataset_path)

    test_dataset = Spo2Dataset(test_dataset_path)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False)

    print('Got {} test examples'.format(len(test_loader.dataset)))

    return test_loader

def measure(cfg, model_name, model, test_loader, device, problem, n, k):
    model.eval()

    test_loss = 0
    test_data = test_loader.dataset.data

    with torch.no_grad():

        for batch_count, (A, label, weight, total_num, item, mask) in enumerate(test_loader):

            input = weight.expand(-1, cfg['data']['total_nodes'], cfg['network']['input_feature'])

            #cuda
            input, A = input.to(device, dtype=torch.float), A.to(device, dtype=torch.float)
            label, weight = label.to(device, dtype=torch.float), weight.to(device, dtype=torch.float)

            output = model(input, A)

            #loss = F.binary_cross_entropy(output, label, weight=charge_weight)
            loss = F.binary_cross_entropy(output, label)

            test_loss += loss.item()
            energy = np.amax(torch.squeeze(output, 0).cpu().numpy()[:total_num, 0])*0.8
            test_data[item]['possibility_x'] = list(zip(test_data[item]['x'], torch.squeeze(output, 0).cpu().numpy()[:total_num, 0].tolist()))

            thresholed_output = torch.squeeze(output, 0).cpu().numpy()[:total_num, 0]
            thresholed_output[thresholed_output > energy] = 1
            thresholed_output[thresholed_output <= energy] = 0

            test_data[item]['predict_x'] = thresholed_output.tolist()
            test_data[item]['energy'] = energy

        avg_test_loss = test_loss / len(test_loader.dataset)

        print('##Test loss : %.6f' %(avg_test_loss))

    s_data = {'cfg':test_loader.dataset.cfg, 'data':test_data, 'loss':avg_test_loss}

    path = f'./dataset/{problem}/output/{model_name}/{n}_{k}'
    # path = f'./dataset/{problem}/output/{n}_{k}'
    file_name = path + '/'+'dataset_output.json'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as fp:
        fp.write(json.dumps(s_data, indent=3))

def measure_MCLP(cfg, model_name, model, test_loader, device, problem, n, k, radius):
    model.eval()

    test_loss = 0
    test_data = test_loader.dataset.data

    with torch.no_grad():

        for batch_count, (A, label, weight, total_num, item, mask) in enumerate(test_loader):

            input = weight.expand(-1, cfg['data']['total_nodes'], cfg['network']['input_feature'])

            #cuda
            input, A = input.to(device, dtype=torch.float), A.to(device, dtype=torch.float)
            label, weight = label.to(device, dtype=torch.float), weight.to(device, dtype=torch.float)

            output = model(input, A)

            #loss = F.binary_cross_entropy(output, label, weight=charge_weight)
            loss = F.binary_cross_entropy(output, label)

            test_loss += loss.item()
            energy = np.amax(torch.squeeze(output, 0).cpu().numpy()[:total_num, 0])*0.8
            test_data[item]['possibility_x'] = list(zip(test_data[item]['x'], torch.squeeze(output, 0).cpu().numpy()[:total_num, 0].tolist()))

            thresholed_output = torch.squeeze(output, 0).cpu().numpy()[:total_num, 0]
            thresholed_output[thresholed_output > energy] = 1
            thresholed_output[thresholed_output <= energy] = 0

            test_data[item]['predict_x'] = thresholed_output.tolist()
            test_data[item]['energy'] = energy

        avg_test_loss = test_loss / len(test_loader.dataset)

        print('##Test loss : %.6f' %(avg_test_loss))

    s_data = {'cfg':test_loader.dataset.cfg, 'data':test_data, 'loss':avg_test_loss}

    path = f'./dataset/{problem}/output/{model_name}/{n}_{k}_{radius}'
    file_name = path + '/'+'dataset_output.json'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as fp:
        fp.write(json.dumps(s_data, indent=3))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN train and validate')
    parser.add_argument(
        'problem',
        help='instance type to process.',
        choices=['p-median', 'p-center', 'LSCP', 'MCLP'],
    )
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size (default: 4)')
    parser.add_argument('--workers', type=int, default=4,
                        help='workers (default: 4)')
    parser.add_argument('--config', dest='config', default='config.json',
                        help='hyperparameter of faster-rcnn in json format')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate (default: 0.005)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
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
    device = torch.device("cuda:1" if not args.no_cuda else "cpu")

    ### MODEL LOADING ###
    sys.path.insert(0, os.path.abspath(f'models/{args.model}'))
    import model
    # importlib.reload(model)
    model = model.GCN(cfg).to(device)
    del sys.path[0]

    if args.problem == 'p-median':
        num_nodes = cfg_data['total_nodes']
        num_medians = 10
        checkpoint_dir = f'./checkpoint/{args.problem}/{args.model}/{num_nodes}_{num_medians}/'
        tb_log_dir = './tb_log/'
        # device = torch.device("cuda:1" if not args.no_cuda else "cpu")

        #test model
        checkpoint_file = f'./checkpoint/{args.problem}/{args.model}/{num_nodes}_{num_medians}/gcn_test_model.pkl'
        print('Load data...')
        dataset_path = f'./dataset/{args.problem}/synthetic/{num_nodes}_{num_medians}'
        test_loader = build_data_loader(args.problem, dataset_path)

        print('Start predicting...')


        model.load_state_dict(torch.load(checkpoint_file)['model_state'])

        measure(cfg, args.model, model, test_loader, device, args.problem, num_nodes, num_medians)


    elif args.problem == 'p-center':
        num_nodes = cfg_data['total_nodes']
        num_centers = 10
        checkpoint_dir = f'./checkpoint/{args.problem}/{args.model}/{num_nodes}_{num_centers}/'
        tb_log_dir = './tb_log/'
        # device = torch.device("cuda:1" if not args.no_cuda else "cpu")

        #test model
        checkpoint_file = f'./checkpoint/{args.problem}/{args.model}/{num_nodes}_{num_centers}/gcn_test_model.pkl'
        print('Load data...')
        dataset_path = f'./dataset/{args.problem}/synthetic/{num_nodes}_{num_centers}'
        test_loader = build_data_loader(args.problem, dataset_path)

        print('Start predicting...')

        model.load_state_dict(torch.load(checkpoint_file)['model_state'])

        measure(cfg, args.model, model, test_loader, device, args.problem, num_nodes, num_centers)

    elif args.problem == 'MCLP':
        num_nodes = cfg_data['total_nodes']
        p = 20
        radius = 15
        checkpoint_dir = f'./checkpoint/{args.problem}/{args.model}/{num_nodes}_{p}_{radius}/'
        tb_log_dir = './tb_log/'
        # device = torch.device("cuda:1" if not args.no_cuda else "cpu")

        #test model
        checkpoint_file = f'./checkpoint/{args.problem}/{args.model}/{num_nodes}_{p}_{radius}/gcn_test_model.pkl'
        print('Load data...')
        dataset_path = f'./dataset/{args.problem}/synthetic/{num_nodes}_{p}_{radius}'
        test_loader = build_data_loader(args.problem, dataset_path)

        print('Start predicting...')

        model.load_state_dict(torch.load(checkpoint_file)['model_state'])

        measure_MCLP(cfg, args.model, model, test_loader, device, args.problem, num_nodes, p, radius)

    elif args.problem == 'LSCP':
        num_nodes = cfg_data['total_nodes']
        radius = 60
        checkpoint_dir = f'./checkpoint/{args.problem}/{num_nodes}_{radius}/'
        tb_log_dir = './tb_log/'

        # device = torch.device("cuda:1" if not args.no_cuda else "cpu")

        #test model
        checkpoint_file = f'./checkpoint/{args.problem}/{args.model}/{num_nodes}_{radius}/gcn_test_model.pkl'
        print('Load data...')
        dataset_path = f'./dataset/{args.problem}/synthetic/{num_nodes}_{radius}'
        test_loader = build_data_loader(args.problem, dataset_path)

        print('Start predicting...')

        model.load_state_dict(torch.load(checkpoint_file)['model_state'])

        measure(cfg, args.model, model, test_loader, device, args.problem, num_nodes, radius)

    else:
        raise Exception("Unrecognized mode.")