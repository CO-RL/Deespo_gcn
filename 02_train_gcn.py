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

    train_dataset_path = dataset_list[:6]
    val_dataset_path = [dataset_list[6], dataset_list[7]]

    if problem == 'p-median':
        train_dataset = SpoDataset(train_dataset_path)
        val_dataset = SpoDataset(val_dataset_path)
    elif problem == 'p-center':
        train_dataset = SpoDataset(train_dataset_path)
        val_dataset = SpoDataset(val_dataset_path)
    elif problem == 'MCLP':
        train_dataset = Spo2Dataset(train_dataset_path)
        val_dataset = Spo2Dataset(val_dataset_path)
    elif problem == 'LSCP':
        train_dataset = Spo2Dataset(train_dataset_path)
        val_dataset = Spo2Dataset(val_dataset_path)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False)

    print('Got {} training examples'.format(len(train_loader.dataset)))
    print('Got {} validation examples'.format(len(val_loader.dataset)))

    return train_loader, val_loader

def train(cfg, model, train_loader, device, optimizer, epoch,  tb_writer, total_tb_it):
    model.train()
    for batch_num, (A, label, weight, total_num, item, mask) in enumerate(train_loader):

        input = weight.expand(-1, cfg['data']['total_nodes'], cfg['network']['input_feature'])

        input, A = input.to(device, dtype=torch.float), A.to(device, dtype=torch.float)
        label, weight = label.to(device, dtype=torch.float), weight.to(device, dtype=torch.float)

        optimizer.zero_grad()

        output = model(input, A)

        loss = F.binary_cross_entropy(output, label)
        loss.backward()
        optimizer.step()

        per_loss = loss.item() / args.batch_size

        tb_writer.add_scalar('train/overall_loss', per_loss, total_tb_it)
        total_tb_it += 1

        if batch_num % 30 == 0:
            print('Epoch [%d/%d] Loss: %.6f' % (epoch, args.epochs, per_loss))

    return total_tb_it

def validate(cfg, model, val_loader, device, tb_writer, total_tb_it):
    model.eval()

    tb_loss = 0

    with torch.no_grad():
        for batch_num, (A, label, weight, total_num, item, mask) in tqdm(enumerate(val_loader)):

            input = weight.expand(-1, cfg['data']['total_nodes'], cfg['network']['input_feature'])

            # cuda
            input, A = input.to(device, dtype=torch.float), A.to(device, dtype=torch.float)
            label, weight = label.to(device, dtype=torch.float), weight.to(device, dtype=torch.float)

            output = model(input, A)

            # mask = mask.to(device, dtype=torch.float)
            # loss = F.binary_cross_entropy(output, label, weight=mask)
            loss = F.binary_cross_entropy(output, label)

            tb_loss += loss.item()

        avg_tb_loss = tb_loss / len(val_loader.dataset)

        print('##Validate loss : %.6f' % (avg_tb_loss))

        tb_writer.add_scalar('val/overall_loss', avg_tb_loss, total_tb_it)

    return avg_tb_loss

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
        num_medians = 20

        checkpoint_dir = f'./checkpoint/{args.problem}/{args.model}/{num_nodes}_{num_medians}/'
        tb_log_dir = './tb_log/'
        # device = torch.device("cuda:1" if not args.no_cuda else "cpu")

        #train model
        name = 'gcn_test'
        tb_writer = SummaryWriter(tb_log_dir + name)
        dataset_path = f'./dataset/{args.problem}/synthetic/{num_nodes}_{num_medians}'
        train_loader, val_loader = build_data_loader(args.problem, dataset_path)

        # model = GCN(cfg).to(device)
        optimizer = optim.Adam(model.parameters(), args.learning_rate, (0.9, 0.999), eps=1e-08)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        total_tb_it = 0
        best_val_loss = 1000

        for epoch in range(args.epochs):
            scheduler.step(epoch)

            total_tb_it = train(cfg, model, train_loader, device, optimizer, epoch, tb_writer, total_tb_it)
            val_loss = validate(cfg, model, val_loader, device, tb_writer, total_tb_it)

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                model_name = checkpoint_dir + name + '_model.pkl'
                os.makedirs(os.path.dirname(model_name), exist_ok=True)
                state = {'epoch': epoch, 'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict()}
                torch.save(state, model_name)

        tb_writer.close()


    elif args.problem == 'p-center':
        num_nodes = cfg_data['total_nodes']
        num_centers = 10
        checkpoint_dir = f'./checkpoint/{args.problem}/{args.model}/{num_nodes}_{num_centers}/'
        tb_log_dir = './tb_log/'

        # device = torch.device("cuda:1" if not args.no_cuda else "cpu")

        #train model
        name = 'gcn_test'
        tb_writer = SummaryWriter(tb_log_dir + name)
        dataset_path = f'./dataset/{args.problem}/synthetic/{num_nodes}_{num_centers}'
        train_loader, val_loader = build_data_loader(args.problem, dataset_path)

        # model = GCN(cfg).to(device)
        optimizer = optim.Adam(model.parameters(), args.learning_rate, (0.9, 0.999), eps=1e-08)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        total_tb_it = 0
        best_val_loss = 1000

        for epoch in range(args.epochs):
            scheduler.step(epoch)

            ####train model
            total_tb_it = train(cfg, model, train_loader, device, optimizer, epoch, tb_writer, total_tb_it)
            val_loss = validate(cfg, model, val_loader, device, tb_writer, total_tb_it)

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                model_name = checkpoint_dir + name + '_model.pkl'
                os.makedirs(os.path.dirname(model_name), exist_ok=True)
                state = {'epoch': epoch, 'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict()}
                torch.save(state, model_name)

        tb_writer.close()

    elif args.problem == 'MCLP':
        num_nodes = cfg_data['total_nodes']
        p = 20
        radius = 15
        checkpoint_dir = f'./checkpoint/{args.problem}/{args.model}/{num_nodes}_{p}_{radius}/'
        tb_log_dir = './tb_log/'

        # device = torch.device("cuda:1" if not args.no_cuda else "cpu")

        #train model
        name = 'gcn_test'
        tb_writer = SummaryWriter(tb_log_dir + name)

        dataset_path = f'./dataset/{args.problem}/synthetic/{num_nodes}_{p}_{radius}'
        train_loader, val_loader = build_data_loader(args.problem, dataset_path)
        # model = GCN(cfg).to(device)
        optimizer = optim.Adam(model.parameters(), args.learning_rate, (0.9, 0.999), eps=1e-08)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        total_tb_it = 0
        best_val_loss = 1000

        for epoch in range(args.epochs):
            scheduler.step(epoch)

            ####train model
            total_tb_it = train(cfg, model, train_loader, device, optimizer, epoch, tb_writer, total_tb_it)
            val_loss = validate(cfg, model, val_loader, device, tb_writer, total_tb_it)

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                model_name = checkpoint_dir + name + '_model.pkl'
                os.makedirs(os.path.dirname(model_name), exist_ok=True)
                state = {'epoch': epoch, 'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict()}
                torch.save(state, model_name)

        tb_writer.close()

    elif args.problem == 'LSCP':
        num_nodes = cfg_data['total_nodes']
        radius = 60
        checkpoint_dir = f'./checkpoint/{args.problem}/{args.model}/{num_nodes}_{radius}/'
        tb_log_dir = './tb_log/'

        # device = torch.device("cuda:1" if not args.no_cuda else "cpu")

        #train model
        name = 'gcn_test'
        tb_writer = SummaryWriter(tb_log_dir + name)

        dataset_path = f'./dataset/{args.problem}/synthetic/{num_nodes}_{radius}'
        train_loader, val_loader = build_data_loader(args.problem, dataset_path)
        # model = GCN(cfg).to(device)
        optimizer = optim.Adam(model.parameters(), args.learning_rate, (0.9, 0.999), eps=1e-08)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        total_tb_it = 0
        best_val_loss = 1000

        for epoch in range(args.epochs):
            scheduler.step(epoch)

            ####train model
            total_tb_it = train(cfg, model, train_loader, device, optimizer, epoch, tb_writer, total_tb_it)
            val_loss = validate(cfg, model, val_loader, device, tb_writer, total_tb_it)

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                model_name = checkpoint_dir + name + '_model.pkl'
                os.makedirs(os.path.dirname(model_name), exist_ok=True)
                state = {'epoch': epoch, 'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict()}
                torch.save(state, model_name)

        tb_writer.close()
    else:
        raise Exception("Unrecognized mode.")