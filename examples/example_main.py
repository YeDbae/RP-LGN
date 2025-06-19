# part of our work (graph conv) is derived from BrainGB, at: https://github.com/HennyJie/BrainGB

from .build_model import build_model
from .modified_args import ModifiedArgs
from .get_transform import get_transform
import argparse
import sys
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
import nni
import os
import random
from typing import List
import logging

from src.dataset import BrainDataset, BrainData
from src.dataset.maskable_list import MaskableList
from src.utils import calculate_bin_edges, get_y
from .train_and_evaluate import train_and_evaluate, evaluate,evaluate2
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# python -m examples.example_main

def seed_everything(seed):
    print(f"seed for seed_everything(): {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # set random seed for numpy
    torch.manual_seed(seed)  # set random seed for CPU
    torch.cuda.manual_seed_all(seed)  # set random seed for all GPUs


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    if args.enable_nni:
        args = ModifiedArgs(args, nni.get_next_parameter())

    # init model
    model_name = str(args.model_name).lower()
    args.model_name = model_name
    # seed_everything(args.seed) # use args.seed for each run
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self_dir = os.path.dirname(os.path.realpath(__file__))

    if args.dataset_name == 'ABIDE':
        root_dir = os.path.join(self_dir, 'datasets/ABIDE/')
    elif args.dataset_name == 'ADHD':
        root_dir = os.path.join(self_dir, 'datasets/ADHD/')
    elif args.dataset_name == 'NYU':
        root_dir = os.path.join(self_dir, 'datasets/NYU/')
    elif args.dataset_name == 'KKI':
        root_dir = os.path.join(self_dir, 'datasets/KKI/')
    elif args.dataset_name == 'UCLA':
        root_dir = os.path.join(self_dir, 'datasets/UCLA/')
    elif args.dataset_name == 'UM':
        root_dir = os.path.join(self_dir, 'datasets/UM/')
    else:
        root_dir = os.path.join(self_dir, 'datasets/')
    dataset = BrainDataset(root=root_dir,
                           name=args.dataset_name,
                           pre_transform=get_transform(args.node_features))
    y = get_y(dataset)
    num_features = dataset[0].x.shape[1]
    nodes_num = dataset.num_nodes

    # if args.model_name == 'gcn':
    #     bin_edges = calculate_bin_edges(dataset, num_bins=args.bucket_num)
    # else:
    #     bin_edges = None

    accs, aucs, macros, exp_accs, exp_aucs, exp_macros,post_accs = [], [], [], [], [], [],[]
    for _ in range(args.repeat):
        seed_everything(random.randint(1, 1000000))  # use random seed for each run
        skf = StratifiedKFold(n_splits=args.k_fold_splits, shuffle=True)
        for train_index, test_index in skf.split(dataset, y):
            model = build_model(args, device, model_name, num_features, nodes_num)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # Adam also
            train_index, test_index = np.array(train_index), np.array(test_index)  # added by lily

            train_index, test_index = train_index.tolist(), test_index.tolist()  # added by lily
            train_set, test_set = dataset[train_index], dataset[test_index]

            train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False)

            # train
            test_acc, test_auc, test_macro = train_and_evaluate(model, train_loader, test_loader,
                                                                  optimizer, device, args)
            best_test_acc, best_test_f1, best_test_auc=0,0,0

            # train_acc, train_auc, train_macro, test_acc, test_auc, test_macro, best_acc, best_f1, best_auc= evaluate(model, device,best_test_acc, best_test_f1, best_test_auc, test_loader)
            test_acc, test_auc, test_macro, best_acc, best_f1, best_auc = evaluate(
                model, device, best_test_acc, best_test_f1, best_test_auc, test_loader,test_loader)
            # test_acc, test_auc, test_macro, best_test_acc = evaluate(model, device,best_test_acc, best_test_f1, best_test_auc, test_loader)
            logging.info(f'(Initial Performance Last Epoch) | test_acc={(test_acc * 100):.2f}, '
                         f'test_f1={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f},'
                         f'best_test_acc={(best_acc * 100):.2f}')

            accs.append(test_acc)
            aucs.append(test_auc)
            macros.append(test_macro)

    result_str = f'(K Fold Final Result)| avg_acc={(np.mean(accs) * 100):.2f} +- {(np.std(accs) * 100): .2f}, ' \
                 f'avg_auc={(np.mean(aucs) * 100):.2f} +- {np.std(aucs) * 100:.2f}, ' \
                 f'avg_macro={(np.mean(macros) * 100):.2f} +- {np.std(macros) * 100:.2f}\n'
    logging.info(result_str)

    with open('result.log', 'a') as f:
        # write all input arguments to f
        input_arguments: List[str] = sys.argv
        f.write(f'{input_arguments}\n')
        f.write(result_str + '\n')
    if args.enable_nni:
        nni.report_final_result(np.mean(aucs))


def count_degree(data: np.ndarray):  # data: (sample, node, node)
    count = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        count[i, :] = np.sum(data[:, i, :] != 0, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str,
                        choices=['ABIDE', 'ADHD', 'UCLA', 'UM', 'KKI', 'NYU'],
                        default="ABIDE")
    parser.add_argument('--view', type=int, default=1)
    parser.add_argument('--node_features', type=str,
                        choices=['identity', 'degree', 'degree_bin', 'LDP', 'node2vec', 'adj', 'diff_matrix',
                                 'eigenvector', 'eigen_norm'],
                        default='adj')
    parser.add_argument('--pooling', type=str,
                        choices=['sum', 'concat', 'mean'],
                        default='concat')
                        
    parser.add_argument('--model_name', type=str, default='sage')
    # gcn_mp_type choices: weighted_sum, bin_concate, edge_weight_concate, edge_node_concate, node_concate
    parser.add_argument('--gcn_mp_type', type=str, default="edge_node_concate")
    # gat_mp_type choices: attention_weighted, attention_edge_weighted, sum_attention_edge, edge_node_concate, node_concate
    parser.add_argument('--gat_mp_type', type=str, default="attention_weighted") 

    parser.add_argument('--enable_nni', action='store_true')
    parser.add_argument('--n_GNN_layers', type=int, default=2)
    parser.add_argument('--n_MLP_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=256)  # 360
    parser.add_argument('--gat_hidden_dim', type=int, default=8)
    parser.add_argument('--edge_emb_dim', type=int, default=256)
    parser.add_argument('--bucket_sz', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=1e-5)    # 1e-4 in BrainGB? #VBLL3e-3
    parser.add_argument('--weight_decay', type=float, default=1e-4)  # 1e-4, original
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--k_fold_splits', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--test_interval', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    parser.add_argument('--seed', type=int, default=112078)
    parser.add_argument('--diff', type=float, default=0.2)
    parser.add_argument('--mixup', type=int, default=1) #[0, 1]
    parser.add_argument('--ratio', type=float, default=0.5)

    main(parser.parse_args())
