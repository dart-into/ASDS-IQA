import torch
import argparse
import random
import numpy as np
from IQASolver import IQASolver

torch.cuda.set_device(0)


def main(config):
    folder_path = {
        'livec': '/path/to/LIVE_WILD',
        'koniq-10k': '/path/to/Koniq-10k/',
        'kadid-10k': '/path/to/kadid10k/',
        'bid': '/path/to/BID/'
    }

    img_num = {
        'livec': list(range(0, 1162)),
        'koniqk': list(range(0, 10073)),
        'kadid-10k': list(range(0, 10125)),
        'bid': list(range(0, 586))
    }

    sel_num = img_num[config.dataset]
    srcc_all = np.zeros(config.train_test_num, dtype=np.float)
    plcc_all = np.zeros(config.train_test_num, dtype=np.float)

    print(f'Training and testing on {config.dataset} dataset for {config.train_test_num} rounds...')

    for i in range(config.train_test_num):
        print(f'Round {i + 1}')
        train_round = i + 1
        random.shuffle(sel_num)

        split_point = int(round(0.8 * len(sel_num)))
        train_index = sel_num[:split_point]
        test_index = sel_num[split_point:]

        solver = IQASolver(config, folder_path[config.dataset], train_index, test_index, train_round)
        srcc_all[i], plcc_all[i] = solver.train()

    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)
    print(f'Testing median SRCC {srcc_med:.4f},\tmedian PLCC {plcc_med:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kadid-10k')
    parser.add_argument('--train_patch_num', type=int, default=1)
    parser.add_argument('--test_patch_num', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr_ratio', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('s', type=int, default=30)
    parser.add_argument('--patch_size', type=int, default=224)
    parser.add_argument('--train_test_num', type=int, default=5)

    config = parser.parse_args()
    main(config)