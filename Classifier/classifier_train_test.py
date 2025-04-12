import torch
import argparse
import random
import numpy as np
from ClassifierSolver import ClassifierSolver
torch.cuda.set_device(0)

def main(config):
    folder_path = {
        'livec': './livec/',
        'koniq-10k': './koniq-10k/',
        'kadid-10k': './kadid-10k/',
        'bid': './bid/'
    }

    img_num = {
        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),
        'kadid-10k': list(range(0, 10125)),
        'bid': list(range(0, 586))
    }

    sel_num = img_num[config.dataset]
    accuracy_all = np.zeros(config.train_test_num, dtype=np.float)

    for i in range(config.train_test_num):
        train_round = i + 1
        random.shuffle(sel_num)
        train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]

        classifier = ClassifierSolver(config, folder_path[config.dataset], train_index, test_index, train_round)
        accuracy_all[i] = classifier.train()

    accuracy_med = np.median(accuracy_all)
    print('Testing median accuracy: %4.4f' % (accuracy_med))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='kadid-10k')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=1)
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=1)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0)
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--epochs', dest='epochs', type=int, default=50)
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224)
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=1)
    parser.add_argument('--mode', dest='mode', type=str, choices=['train', 'test'], default='train')
    config = parser.parse_args()
    main(config)