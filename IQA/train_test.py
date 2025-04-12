import torch
import argparse
import random
import numpy as np

from IQASolver import IQASolver

torch.cuda.set_device(0)


def main(config):
    
    result_file_name='IQA_train.txt'
    
    folder_path = {
        'live': './LIVE/',
        'csiq': './CSIQ/',
        'tid2013': './TID2013/',
        'livec': '/home/y222212028/MetaIQA/MetaIQA-master/LIVE_WILD',
        'koniq-10k': './KonIQ-10k/',
        'cid2013': './CID2013/',
        'kadid-10k': '/home/user/MetaIQA/MetaIQA-master/kadid10k/',
        'SPAQ': './SPAQ/'
    }

    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),
        'cid2013': list(range(0, 6)),
        'kadid-10k': list(range(0, 4050)),
        'SPAQ': list(range(0, 11124))
    }

    sel_num = img_num[config.dataset]

    srcc_all = np.zeros(config.train_test_num, dtype=np.float)
    plcc_all = np.zeros(config.train_test_num, dtype=np.float)

    print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))

    # 打开结果文件以追加模式
    with open(result_file_name, 'a') as result_file:
        for i in range(config.train_test_num):
            print('Round %d' % (i + 1))
            train_round = i + 1

            random.shuffle(sel_num)

            train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
            test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]

            solver = IQASolver(config, folder_path[config.dataset], train_index, test_index, train_round)
            srcc_all[i], plcc_all[i] = solver.train()

            # 将每轮的结果写入文件
            result_file.write(f'Round {i + 1}: SRCC = {srcc_all[i]:.4f}, PLCC = {plcc_all[i]:.4f}\n')

    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)

    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))

    # 也将中位数结果写入文件
    with open(result_file_name, 'a') as result_file:
        result_file.write(f'Median SRCC = {srcc_med:.4f}, Median PLCC = {plcc_med:.4f}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='kadid-10k',
                        help='Support datasets: livec|koniq-10k|cid2013|live|csiq|tid2013|SPAQ|kadid-10k')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=1,
                        help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=1,
                        help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10,
                        help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=30, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                        help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=5, help='Train-test times')

    config = parser.parse_args()
    main(config)
