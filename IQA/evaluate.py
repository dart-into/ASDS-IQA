import torch
import argparse
import random
import numpy as np

from IQAtestSolver import IQAtestSolver

torch.cuda.set_device(0)


import numpy as np
import random

def main(config):
    
    result_file_name='IQA_test_LIVEC.txt'

    folder_path = {
        'live': './LIVE/',
        'csiq': './CSIQ/',
        'tid2013': './TID2013/',
        'livec': '/home/user/MetaIQA/MetaIQA-master/LIVE_WILD',
        'koniq-10k': '/home/user/Dataset/Koniq-10k/',
        'cid2013': './CID2013/',
        'kadid-10k': '/home/user/MetaIQA/MetaIQA-master/kadid10k/',
        'SPAQ': './SPAQ/',
        'bid':'/home/user/Dataset/BID/'
    }

    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),
        'cid2013': list(range(0, 6)),
        'kadid-10k': list(range(0, 2430)),
        'SPAQ': list(range(0, 11124)),
        'bid': list(range(0, 586))
    }

    sel_num = img_num[config.dataset]

    srcc_all = np.zeros(config.train_test_num, dtype=np.float)
    plcc_all = np.zeros(config.train_test_num, dtype=np.float)

    print('正在测试 %s 数据集，共 %d 轮...' % (config.dataset, config.train_test_num))

    # 以追加模式打开结果文件
    with open(result_file_name, 'a') as result_file:
        for i in range(config.train_test_num):
            print('第 %d 轮' % (i + 1))
            test_round = i + 1

            random.shuffle(sel_num)

            # 使用所有样本进行测试
            test_index = sel_num  # 直接使用所有索引进行测试

            # 假设有一个方法来加载预训练模型
            solver = IQAtestSolver(config, folder_path[config.dataset], None, test_index, test_round)
            srcc_all[i], plcc_all[i] = solver.test()  # 更改为假设的测试方法

            # 将每一轮的结果写入文件
            result_file.write(f'第 {i + 1} 轮: SRCC = {srcc_all[i]:.4f}, PLCC = {plcc_all[i]:.4f}\n')

    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)

    print('测试的中位数 SRCC %4.4f,\t中位数 PLCC %4.4f' % (srcc_med, plcc_med))

    # 也将中位数结果写入文件
    with open(result_file_name, 'a') as result_file:
        result_file.write(f'中位数 SRCC = {srcc_med:.4f}, 中位数 PLCC = {plcc_med:.4f}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='livec',
                        help='Support datasets: livec|koniq-10k|cid2013|live|csiq|tid2013|SPAQ|kadid-10k|bid')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=1,
                        help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=1,
                        help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10,
                        help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=50, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                        help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=1, help='Train-test times')

    config = parser.parse_args()
    main(config)
