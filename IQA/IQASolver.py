import torch
from scipy import stats
import numpy as np
from tqdm import tqdm
from torch import nn
from skimage import transform
import torch.nn.functional as F
import torch.hub
from functools import partial
from network import DACNN
import dataloader.dataLoder as data_loader
import torchvision.models as models
import os
import timm

class BaselineModel1(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(BaselineModel1, self).__init__()
        self.fc1 = nn.Linear(inputsize, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.02)
            #     m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        out = self.fc1(x)

        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        return out


class Net(nn.Module):
    def __init__(self , net1, net2):
        super(Net, self).__init__()
        self.net1 = net1
        self.net2 = net2
        self.conv1 = nn.Conv2d(in_channels=24,out_channels=48,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(in_channels=48,out_channels=64,kernel_size=3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=160,kernel_size=3,stride=2,padding=1)
        self.conv4 = nn.Conv2d(in_channels=160,out_channels=256,kernel_size=3,stride=2,padding=1)

    def forward(self, x):
        endpoints = self.net1(x)
        a0 = endpoints[0]  # [1, 24, 56, 56]
        a1 = endpoints[1]  # [1, 48, 56, 56]
        a0 = self.conv1(a0)
        a1 = a0 + a1 
        a2 = endpoints[2]  # [1, 64, 28, 28]
        a1 = self.conv2(a1)
        a2 = a1 + a2 
        a3 = endpoints[3]  # [1, 160, 14, 14]
        a2 = self.conv3(a2)
        a3 = a2 + a3 
        a4 = endpoints[4]  # [1, 256, 7, 7]
        a3 = self.conv4(a3)
        a4 = a3 + a4 
        a4 = a4.mean(dim=[2, 3])
        x = self.net2(a4)
        return x

class IQASolver(object):
    """training and testing"""
    def __init__(self, config, path, train_idx, test_idx, train_round):
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num

        self.config = config
        self.train_round = train_round

        pretrained_cfg_overlay = {'file': r"/home/user/DACNN-main/tf_efficientnetv2_s_21k-6337ad01.pth"}
        self.net1 = timm.create_model('tf_efficientnetv2_s.in21k', pretrained_cfg_overlay=pretrained_cfg_overlay,
                                        features_only=True, pretrained=True)
        self.net2 = BaselineModel1(1, 0.5, 256)
        self.model = Net(net1=self.net1, net2=self.net2)
        self.model.cuda()
        self.model.train(True)

        self.l1_loss = torch.nn.L1Loss().cuda()
        self.lr = config.lr 
        self.lrratio = config.lr_ratio 
        self.weight_decay = config.weight_decay


        paras = [{'params': self.model.parameters(), 'lr': self.lr * self.lrratio}]

        self.optimizer = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(config.dataset,
                                              path,
                                              train_idx,
                                              config.patch_size,
                                              config.train_patch_num,
                                              batch_size=config.batch_size,
                                              istrain=True)
        test_loader = data_loader.DataLoader(config.dataset,
                                             path,
                                             test_idx,
                                             config.patch_size,
                                             config.test_patch_num,
                                             istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self):
        """Training"""
        best_srcc = 0.0
        best_plcc = 0.0

        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            for img, label in tqdm(self.train_data):

                img = img.cuda()
                label = label.cuda()

                self.optimizer.zero_grad()

                pred = self.model(img)

                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                loss = self.l1_loss(pred.squeeze(), label.float().detach())
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

            test_srcc, test_plcc = self.test(self.test_data)

            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                torch.save(self.model.state_dict(),"./models/densenet_"+ str(self.config.dataset) + "_" + str(self.train_round) + ".pth")
                
            print('%d\t\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc))

         
            lr = (self.lr * 9) / pow(10, (t // 10))
            self.paras = [{'params': self.model.parameters(), 'lr': lr}]
            self.optimizer = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        self.model.train(False)
        pred_scores = []
        gt_scores = []

        for img, label in tqdm(data):
            img = img.cuda()
            label = label.cuda()

            pred = self.model(img)

            pred_scores.append(float(pred.item()))
            gt_scores = gt_scores + label.cpu().tolist()

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)

        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        self.model.train(True)
        return test_srcc, test_plcc
