import torch
from scipy import stats
import numpy as np
from tqdm import tqdm
from torch import nn
import torch.hub
import dataloader.dataLoder as data_loader
import timm


class Regressor(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(Regressor, self).__init__()
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
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        a0 = endpoints[0]  # [1, 24, 112, 112]
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

class IQAtestSolver(object):
    """training and testing"""
    def __init__(self, config, path, train_idx, test_idx, train_round):
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num

        self.config = config
        self.train_round = train_round

        pretrained_cfg_overlay = {'file': r"/path/to/tf_efficientnetv2_s_21k-6337ad01.pth"}
        self.net1 = timm.create_model('tf_efficientnetv2_s.in21k', pretrained_cfg_overlay=pretrained_cfg_overlay,
                                        features_only=True, pretrained=False)
        self.net2 = Regressor(1, 0.5, 256)
        self.model = Net(net1=self.net1, net2=self.net2)
        self.model.cuda()
        self.model.train(True)

        weight_path = 'path/to/model.pt'
        print("Loading...",weight_path)
        state_dict = torch.load(weight_path)
        self.model.load_state_dict(state_dict)
        self.l1_loss = torch.nn.L1Loss().cuda()
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay


        paras = [{'params': self.model.parameters(), 'lr': self.lr * self.lrratio}]
        self.optimizer = torch.optim.Adam(paras, weight_decay=self.weight_decay)
        test_loader = data_loader.DataLoader(config.dataset,
                                             path,
                                             test_idx,
                                             config.patch_size,
                                             config.test_patch_num,
                                             istrain=False)
        self.test_data = test_loader.get_data()

    def test(self):
        """Testing"""
        data = self.test_data
        self.model.train(False)
        pred_scores = []
        gt_scores = []

        for img, label in tqdm(data):
            img = img.cuda()
            label = label.cuda()

            pred = self.model(img)

            pred_scores.append(float(pred.item()))
            gt_scores = gt_scores + label.cpu().tolist()
            break

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)

        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        self.model.train(True)
        return test_srcc, test_plcc
