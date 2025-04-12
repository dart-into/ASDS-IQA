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
from efficientnet_pytorch import EfficientNet
import timm
from einops import rearrange
import matplotlib.pyplot as plt
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model
# from timm.models.vision_transformer import _cfg, Mlp, Block

# def visualize_feature_maps2(feature_map, name):
#
#     feature_maps = feature_map.cpu().detach().numpy()
#     x = name
#     filename2 = f"features/all_{x}.png"
#
#     summed_map = np.sum(feature_maps, axis=0)
#     map_min = np.min(summed_map)
#     map_max = np.max(summed_map)
#     normalized_map = (summed_map - map_min) / (map_max - map_min)
#     plt.imshow(normalized_map, cmap='viridis')
#     plt.axis('off')
#     plt.savefig(filename2)
#
#
#
# def visualize_feature_maps1(feature_map, name, size):
#
#     feature_maps = feature_map.cpu().detach().numpy()
#
#     x = name
#     filename1 = f"features/image_9_{x}.png"
#
#     num_channels = feature_maps.shape[0]
#     num_cols = size  # 每行显示16个通道
#     num_rows = num_channels // num_cols + 1
#
#     fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))
#
#     for i in range(num_channels):
#         row = i // num_cols
#         col = i % num_cols
#         ax = axes[row, col]
#         ax.imshow(feature_maps[i], cmap='viridis')  # 使用viridis色彩映射
#         ax.axis('off')
#
#     # 隐藏多余的子图
#     for i in range(num_channels, num_rows * num_cols):
#         row = i // num_cols
#         col = i % num_cols
#         fig.delaxes(axes[row, col])
#
#     plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.05, hspace=0.05)
#
#     if filename1:
#         plt.savefig(filename1)  # 保存可视化特征图
#     else:
#         plt.show()
    
    
      

class Net(nn.Module):
    def __init__(self, net1, net3, net4, vit, linear):
        super(Net, self).__init__()
        self.Net1 = net1
    
    def forward(self, x1, x3, x5):
       
        X1 = self.Net1(x1)
        
        
        x1 = rearrange(X1, 'b c h w  -> (b c) h w ', h=16, w=16)
        visualize_feature_maps2(x1, name = 'sa_dense')
        visualize_feature_maps1(x1, name = 'sa_dense', size=16)
        
        return X1
    
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
        a0 = endpoints[0]  # [1, 24, 112, 112]
        x0 = rearrange(a0, 'b c h w  -> (b c) h w ', h=112, w=112)
        visualize_feature_maps1(x0, name = '1', size=24)
        a1 = endpoints[1]  # [1, 48, 56, 56]
        # print("Shape of a1:", a1.shape)
        a0 = self.conv1(a0)
        # print("Shape of a0:", a0.shape)
        x1 = rearrange(a1, 'b c h w  -> (b c) h w ', h=56, w=56)
        visualize_feature_maps1(x1, name = '2', size=48)
        a1 = a0 + a1 
        x2 = rearrange(a1, 'b c h w  -> (b c) h w ', h=56, w=56)
        visualize_feature_maps1(x2, name = '3', size=48)
        print("Shape of a1:", a1.shape)
        x1 = rearrange(a1, 'b c h w  -> (b c) h w ', h=56, w=56)
        # print("Shape of x1:", x1.shape)
        visualize_feature_maps1(x1, name = '1', size=24)
        a2 = endpoints[2]  # [1, 64, 28, 28]
        # print("Shape of a2:", a2.shape)
        a1 = self.conv2(a1)
        a2 = a1 + a2
        a3 = endpoints[3]  # [1, 160, 14, 14]
        # print("Shape of a3:", a3.shape)
        a2 = self.conv3(a2)
        a3 = a2 + a3 
        a4 = endpoints[4]  # [1, 256, 7, 7]
        # print("Shape of a4:", a4.shape)
        a3 = self.conv4(a3)
        a4 = a3 + a4 
        # print("Shape of a4:", a4.shape)
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

        pretrained_cfg_overlay = {'file': r"/home/user/DACNN-main/tf_efficientnetv2_s_21k-6337ad01.pth"}
        self.net1 = timm.create_model('tf_efficientnetv2_s.in21k', pretrained_cfg_overlay=pretrained_cfg_overlay,
                                        features_only=True, pretrained=False)
        self.net2 = BaselineModel1(1, 0.5, 256)
        self.model = Net(net1=self.net1, net2=self.net2)
        self.model.cuda()
        self.model.train(True)
        
        #########################读取权重##########################
        weight_path = 'models/NewType1_top7.pth'
        # weight_path = 'models/bid_top10.pth'
        print("Loading...",weight_path)
        state_dict = torch.load(weight_path)
        self.model.load_state_dict(state_dict)
        #########################读取权重##########################

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
