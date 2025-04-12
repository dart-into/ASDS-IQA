import torch
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import torch.hub
import dataloader.dataLoder as data_loader
import torchvision.models as models
import numpy as np


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
        # out = nn.functional.softmax(out, dim=1)
        return out



class Net(nn.Module):
    def __init__(self, resnet, net):
        super(Net, self).__init__()
        self.resnet_layer = resnet
        self.net = net

    def forward(self, x):
        x = self.resnet_layer(x)
        x = self.net(x)
        return x
    
    def temperature_scaling(self, logits, temperature=1.0):
        """应用温度缩放"""
        return logits / temperature


class ClassifierSolver(object):
    """training and testing"""

    def __init__(self, config, path, train_idx, test_idx, train_round):
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.config = config
        self.train_round = train_round
        self.thresholds = None  # 初始化阈值


        self.net1 = models.densenet121(pretrained=True)
        self.net2 = BaselineModel1(25, 0.5, 1000)
        self.model = Net(resnet=self.net1, net=self.net2)
        self.model.cuda()
        self.model.train(True)

        self.l1_loss = torch.nn.L1Loss().cuda()
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay

        paras = [{'params': self.model.parameters(), 'lr': self.lr * self.lrratio}]

        self.optimizer = torch.optim.Adam(paras, weight_decay=self.weight_decay)



        # if config.mode == 'train':
        train_loader = data_loader.DataLoader(config.dataset,
                                  path,
                                  train_idx,
                                  config.patch_size,
                                  config.train_patch_num,
                                  batch_size=config.batch_size,
                                  istrain=True)
        self.train_data = train_loader.get_data()
        test_loader = data_loader.DataLoader(config.dataset,
                                 path,
                                 test_idx,
                                 config.patch_size,
                                 config.test_patch_num,
                                 istrain=False)
        self.test_data = test_loader.get_data()
        
    def train(self):  
        """训练方法"""  
        best_accuracy = 0.0  

        # 打印并准备文件记录头部信息  
        print('Epoch\tTrain_Loss\tTrain_Accuracy\tTest_Accuracy')  
        with open(f'#6_{self.config.dataset}_epoch_results.txt', 'a+') as f:  
            f.write('Epoch\tTrain_Loss\tTrain_Accuracy\tTest_Accuracy\n')  

            for t in range(self.epochs):  
                epoch_loss = []  
                correct = 0  
                total = 0  

                # 训练阶段  
                self.model.train()  # 设置模型为训练模式  
                for img, label in tqdm(self.train_data):  
                    # print(type(img))
                    img = img.cuda()  
                    label = label.cuda()  

                    self.optimizer.zero_grad()  

                    pred = self.model(img)  

                    # 计算损失  
                    pred = pred.float()          # 确保预测值为 Float 类型  
                    label = label.long()         # 确保标签为 Long 类型  

                    loss = self.criterion(pred, label)  
                    epoch_loss.append(loss.item())  
                    loss.backward()  
                    self.optimizer.step()  

                    # 计算准确率  
                    _, predicted = torch.max(pred.data, 1)  
                    total += label.size(0)  
                    correct += (predicted == label).sum().item()  

                train_accuracy = correct / total  

                # 评估阶段  
                test_accuracy = self.evaluate(self.test_data)  

                if test_accuracy > best_accuracy:  
                    best_accuracy = test_accuracy  
                    torch.save(self.model.state_dict(),  
                               f"./models/ClassificationModel_{self.config.dataset}_{self.train_round}.pth")  

                # 打印并记录到文件  
                avg_loss = sum(epoch_loss) / len(epoch_loss)  
                print(f'{t + 1}\t\t{avg_loss:.3f}\t\t{train_accuracy:.4f}\t\t{test_accuracy:.4f}')  
                f.write(f'{t + 1}\t\t{avg_loss:.3f}\t\t{train_accuracy:.4f}\t\t{test_accuracy:.4f}\n')  

                # 学习率调整  
                lr = (self.lr * 9) / (10 ** (t // 10))  
                self.optimizer.param_groups[0]['lr'] = lr  # 更新学习率  

        print(f'最佳测试准确率 {best_accuracy:.4f}')  
        return best_accuracy

    def evaluate(self, test_data):
        """评估方法"""
        self.model.eval()  # 设置模型为评估模式
        correct = 0
        total = 0

        with torch.no_grad():  # 禁用梯度追踪
            for img, label in test_data:
                img = img.cuda()
                label = label.cuda()

                pred = self.model(img)
                _, predicted = torch.max(pred.data, 1)

                total += label.size(0)
                correct += (predicted == label).sum().item()

        accuracy = correct / total
        return accuracy


#温度缩放自适应
    def calculate_thresholds(self, predictions):
        """计算自适应阈值"""
        predictions = np.array(predictions)
        self.thresholds = np.mean(predictions, axis=0) + np.std(predictions, axis=0)
        self.thresholds = self.thresholds.flatten()  # 确保 thresholds 是一维数组

    def vote_on_predictions(self, model_weights_path, temperature=1.5):
        """对测试数据进行投票预测"""
        # 加载模型权重
        self.model.load_state_dict(torch.load(model_weights_path))
        self.model.eval()

        arr = np.zeros(25, dtype=int)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        predictions = []  # 用于存储模型输出

        with torch.no_grad():
            for img, label in tqdm(self.test_data):
                img = img.to(device)
                logits = self.model(img)  # 获取 logits 输出

                # 对 logits 应用温度缩放
                scaled_logits = self.model.temperature_scaling(logits, temperature)
                
                # 使用 softmax 计算概率
                pred = F.softmax(scaled_logits, dim=1)
                predictions.append(pred.cpu().numpy())

        # 计算自适应阈值
        self.calculate_thresholds(predictions)

        # 对每个预测进行投票
        for pred in predictions:
            for i in range(25):
                if pred[0, i] > self.thresholds[i]:  # 使用一维的阈值
                    arr[i] += 1

        np.savetxt('vote.txt', arr, fmt='%d')
        return arr