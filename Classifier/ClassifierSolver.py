import torch
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import torch.hub
import dataloader.dataLoder as data_loader
import torchvision.models as models
import numpy as np
import math


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
    def __init__(self, net1, net2):
        super(Net, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        return x

    def temperature_scaling(self, logits, temperature=1.0):
        return logits / temperature


class ClassifierSolver:
    def __init__(self, config, path, train_idx, test_idx, train_round):
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.config = config
        self.train_round = train_round
        self.thresholds = None
        self.net1 = models.densenet121(pretrained=True)
        self.net2 = Regressor(25, 0.5, 1000)
        self.model = Net(net1=self.net1, net2=self.net2)
        self.model.cuda()
        self.model.train(True)
        self.l1_loss = torch.nn.L1Loss().cuda()
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay

        paras = [{'params': self.model.parameters(), 'lr': self.lr * self.lrratio}]
        self.optimizer = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(
            config.dataset,
            path,
            train_idx,
            config.patch_size,
            config.train_patch_num,
            batch_size=config.batch_size,
            istrain=True
        )
        self.train_data = train_loader.get_data()

        test_loader = data_loader.DataLoader(
            config.dataset,
            path,
            test_idx,
            config.patch_size,
            config.test_patch_num,
            istrain=False
        )
        self.test_data = test_loader.get_data()

    def train(self):
        best_accuracy = 0.0

        for t in range(self.epochs):
            epoch_loss = []
            correct = 0
            total = 0

            self.model.train()
            for img, label in tqdm(self.train_data):
                img = img.cuda()
                label = label.cuda()

                self.optimizer.zero_grad()
                pred = self.model(img)
                pred = pred.float()
                label = label.long()

                loss = self.criterion(pred, label)
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(pred.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

            train_accuracy = correct / total
            test_accuracy = self.evaluate(self.test_data)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(
                    self.model.state_dict(),
                    f"./models/ClassificationModel_{self.config.dataset}_{self.train_round}.pth"
                )

            lr = (self.lr * 9) / (10 ** (t // 10))
            self.optimizer.param_groups[0]['lr'] = lr

        return best_accuracy

    def evaluate(self, test_data):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for img, label in test_data:
                img = img.cuda()
                label = label.cuda()

                pred = self.model(img)
                _, predicted = torch.max(pred.data, 1)

                total += label.size(0)
                correct += (predicted == label).sum().item()

        return correct / total

    def calculate_thresholds(self, predictions):
        predictions = np.array(predictions)
        self.thresholds = np.mean(predictions, axis=0) + np.std(predictions, axis=0)
        self.thresholds = self.thresholds.flatten()

    def vote_on_predictions(self, model_weights_path, temperature=1.5):
        self.model.load_state_dict(torch.load(model_weights_path))
        self.model.eval()

        arr = np.zeros(25, dtype=int)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        predictions = []

        with torch.no_grad():
            for img, label in tqdm(self.test_data):
                img = img.to(device)
                logits = self.model(img)
                scaled_logits = self.model.temperature_scaling(logits, temperature)
                pred = F.softmax(scaled_logits, dim=1)
                predictions.append(pred.cpu().numpy())

        self.calculate_thresholds(predictions)

        for pred in predictions:
            for i in range(25):
                if pred[0, i] > self.thresholds[i]:
                    arr[i] += 1

        return arr