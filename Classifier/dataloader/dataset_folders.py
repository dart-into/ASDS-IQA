import torch.utils.data as data
import os
import numpy as np
import csv
from PIL import Image

class LIVEC2Folder(data.Dataset):
    def __init__(self, root: str, index: list, transform: callable, patch_num: int):
        self.imgnames = []
        self.noise = []
        csv_file = os.path.join(root, 'livec_image_labeled_by_per_noise.csv')

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV 文件 '{csv_file}' 不存在。")

        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.imgnames.append(row['image'])
                self.noise.append(float(row['noise']))

        self.samples = []
        for i in index:
            if i < len(self.imgnames):
                for _ in range(patch_num):
                    label = int(self.noise[i]) - 1
                    self.samples.append((os.path.join(root, 'images', self.imgnames[i]), label))

        self.transform = transform

    def __getitem__(self, index: int):
        img_path, target = self.samples[index]
        img = pil_loader(img_path)
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.samples)

class Kadid_10kFolder(data.Dataset):
    def __init__(self, root: str, index: list, transform: callable, patch_num: int):
        self.imgnames = []
        self.noise = []
        csv_file = os.path.join(root, 'image_labeled_by_per_noise.csv')

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV 文件 '{csv_file}' 不存在。")

        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.imgnames.append(row['image'])
                self.noise.append(float(row['noise']))

        self.samples = []
        for i in index:
            if i < len(self.imgnames):
                for _ in range(patch_num):
                    label = int(self.noise[i]) - 1
                    self.samples.append((os.path.join(root, 'images', self.imgnames[i]), label))

        self.transform = transform

    def __getitem__(self, index: int):
        img_path, target = self.samples[index]
        img = pil_loader(img_path)
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.samples)

class Koniq_10kFolder(data.Dataset):
    def __init__(self, root, index, transform, patch_num):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'koniq10k_scores_and_distributions.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                mos_all.append(np.array(float(row['MOS'])).astype(np.float32))

        self.samples = []
        for i, item in enumerate(index):
            for _ in range(patch_num):
                self.samples.append((os.path.join(root, 'images', imgname[item]), mos_all[item]))

        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

class bidFolder(data.Dataset):
    def __init__(self, root, index, transform, patch_num):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'grades.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                mos_all.append(np.array(float(row['MOS'])).astype(np.float32))

        self.samples = []
        for i, item in enumerate(index):
            for _ in range(patch_num):
                self.samples.append((os.path.join(root, 'ImageDatabase', imgname[item]), mos_all[item]))

        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')