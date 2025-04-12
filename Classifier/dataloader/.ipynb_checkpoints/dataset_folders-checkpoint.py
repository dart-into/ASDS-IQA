import torch.utils.data as data
from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
import csv
import torchvision.utils as utils
#from openpyxl import load_workbook
from PIL import Image

class LIVEFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        self.patch_num = patch_num

        refpath = os.path.join(root, 'refimgs')
        refname = getFileName(refpath, '.bmp')

        jp2kroot = os.path.join(root, 'jp2k')
        jp2kname = self.getDistortionTypeFileName(jp2kroot, 227)

        jpegroot = os.path.join(root, 'jpeg')
        jpegname = self.getDistortionTypeFileName(jpegroot, 233)

        wnroot = os.path.join(root, 'wn')
        wnname = self.getDistortionTypeFileName(wnroot, 174)

        gblurroot = os.path.join(root, 'gblur')
        gblurname = self.getDistortionTypeFileName(gblurroot, 174)

        fastfadingroot = os.path.join(root, 'fastfading')
        fastfadingname = self.getDistortionTypeFileName(fastfadingroot, 174)

        imgpath = jp2kname + jpegname + wnname + gblurname + fastfadingname

        dmos = scipy.io.loadmat(os.path.join(root, 'dmos_realigned.mat'))
        labels = dmos['dmos_new'].astype(np.float32)

        orgs = dmos['orgs']
        refnames_all = scipy.io.loadmat(os.path.join(root, 'refnames_all.mat'))
        refnames_all = refnames_all['refnames_all']

        sample = []

        for i in range(0, len(index)):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = train_sel * ~orgs.astype(np.bool_)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[1].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((imgpath[item], labels[0][item]))
                # print(self.imgpath[item])
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        # print('self.patch_num', self.patch_num)
        # if self.patch_num == 5:
        #     print('path', path)
        #     utils.save_image(sample, "D:/00CODE00/Chiralnet_test/images/{}.bmp".format(index), normalize=True)

        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = '%s%s%s' % ('img', str(index), '.bmp')
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename

class LIVEChallengeFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        imgpath = scipy.io.loadmat(os.path.join(root, 'Data', 'AllImages_release.mat'))
        imgpath = imgpath['AllImages_release']
        imgpath = imgpath[7:1169]
        mos = scipy.io.loadmat(os.path.join(root, 'Data', 'AllMOS_release.mat'))
        labels = mos['AllMOS_release'].astype(np.float32)
        labels = labels[0][7:1169]

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'images', imgpath[item][0][0]), labels[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        print(f"Sample path: ",path)
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)

        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length
    
class LIVEC2Folder(data.Dataset):
    def __init__(self, root: str, index: list, transform: callable, patch_num: int):
        self.imgnames = []
        self.noise = []

        # CSV 文件路径
        csv_file = os.path.join(root, 'livec_image_labeled_by_per_noise.csv')
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV 文件 '{csv_file}' 不存在。")

        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.imgnames.append(row['image'])
                self.noise.append(float(row['noise']))  # 将 dmos 改为 noise

        self.samples = []
        
        # 创建样本列表
        for i in index:
            if i < len(self.imgnames):  # 确保索引在范围内
                for _ in range(patch_num):
                    # 转换噪声值为标签 (从 1-25 转换为 0-24)
                    label = int(self.noise[i]) - 1  # 减去 1
                    self.samples.append((os.path.join(root, 'images', self.imgnames[i]), label))

        self.transform = transform

    def __getitem__(self, index: int):
        img_path, target = self.samples[index]
        img = pil_loader(img_path)  # 假设 pil_loader 函数已定义
        
        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.samples)

    
class CSIQFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        refpath = os.path.join(root, 'src_imgs')
        refname = getFileName(refpath, '.png')
        txtpath = os.path.join(root, 'im_names.txt')
        mostxt = os.path.join(root, 'scores.txt')
        fh = open(txtpath, 'r')
        lh = open(mostxt, 'r')
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[0]))
            ref_temp = words[0].split(".")
            ref_tempp = ref_temp[0].split("/")
            refnames_all.append(ref_tempp[1] + '.' + ref_temp[-1])
        for line in lh:
            line = line.split('\n')
            words = line[0].split()
            target.append(words[0])

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        sample = []

        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((os.path.join(root, 'dst_imgs', imgnames[item]), labels[item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)

        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

class CID2013Folder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        imgnames = []
        target = []
        print('index', index)
        for i in range(len(index)):
            n = int(index[i])
            if n == 0:
                txtpath = os.path.join(root, 'IS1.txt')
                fh = open(txtpath, 'r')
                for line in fh:
                    line = line.split('\n')
                    words = line[0].split()
                    imgnames.append((words[0]))
                    target.append(words[1])

            elif n == 1:
                txtpath = os.path.join(root, 'IS2.txt')
                fh = open(txtpath, 'r')
                for line in fh:
                    line = line.split('\n')
                    words = line[0].split()
                    imgnames.append((words[0]))
                    target.append(words[1])

            elif n == 2:
                txtpath = os.path.join(root, 'IS3.txt')
                fh = open(txtpath, 'r')
                for line in fh:
                    line = line.split('\n')
                    words = line[0].split()
                    imgnames.append((words[0]))
                    target.append(words[1])

            elif n == 3:
                txtpath = os.path.join(root, 'IS4.txt')
                fh = open(txtpath, 'r')
                for line in fh:
                    line = line.split('\n')
                    words = line[0].split()
                    imgnames.append((words[0]))
                    target.append(words[1])

            elif n == 4:
                txtpath = os.path.join(root, 'IS5.txt')
                fh = open(txtpath, 'r')
                for line in fh:
                    line = line.split('\n')
                    words = line[0].split()
                    imgnames.append((words[0]))
                    target.append(words[1])

            elif n == 5:
                txtpath = os.path.join(root, 'IS6.txt')
                fh = open(txtpath, 'r')
                for line in fh:
                    line = line.split('\n')
                    words = line[0].split()
                    imgnames.append((words[0]))
                    target.append(words[1])

        labels = np.array(target).astype(np.float32)


        sample = []
        for i in range(len(imgnames)):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'images', imgnames[i]+'.jpg'), labels[i]))




        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        # print("path:{}  target:{}".format(path, target))

        sample = pil_loader(path)
        sample = self.transform(sample)

        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

class Kadid_10kFolder(data.Dataset):
    
    def __init__(self, root: str, index: list, transform: callable, patch_num: int):
        self.imgnames = []
        self.noise = []

        # CSV 文件路径
        csv_file = os.path.join(root, 'image_labeled_by_per_noise.csv')
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV 文件 '{csv_file}' 不存在。")

        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.imgnames.append(row['image'])
                self.noise.append(float(row['noise']))  # 将 dmos 改为 noise

        self.samples = []
        
        # 创建样本列表
        for i in index:
            if i < len(self.imgnames):  # 确保索引在范围内
                for _ in range(patch_num):
                    # 转换噪声值为标签 (从 1-25 转换为 0-24)
                    label = int(self.noise[i]) - 1  # 减去 1
                    self.samples.append((os.path.join(root, 'images', self.imgnames[i]), label))

        self.transform = transform

    def __getitem__(self, index: int):
        img_path, target = self.samples[index]
        img = pil_loader(img_path)  # 假设 pil_loader 函数已定义
        
        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.samples)

class SPAQdataFloder:
    def __init__(self,root,index,transform,patch_num):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'MOS and Image attribute scores.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['Image name'])
                mos = np.array(float(row['MOS'])).astype(np.float32)
                mos_all.append(mos)

            sample = []
            for i, item in enumerate(index):
                for aug in range(patch_num):
                    sample.append((os.path.join(root, 'TestImage', imgname[item]),mos_all[item]))

            self.samples = sample
            self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

class Koniq_10kFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'koniq10k_scores_and_distributions.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                mos = np.array(float(row['MOS'])).astype(np.float32)
                mos_all.append(mos)

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'images', imgname[item]), mos_all[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length
    
class bidFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'grades.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                mos = np.array(float(row['MOS'])).astype(np.float32)
                mos_all.append(mos)

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'ImageDatabase', imgname[item]), mos_all[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

class TID2013Folder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        refpath = os.path.join(root, 'reference_images')
        refname = getTIDFileName(refpath, '.bmp.BMP')
        txtpath = os.path.join(root, 'mos_with_names.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[1]))
            target.append(words[0])
            ref_temp = words[1].split("_")
            refnames_all.append(ref_temp[0][1:])
        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        sample = []
        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((os.path.join(root, 'distorted_images', imgnames[item]), labels[item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename

def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')