import torch
import torchvision
import dataloader.dataset_folders as dataset_folders


class DataLoader:
    def __init__(self, dataset, path, img_indx, patch_size, patch_num, batch_size=1, istrain=True):
        self.batch_size = batch_size
        self.istrain = istrain

        base_transforms = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ]


        resize_size = (384, 384)
        transforms = [torchvision.transforms.Resize(resize_size)]

        if istrain:
            transforms.append(torchvision.transforms.RandomHorizontalFlip())

        transforms.append(torchvision.transforms.RandomCrop(size=patch_size))
        transforms.extend(base_transforms)

        self.transforms = torchvision.transforms.Compose(transforms)

        dataset_mapping = {
            'bid': dataset_folders.bidFolder,
            'livec': dataset_folders.LIVEChallengeFolder,
            'koniq-10k': dataset_folders.Koniq_10kFolder,
            'kadid-10k': dataset_folders.Kadid_10kFolder
        }

        self.data = dataset_mapping[dataset](
            root=path,
            index=img_indx,
            transform=self.transforms,
            patch_num=patch_num
        )

    def get_data(self):
        return torch.utils.data.DataLoader(
            self.data,
            batch_size=self.batch_size if self.istrain else 1,
            shuffle=self.istrain,
            num_workers=4
        )