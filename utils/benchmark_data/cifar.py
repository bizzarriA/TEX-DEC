import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

class ADCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, target_classes=[0]):
        self.cifar10 = CIFAR10(root=root, train=train, download=True, transform=transform)
        self.transform = transform
        self.target_classes = target_classes

        self.data = []
        self.labels = []

        for img, label in self.cifar10:
            if train:
                # If train add target class
                if label in target_classes:
                    self.data.append(img)
                    self.labels.append(label)
            else:
                # Else add only OOD
                if label not in target_classes:
                    self.data.append(img)
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]

        label = self.labels[idx]

        return img, label, torch.tensor(0)

