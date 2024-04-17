from build_augmented_data import *

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import precision_score, recall_score

import os
import pickle
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Microplastics_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Root directory containing class subdirectories with images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(root_dir)
        self.samples = self._make_dataset(root_dir, self.class_to_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target

    def _find_classes(self, directory):
        classes = [class_entry.name for class_entry in os.scandir(directory) if class_entry.is_dir()]
        classes.sort()
        class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, directory, class_to_idx):
        images = []
        for target in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target]
            target_dir = os.path.join(directory, target)
            for root, _, filenames in sorted(os.walk(target_dir)):
                for filename in sorted(filenames):
                    path = os.path.join(root, filename)
                    sample = (path, class_index)
                    images.append(sample)
        return images

    def count_instances_per_class(self):
        instances_per_class = {cls: 0 for cls in self.classes}
        for _, class_index in self.samples:
            instances_per_class[self.classes[class_index]] += 1
        return instances_per_class

    def print_instances_per_class(self):
        instances_per_class = self.count_instances_per_class()
        for cls, count in instances_per_class.items():
            print(f"Class '{cls}': {count} instances")


class SyntheticDataChooser_CNN(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 58 * 58, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # print(x.size())
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.size())
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # print(x.size())
        x = F.relu(self.fc1(x))
        # print(x.size())
        x = F.relu(self.fc2(x))
        # print(x.size())
        x = self.fc3(x)
        # print(x.size())
        # exit()
        return x


class SyntheticDataChooser_ResNet50(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        self.resnet = models.resnet50(pretrained=self.pretrained)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


def get_accuracy(test_loader):
    total = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(correct, total)
    accuracy = (100 * correct // total) / 100
    print(f'Accuracy of the network on the {len(test_dataset)} test images: {accuracy}')

    return accuracy


def get_precision(test_loader):
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    print(f'Precision of the network on the {len(test_dataset)} test images: {precision}')

    return precision


def get_recall(test_loader):
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    print(f'Recall of the network on the {len(test_dataset)} test images: {recall}')

    return recall


def imshow(img):
    if isinstance(img, torch.Tensor):
        # If input is a PyTorch tensor
        npimg = img.cpu()  # Move tensor to CPU
        npimg = npimg / 2 + 0.5  # unnormalize
        npimg = npimg.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    elif isinstance(img, np.ndarray):
        # If input is a NumPy array
        plt.imshow(img)
        plt.show()