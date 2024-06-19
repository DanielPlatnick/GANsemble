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




# Image dimensions: (1167, 875)



os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# exit()

""" This file contains the CNN and resnet model that is used to test the effectiveness of each augmentation strategy in a controlled experiment """

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
        self.fc1 = nn.Linear(16 * 58*58, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # print(x.size())
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.size())
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print(x.size())
        x = F.relu(self.fc1(x))
        # print(x.size())
        x = F.relu(self.fc2(x))
        # print(x.size())
        x = self.fc3(x)
        # print(x.size())
        # exit()
        return x

    ## this code is for inspecting the dimensions at each layer during the forward pass of the network    
    # def forward(self, x):
    #     print(x.size())
    #     x = self.pool(F.relu(self.conv1(x)))
    #     print(x.size())
    #     x = self.pool(F.relu(self.conv2(x)))
    #     print(x.size())
    #     x = torch.flatten(x, 1) # flatten all dimensions except batch
    #     print(x.size())
    #     x = F.relu(self.fc1(x))
    #     print(x.size())
    #     x = F.relu(self.fc2(x))
    #     print(x.size())
    #     x = self.fc3(x)
    #     print(x.size())
    #     # exit()
    #     return x



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
    accuracy = (100*correct//total)/100
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








model_list = ['CNN', 'Resnet50_base', 'Resnet50_pretrained']


# for table 1 study
# sample_size_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# sample_size_list = [x for x in sample_size_list  if x == 80]
num_epochs = 150
batch_size = 4

# run = 5

for run in range(1,4):

    metrics_list = []




    AUG_STRAT = 4
    CURR_NUM_SAMPLES = 50
    CURR_MODEL = model_list[2]
    # if using pre-trained then make sure to set Resnet50 with pretrained=True while calling the Resnet50 class

    data_processing_dir = os.getcwd() + '\\data_processing\\'

    training_dir = data_processing_dir + f'augmented_datasets\\aug_data_{CURR_NUM_SAMPLES}_samples\\aug_strategy_{AUG_STRAT}\\'

    #  no augment no RESAMPLE
    # training_dir = data_processing_dir + 'raw_data\\polar\\'

    test_dir = data_processing_dir + 'evaluation_set\\'

    # setting weight storage
    aug_models_dir = f'info_data_chooser\\{CURR_MODEL}\\weights_data_chooser\\'
    if not os.path.exists(aug_models_dir): os.mkdir(aug_models_dir)
    current_model = training_dir.split('\\')[-2]

    #### CHANGE BACK
    model_weights_path = aug_models_dir + f'\\n{CURR_NUM_SAMPLES}_run{run}_{CURR_MODEL}_{current_model}_{num_epochs}epochs.pth'
    # model_weights_path = aug_models_dir + f'\\n{CURR_NUM_SAMPLES}_run{run}_{CURR_MODEL}_no_resample_{num_epochs}epochs.pth'
    # model_weights_path = aug_models_dir + f'\\n{CURR_NUM_SAMPLES}_run{run}_{CURR_MODEL}_resample_{num_epochs}epochs.pth'

    # setting acc, precision, recall storage
    aug_performance_dir = f'info_data_chooser\\{CURR_MODEL}\\acc_prec_recall\\'
    if not os.path.exists(aug_performance_dir): os.mkdir(aug_performance_dir)


    # image dimensions are 865x1167 pixels with 3 channels
    # model = SyntheticDataChooser_CNN()

    # Set pre-training or not
    model = SyntheticDataChooser_ResNet50(pretrained=True)

    model_used = str(model)
    model_used = model_used.split('(')[0]
    height = 875
    # pad to (3,1180,1180)
    width = 1167


    # transforms
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Pad((6,153,7,152), fill=255), transforms.Resize((244,244), antialias=True), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = Microplastics_Dataset(root_dir=training_dir, transform=transform)
    test_dataset = Microplastics_Dataset(root_dir=test_dir, transform=transform)



    # print(training_dir)
    # print(test_dir)
    # print(aug_models_dir)
    # print(model_weights_path)
    # print(aug_performance_dir)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    classes = tuple([class_dir for class_dir in os.listdir(training_dir)])
    print(classes)

    num_classes = len(train_dataset.classes)

    # verify model architecture
    # print(model)



    # Image dimensions: (1167, 875)
    # show some images
    # for i in range(2):
    #     dataiter = iter(train_loader)
    #     images, labels = next(dataiter)

    # # ## show images
    # imshow(torchvision.utils.make_grid(images))
    # print(labels)
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    # # Display the first image
    # first_image.show()

    # # Print the dimensions of the first image
    # print("Image dimensions:", first_image.size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00051, momentum=0.9)

    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # for param in model.parameters():
    #     print(param.device)

    ## optionally load model to continue training from checkpoint
    #### model.load_state_dict(torch.load(model_weights_path))


    ######## Training loop ########
    ## specified epochs at top for pathing
    # num_epochs = x

    for epoch in tqdm(range(num_epochs), desc='Epochs', unit='epoch'):
        running_loss = 0.0
        tqdm_train_loader = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')
        # Use tqdm to create a loading bar for the inner loop
        for i, data in enumerate(tqdm_train_loader, 0):
            if model_used == 'SyntheticDataChooser_ResNet50':
                if model.pretrained == True:
                            # Freeze all layers
                    for param in model.resnet.parameters():
                        param.requires_grad = False

                    # Unfreeze the last fully connected layer
                    for param in model.resnet.fc.parameters():
                        param.requires_grad = True

                    if epoch >= 15:
                        for param in model.resnet.parameters():
                            param.requires_grad = True
                else:
                    pass

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            tqdm_train_loader.set_postfix(loss=running_loss / (i + 1))

        # Print the average loss for the epoch
        average_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Average Loss: {average_loss:.3f}')

    # Save the trained model
    torch.save(model.state_dict(), model_weights_path)


    # # exit('model trained')



    # # Print some information to verify correctness
    # # print("Number of classes:", len(train_dataset.classes))
    # # print("Class names:", train_dataset.classes)
    # # print("Class to index mapping:", train_dataset.class_to_idx)
    print("Number of samples in training dataset:", len(train_dataset))
    print("Number of samples in test dataset:", len(test_dataset))


    # Iterate through the test dataset
    for index in range(len(test_dataset)):
        image, label = test_dataset[index]


    # Initialize the model
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)
    labels = labels.to(device)



    model.load_state_dict(torch.load(model_weights_path))

    model.eval()

    outputs = model(images)
    outputs.to(device)


    # # print images
    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))


    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                for j in range(4)))



    curr_model_accuracy = round(get_accuracy(test_loader=test_loader),2)
    curr_model_precision = round(get_precision(test_loader=test_loader),2)
    curr_model_recall = round(get_recall(test_loader=test_loader),2)
    metrics_list.append([AUG_STRAT,[curr_model_accuracy, curr_model_precision, curr_model_recall]])
    # metrics_list.append(['No resampling',[curr_model_accuracy, curr_model_precision, curr_model_recall]])
    # metrics_list.append(['Resampling',[curr_model_accuracy, curr_model_precision, curr_model_recall]])
    # print(metrics_list)


    # update path of where to store your table info
    table_curr_metrics_list = metrics_list
    table_pickle_path = aug_performance_dir + f'n{CURR_NUM_SAMPLES}_dcm_table_{CURR_MODEL}_aug{AUG_STRAT}_metrics.pkl'
    # table_pickle_path = aug_performance_dir + f'n{CURR_NUM_SAMPLES}_dcm_table_{CURR_MODEL}_no_resample_metrics.pkl'
    # table_pickle_path = aug_performance_dir + f'n{CURR_NUM_SAMPLES}_dcm_table_{CURR_MODEL}_resample_metrics.pkl'


    ## if pkl not exist, create it
    if not os.path.exists(table_pickle_path):
        with open(table_pickle_path, 'wb') as f:
            pickle.dump([],f)


    ## load table 1 metrics list
    with open(table_pickle_path, 'rb') as file:
        table_metrics_list_pkl = pickle.load(file)
    print(table_metrics_list_pkl)

    ## append metrics from current neural network
    table_metrics_list_pkl.append(table_curr_metrics_list)

    ## overwrite pkl file with new file containing appended table 1 metrics list
    with open(table_pickle_path, 'wb') as file:
        pickle.dump(table_metrics_list_pkl, file)

    ## verify update of metrics list
    with open(table_pickle_path, 'rb') as file:
        test_metrics = pickle.load(file)
    print(test_metrics)


# correct_pred = {classname: 0 for classname in classes}
# total_pred = {classname: 0 for classname in classes}

# # again no gradients needed
# with torch.no_grad():
#     for data in test_loader:
#         images, labels = data
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predictions = torch.max(outputs, 1)
#         # collect the correct predictions for each class
#         for label, prediction in zip(labels, predictions):
#             if label == prediction:
#                 correct_pred[classes[label]] += 1
#             total_pred[classes[label]] += 1


# # print accuracy for each class
# for classname, correct_count in correct_pred.items():
#     accuracy = 100 * float(correct_count) / total_pred[classname]
#     print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

# acc_list.append((AUG_STRAT, accuracy))
# print(acc_list)







