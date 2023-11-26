from build_augmented_data import *
import os
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR

# IF SOME WEIRD ERROR STARTS OCCURING WITH DIRS THAT WONT GO AWAY THEN MAYBE ITS BECAUSE U   PLACED SOMETHING SOMEEWHERE
# Image dimensions: (1167, 875)



os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# exit()

""" This file contains the CNN model that is used to test the effectiveness of each augmentation strategy in a controlled experiment """

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


class SyntheticDataChooser(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
    
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 70*70, 256)
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
    # def forward(self, x):
        # print(x.size())
        # x = self.pool(F.relu(self.conv1(x)))
        # print(x.size())
        # x = self.pool(F.relu(self.conv2(x)))
        # print(x.size())
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print(x.size())
        # x = F.relu(self.fc1(x))
        # print(x.size())
        # x = F.relu(self.fc2(x))
        # print(x.size())
        # x = self.fc3(x)
        # print(x.size())
        # # exit()
        # return x




class SyntheticDataChooser_ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x

    
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

# Assuming the image dimensions are 865x1167 pixels with 3 channels
model = SyntheticDataChooser_ResNet50()
height = 875
# pad to (3,1180,1180)
width = 1167

# transforms.RandomCrop(512,512),transforms.Resize((244, 244)),
# Normalizing the images

#resnet
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Pad((6,153,7,152), fill=255), transforms.Resize((244,244)), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# transform = transforms.Compose(
#     [transforms.ToTensor(), transforms.Pad((6,153,7,152), fill=255), transforms.Resize((295,295)), 
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


AUG_STRAT = 6

training_dir = f'C:\\Users\\Owner\\Desktop\\microplastics_data_generation_private\\data_processing\\augmented_datasets\\aug_data_30_samples\\aug_strategy_{AUG_STRAT}'
test_dir = 'C:\\Users\\Owner\\Desktop\\microplastics_data_generation_private\\data_processing\\raw_data\\polar'
aug_models_dir = 'C:\\Users\\Owner\\Desktop\\microplastics_data_generation_private\\models'
current_model = training_dir.split('\\')[-1]
model_weights_path = f'C:\\Users\\Owner\\Desktop\\microplastics_data_generation_private\\models\\{current_model}.pth'

train_dataset = Microplastics_Dataset(root_dir=training_dir, transform=transform)
test_dataset = Microplastics_Dataset(root_dir=test_dir, transform=transform)
acc_list = []
for i in range(AUG_STRAT,len(os.listdir(training_dir))):
    # Create DataLoaders
    AUG_STRAT+=1
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    classes = tuple([class_dir for class_dir in os.listdir(training_dir)])
    print(classes)

    num_classes = len(train_dataset.classes)
    # print(model)
    # Image dimensions: (1167, 875)
    # Define loss function and optimizer
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # ## show images
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




    # Training loop
    num_epochs = 50
    # model.load_state_dict(torch.load(model_weights_path))


    for epoch in tqdm(range(num_epochs), desc='Epochs', unit='epoch'):
        running_loss = 0.0
        tqdm_train_loader = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')
        # Use tqdm to create a loading bar for the inner loop
        for i, data in enumerate(tqdm_train_loader, 0):
            if SyntheticDataChooser_ResNet50:
                            # Freeze all layers
                for param in model.resnet.parameters():
                    param.requires_grad = False

                # Unfreeze the last fully connected layer
                for param in model.resnet.fc.parameters():
                    param.requires_grad = True

                if epoch >= 15:
                    for param in model.resnet.parameters():
                        param.requires_grad = True

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


    correct = 0
    total = 0

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
    print(f'Accuracy of the network on the 210 test images: {100 * correct // total} %')


    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    acc_list.append((AUG_STRAT, accuracy))
    print(acc_list)
    # # Print some information to verify correctness
    # print("Number of classes:", len(train_dataset.classes))
    # print("Class names:", train_dataset.classes)
    # print("Class to index mapping:", train_dataset.class_to_idx)
    # print("Number of samples in training dataset:", len(train_dataset))
    # print("Number of samples in test dataset:", len(test_dataset))


    model.eval()

    # Iterate through the test dataset
    for index in range(5): 
        image, label = test_dataset[index]
        image = image.unsqueeze(0)  # Add a batch dimension

        # Move the data to the GPU if available
        image = image.to(device)

        # Forward pass
        with torch.no_grad():
            output = model(image)

        # Get the predicted label
        _, predicted = torch.max(output, 1)

        # Print the ground truth and predicted labels
        print(f'Example {index + 1}: Ground Truth - {classes[label]}, Predicted - {classes[predicted.item()]}')


    # for index in range(50):
    #     image, label = test_dataset[index]
    #     image = image.unsqueeze(0)  # Add a batch dimension
    #     image = image.to(device)

    #     # Set the model to evaluation mode
    #     model.eval()

    #     # Forward pass
    #     with torch.no_grad():
    #         output = model(image)
    #         _, predicted = torch.max(output, 1)

    #     # Print results
    #     print(f'Example {index + 1}:')
    #     print(f'Ground Truth: {classes[label]}, Predicted: {classes[predicted.item()]}')
    #     print('---')