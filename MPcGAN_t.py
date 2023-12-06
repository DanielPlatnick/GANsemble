import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import platform
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split

os_name = platform.system()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


torch.manual_seed(42)

# Define transform for loading the data
# transform = transforms.Compose(transforms.ToTensor())
transform = None

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='.\\data', train=True, download=False, transform=transform)
test_dataset = datasets.CIFAR10(root='.\\data', train=False, download=False, transform=transform)

# Split the data into training and testing sets                                     # make 0.14 later
trainX, _, trainy, _ = train_test_split(train_dataset.data, train_dataset.targets, test_size=0.1, random_state=42)
testX, _, testy, _ = train_test_split(test_dataset.data, test_dataset.targets, test_size=0.1, random_state=42)

type(trainy)
X_train = np.array(trainX)
trainy = np.array(trainy)
X_test = np.array(testX)
testy = np.array(testy)

y_test = testy.reshape(-1,1)
y_train = trainy.reshape(-1,1)

print(y_test[0], y_test[-1])
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# print(trainX.shape, len(trainy))
# print(testX.shape, len(testy))
# print(type(trainX[0]))
# print(testy[0], testy[-1])

# Check if it's Windows
if os_name == 'Windows':
	gan_data_dir = "C:\\Users\\Owner\\Desktop\\microplastics_data_generation_private\\data_processing\\gan_dataset\\"
elif os_name == 'Darwin':
	gan_data_dir = "data_processing/gan_dataset/"
else:
	print(f"Unknown operating system: {os_name}")
	exit()

def load_gan_training_data(data_dir, image_size=(32,32), test_split=0.2):
	
	images = []
	labels = []
	class_folders = os.listdir(data_dir)

	for class_index, class_folder in enumerate(class_folders):
		class_path = os.path.join(data_dir, class_folder)
		# print(class_index, class_path)

		image_files = [f for f in os.listdir(class_path) if f.endswith('.png')]

		for image_file in image_files:
			image_path = os.path.join(class_path, image_file)
            # print(image_path)
		
			image = Image.open(image_path)
			image = image.resize(image_size)
			image_array = np.array(image)



			images.append(image_array)
			labels.append(class_index)

    # Convert lists to NumPy arrays
	images = np.array(images)
	labels = np.array(labels)

    # Split the data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_split, random_state=42)
	y_test = y_test.reshape(-1,1)
	y_train = y_train.reshape(-1,1)

	return (X_train, y_train), (X_test, y_test)





(GAN_trainX, GAN_trainy), (GAN_testX, GAN_testy) = load_gan_training_data(data_dir=gan_data_dir, image_size=(32, 32))


print(GAN_testy[0], GAN_testy[-1])
print(GAN_trainX.shape, GAN_trainy.shape)
print(GAN_testX.shape, GAN_testy.shape)
# # plot 5 images
# for i in range(5):
# 	plt.subplot(1, 5, 1 + i)
# 	plt.axis('off')
# 	plt.imshow(GAN_trainX[i])
# plt.show()


print(gan_data_dir)

class Discriminator(nn.Module):
    def __init__(self, in_shape=(32,32,3), n_classes=10):
        super(Discriminator, self).__init__()
        self.in_shape = in_shape
        self.n_classes = n_classes
        self.embedding = nn.Embedding(n_classes, 50)
        self.fc1 = nn.Linear(50, in_shape[0]*in_shape[1])
        self.conv1 = nn.Conv2d(in_shape[2]+1, 128, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.fc2 = nn.Linear(128*8*8, 1)

    def forward(self, image, label):
        # label input
        li = self.embedding(label)
        li = self.fc1(li)
        li = li.view(-1, 1, self.in_shape[0], self.in_shape[1])
        # image input
        in_image = image.view(-1, self.in_shape[2], self.in_shape[0], self.in_shape[1])
        # concat label as a channel
        merge = torch.cat((in_image, li), 1)
        # downsample
        fe = F.leaky_relu(self.conv1(merge), 0.2)
        # downsample
        fe = F.leaky_relu(self.conv2(fe), 0.2)
        # flatten feature maps
        fe = fe.view(fe.size(0), -1)
        # output
        out_layer = torch.sigmoid(self.fc2(fe))
        return out_layer

test_discr = Discriminator()
print(test_discr)


class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes=10):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        # Label input
        self.embedding = nn.Embedding(n_classes, 50)
        self.fc1 = nn.Linear(50, 8*8)

        # Image generator input
        self.fc2 = nn.Linear(latent_dim, 128*8*8)

        # Upsampling layers
        self.upsample1 = nn.ConvTranspose2d(128+1, 128, kernel_size=4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)

        # Output layer
        self.out = nn.Conv2d(128, 3, kernel_size=8, padding=3)

    def forward(self, z, labels):
        # Label input
        li = self.embedding(labels)
        li = self.fc1(li)
        li = li.view(-1, 1, 8, 8)

        # Image generator input
        z = self.fc2(z)
        z = z.view(-1, 128, 8, 8)

        # Concatenate label as a channel
        merge = torch.cat((z, li), 1)

        # Upsample to 16x16
        x = F.leaky_relu(self.upsample1(merge), 0.2)

        # Upsample to 32x32
        x = F.leaky_relu(self.upsample2(x), 0.2)

        # Output
        out = torch.tanh(self.out(x))

        return out

test_gen = Generator(100, n_classes=10)
print(test_gen)


class GAN(nn.Module):
    def __init__(self, g_model, d_model):
        super(GAN, self).__init__()
        self.g_model = g_model
        self.d_model = d_model
        self.d_model.requires_grad_(False)  # Set discriminator to not trainable.

    def forward(self, gen_noise, gen_label):
        gen_output = self.g_model(gen_noise, gen_label)
        gan_output = self.d_model(gen_output, gen_label)
        return gan_output


latent_dim = 100
# Define the GAN model
# create the discriminator
d_model = Discriminator()
# create the generator
g_model = Generator(latent_dim)
gan_model = GAN(g_model, d_model)

# Define the optimizer
optimizer = Adam(gan_model.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Define the loss function
criterion = nn.BCELoss()


# create the gan
# load image data
# dataset = load_real_samples()
print(gan_model)
print(d_model)
print(g_model)


def custom_transform(x):
    return x.numpy()