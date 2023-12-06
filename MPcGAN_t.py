import os

from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import platform
import numpy as np
from numpy import zeros, ones
from numpy.random import randint
from numpy.random import randn
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd.variable import Variable
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split


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


def load_real_samples():
        
    transform = None
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='.\\data', train=True, download=True, transform=transform)
    # test_dataset = datasets.CIFAR10(root='.\\data', train=False, download=True, transform=transform)

    trainX, trainy = np.array(train_dataset.data), np.array(train_dataset.targets).reshape(-1,1)
    # print(trainX.shape, trainy.shape)


    # convert to floats and scale
    X = trainX.astype('float32')
    # normalize from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5   #Generator uses tanh activation to rescale 

    # Normalize pixel values to the range [0, 1]
    # image_array = image_array / 255.0
                            #original images to -1 to 1 to match the output of generator.
    return [X, trainy]

# # select real samples
# pick a batch of random real samples to train the GAN
#In fact, we will train the GAN on a half batch of real images and another 
#half batch of fake images. 
#For each real image we assign a label 1 and for fake we assign label 0. 
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset  
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels and assign to y (don't confuse this with the above labels that correspond to cifar labels)
    y = ones((n_samples, 1))  #Label=1 indicating they are real
    y = torch.from_numpy(y)


    return [X, labels], y


  # generates random noise of latent vect dims as well as a random class label
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]


# use the generator to generate n fake examples, with class labels
#Supply the generator, latent_dim and number of samples as input.
#Use the above latent point generator to generate latent points. 
# def generate_fake_samples(generator, latent_dim, n_samples):
# 	# generate points in latent space
# 	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
# 	# predict outputs
# 	images = generator.predict([z_input, labels_input])
# 	# create class labels
# 	y = zeros((n_samples, 1))  #Label=0 indicating they are fake
# 	return [images, labels_input], y

# modified for pytorch
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    
    # convert inputs to PyTorch tensors
    z_input = torch.from_numpy(z_input).float().to(device)
    labels_input = torch.from_numpy(labels_input).long().to(device)
    
    # predict outputs
    generator.eval()  # set the generator to evaluation mode
    with torch.no_grad():
        images = generator(z_input, labels_input)

    
    # create class labels
    y = torch.zeros((n_samples, 1)).to(device)  # Label=0 indicating they are fake
    
    return [images, labels_input], y



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
        li = li.view(-1, 1, self.in_shape[0], self.in_shape[1])  # Change the dimensions to (batch_size, channels, height, width)

        # image input
        # in_image = image.permute(0, 3, 1, 2)  # Change the dimensions to (batch_size, channels, height, width)
        in_image = image
        # concat label as a channel

        print(in_image.shape, li.shape)
        merge = torch.cat((in_image, li), 1)  # Concatenate along the channel dimension
        print(merge.shape)
        # exit()
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
        self.upsample1 = nn.ConvTranspose2d(129, 128, kernel_size=4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)

        # Output layer
        self.out = nn.Conv2d(128, 3, kernel_size=3, padding=1)

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



def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            X_real = X_real.permute(0, 3, 1, 2)  
            # labels_real = labels_real.permute(0, 3, 1, 2)  

            # update discriminator model weights
            d_model.zero_grad()

            output = d_model(X_real, labels_real)

            output = output.float()
            y_real = y_real.float()

            output = output.to(device)
            y_real = y_real.to(device)
            output.requires_grad_()
            y_real.requires_grad_()

            d_loss_real = torch.nn.functional.binary_cross_entropy(output, y_real)
            d_loss_real.backward()

            # generate 'fake' examples
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            


            # update discriminator model weights
            output_fake = d_model(X_fake.detach(), labels)

            output_fake = output_fake.float()
            y_fake = y_fake.float()

            output_fake = output_fake.to(device)
            y_fake = y_fake.to(device)
            output_fake.requires_grad_()
            y_fake.requires_grad_()

            d_loss_fake = torch.nn.functional.binary_cross_entropy(output_fake, y_fake)


            d_loss_fake.backward()
            print(d_loss_fake)
            d_loss = 0.5 * (d_loss_real.item() + d_loss_fake.item())

            print(d_loss)
            d_model.optimizer.step()

            # prepare points in latent space as input for the generator
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
            
            # create inverted labels for the fake samples
            y_gan = Variable(torch.ones(n_batch, 1))
            
            # update the generator via the weight-frozen discriminator's error
            g_model.zero_grad()
            g_loss = gan_model(z_input, labels_input, y_gan)
            g_loss.backward()
            g_model.optimizer.step()

            # Print losses on this batch
            print('Epoch>%d, Batch%d/%d, d1=%.3f, d2=%.3f d3=%.3f g=%.3f ' %
                (i+1, j+1, bat_per_epo, d_loss_real.item(), d_loss_fake.item(), d_loss, g_loss.item()))

    # save the generator model
    torch.save(g_model.state_dict(), 'generator_weights\\cifar_conditional_generator_25epochs.pth')


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
train_dataset = datasets.CIFAR10(root='.\\data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='.\\data', train=False, download=True, transform=transform)

# Split the data into training and testing sets                                     # make 0.14 later
# trainX, _, trainy, _ = train_test_split(train_dataset.data, train_dataset.targets, test_size=.1, random_state=42)
# testX, _, testy, _ = train_test_split(test_dataset.data, test_dataset.targets, test_size=.1, random_state=42)

X_train, X_test = np.array(train_dataset.data), np.array(test_dataset.data)
y_train, y_test = np.array(train_dataset.targets).reshape(-1,1), np.array(test_dataset.targets).reshape(-1,1)


print(y_test[0], y_test[-1])
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# print(trainX.shape, len(trainy))
# print(testX.shape, len(testy))
# print(type(trainX[0]))
# print(testy[0], testy[-1])

# Check if it's Windows
if os_name == 'Windows':
	gan_data_dir = "data_processing\\gan_dataset\\"
elif os_name == 'Darwin':
	gan_data_dir = "data_processing/gan_dataset/"
else:
	print(f"Unknown operating system: {os_name}")
	exit()


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



latent_dim = 100
# Define the GAN model
# create the discriminator
d_model = Discriminator()
# create the generator
g_model = Generator(latent_dim)
gan_model = GAN(g_model, d_model)
# Define the GAN model optimizer
optimizer = Adam(gan_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
# Define the loss function
criterion = nn.BCELoss()
dataset = load_real_samples()
dataset = [torch.from_numpy(tensor).to(device) for tensor in dataset]
# send everything to gpu
# X_train.to(device)
# X_test.to(device)
# y_train.to(device)
# y_test.to(device)
# dataset.to(device)
d_model.to(device)
g_model.to(device)
gan_model.to(device)


train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=2)

# create the gan
# load image data
# dataset = load_real_samples()
# print(gan_model)
# print(d_model)
# print(g_model)