import os
import numpy as np
import platform
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate
from matplotlib import pyplot as plt
from keras.datasets.cifar10 import load_data
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


print("TensorFlow version:", tf.__version__)
print("TensorFlow is installed at:", tf.__file__)
# d = 'C:\Users\Owner\Desktop\microplastics_data_generation_private\\tf-gpu-env'
# exit()
CUDA_VISIBLE_DEVICES=1
(trainX, trainy), (testX, testy) = load_data()
print(trainX.shape, trainy.shape)
print(testX.shape, testy.shape)
print(type(trainX[0]))
print(testy[0], testy[-1])

os_name = platform.system()
print(os_name)
if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
    print('GPU device is available.')
else:
    print('GPU device is not available.')

# Alternatively, you can use this code to list all available GPUs
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Check if it's Windows
if os_name == 'Windows':
	gan_data_dir = "data_processing\\gan_dataset\\"
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





# Example usage
print(gan_data_dir)


#############################################################################
#Define generator, discriminator, gan and other helper functions
#We will use functional way of defining model as we have multiple inputs; 
#both images and corresponding labels. 
########################################################################

# define the standalone discriminator model
#Given an input image, the Discriminator outputs the likelihood of the image being real.
#Binary classification - true or false (1 or 0). So using sigmoid activation.

#Unlike regular GAN here we are also providing number of classes as input. 
#Input to the model will be both images and labels. 
def define_discriminator(in_shape=(32,32,3), n_classes=10):
	
    # label input
	in_label = Input(shape=(1,))  #Shape 1
	print(in_label.shape)
	# exit()
	# embedding for categorical input
    #each label (total 10 classes for cifar), will be represented by a vector of size 50. 
    #This vector of size 50 will be learnt by the discriminator
	li = Embedding(n_classes, 50)(in_label) #Shape 1,50
	# print(li.shape)
	# exit()
	# scale up to image dimensions with linear activation
	n_nodes = in_shape[0] * in_shape[1]  #32x32 = 1024. 
	li = Dense(n_nodes)(li)  #Shape = 1, 1024
	# reshape to additional channel
	li = Reshape((in_shape[0], in_shape[1], 1))(li)  #32x32x1


	# image input
	in_image = Input(shape=in_shape) #32x32x3
	print(in_image.shape, li.shape)

	# concat label as a channel
	merge = Concatenate()([in_image, li]) #32x32x4 (4 channels, 3 for image and the other for labels)
    
	# downsample: This part is same as unconditional GAN upto the output layer.
    #We will combine input label with input image and supply as inputs to the model. 
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge) #16x16x128
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe) #8x8x128
	fe = LeakyReLU(alpha=0.2)(fe)
	# flatten feature maps
	fe = Flatten()(fe)  #8192  (8*8*128=8192)
	# dropout
	fe = Dropout(0.4)(fe)
	# output
	out_layer = Dense(1, activation='sigmoid')(fe)  #Shape=1
    
	# define model
    ##Combine input label with input image and supply as inputs to the model. 
	model = Model([in_image, in_label], out_layer)
	# compile model
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

test_discr = define_discriminator()
print(test_discr.summary())


# define the standalone generator model
#latent vector and label as inputs

def define_generator(latent_dim, n_classes=10):
    
	# label input
	in_label = Input(shape=(1,))  #Input of dimension 1
	# embedding for categorical input
    #each label (total 10 classes for cifar), will be represented by a vector of size 50. 
	li = Embedding(n_classes, 50)(in_label) #Shape 1,50
    
	# linear multiplication
	n_nodes = 8 * 8  # To match the dimensions for concatenation later in this step.  
	li = Dense(n_nodes)(li) #1,64
	# reshape to additional channel
	li = Reshape((8, 8, 1))(li)
    
    
	# image generator input
	in_lat = Input(shape=(latent_dim,))  #Input of dimension 100
    

	# vector starts as the size which is 2 factorsof2 smaller than the input images     (factors of 2 are based on 2 convolutional layers of stride length 2)


	# foundation for 8x8 image
    # We will reshape input latent vector into 8x8 image as a starting point. 
    #So n_nodes for the Dense layer can be 128x8x8 so when we reshape the output 
    #it would be 8x8x128 and that can be slowly upscaled to 32x32 image for output.
    #Note that this part is same as unconditional GAN until the output layer. 
    #While defining model inputs we will combine input label and the latent input.
	n_nodes = 128 * 8 * 8

	gen = Dense(n_nodes)(in_lat)  #shape=8192
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((8, 8, 128))(gen) #Shape=8x8x128
	# merge image gen and label input

	merge = Concatenate()([gen, li])  #Shape=8x8x129 (Extra channel corresponds to the label)
	# upsample to 16x16
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge) #16x16x128
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample to 32x32
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen) #32x32x128
	gen = LeakyReLU(alpha=0.2)(gen)
	# output
	out_layer = Conv2D(3, (8,8), activation='tanh', padding='same')(gen) #32x32x3
	# define model
	model = Model([in_lat, in_label], out_layer)
	return model   #Model not compiled as it is not directly trained like the discriminator.

test_gen = define_generator(100, n_classes=10)
print(test_gen.summary())

# #Generator is trained via GAN combined model. 
# define the combined generator and discriminator model, for updating the generator
#Discriminator is trained separately so here only generator will be trained by keeping
#the discriminator constant. 
def define_gan(g_model, d_model):
	d_model.trainable = False  #Discriminator is trained separately. So set to not trainable.
    
    ## connect generator and discriminator...

	#1. pass noise and label through generator and generate fake image
	# first, get noise and label inputs from generator model
	gen_noise, gen_label = g_model.input  #Latent vector size and label size
	# get image output from the generator model
	gen_output = g_model.output  #32x32x3
    
	print(gen_noise.dtype, gen_label.dtype)
	# exit()

	#2. pass fake image and class label through discriminator model to get the generator error signal
	# generator image output and corresponding input label are inputs to discriminator
	gan_output = d_model([gen_output, gen_label])
	# define gan model as taking noise and label and outputting a classification
	model = Model([gen_noise, gen_label], gan_output)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# load cifar images
def load_real_samples():

	(trainX, trainy), (_, _) = load_data()  
	print(trainX.shape, trainy.shape)
	# (trainX, trainy), (_, _) = load_gan_training_data(data_dir=gan_data_dir, image_size=(32, 32))

	
	# convert to floats and scale
	X = trainX.astype('float32')
	# normalize from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5   #Generator uses tanh activation so rescale 

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
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	y = zeros((n_samples, 1))  #Label=0 indicating they are fake

	return [images, labels_input], y

# train the generator and discriminator
#We loop through a number of epochs to train our Discriminator by first selecting
#a random batch of images from our true/real dataset.
#Then, generating a set of images using the generator. 
#Feed both set of images into the Discriminator. 
#Finally, set the loss parameters for both the real and fake images, as well as the combined loss. 
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	half_batch = int(n_batch / 2)  #the discriminator model is updated for a half batch of real samples 
                            #and a half batch of fake samples, combined a single batch. 
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			
             # Train the discriminator on real and fake images, separately (half batch each)
        # separate training is more effective
			
            # get randomly selected 'real' samples
			[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)

            # update discriminator model weights
            ##train_on_batch allows you to update weights based on a collection 
            #of samples you provide
			d_loss_real, _ = d_model.train_on_batch([X_real, labels_real], y_real)
            
			# generate 'fake' examples
			[X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update discriminator model weights
			d_loss_fake, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) #Average loss 
            
			# prepare points in latent space as input for the generator
			[z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
            
            # The generator wants the discriminator to label the generated samples
        # as valid (ones)
        #This is where the generator is trying to trick discriminator into believing
        #the generated image is true (hence value of 1 for y)	
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
             # Generator is part of combined model where it got directly linked with the discriminator
        # Train the generator with latent_dim as x and 1 as y. 
        # 1 as the output as it is adversarial and if generator did a great
        # job of fooling the discriminator then the output would be 1 (true)
			# update the generator via the weight-frozen discriminator's error
			print(z_input.dtype, labels_input.dtype, y_gan.dtype)
			g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
			# Print losses on this batch
			print('Epoch>%d, Batch%d/%d, d1=%.3f, d2=%.3f d3=%.3f g=%.3f ' %
				(i+1, j+1, bat_per_epo, d_loss_real, d_loss_fake, d_loss, g_loss))
	# save the generator model
	g_model.save('generator_weights\\cifar_conditional_generator_25epochs.h5')

#Train the GAN

# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples()
# train model
# train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=2)


# exit()
##########################################################
# Now, let us load the generator model and generate images
# # Lod the trained model and generate a few images
# from numpy import asarray
# from numpy.random import randn
# from numpy.random import randint
# from keras.models import load_model
# import numpy as np
# # 

# #Note: CIFAR10 classes are: airplane, automobile, bird, cat, deer, dog, frog, horse,
# # ship, truck

# # load model
# model = load_model('generator_weights\\cifar_conditional_generator_2epochs.h5')

# # generate multiple images

# latent_points, labels = generate_latent_points(100, 100)
# # specify labels - generate 10 sets of labels each gping from 0 to 9
# labels = asarray([x for _ in range(10) for x in range(10)])
# # generate images
# X  = model.predict([latent_points, labels])
# # scale from [-1,1] to [0,1]
# X = (X + 1) / 2.0
# X = (X*255).astype(np.uint8)
# # plot the result (10 sets of images, all images in a column should be of same class in the plot)
# # Plot generated images 
# def show_plot(examples, n):
# 	for i in range(n * n):
# 		plt.subplot(n, n, 1 + i)
# 		plt.axis('off')
# 		plt.imshow(examples[i, :, :, :])
# 	plt.show()
    
# show_plot(X, 3)




