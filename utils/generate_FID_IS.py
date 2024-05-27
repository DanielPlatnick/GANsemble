import os

from MPcGAN import *
import matplotlib.pyplot as plt
from numpy import asarray
from tensorflow.keras.models import Model
from PIL import Image
# import torchvision.transforms as transforms
import tensorflow as tf
from scipy.linalg import sqrtm
import pandas as pd
import pickle

def generate_image_labels():
    # Define the counts for each label
    counts = {
        "cellulosic": 19,
        "polyacetal": 5,
        "polyamide": 19,
        "polyethylene": 26,
        "polyethylene_ter": 16,
        "polypropylene": 20,
        "polystyrene": 17,
        "polyurethane": 27,
        "polyvynylchloride": 23,
        "silica": 38
    }

    gen_image_labels = []
    for label, count in enumerate(counts.values()):
        gen_image_labels.extend([label] * count)

    return gen_image_labels


# def load_images_from_directory(directory):
#     images = []
#     for filename in os.listdir(directory):
#         if filename.endswith('.jpg') or filename.endswith('.png'):  # Add other file types if needed
#             img = Image.open(os.path.join(directory, filename))
#             # img = img.resize((299, 299))  # Resize if needed
#             images.append(np.asarray(img))
#     return np.array(images)


def calculate_fid(model, images1, images2):
    # Calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)

    # Calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)

    # Calculate sqrt of product between covariances
    covmean = sqrtm(sigma1.dot(sigma2))

    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def calculate_inception_score(model, images, n_split=10, eps=1E-16):
    # Predict class probabilities
    preds = model.predict(images)

    # Split into groups
    scores = []
    n_part = preds.shape[0] // n_split
    for i in range(n_split):
        part = preds[i * n_part: (i + 1) * n_part, :]
        kl_div = part * (np.log(part + eps) - np.log(np.expand_dims(np.mean(part, axis=0), 0) + eps))
        kl_div = np.mean(np.sum(kl_div, axis=1))
        scores.append(np.exp(kl_div))
    return np.mean(scores), np.std(scores)


def calculate_fid_for_directories(groundtruth_dir, augstrat_dir):
    # Load images
    ground_truth_images = load_images_from_directory(groundtruth_dir)
    generated_images = load_images_from_directory(augstrat_dir)

    # Load pre-trained InceptionV3 model
    model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

    # Scale images
    ground_truth_images_scaled = ground_truth_images / 255.0
    generated_images_scaled = generated_images / 255.0
    print(ground_truth_images_scaled.shape, generated_images_scaled.shape)

    # Calculate FID
    fid_value = calculate_fid(model, ground_truth_images_scaled, generated_images_scaled)
    return fid_value


def calculate_is_for_directory(directory):
    # Load images
    generated_images = load_images_from_directory(directory)

    # Load pre-trained InceptionV3 model (for classification)
    model = tf.keras.applications.InceptionV3(include_top=True, input_shape=(299,299,3))

    # Scale images
    generated_images_scaled = generated_images / 255.0

    # Calculate IS
    is_mean, is_std = calculate_inception_score(model, generated_images_scaled)
    return is_mean, is_std


def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Add other file types if needed
            img = Image.open(os.path.join(directory, filename)).convert('RGB')
            img = img.resize((299, 299))  # Resize if needed
            images.append(np.asarray(img))
    return np.array(images)






print(calculate_fid_for_directories(os.getcwd() + '\\FID_IS_groundtruth_299\\', os.getcwd() + '\\FID_IS_baseline-1_299\\'))
# for i in range(len(dirs)):

#     fid_value = calculate_fid_for_directories(groundtruth_dir, dirs[i])
#     is_mean, is_std = calculate_is_for_directory(dirs[i])

#     print(f'fid = {fid_value}  \n   is_mean = {is_mean}, is_std = {is_std}')

# strategy_dirs = {
#     'No resample': 'FID_IS_no_resample_299/',
#     'Resample': 'FID_IS_resample_299/',
#     'Aug1': 'FID_IS_aug1_299/',
#     'Aug7': 'FID_IS_aug7_299/',
#     'Aug4': 'FID_IS_aug4_299/'
# }




# # Dataframe to store the results
# results_df = pd.DataFrame(columns=strategy_dirs.keys(), index=['FID', 'IS Mean', 'IS Std'])

# # Calculate FID and IS for each strategy
# for strategy, dir in strategy_dirs.items():
#     # FID
#     fid_value = calculate_fid_for_directories(groundtruth_dir, dir)
#     results_df.at['FID', strategy] = fid_value

#     # IS
#     is_mean, is_std = calculate_is_for_directory(dir)
#     results_df.at['IS Mean', strategy] = is_mean
#     results_df.at['IS Std', strategy] = is_std

# # Display the DataFrame
# print(results_df)

# results_df.to_pickle('FID_IS_data.pkl')


# ##______________________________________________________________________##

# ### Generating SYMP data for FID and IS calculations

# dataset_dict = {
#     0: "cellulosic",
#     1: "polyacetal",
#     2: "polyamide",
#     3: "polyethylene",
#     4: "polyethylene_ter",
#     5: "polypropylene",
#     6: "polystyrene",
#     7: "polyurethane",
#     8: "polyvynylchloride",
#     9: "silica"
# }


# for i in range(5):
        

#     # output_dir = 'baseline-1_SYMP_299'


#     model_dir_path = 'MPcGAN_output_n210/MPcGAN_weights_n210/'

#     models = ['baseline-best_MPcGAN_gen_epochs.h5',
#             'baseline_resamplebest_MPcGAN_gen_epochs.h5', 'aug1best_MPcGAN_gen_epochs.h5',
#             'aug7best_MPcGAN_gen_120epochs.h5', 'aug4best_MPcGAN_gen_epochs.h5']

   
#     output_dir = f'per_class_FID_IS_aug4-{i+1}_299'
#     if not os.path.exists(output_dir): os.mkdir(output_dir)


#     # choose model
#     model_path = model_dir_path + models[-1] 
#     model = load_model(model_path)


#     # Generate and print the labels
#     generated_labels = generate_image_labels()

#     generated_labels_100 = generated_labels[:100]
#     generated_labels_200 = generated_labels[100:200]
#     generated_labels_10 = generated_labels[200:]

#     gen_funcs = [generated_labels_100, generated_labels_200, generated_labels_10]

#     for i in range(len(gen_funcs)):

#         curr_gen_func = gen_funcs[i]

#         if curr_gen_func == generated_labels_100:
#             latent_points, labels = generate_latent_points(latent_dim=100, n_samples=100)
#             labels = asarray(generated_labels_100)
#             section = 0

#         if curr_gen_func == generated_labels_200:
#             latent_points, labels = generate_latent_points(latent_dim=100, n_samples=100)
#             labels = asarray(generated_labels_200)
#             section = 100

#         if curr_gen_func == generated_labels_10:
#             latent_points, labels = generate_latent_points(latent_dim=100, n_samples=10)
#             labels = asarray(generated_labels_10)
#             section = 200

#         # labels = asarray([x for _ in range(10) for x in range(10)])
#         print(labels, latent_points.shape)

#         str_labels = [dataset_dict[i] for i in labels]
#         print(str_labels, len(str_labels))


#         X  = model.predict([latent_points, labels])

#         # scale from [-1,1] to [0,1]
#         X = (X + 1) / 2.0
#         X = (X*255).astype(np.uint8)
#         # print(len(X))
#         # plt.imshow(X[0])
#         # plt.show()

#         for i in range(len(X)):
#             image = Image.fromarray(X[i])                          
#             image_path = os.path.join(output_dir, f'{str_labels[i]}_{i+section}.png')


#             ## resize for FID and IS inception model
#             resize_transform = transforms.Resize((299, 299))
#             image = resize_transform(image)

#             image.save(image_path)


#             # print(len(os.listdir('FID_IS_SYMP_299')))
