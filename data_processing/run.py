from build_augmented_data import *
from create_gan_data import *



""" 
Main file for GANsemble 
"""

## success
## 1. create uncorrelated test dir for later
data_processing_dir = os.getcwd() + "\\data_processing\\"
raw_data_dir = data_processing_dir + 'raw_data\\'
# create_unbiased_eval_set(raw_data_dir, 2)


## need to update data augmentation types
## 2. build the augmented datasets

# combos = combine_augment_strategies()
# combos = [strat_combos for strat_combos in combos]
# augment_strategy_list = [combos[0],combos[3]]
# for i, comb in enumerate(combos):
#     print(i, comb)

# print(augment_strategy_list, combos)

# build_data(100, augment_strategy_list)



# 3. build baseline 1: no augmentation with resampling to fix imbalance
#    build baseline 2: imbalanced with no augmentation  -  just copy raw_data dir
   



# 4. run data_chooser_model to empirically choose the best 3 augmentation strategy

#  testing against baseline:
#         data aug + cGAN samples
#         only cGAN samples as data augmentation imbalance fix for classifier
#         


# 5. train MPcGAN on the top 3 augmented datasets 
#    add generated training samples from MPcGAN to original data and see if can use it to improve the accuracy
#    use a classifier to see how cGAN samples impacts performance:
        #  experiment with cgan to generate samples in different ways

# 6. use a model 1: CNN and also a model 2: resnet to test the MPcGAN-aided synthetic datasets versus baseline methods such as resampling as well as no resampling and no augmentation



# raw_data_dir = os.getcwd() + '\\raw_data\\'
# polar_data_dir = raw_data_dir + 'polar\\'
# augmented_data_dir = os.getcwd() + '\\augmented_datasets\\'
# first_image_comparison = polar_data_dir + 'Silica\\Silica_95.png' 


# valid_dataset_size_check, invalid_dataset_size_check, class_dir_size_list, plastic_obs, non_plastic_obs = validate_data(polar_data_dir, first_image_comparison) 
# print(f'Number of valid datapoints: {valid_dataset_size_check}, invalid datapoints: {invalid_dataset_size_check}')
# print(f'Sizes of each class: {class_dir_size_list}, number of plastic obs: {plastic_obs}, number of non-plastic: {non_plastic_obs}')



# class_dir_lengths, class_num_required_samples = get_num_needed_samples(polar_data_dir, 30)
# print(f'Total dataset size: {sum(class_dir_lengths)}')
# print(f'Number of samples to create for each class: {class_num_required_samples}')

# # create_charts()
# testing_image_path = polar_data_dir + 'Silica\\Silica_95.png' 
# testing_image = Image.open(testing_image_path)
# testing_image = np.array(testing_image)
# combos = combine_augment_strategies()
# combos = [strat_combos for strat_combos in combos]
# augment_strategy_list = combos
# for i, comb in enumerate(combos):
#     print(i, comb)


