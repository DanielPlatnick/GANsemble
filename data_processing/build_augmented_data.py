from data_preprocessing import *


#NOTE: Maybe shuffle the class dir list before building the datasets and set the seed. Eitherway just keep in mind the bias and the effects of simulated randomness

# Checking to make sure each example is the same dimensions
def validate_data(data_dir, image_1_path):
    valid_dataset_size_check = 0
    invalid_dataset_size_check = 0
    plastic_obs = 0
    non_plastic_obs = 0
    image_1 = cv2.imread(image_1_path)
    image_1_size = np.shape(image_1)

    class_dir_size_list = []

    for class_dir in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_dir)
        class_dir_size_list.append(len(os.listdir(class_dir)))
        if 'Cellulosic' in class_dir or 'Silica' in class_dir:
            non_plastic_obs += len(os.listdir(class_dir))
        else:
            plastic_obs += len(os.listdir(class_dir))
        for data_instance in os.listdir(class_dir):
            data_instance = os.path.join(class_dir, data_instance)
            data_instance = cv2.imread(data_instance)
            if np.shape(data_instance) == image_1_size:
                valid_dataset_size_check += 1
            else:
                invalid_dataset_size_check += 1

    return valid_dataset_size_check, invalid_dataset_size_check, class_dir_size_list, plastic_obs, non_plastic_obs


def get_num_needed_samples(data_dir, synth_data_class_size):
    class_dir_list = os.listdir(data_dir)
    num_classes = len(class_dir_list)
    print(class_dir_list)
    print(f'Number of classes: {num_classes}')

    class_dir_lengths = []
    class_num_required_samples = []
    for class_dir in class_dir_list:
        class_list = os.listdir(data_dir + class_dir)
        class_dir_lengths.append(len(class_list))
        class_num_required_samples.append((str(class_dir), synth_data_class_size))
        # print(class_list)

    return class_dir_lengths, class_num_required_samples


def build_datasets():
    augmented_data_path = os.getcwd() + "\\augmented_datasets"
    if not os.path.exists(augmented_data_path):
        os.mkdir(augmented_data_path)
    return None


def create_charts():
    raw_data_dir = os.getcwd() + '\\raw_data\\'
    polar_data_dir = raw_data_dir + 'polar\\'
    testing_image_path = polar_data_dir + 'Silica\\Silica_95.png' 
    testing_image = Image.open(testing_image_path)
    testing_image = np.array(testing_image)

    plt.imshow(testing_image)
    plt.show()


    combos = combine_augment_strategies()
    combos = [strat_combos for strat_combos in combos]
    listed_combos = combos
    for i, comb in enumerate(listed_combos):
        print(i, comb)

    for combo in combos[14]:
        print(combo)
        testing_image = combo(testing_image)

    plt.imshow(testing_image)
    plt.show()



# GOAL: 
# For augment strategy in augment strategies:
#   Xin Tian et al. has provided an extensive study considering the amount of synthetic samples required to maximize the performance of Deep learning models using augmented data
#   That amount is shown to be 40 empirically in the works: xxxxx

# Main build_augmented_data.py
# Working with the polar data and treating them as images
raw_data_dir = os.getcwd() + '\\raw_data\\'
polar_data_dir = raw_data_dir + 'polar\\'
augmented_data_dir = os.getcwd() + '\\augmented_datasets\\'
first_image_comparison = polar_data_dir + 'Silica\\Silica_95.png' 


valid_dataset_size_check, invalid_dataset_size_check, class_dir_size_list, plastic_obs, non_plastic_obs = validate_data(polar_data_dir, first_image_comparison) 
print(f'Number of valid datapoints: {valid_dataset_size_check}, invalid datapoints: {invalid_dataset_size_check}')
print(f'Sizes of each class: {class_dir_size_list}, number of plastic obs: {plastic_obs}, number of non-plastic: {non_plastic_obs}')



class_dir_lengths, class_num_required_samples = get_num_needed_samples(polar_data_dir, 30)
print(f'Total dataset size: {sum(class_dir_lengths)}')
print(f'Number of samples to create for each class: {class_num_required_samples}')

# create_charts()
testing_image_path = polar_data_dir + 'Silica\\Silica_95.png' 
testing_image = Image.open(testing_image_path)
testing_image = np.array(testing_image)
combos = combine_augment_strategies()
combos = [strat_combos for strat_combos in combos]
augment_strategy_list = combos
# for i, comb in enumerate(listed_combos):
#     print(i, comb)

# for combo in combos[14]:
#     print(combo)
#     testing_image = combo(testing_image)
num_samples_needed = 2
if not os.path.exists(augmented_data_dir):
    os.makedirs(augmented_data_dir)

# For each augmentation strategy, create a synthetic dataset of size: num_samples_needed * num_classes
for aug_strat in augment_strategy_list:
    aug_strat = augment_strategy_list[11]
    print(aug_strat)

    for obs_class in os.listdir(polar_data_dir):
        class_dir = polar_data_dir + obs_class

        class_obs_list = os.listdir(class_dir)
        num_real_samples = len(class_obs_list)
        # print(class_dir_lengths, class_num_required_samples)
        num_samples_generated = 0

        while num_samples_generated < num_samples_needed:
            # Calculate the number of samples to generate in the current iteration
            samples_to_generate = min(num_real_samples, num_samples_needed - num_samples_generated)

            # Generate synthetic samples in the current batch
            if samples_to_generate < num_real_samples:
                for sample in range(samples_to_generate):
                    real_sample_path = os.path.join(class_dir, class_obs_list[sample])
                    real_sample = Image.open(real_sample_path)
                    real_sample = np.array(real_sample)
                    # print(real_sample)
                    print(real_sample_path)
                    # plt.imshow(real_sample)
                    # plt.show()

                    for strat in aug_strat:
                        real_sample = strat(real_sample)
                    
                    augmented_image = Image.fromarray(real_sample.astype('uint8'))
                    augmented_data_class_dir = augmented_data_dir + obs_class
                    if not os.path.exists(augmented_data_class_dir): os.mkdir(augmented_data_class_dir)
                    class_name = class_obs_list[sample].split('_')[0]
                    png_path = os.path.join(augmented_data_class_dir, f"augstrat_{augment_strategy_list.index(aug_strat)}_{class_name}_{num_samples_generated + 1}.png")
                    augmented_image.save(png_path)

                    # plt.imshow(augmented_image)
                    # plt.show()
                    
                    num_samples_generated += 1
                # Breaks out of loop after exhaustively 
                break
            
            if samples_to_generate >= num_real_samples:
                for sample in range(num_real_samples):
                    real_sample_path = os.path.join(class_dir, class_obs_list[sample])
                    real_sample = Image.open(real_sample_path)
                    real_sample = np.array(real_sample)
                    # print(real_sample)
                    print(real_sample_path)

                    for strat in aug_strat:
                        real_sample = strat(real_sample)
                    
                    augmented_image = Image.fromarray(real_sample.astype('uint8'))
                    augmented_data_class_dir = augmented_data_dir + obs_class
                    if not os.path.exists(augmented_data_class_dir): os.mkdir(augmented_data_class_dir)
                    class_name = class_obs_list[sample].split('_')[0]
                    png_path = os.path.join(augmented_data_class_dir, f"augstrat_{augment_strategy_list.index(aug_strat)}_{class_name}_{num_samples_generated + 1}.png")
                    augmented_image.save(png_path)
                
                    num_samples_generated += 1
                    print(real_sample_path)


            # Check if there are remaining samples to generate
            remaining_synth_needed = num_samples_needed - num_samples_generated

            if remaining_synth_needed > 0 and remaining_synth_needed < num_real_samples:
                # Randomly choose from the remaining samples
                for _ in range(remaining_synth_needed):
                    random_sample_path = random.choice(class_obs_list)
                    random_sample_path = os.path.join(class_dir, random_sample_path)
                    real_sample = Image.open(real_sample_path)
                    real_sample = np.array(real_sample)

                    for strat in aug_strat:
                        real_sample = strat(real_sample)
                    
                    augmented_image = Image.fromarray(real_sample.astype('uint8'))
                    augmented_data_class_dir = augmented_data_dir + obs_class
                    if not os.path.exists(augmented_data_class_dir): os.mkdir(augmented_data_class_dir)
                    class_name = class_obs_list[sample].split('_')[0]
                    png_path = os.path.join(augmented_data_class_dir, f"augstrat_{augment_strategy_list.index(aug_strat)}_{class_name}_{num_samples_generated + 1}.png")
                    augmented_image.save(png_path)





                    num_samples_generated += 1
                    print(random_sample_path)

        print("Total samples generated:", num_samples_generated)
        print(num_samples_generated, num_samples_needed, class_dir)
    break        

        






# class_dir_list = os.listdir(polar_data_dir)
# num_classes = len(class_dir_list)
# print(class_dir_list)
# print(f'Number of classes: {num_classes}')




# # SAVE THIS CODE LATER FOR GENERATING PLOTS
# testing_image_path = polar_data_dir + 'Silica\\Silica_95.png' 
# testing_image = Image.open(testing_image_path)
# testing_image = np.array(testing_image)
# print(np.shape(testing_image))
# plt.imshow(testing_image)
# plt.show()

# testing_image = Image.open(testing_image_path)
# testing_image = np.array(testing_image)
# print(np.shape(testing_image))

# # just add extra loop to go through iterable
# combos = combine_augment_strategies()
# combos = [strat_combos for strat_combos in combos]
# listed_combos = combos
# for i, comb in enumerate(listed_combos):
#     print(i, comb)

# for combo in combos[12]:
#     print(combo)
#     testing_image = combo(testing_image)

# plt.imshow(testing_image)
# plt.show()


# # # # Load the image
# # testing_image_path = polar_data_dir + 'Silica/Silica_95.png' 
# # testing_image = Image.open(testing_image_path)
# # print(np.shape(testing_image))
# # plt.imshow(testing_image)
# # plt.show()
