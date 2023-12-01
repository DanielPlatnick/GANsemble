import shutil
from build_augmented_data import *


"""
Chosen dataset based on CNN combines with the original dataset to balance it and train cGAN
"""


# when training the cGAN on data for real, only use transformations that preserve the invertible property (so the cGAN will not produce samples out of the training set distribution)
def generate_gan_data_paths(data_processing_dir, desired_enriched_class_size=35):

    chosen_dir = data_processing_dir + "augmented_datasets\\aug_data_30_samples\\aug_strategy_4\\"
    raw_data_dir = data_processing_dir + "raw_data\\polar\\"
    enriched_data_dir = data_processing_dir + 'gan_dataset\\'
    if not os.path.exists(enriched_data_dir): os.mkdir(enriched_data_dir)
    raw_class_sizes = []

    for class_dir in os.listdir(raw_data_dir):
        class_dir_path = os.path.join(raw_data_dir, class_dir) + '\\'
        raw_class_sizes.append(len(os.listdir(class_dir_path)))

    list_of_class_lists = []

    for class_dir in range(len(os.listdir(raw_data_dir))):

        class_list = os.listdir(raw_data_dir)
        raw_class_dir_path = os.path.join(raw_data_dir, class_list[class_dir]) + '\\'
        synth_class_dir_path = os.path.join(chosen_dir, class_list[class_dir]) + '\\'

        if not os.path.exists(enriched_data_dir + class_list[class_dir]): os.mkdir(enriched_data_dir + class_list[class_dir])

        starting_raw_data_list = os.listdir(raw_class_dir_path)
        starting_synth_data_list = os.listdir(synth_class_dir_path)
        curr_enriched_class = []

        if len(curr_enriched_class) < desired_enriched_class_size:
            # use ground truth samples in enriched data until out of them
            for raw_sample in starting_raw_data_list:
                if len(curr_enriched_class) < desired_enriched_class_size:
                    raw_sample_path = raw_class_dir_path + raw_sample
                    curr_enriched_class.append(raw_sample_path)

            # if len of enriched data class still under the desired amount, keep appending synthetic data
        if len(curr_enriched_class) < desired_enriched_class_size:
            for _ in range(len(starting_synth_data_list)):
                if len(curr_enriched_class) < desired_enriched_class_size:
                    synth_sample_path = synth_class_dir_path + starting_synth_data_list.pop(random.randrange(len(starting_synth_data_list)))
                    curr_enriched_class.append(synth_sample_path)
        list_of_class_lists.append(curr_enriched_class)

    return list_of_class_lists


# copy each file into new dataset dir from enriched paths list
def generate_gan_dataset(enriched_data_paths, enriched_dataset_dir):
    enriched_class_list = os.listdir(enriched_dataset_dir)
    for class_path_list in range(len(enriched_class_list)):
        enriched_class_dir = enriched_dataset_dir + enriched_class_list[class_path_list] + '\\'
        print(enriched_data_paths[class_path_list])
        print(enriched_class_dir)

        for obs_path in range(len(enriched_data_paths[class_path_list])):
            print(obs_path)
            print(enriched_class_dir)
            shutil.copy(enriched_data_paths[class_path_list][obs_path], enriched_class_dir)



def generate_gan_baseline_data_paths(data_processing_dir, desired_enriched_class_size=35):

    raw_data_dir = data_processing_dir + "raw_data\\polar\\"
    enriched_data_dir = data_processing_dir + 'gan_baseline_dataset\\'
    if not os.path.exists(enriched_data_dir): os.mkdir(enriched_data_dir)
    raw_class_sizes = []

    for class_dir in os.listdir(raw_data_dir):
        class_dir_path = os.path.join(raw_data_dir, class_dir) + '\\'
        raw_class_sizes.append(len(os.listdir(class_dir_path)))

    list_of_class_lists = []

    for class_dir in range(len(os.listdir(raw_data_dir))):

        class_list = os.listdir(raw_data_dir)
        raw_class_dir_path = os.path.join(raw_data_dir, class_list[class_dir]) + '\\'

        if not os.path.exists(enriched_data_dir + class_list[class_dir]): os.mkdir(enriched_data_dir + class_list[class_dir])

        # starting_raw_data_list = os.listdir(raw_class_dir_path)
        curr_enriched_class = []

        while len(curr_enriched_class) < desired_enriched_class_size:
            # use ground truth samples with replacement
            starting_raw_data_list = os.listdir(raw_class_dir_path)

            
            if len(curr_enriched_class) < len(starting_raw_data_list):
                for raw_sample in starting_raw_data_list:
                    if len(curr_enriched_class) < desired_enriched_class_size:
                        raw_sample_path = raw_class_dir_path + raw_sample
                        curr_enriched_class.append(raw_sample_path)
            else:
                curr_enriched_class.append(random.choice(starting_raw_data_list))

            # if len(curr_enriched_class) < desired_enriched_class_size:
        list_of_class_lists.append(curr_enriched_class)
    
    return list_of_class_lists
                




data_processing_dir = os.getcwd() + "\\data_processing\\"
enriched_dataset_dir = data_processing_dir + 'gan_dataset\\'

# gan_dataset_paths = generate_gan_data_paths(data_processing_dir=data_processing_dir, desired_enriched_class_size=35)

# gan_dataset_size = sum([len(x) for x in gan_dataset_paths])
# print(gan_dataset_size)

# generate_gan_dataset(gan_dataset_paths, enriched_dataset_dir)
gan_baseline_dataset_paths = generate_gan_baseline_data_paths(data_processing_dir=data_processing_dir, desired_enriched_class_size=35)
print([len(x) for x in gan_baseline_dataset_paths])
