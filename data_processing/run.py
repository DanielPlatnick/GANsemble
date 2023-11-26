from build_augmented_data import *



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
for i, comb in enumerate(combos):
    print(i, comb)



# Build the synthesized dataset

# build_data(40, augment_strategy_list)
