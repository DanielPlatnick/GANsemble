from data_preprocessing import *


# all spectra data is in the form of a 2d array
raw_data_dir = os.getcwd() + '\\data_processing\\raw_data\\'

# preprocessing spectra data
polar_data_dir = raw_data_dir + 'polar\\'



# SAVE THIS CODE LATER FOR GENERATING PLOTS
testing_image_path = polar_data_dir + 'Silica\\Silica_95.png' 
testing_image = Image.open(testing_image_path)
testing_image = np.array(testing_image)
print(np.shape(testing_image))


# just add extra loop to go through iterable
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


# # # Load the image
# testing_image_path = polar_data_dir + 'Silica/Silica_95.png' 
# testing_image = Image.open(testing_image_path)
# print(np.shape(testing_image))
# plt.imshow(testing_image)
# plt.show()

