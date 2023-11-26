import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import *
from pylab import *
import random
from scipy.ndimage import map_coordinates, gaussian_filter
from itertools import combinations


''' Augmentation functions expect and return a numpy array '''

# Data augmentation method 1 - Horizontal shifting and horizontal flipping
def augment_strategy_1(raw_image):
    shift_value = random.randint(-270, 270)  # Change this to the number of pixels you want to shift the image
    augmented_image = np.roll(raw_image, shift=shift_value, axis=1)

    flip_chance = random.randint(0,1)
    if flip_chance == 0:
        augmented_image = np.flip(augmented_image, axis=1)

    return augmented_image


# Data augmentation method 2 - Varying elastic deformation and random rotation      Note: values started a=50 s=5  --> a-[15,50] s->[2.5,4]
def augment_strategy_2(raw_image, random_state=None, border=30, alpha_range=[10,23], sigma_range=[2.8,3.82]):

    assert len(raw_image.shape) == 3
    alpha = random.randint(alpha_range[0], alpha_range[-1])
    sigma = random.uniform(sigma_range[0], sigma_range[-1])

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = raw_image.shape

    distorted_channels = []
    for channel in range(shape[2]):
        dx = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

        distorted_channel = map_coordinates(raw_image[:,:,channel], indices, order=1).reshape(shape[:2])
        distorted_channels.append(distorted_channel)

    distorted_image = np.stack(distorted_channels, axis=-1)
    distorted_image = Image.fromarray(distorted_image.astype('uint8'))
    width, height = distorted_image.size

    # Add border to fix colour distortion along the edges due to elastic deformation
    for i in range(0, width):
        for j in range(0, border):
            distorted_image.putpixel((i,j), (255,255,255)) # top border
            distorted_image.putpixel((i,height-j-1), (255,255,255)) # bottom border

    for i in range(0, height):
        for j in range(0, border):
            distorted_image.putpixel((j,i), (255,255,255)) # left border
            distorted_image.putpixel((width-j-1,i), (255,255,255)) # right border

    # Add random rotations
    rotation_values = [0, 90, 180, 270]
    rotation_val = random.choice(rotation_values)
    distorted_image = distorted_image.rotate(rotation_val)

    # Apply white border if necessary
    if rotation_val == 90 or rotation_val == 270:
        for i in range(0, height):
            for j in range(0, 150):
                distorted_image.putpixel((j,i), (255,255,255)) # left border
                distorted_image.putpixel((width-j-1,i), (255,255,255)) # right border

    augmented_image = np.array(distorted_image)

    return augmented_image


# Data augmentation method 3 - Random zoom and rotations
def augment_strategy_3(input_image, min_zoom = 1.05, max_zoom = 1.34):

    input_image = Image.fromarray(input_image.astype('uint8'))

    width, height = input_image.size

 
    zoom_level = random.uniform(min_zoom, max_zoom)

    # Calculate the dimensions for the zoomed area
    zoom_area_width = int(width / zoom_level)
    zoom_area_height = int(height / zoom_level)

    # Calculate the position of the zoomed area
    left = (width - zoom_area_width) // 2
    top = (height - zoom_area_height) // 2
    right = left + zoom_area_width
    bottom = top + zoom_area_height

    img_cropped = input_image.crop((left, top, right, bottom))
    augmented_image = img_cropped.resize((width, height))

    rotation_values = [0, 90, 180, 270]
    rotation_val = random.choice(rotation_values)
    augmented_image = augmented_image.rotate(rotation_val)

    # Apply white border if necessary
    if rotation_val == 90 or rotation_val == 270:
        for i in range(0, height):
            for j in range(0, 150):
                augmented_image.putpixel((j,i), (255,255,255)) # left border
                augmented_image.putpixel((width-j-1,i), (255,255,255)) # right border

    augmented_image = np.array(augmented_image)

    return augmented_image


# Data augmentation method 4 - Random circular and horizontal masking
def augment_strategy_4(input_image, num_circles=40, num_rectangles=15):

    img_array = input_image

    for i in range(num_circles):
        # Specify the radius of the circle within the range
        circle_radius = random.randint(5, 10)

        # Calculate the center of the image
        center = (img_array.shape[1] // 2, img_array.shape[0] // 2)
        center_x, center_y = center

        # Move center to a random range and spreading it out
        if i < num_circles // 2:
            center_x += random.randint(-175, 175)
            center_y += random.randint(-175, 0)
        else:
            center_x += random.randint(-175, 175)
            center_y += random.randint(0, 175)

        center = (center_x, center_y)

        # Create a mask for the circle
        y, x = np.ogrid[:img_array.shape[0], :img_array.shape[1]]
        mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2) <= circle_radius ** 2

        # Turn the specified shape to be white
        img_array[mask] = [255, 255, 255]

    for i in range(num_rectangles):
        # Specify the dimensions of the rectangle
        rect_width = random.randint(2,7)
        rect_height = img_array.shape[0]

        # Specify the top-left corner of the rectangle
        rect_x = random.randint(200, img_array.shape[1] - (rect_width+200))
        rect_y = random.randint(0, img_array.shape[0] - rect_height)

        # Create a mask for the rectangle
        mask = (
            (x >= rect_x) & (x < rect_x + rect_width) &
            (y >= rect_y) & (y < rect_y + rect_height)
        )

        # Turn the specified shape to be white
        img_array[mask] = [255, 255, 255]

    # Convert the NumPy array back to an image
    result_image = img_array
    
    return result_image


# Combine the strategies in a factorial study design            #LOOK INTO SPECIFICS OF NUMBER OF STUDIES GENERATED
def combine_augment_strategies(augment_strategies=[augment_strategy_1, augment_strategy_2, augment_strategy_3, augment_strategy_4]):

    solo_combinations = [[strat] for strat in augment_strategies]
    pairwise_combinations = list(combinations(augment_strategies, 2))
    three_way_combinations = list(combinations(augment_strategies, 3))
    four_way_combinations = list(combinations(augment_strategies, 4))

    # Decide either 2-step or full factorial designed study
    # all_combinations = solo_combinations + pairwise_combinations 
    all_combinations = solo_combinations + pairwise_combinations + three_way_combinations + four_way_combinations

    return all_combinations

