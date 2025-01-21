# visualization.py

import os
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from PIL import Image

# Visualizes 5 random images from the dataset
def visualize_images(path, target_size=(256, 256), num_images=5):
    image_filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    if not image_filenames:
        raise ValueError("No images found in the specified path")

    # Select random images
    selected_images = random.sample(image_filenames, min(num_images, len(image_filenames)))

    fig, axes = plt.subplots(1, num_images, figsize=(8, 8), facecolor='white')

    # Display each image
    for i, image_filename in enumerate(selected_images):
        image_path = os.path.join(path, image_filename)
        image = Image.open(image_path)
        image = image.resize(target_size)

        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(image_filename)

    plt.tight_layout()
    plt.show()
