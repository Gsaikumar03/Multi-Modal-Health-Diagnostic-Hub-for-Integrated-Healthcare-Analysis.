import os
import random
from PIL import Image
import matplotlib.pyplot as plt


def load_random_images(base_path: str, split: str, label: str, num_images: int = 5) -> None:
    """
    Load and display random images from dataset
    """
    image_dir = os.path.join(base_path, split, label)
    images = os.listdir(image_dir)

    selected_images = random.sample(images, min(num_images, len(images)))

    plt.figure(figsize=(12, 4))

    for idx, img_name in enumerate(selected_images):
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        plt.subplot(1, len(selected_images), idx + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(label)

    plt.show()


def check_image_properties(base_path: str, split: str, label: str) -> None:
    """
    Print size and mode of a sample image
    """
    image_dir = os.path.join(base_path, split, label)
    img_name = os.listdir(image_dir)[0]
    img_path = os.path.join(image_dir, img_name)

    img = Image.open(img_path)
    print("\nSample Image Properties:")
    print("Size:", img.size)
    print("Mode:", img.mode)
