import os
import random
import cv2
import numpy as np

IMG_SIZE = 150

def preprocess_user_data(directory, categories):
    data = []

    for category in categories:
        path = os.path.join(directory, category)
        label = categories.index(category)

        if not os.path.isdir(path):
            print(f"Skipping invalid category path: {path}")
            continue

        has_images = False  # Track if valid images are found
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            try:
                if not img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    print(f"Skipping non-image file: {img_path}")
                    continue

                img_arr = cv2.imread(img_path)
                if img_arr is None:
                    print(f"Failed to read image: {img_path}")
                    continue

                img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
                data.append([img_arr, label])
                has_images = True
            except Exception as e:
                print(f"Error processing file {img_path}: {e}")

        if not has_images:
            print(f"No valid images found in category: {category}")

    if not data:
        raise ValueError("No valid image data found in the provided dataset.")

    random.shuffle(data)
    X, y = [], []
    for features, label in data:
        X.append(features)
        y.append(label)

    X = np.array(X) / 255.0
    y = np.array(y)

    return X, y