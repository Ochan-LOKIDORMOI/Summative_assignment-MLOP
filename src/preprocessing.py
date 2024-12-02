import os
import cv2
import numpy as np

def load_data(data_dir):
    """Load images and labels from the specified directory."""
    images = []
    labels = []
    class_names = os.listdir(data_dir)

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, filename)
                if os.path.isfile(img_path):
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (150, 150))  # Resize to expected input shape
                    images.append(img)
                    labels.append(class_names.index(class_name))  # Assign label based on folder index

    X = np.array(images, dtype='float32') / 255.0  # Normalize pixel values
    y = np.array(labels)
    return X, y