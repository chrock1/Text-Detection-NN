import tensorflow_datasets as tfds
import numpy as np

# Step 1: Download the EMNIST dataset
dataset, info = tfds.load('emnist/letters', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# Step 2: Convert the TensorFlow dataset to NumPy arrays
def convert_to_numpy(dataset):
    images = []
    labels = []
    for image, label in tfds.as_numpy(dataset):
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)

# Convert train and test datasets to numpy
X_train, y_train = convert_to_numpy(train_dataset)
X_test, y_test = convert_to_numpy(test_dataset)

# Step 3: Save the dataset to .npz format
np.savez('emnist_letters.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
print("Dataset saved to emnist_letters.npz")
