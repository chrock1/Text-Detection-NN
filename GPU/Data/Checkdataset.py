import numpy as np
import collections


# Load the dataset (assuming this is how you loaded it)
data = np.load('emnist_letters.npz')

# Check the shapes and content
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

# Print the shapes and unique labels
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Unique labels in the training set:", np.unique(y_train))
print("Unique labels in the test set:", np.unique(y_test))

# Count the occurrences of each label in the training set
train_label_counts = collections.Counter(y_train)
test_label_counts = collections.Counter(y_test)

print("Training label distribution:", train_label_counts)
print("Test label distribution:", test_label_counts)
