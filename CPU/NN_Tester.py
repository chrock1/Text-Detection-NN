import numpy as np
import tensorflow as tf
from Header import NN_func

# Step 3: Load the saved dataset

data = np.load(NN_func.data_path('emnist_letters.npz'))


X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

X_train_rotated_flipped = np.array([NN_func.rotate_flip_normalize(img) for img in X_train])
X_test_rotated_flipped = np.array([NN_func.rotate_flip_normalize(img) for img in X_test])

# Adjust labels from [1, 26] to [0, 25]
y_train -= 1
y_test -= 1

# One-hot encode labels to match the output shape of the model
y_train = tf.keras.utils.to_categorical(y_train, num_classes=26)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=26)

loaded_model = tf.keras.models.load_model(NN_func.model_path('emnist_trained_model.keras'))
print("Model loaded successfully!")

# Step 10: Test the Model with 10 Random Samples and Display them in One Plot

# Example usage after training or loading the model
# Ensure your model is defined and trained or loaded before calling this function
NN_func.predict_and_plot_random_samples(loaded_model, X_test_rotated_flipped, y_test, num_samples=10)