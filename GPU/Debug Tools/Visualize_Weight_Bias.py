import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import sample
import os
import sys

# Get the parent directory of the current script
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the parent directory to the system path
sys.path.append(parent_directory)

# Import the module from the parent directory
from Header import NN_func  # Now you can import your custom module


# Step 3: Load the saved dataset
data = np.load(NN_func.data_path('emnist_letters.npz'))

# Extract the training and testing datasets
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']


# Adjust labels from [1, 26] to [0, 25]
y_train -= 1
y_test -= 1

# One-hot encode labels to match the output shape of the model
y_train = tf.keras.utils.to_categorical(y_train, num_classes=26)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=26)

loaded_model = tf.keras.models.load_model(NN_func.model_path('emnist_trained_model.h5'))
print("Model loaded successfully!")

for layer in loaded_model.layers:
    # Check if the layer has weights (some layers, like dropout, do not have weights)
    if len(layer.get_weights()) > 0:
        # Retrieve weights and biases
        weights, biases = layer.get_weights()

        # Compute statistics for weights
        weight_mean = np.mean(weights)
        weight_std = np.std(weights)
        weight_min = np.min(weights)
        weight_max = np.max(weights)

        # Compute statistics for biases
        bias_mean = np.mean(biases)
        bias_std = np.std(biases)
        bias_min = np.min(biases)
        bias_max = np.max(biases)

        # Print statistics
        print(f"Layer: {layer.name}")
        print(f"  Weights - mean: {weight_mean:.4f}, std: {weight_std:.4f}, min: {weight_min:.4f}, max: {weight_max:.4f}")
        print(f"  Biases  - mean: {bias_mean:.4f}, std: {bias_std:.4f}, min: {bias_min:.4f}, max: {bias_max:.4f}\n")

# Determine the number of layers with weights and biases
layers_with_weights = [layer for layer in loaded_model.layers if len(layer.get_weights()) > 0]

# Number of subplots (3 rows and enough columns to fit all layers)
num_layers = len(layers_with_weights)
num_cols = (num_layers + 2) // 3  # Calculate the number of columns needed

# Create a figure with multiple subplots
fig, axs = plt.subplots(3, num_cols, figsize=(5 * num_cols, 15))  # 3 rows

# Flatten the axs array for easy iteration
axs = axs.flatten()

# Loop over each layer with weights and biases
for idx, layer in enumerate(layers_with_weights):
    # Retrieve weights and biases
    weights, biases = layer.get_weights()

    # Flatten the weights and biases for plotting
    flat_weights = weights.flatten()
    flat_biases = biases.flatten()

    # Plot on the current subplot axis
    ax1 = axs[idx]

    # Plot weights distribution on the first y-axis
    ax1.hist(flat_weights, bins=30, color='blue', alpha=0.5, label='Weights')
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Weights Frequency", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis sharing the same x-axis for biases
    ax2 = ax1.twinx()
    ax2.hist(flat_biases, bins=30, color='orange', alpha=0.5, label='Biases')
    ax2.set_ylabel("Biases Frequency", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Add title for each subplot
    ax1.set_title(f"Layer: {layer.name}")

# Hide any unused subplots
for ax in axs[num_layers:]:
    ax.remove()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()