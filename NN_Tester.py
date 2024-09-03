import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import sample

# Step 1: Pin GPU Memory
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Step 2: Enable Mixed Precision Training
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# Step 3: Load the saved dataset
data = np.load('emnist_letters.npz')

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
loaded_model = tf.keras.models.load_model('emnist_trained_model.h5')
print("Model loaded successfully!")

# Step 10: Test the Model with 10 Random Samples and Display them in One Plot
# Function to predict and plot random samples
def predict_and_plot_random_samples(model, X_test, y_test, num_samples=10):
    # Randomly select 10 indices from the test dataset
    random_indices = sample(range(X_test.shape[0]), num_samples)
    random_images = X_test[random_indices]
    random_labels = y_test[random_indices]

    plt.figure(figsize=(15, 5))
    for i, (image, label) in enumerate(zip(random_images, random_labels)):
        # Ensure image is preprocessed correctly (normalized and reshaped)
        image_processed = image.reshape(1, 28, 28, 1).astype('float32') / 255.0
        
        # Debugging: Print shape and type information
        print(f"Input image shape: {image_processed.shape}, dtype: {image_processed.dtype}")

        # Get the prediction from the model
        prediction = model.predict(image_processed)
        
        # Debugging: Print prediction details
        print(f"Raw prediction output: {prediction}")
        
        # Extract the predicted class using argmax
        predicted_class = tf.argmax(prediction[0]).numpy()
        true_class = tf.argmax(label).numpy()

        # Debugging: Print predicted and true classes
        print(f"Predicted class: {predicted_class}, True class: {true_class}")

        # Plot the image and prediction
        plt.subplot(2, 5, i + 1)
        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.title(f'True: {chr(true_class + 65)}\nPred: {chr(predicted_class + 65)}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage after training or loading the model
# Ensure your model is defined and trained or loaded before calling this function
predict_and_plot_random_samples(loaded_model, X_test, y_test, num_samples=10)