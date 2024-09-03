import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from random import sample

current_directory = os.getcwd()
current_directory = os.path.join(current_directory,'CPU')

DATA_PATH = os.path.join(current_directory,'Data','emnist_letters.npz')
MODEL_PATH = os.path.join(current_directory, 'Models','emnist_trained_model.h5')
AUGMENT = False

#Pin GPU Memory
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

#Enable Mixed Precision Training
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')


def data_path(file_name):

    path = os.path.join(current_directory, 'Data',file_name)
    return path

def model_path(file_name):
    path = os.path.join(current_directory, 'Models',file_name)
    return path

def rotate_and_flip(image):
    # Rotate the image 90 degrees clockwise
    rotated_image = np.rot90(image, k=3)  # Rotate 90 degrees clockwise (equivalent to counter-clockwise three times)
    
    # Flip the image horizontally
    flipped_image = np.fliplr(rotated_image)
    
    return flipped_image


augmenter = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.3),
    tf.keras.layers.RandomContrast(0.3),
    tf.keras.layers.RandomTranslation(0.2, 0.2)
])

def preprocess(image, label):
    # Apply data augmentation
    # image = augmenter(image)
    if (AUGMENT):
        image = augmenter(image)
    else:
        image = image
    return image, label

def predict_and_plot_random_samples(model, X_test, y_test, num_samples=10):
    # Randomly select 10 indices from the test dataset
    random_indices = sample(range(X_test.shape[0]), num_samples)
    random_images = X_test[random_indices]
    random_labels = y_test[random_indices]

    # Set up the plot
    plt.figure(figsize=(15, 5))
    for i, (image, label) in enumerate(zip(random_images, random_labels)):
        # Ensure the image is preprocessed correctly (normalized and reshaped)
        image_processed = image.reshape(1, 28, 28, 1).astype('float32') / 255.0
        
        # Debugging: Print the shape and type of the input image
        print(f"Input image shape: {image_processed.shape}, dtype: {image_processed.dtype}")

        # Get the prediction from the model
        prediction = model.predict(image_processed)
        
        # Debugging: Print the raw prediction output
        print(f"Raw prediction output: {prediction}")
        
        # Ensure we are using the correct axis for argmax to get the predicted class
        predicted_class = np.argmax(prediction[0])  # Use np.argmax to get the predicted class
        true_class = np.argmax(label)  # Use np.argmax to get the true class

        # Debugging: Print predicted and true classes
        print(f"Predicted class: {predicted_class}, True class: {true_class}")

        # Plot the image and prediction
        plt.subplot(2, 5, i + 1)
        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.title(f'True: {chr(true_class + 65)}\nPred: {chr(predicted_class + 65)}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

class GradientLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, train_data):
        super(GradientLoggingCallback, self).__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.train_data = train_data  # Save the training data for gradient computation

    def on_epoch_end(self, epoch, logs=None):
        # Get a batch of data to compute gradients
        x_batch, y_batch = next(iter(self.train_data))

        with tf.GradientTape() as tape:
            # Watch the trainable weights
            tape.watch(self.model.trainable_weights)
            
            # Make a forward pass through the model
            y_pred = self.model(x_batch, training=True)

            # Compute the loss
            loss = self.model.compiled_loss(y_batch, y_pred)

        # Compute the gradients with respect to the trainable weights
        gradients = tape.gradient(loss, self.model.trainable_weights)

        # Log gradients to TensorBoard
        with self.file_writer.as_default():
            for weight, grad in zip(self.model.trainable_weights, gradients):
                if grad is not None:
                    tf.summary.histogram(f'{weight.name}/gradients', grad, step=epoch)
        self.file_writer.flush()