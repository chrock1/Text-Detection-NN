import numpy as np
import matplotlib as plt
import tensorflow as tf
from random import sample

# Step 3: Load the saved dataset
data = np.load('emnist_letters.npz')

# Extract the training and testing datasets
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

loaded_model = tf.keras.models.load_model('emnist_trained_model.h5')
print("Model loaded successfully!")
# Step 10: Test the Model with 10 Random Samples and Display them in One Plot
def predict_and_plot_random_samples(model, X_test, y_test, num_samples=10):
    # Randomly select 10 indices from the test dataset
    random_indices = sample(range(X_test.shape[0]), num_samples)
    random_images = X_test[random_indices]
    random_labels = y_test[random_indices]

    # Set up the plot
    plt.figure(figsize=(15, 5))
    for i, (image, label) in enumerate(zip(random_images, random_labels)):
        # Preprocess image for prediction
        image_processed = image.reshape(1, 28, 28, 1).astype('float32') / 255.0
        prediction = model.predict(image_processed)
        predicted_class = tf.argmax(prediction[0]).numpy()
        true_class = tf.argmax(label).numpy()

        # Plot the image and prediction
        plt.subplot(2, 5, i + 1)
        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.title(f'True: {chr(true_class + 65)}\nPred: {chr(predicted_class + 65)}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Call the function to test 10 random samples and display them
predict_and_plot_random_samples(loaded_model, X_test, y_test, num_samples=10)