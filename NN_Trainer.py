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

# Step 4: Optimize Data Pipeline with Enhanced Data Augmentation
batch_size = 256  # Smaller batch size to introduce more noise during training
augmenter = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

def preprocess(image, label):
    # Apply data augmentation
    image = augmenter(image)
    return image, label

# Convert datasets to tf.data objects and set the correct order for caching
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.map(preprocess).shuffle(10000).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

# Step 5: Define a Simpler Neural Network Model with Stronger Regularization
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),  # Increase dropout rate

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),  # Increase dropout rate

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # L2 Regularization
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(26, activation='softmax')
])

# Step 6: Compile the Model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Step 7: Implement Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Step 8: Train the Model
history = model.fit(train_dataset, epochs=50, validation_data=test_dataset, callbacks=[early_stopping])

# Step 9: Evaluate the Model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

model.save('emnist_trained_model.h5')
print("Model saved successfully!")

# Step 10: Load the Saved Model for Future Use
loaded_model = tf.keras.models.load_model('emnist_trained_model.h5')
print("Model loaded successfully!")

# Step 11: Define the Function to Test Random Samples
def predict_and_plot_random_samples(model, X_test, y_test, num_samples=10):
    random_indices = sample(range(X_test.shape[0]), num_samples)
    random_images = X_test[random_indices]
    random_labels = y_test[random_indices]

    plt.figure(figsize=(15, 5))
    for i, (image, label) in enumerate(zip(random_images, random_labels)):
        image_processed = image.reshape(1, 28, 28, 1).astype('float32') / 255.0
        prediction = model.predict(image_processed)
        predicted_class = tf.argmax(prediction[0]).numpy()
        true_class = tf.argmax(label).numpy()

        plt.subplot(2, 5, i + 1)
        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.title(f'True: {chr(true_class + 65)}\nPred: {chr(predicted_class + 65)}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Call the function to test 10 random samples and display them
predict_and_plot_random_samples(loaded_model, X_test, y_test, num_samples=10)