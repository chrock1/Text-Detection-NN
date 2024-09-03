import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import sample
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from datetime import datetime
from sklearn.utils import class_weight

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
def rotate_and_flip(image):
    # Rotate the image 90 degrees clockwise
    rotated_image = np.rot90(image, k=3)  # Rotate 90 degrees clockwise (equivalent to counter-clockwise three times)
    
    # Flip the image horizontally
    flipped_image = np.fliplr(rotated_image)
    
    return flipped_image


X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

X_train_rotated_flipped = np.array([rotate_and_flip(img) for img in X_train])
X_test_rotated_flipped = np.array([rotate_and_flip(img) for img in X_test])

# Adjust labels from [1, 26] to [0, 25]
y_train -= 1
y_test -= 1

# One-hot encode labels to match the output shape of the model
y_train = tf.keras.utils.to_categorical(y_train, num_classes=26)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=26)

# Check the class distribution in the training dataset
class_distribution = np.sum(y_train, axis=0)
print("Class distribution in the training set:", class_distribution)

# Compute class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))

# Convert class weights to a dictionary
class_weight_dict = dict(enumerate(class_weights))

# Step 4: Optimize Data Pipeline with Enhanced Data Augmentation
batch_size = 8  # Smaller batch size to introduce more noise during training
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
    image = image
    return image, label

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_rotated_flipped, y_train))
train_dataset = train_dataset.map(preprocess).shuffle(10000).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test_rotated_flipped, y_test))
test_dataset = test_dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

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


log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
gradient_callback = GradientLoggingCallback(log_dir, train_dataset)

# Step 5: Define a Model with Stronger Regularization
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(128, (3, 3), activation='selu', kernel_initializer='lecun_normal', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='selu', kernel_initializer='lecun_normal'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),  # Dropout works well with SELU to induce self-normalizing properties

    tf.keras.layers.Conv2D(256, (3, 3), activation='selu', kernel_initializer='lecun_normal'),
    tf.keras.layers.Conv2D(256, (3, 3), activation='selu', kernel_initializer='lecun_normal'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='selu', kernel_initializer='lecun_normal'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(26, activation='softmax')
])

# model = tf.keras.models.load_model('emnist_trained_model.h5')
# print("Model loaded successfully!")

# Step 6: Compile the Model with an Initial Learning Rate
initial_learning_rate = 0.00001
optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate,momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Step 7: Implement Early Stopping and TensorBoard for Monitoring
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Step 8: Define the Learning Rate Scheduler
def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    elif epoch < 20:
        return lr * 0.5  # Reduce LR by half after 10 epochs
    elif epoch < 30:
        return lr * 0.1  # Further reduce after 20 epochs
    else:
        return lr * 0.01  # Further reduce after 30 epochs

lr_schedule_callback = LearningRateScheduler(lr_schedule)

reduce_lr_callback = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, min_lr=1e-8, verbose=1)

# Step 9: Train the Model with the Learning Rate Scheduler
history = model.fit(
    train_dataset, 
    epochs=100,  # Increase the number of epochs
    validation_data=test_dataset, 
    class_weight=class_weight_dict,  # Add class weights
    callbacks=[early_stopping, tensorboard_callback, lr_schedule_callback, gradient_callback, reduce_lr_callback]  # Include the LR scheduler
)

# Step 10: Save the Trained Model
model.save('emnist_trained_model.h5')
print("Model saved successfully!")


# Step 9: Evaluate the Model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')


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

# Example usage after training or loading the model
# Ensure your model is defined and trained or loaded before calling this function
predict_and_plot_random_samples(model, X_test_rotated_flipped, y_test, num_samples=10)