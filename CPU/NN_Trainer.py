import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight  # Use sklearn in your code as usual
from Header import NN_func

BATCH_NUM = 264

# Step 3: Load the saved dataset
data = np.load(NN_func.data_path('emnist_letters.npz'))

# Extract the training and testing datasets
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

X_train_rotated_flipped = np.array([NN_func.rotate_and_flip(img) for img in X_train])
X_test_rotated_flipped = np.array([NN_func.rotate_and_flip(img) for img in X_test])

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
class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))

# Convert class weights to a dictionary
class_weight_dict = dict(enumerate(class_weights))

# Step 4: Optimize Data Pipeline with Enhanced Data Augmentation
batch_size = BATCH_NUM  # Smaller batch size to introduce more noise during training

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_rotated_flipped, y_train))
train_dataset = train_dataset.map(NN_func.preprocess).shuffle(10000).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test_rotated_flipped, y_test))
test_dataset = test_dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)


log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
gradient_callback = NN_func.GradientLoggingCallback(log_dir, train_dataset)

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
    tf.keras.layers.Dense(26, activation='softmax')  # Update the output layer to 26 neurons
])


# model = tf.keras.models.load_model('emnist_trained_model.h5')
# print("Model loaded successfully!")

# Step 6: Compile the Model with an Initial Learning Rate
# Compile with an adaptive learning rate
initial_learning_rate = 0.00001
optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate, momentum=0.9, nesterov=True)
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
model.save(NN_func.model_path('emnist_trained_model.h5'))
print("Model saved successfully!")


# Step 9: Evaluate the Model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Example usage after training or loading the model
# Ensure your model is defined and trained or loaded before calling this function
NN_func.predict_and_plot_random_samples(model, X_test_rotated_flipped, y_test, num_samples=10)