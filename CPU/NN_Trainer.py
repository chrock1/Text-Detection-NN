import os

# Ensure threading settings are configured before any TensorFlow operation
os.environ["OMP_NUM_THREADS"] = "4"  # Set the number of OpenMP threads
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"  # Set the number of intra-op parallelism threads
os.environ["TF_NUM_INTEROP_THREADS"] = "2"  # Set the number of inter-op parallelism threads

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from Header import NN_func

# Optionally, disable eager execution for faster graph mode execution
# tf.compat.v1.disable_eager_execution()

BATCH_NUM = 32  # Adjust batch size for CPU optimization

# Load dataset
data = np.load(NN_func.data_path('emnist_letters.npz'))
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

X_train_rotated_flipped = np.array([NN_func.rotate_and_flip(img) for img in X_train])
X_test_rotated_flipped = np.array([NN_func.rotate_and_flip(img) for img in X_test])

y_train -= 1
y_test -= 1

y_train = tf.keras.utils.to_categorical(y_train, num_classes=26)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=26)

class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
class_weight_dict = dict(enumerate(class_weights))

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_rotated_flipped, y_train))
train_dataset = train_dataset.map(NN_func.preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(10000).batch(BATCH_NUM).cache().prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_rotated_flipped, y_test))
test_dataset = test_dataset.batch(BATCH_NUM).cache().prefetch(tf.data.AUTOTUNE)

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
gradient_callback = NN_func.GradientLoggingCallback(log_dir, train_dataset)

model = tf.keras.Sequential([
    tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.SeparableConv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.SeparableConv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(26, activation='softmax')
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
lr_schedule_callback = LearningRateScheduler(lambda epoch, lr: lr * 0.9 if epoch > 10 else lr)
reduce_lr_callback = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, min_lr=1e-8, verbose=1)

history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=test_dataset,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, tensorboard_callback, lr_schedule_callback, gradient_callback, reduce_lr_callback]
)

model.save(NN_func.model_path('emnist_trained_model.h5'))
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')
NN_func.predict_and_plot_random_samples(model, X_test_rotated_flipped, y_test, num_samples=10)
