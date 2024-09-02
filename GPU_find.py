import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs available: ", gpus)
else:
    print("No GPU found. Please check your CUDA and cuDNN installation.")
