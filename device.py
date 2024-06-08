import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
print("Number of GPUs:", len(physical_devices))

if len(physical_devices) > 0:
  print("GPU details:", physical_devices[0])
else:
  print("No GPUs available. TensorFlow is likely using CPU.")
