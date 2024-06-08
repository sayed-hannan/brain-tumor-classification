import matplotlib.pyplot as  plt

# Inside data_utils.py
def print_shapes(dataset):
    for image, label in dataset.take(1):
        print(f' Images shape: {image.shape}, Labels shape: {label.shape}')
    return dataset

# Function to plot images with labels
def plot_images(images, labels, class_names, max_images=25):
    plt.figure(figsize=(10, 10))
    num_images = min(len(images), max_images)
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[tf.argmax(labels[i]).numpy()])
        plt.axis("off")
    plt.show()