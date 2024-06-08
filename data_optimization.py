from imports import *


class DatasetOptimizer:
    """
    Class for optimizing TensorFlow datasets by caching and prefetching.

    Attributes:
        buffer_size (int): Size of the buffer used for prefetching. Default is tf.data.AUTOTUNE.
    """

    def __init__(self, buffer_size=tf.data.AUTOTUNE):
        """
        Initializes DatasetOptimizer with a specified buffer size.

        Args:
            buffer_size (int): Size of the buffer used for prefetching. Default is tf.data.AUTOTUNE.
        """
        self.buffer_size = buffer_size

    def optimize(self, train_dataset, val_dataset, test_dataset=None):
        """
        Optimizes the input datasets by caching and prefetching.

        Args:
            train_dataset (tensorflow.data.Dataset): Training dataset.
            val_dataset (tensorflow.data.Dataset): Validation dataset.
            test_dataset (tensorflow.data.Dataset, optional): Test dataset. Defaults to None.

        Returns:
            tuple: A tuple containing the optimized training, validation, and test datasets (if provided).
        """
        train_dataset = train_dataset.cache().prefetch(buffer_size=self.buffer_size)
        val_dataset = val_dataset.cache().prefetch(buffer_size=self.buffer_size)

        if test_dataset:
            test_dataset = test_dataset.cache().prefetch(buffer_size=self.buffer_size)
            return train_dataset, val_dataset, test_dataset

        return train_dataset, val_dataset


if __name__ == "__main__":
    # Define dataset paths and parameters
    train_data = "../data/Training"
    test_data = "../data/Testing"
    batch_size = 32
    image_size = (224, 224)

    # Load datasets
    dataset_loader = DatasetLoader(train_data, test_data, batch_size, image_size)
    train_dataset, val_dataset, test_dataset, class_names = dataset_loader.load_datasets()

    # Check if datasets are loaded successfully
    if train_dataset is not None and val_dataset is not None and test_dataset is not None:
        print("Datasets loaded successfully.")

        # Apply data augmentation
        augmenter = DataAugmentation()
        train_dataset_augmented, val_dataset, test_dataset = augmenter.apply(train_dataset, val_dataset, test_dataset)

        # Check if data augmentation is successful
        if train_dataset_augmented is not None:
            print("Augmented datasets created successfully.")

            # Optimize datasets
            optimizer = DatasetOptimizer()
            train_dataset_optimized, val_dataset_optimized, test_dataset_optimized = optimizer.optimize(train_dataset_augmented, val_dataset, test_dataset)

            # Print shapes of optimized datasets
            print_shapes(train_dataset_optimized)

            # Check if datasets are optimized successfully
            if train_dataset_optimized and val_dataset_optimized:
                print("Datasets optimized successfully.")
            else:
                print("Failed to optimize datasets.")
        else:
            print("Failed to augment datasets.")
    else:
        print("Failed to load datasets.")
