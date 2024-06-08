from imports import *


class DataAugmentation:
    """
    Class for applying data augmentation techniques to image datasets.

    Attributes:
        data_augmentation (tensorflow.keras.Sequential): Sequential model containing data augmentation layers.
        rescaling (tensorflow.keras.layers.Rescaling): Rescaling layer for normalizing pixel values.
    """

    def __init__(self):
        """
        Initializes DataAugmentation class with default augmentation techniques and rescaling layer.
        """
        self.data_augmentation = Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
            layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
        ])
        self.rescaling = layers.Rescaling(1./255)

    def apply(self, train_dataset, val_dataset, test_dataset):
        """
        Applies data augmentation to training dataset and rescaling to validation and test datasets.

        Args:
            train_dataset (tensorflow.data.Dataset): Training dataset containing images and labels.
            val_dataset (tensorflow.data.Dataset): Validation dataset containing images and labels.
            test_dataset (tensorflow.data.Dataset): Test dataset containing images and labels.

        Returns:
            tensorflow.data.Dataset, tensorflow.data.Dataset, tensorflow.data.Dataset: Augmented training dataset,
            validation dataset with rescaling, and test dataset with rescaling.
        """
        try:
            train_dataset_augmented = train_dataset.map(lambda x, y: (self.data_augmentation(x, training=True), y))
            val_dataset = val_dataset.map(lambda x, y: (self.rescaling(x), y))
            test_dataset = test_dataset.map(lambda x, y: (self.rescaling(x), y))

            return train_dataset_augmented, val_dataset, test_dataset
        except Exception as e:
            print(f"An error occurred during data augmentation: {e}")
            return None, None, None

def main():
    """
    Main function for loading datasets, applying data augmentation, and running the script.
    """
    train_data = "../data/Training"
    test_data = "../data/Testing"
    batch_size = 32
    image_size = (224, 224)

    dataset_loader = DatasetLoader(train_data, test_data, batch_size, image_size)
    train_dataset, val_dataset, test_dataset, class_names = dataset_loader.load_datasets()

    if train_dataset is None:
        print("Failed to load datasets.")
        return

    print("Datasets loaded successfully.")

    augmenter = DataAugmentation()
    train_dataset_augmented, val_dataset, test_dataset = augmenter.apply(train_dataset, val_dataset, test_dataset)

    if train_dataset_augmented is not None:
        print("Augmented datasets created successfully.")
    else:
        print("Failed to augment datasets.")

if __name__ == "__main__":
    main()
