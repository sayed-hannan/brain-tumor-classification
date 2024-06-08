from imports import*


class DatasetLoader:
    def __init__(self, train_data_dir, test_data_dir, batch_size=32, image_size=(224, 224), validation_split=0.1, seed=42):
        """
        Initialize the DatasetLoader with directory paths and parameters.

        Parameters:
        train_data_dir (str): Path to the training data directory.
        test_data_dir (str): Path to the testing data directory.
        batch_size (int): Number of samples per batch of computation.
        image_size (tuple): Size to resize images to after they are read from disk.
        validation_split (float): Fraction of training data to be used as validation data.
        seed (int): Random seed for shuffling and transformations.
        """
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.validation_split = validation_split
        self.seed = seed
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_names = None

    def load_datasets(self):
        """
        Load training, validation, and test datasets from the specified directories.

        Returns:
        tuple: A tuple containing the training dataset, validation dataset, test dataset, and class names.
        """
        try:

            # Verify that the directories exist
            if not os.path.exists(self.train_data_dir):
                raise FileNotFoundError(f"Training data directory '{self.train_data_dir}' does not exist.")
            if not os.path.exists(self.test_data_dir):
                raise FileNotFoundError(f"Testing data directory '{self.test_data_dir}' does not exist.")

            # Create a TensorFlow dataset for training and validation
            self.train_dataset = tf.keras.utils.image_dataset_from_directory(
                directory=self.train_data_dir,
                labels="inferred",
                label_mode="categorical",
                color_mode="rgb",
                batch_size=self.batch_size,
                image_size=self.image_size,
                shuffle=True,
                seed=self.seed,
                validation_split=self.validation_split,
                subset="training"
            )

            self.val_dataset = tf.keras.utils.image_dataset_from_directory(
                directory=self.train_data_dir,
                labels="inferred",
                label_mode="categorical",
                color_mode="rgb",
                batch_size=self.batch_size,
                image_size=self.image_size,
                shuffle=True,
                seed=self.seed,
                validation_split=self.validation_split,
                subset="validation"
            )

            # Create a TensorFlow dataset for testing
            self.test_dataset = tf.keras.utils.image_dataset_from_directory(
                directory=self.test_data_dir,
                labels="inferred",
                label_mode="categorical",
                color_mode="rgb",
                batch_size=self.batch_size,
                image_size=self.image_size,
                shuffle=True,
                seed=self.seed,
            )

            # Check class names
            self.class_names = self.train_dataset.class_names
            print("Class names:", self.class_names)

            return self.train_dataset, self.val_dataset, self.test_dataset, self.class_names

        except Exception as e:
            print(f"An error occurred while loading the datasets: {e}")
            return None, None, None, None

if __name__ == "__main__":
    # Example usage
    # train_data = "../data/Training"
    # test_data = "../data/Testing"

    project_dir = os.path.dirname(os.path.abspath(__file__))  # Assuming trainer.py is in src
    data_dir = os.path.join(project_dir, "data")
    train_data = os.path.join(data_dir, "Training")
    test_data = os.path.join(data_dir, "Testing")

    batch_size = 32
    image_size = (224, 224)

    dataset_loader = DatasetLoader(train_data, test_data, batch_size, image_size)
    train_dataset, val_dataset, test_dataset, class_names = dataset_loader.load_datasets()

    # print_shapes(train_dataset)
   

    if train_dataset is not None:
        print("Datasets loaded successfully.")
    else:
        print("Failed to load datasets.")
