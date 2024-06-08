from imports import*


# Define paths to the training and testing data directories
train_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data','Training')
test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'Testing')

# Check if the data directories exist
if not os.path.exists(train_data_dir):
    raise FileNotFoundError(f"Training data directory '{train_data_dir}' does not exist.")
if not os.path.exists(test_data_dir):
    raise FileNotFoundError(f"Testing data directory '{test_data_dir}' does not exist.")

class Trainer:
    def __init__(self, train_data_dir, test_data_dir, num_classes, num_epochs=10):
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_epochs = num_epochs

    def train_model(self, model_save_path):
        try:
            # Load datasets
            dataset_loader = DatasetLoader(self.train_data_dir, self.test_data_dir, self.batch_size, self.image_size)
            train_dataset, val_dataset, test_dataset, class_names = dataset_loader.load_datasets()

            if train_dataset is not None and val_dataset is not None and test_dataset is not None:
                print("Datasets loaded successfully.")
                
                # Apply data augmentation
                augmenter = DataAugmentation()
                train_dataset_augmented, val_dataset_augmented, test_dataset_augmented = augmenter.apply(train_dataset, val_dataset, test_dataset)
                
                if train_dataset_augmented is not None and val_dataset_augmented is not None and test_dataset_augmented is not None:
                    print("Datasets augmented successfully.")
                    
                    # Optimize datasets
                    optimizer = DatasetOptimizer()
                    train_dataset_optimized, val_dataset_optimized, test_dataset_optimized = optimizer.optimize(train_dataset_augmented, val_dataset_augmented, test_dataset_augmented)
                    
                    if train_dataset_optimized is not None and val_dataset_optimized is not None and test_dataset_optimized is not None:
                        print("Datasets optimized successfully.")

                        # Build the model
                        model = BaseModel(num_classes=self.num_classes)

                        # Compile the model
                        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                        # Define callbacks
                        callbacks = [
                            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                            ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max')
                        ]

                        # Train the model
                        history = model.fit(
                            train_dataset_optimized,
                            validation_data=val_dataset_optimized,
                            epochs=self.num_epochs,
                            callbacks=callbacks
                        )

                        # Save the trained model
                        model.save(model_save_path)
                        print(f"Model saved at {model_save_path}")

                    else:
                        print("Failed to optimize datasets.")
                else:
                    print("Failed to augment datasets.")
            else:
                print("Failed to load datasets.")

        except Exception as e:
            print(f"An error occurred during training: {e}")

if __name__ == "__main__":
    # Example usage
    train_data_dir = "../data/Training"
    test_data_dir = "../data/Testing"
    batch_size = 32
    image_size = (224, 224)
    num_epochs = 10
    num_classes = 4
    model_save_path = "../models/base_model_model.keras"

    # Instantiate the CustomVGG16Trainer
    trainer = Trainer(train_data_dir, test_data_dir, num_classes, num_epochs)

    # Train the model
    trainer.train_model(model_save_path)
