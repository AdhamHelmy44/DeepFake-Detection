import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import xception


class DataPreparer:
    """Handles creating training and validation data generators."""

    def __init__(self, train_dir, val_dir, image_size, batch_size):
        """
        Initializes the DataPreparer with paths and settings.

        Args:
            train_dir (str): Path to the training dataset.
            val_dir (str): Path to the validation dataset.
            image_size (tuple): The (width, height) of the images.
            batch_size (int): The number of images per batch.
        """
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.image_size = image_size
        self.batch_size = batch_size

    def get_data_generators(self):
        """
        Creates and returns training and validation ImageDataGenerators.

        The generators will preprocess images using the Xception standard,
        and apply data augmentation to the training set.
        """
        print("--- Preparing Data Generators ---")

        # Apply data augmentation to the training data to improve generalization
        train_datagen = ImageDataGenerator(
            preprocessing_function=xception.preprocess_input,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # For validation data, we only apply the Xception preprocessing
        val_datagen = ImageDataGenerator(
            preprocessing_function=xception.preprocess_input
        )

        # Create the generator for training data
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary'  # For REAL vs. FAKE classification
        )

        # Create the generator for validation data
        validation_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False  # No need to shuffle validation data
        )

        print("--- Data Generators Ready ---")
        return train_generator, validation_generator