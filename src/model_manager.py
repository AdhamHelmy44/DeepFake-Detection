import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import xception


class ModelManager:
    """Manages loading models and running frame predictions."""

    def __init__(self, models_folder_path):
        self.models = self._load_all_models(models_folder_path)

    def _load_all_models(self, folder_path):
        """Scans a folder, finds all .keras files, and loads them."""
        loaded_models = {}
        print("--- Loading All Specialist Models ---")
        if not os.path.isdir(folder_path):
            print(f"Error: Models folder not found at {folder_path}")
            return None
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith((".keras", ".h5")):
                model_name = os.path.splitext(filename)[0]
                full_path = os.path.join(folder_path, filename)
                try:
                    print(f"Loading '{model_name}'...")
                    loaded_models[model_name] = tf.keras.models.load_model(full_path, compile=False)
                except Exception as e:
                    print(f"Error loading model {filename}: {e}")
        print(f"--- Successfully loaded {len(loaded_models)} models ---\n")
        return loaded_models

    def _preprocess_image(self, image_path, target_size):
        """Preprocesses a single image for an Xception model."""
        try:
            img = Image.open(image_path).convert('RGB').resize(target_size)
            img_array = np.array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            return xception.preprocess_input(img_batch)
        except Exception:
            return None

    def get_frame_verdict(self, image_path, target_size):
        """Gets the ensemble verdict for a single frame."""
        preprocessed_image = self._preprocess_image(image_path, target_size)
        if preprocessed_image is None:
            return "ERROR"

        # Early exit if any model votes "FAKE"
        for model in self.models.values():
            score = model.predict(preprocessed_image, verbose=0)[0][0]
            if score < 0.5:
                return "FAKE"
        return "REAL"