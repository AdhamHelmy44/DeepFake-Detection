import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.model_manager import ModelManager  # We can reuse the preprocessing logic


class ModelEvaluator:
    """
    Handles loading a test dataset and evaluating multiple models to compare
    their performance on metrics like F1-score, precision, and recall.
    """

    def __init__(self, models_to_test, test_data_path, image_size=(224, 224)):
        self.models = models_to_test
        self.test_data_path = test_data_path
        self.image_size = image_size
        # We need a temporary ModelManager to access its preprocessing function
        # This is a bit of a workaround; a better design might be to make
        # preprocessing a static function. For now, this works.
        self._mm = ModelManager(models_folder_path='')  # empty path
        self.test_paths, self.y_true = self._prepare_test_dataset()

    def _prepare_test_dataset(self):
        """Loads all file paths and labels from the test directory."""
        print(f"--- Loading Test Dataset from: {self.test_data_path} ---")
        test_paths, y_true = [], []

        real_path = os.path.join(self.test_data_path, 'real')
        fake_path = os.path.join(self.test_data_path, 'fake')

        # Load real images (label = 1)
        for img_file in os.listdir(real_path):
            if img_file.lower().endswith(('.png', '.jpg')):
                test_paths.append(os.path.join(real_path, img_file))
                y_true.append(1)

        # Load fake images (label = 0)
        for img_file in os.listdir(fake_path):
            if img_file.lower().endswith(('.png', '.jpg')):
                test_paths.append(os.path.join(fake_path, img_file))
                y_true.append(0)

        print(f"Found {len(test_paths)} images for evaluation.")
        return test_paths, np.array(y_true)

    def run_comparison(self):
        """
        Evaluates all models on the test set and returns a comparison DataFrame.
        """
        if not self.test_paths:
            print("Test data is empty. Cannot run comparison.")
            return None

        all_metrics = []

        for model_name, model in self.models.items():
            print(f"Evaluating model: {model_name}...")
            y_pred = []

            for img_path in tqdm(self.test_paths, desc=f"Predicting with {model_name}"):
                # Preprocess the image
                preprocessed_img = self._mm._preprocess_image(img_path, self.image_size)
                if preprocessed_img is not None:
                    # Get prediction and convert to 0 or 1
                    score = model.predict(preprocessed_img, verbose=0)[0][0]
                    y_pred.append(1 if score >= 0.5 else 0)
                else:
                    # Handle cases where image can't be loaded
                    y_pred.append(-1)  # Placeholder for error

            # Filter out errors before calculating metrics
            valid_indices = [i for i, label in enumerate(y_pred) if label != -1]
            y_true_valid = self.y_true[valid_indices]
            y_pred_valid = np.array(y_pred)[valid_indices]

            # Calculate metrics
            metrics = {
                'Model': model_name,
                'Accuracy': accuracy_score(y_true_valid, y_pred_valid),
                'Precision': precision_score(y_true_valid, y_pred_valid, zero_division=0),
                'Recall': recall_score(y_true_valid, y_pred_valid, zero_division=0),
                'F1-Score': f1_score(y_true_valid, y_pred_valid, zero_division=0)
            }
            all_metrics.append(metrics)

        # Create and return a DataFrame
        results_df = pd.DataFrame(all_metrics).sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
        return results_df