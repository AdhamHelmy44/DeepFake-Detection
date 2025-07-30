import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, RandomFlip, RandomRotation, \
    RandomZoom
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.visualization import plot_training_history, plot_confusion_matrix


class ModelTrainer:
    """Handles building, training, and evaluating the specialist model."""

    def __init__(self, config, method_name):
        self.config = config
        self.method_name = method_name
        self.model = self._build_model()
        self.history = {}

    def _build_model(self):
        """Builds the custom Xception model with data augmentation."""
        image_input = Input(shape=(*self.config.IMG_SIZE, 3), name='image_input')
        x = RandomFlip("horizontal")(image_input)
        x = RandomRotation(0.1)(x)
        x = RandomZoom(0.1)(x)

        base_model = Xception(weights='imagenet', include_top=False, input_shape=(*self.config.IMG_SIZE, 3))
        base_model.trainable = False
        x = base_model(x, training=False)

        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.5)(x)
        output = Dense(1, activation='sigmoid', dtype='float32')(x)

        return Model(inputs=image_input, outputs=output)

    def train_head(self, train_ds, val_ds):
        """Stage 1: Train only the classification head."""
        print(f"\n--- STAGE 1: Training classification head for {self.method_name} ---")
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE_HEAD),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        history_head = self.model.fit(
            train_ds,
            epochs=self.config.EPOCHS_HEAD,
            validation_data=val_ds,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)]
        )
        self.history.update(history_head.history)

    def fine_tune(self, train_ds, val_ds):
        """Stage 2: Unfreeze layers and fine-tune the model."""
        print(f"\n--- STAGE 2: Fine-tuning model for {self.method_name} ---")
        base_model_layer = self.model.get_layer('xception')
        base_model_layer.trainable = True

        # Fine-tune only the top blocks of Xception
        for layer in base_model_layer.layers[:-30]:
            layer.trainable = False

        self.model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE_FINETUNE),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )

        model_path = os.path.join(self.config.OUTPUT_DIR, f'best_model_{self.method_name}.keras')
        history_finetune = self.model.fit(
            train_ds,
            epochs=self.config.EPOCHS_HEAD + self.config.EPOCHS_FINETUNE,
            initial_epoch=self.config.EPOCHS_HEAD,
            validation_data=val_ds,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1),
                ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1)
            ]
        )
        for key in history_finetune.history:
            self.history[key].extend(history_finetune.history[key])

    def evaluate(self, test_ds):
        """Evaluates the final model on the test set."""
        print(f"\n--- EVALUATING {self.method_name} SPECIALIST MODEL ---")
        if test_ds is None:
            print("Test dataset is empty. Skipping evaluation.")
            return

        model_path = os.path.join(self.config.OUTPUT_DIR, f'best_model_{self.method_name}.keras')
        if os.path.exists(model_path):
            self.model.load_weights(model_path)

        results = self.model.evaluate(test_ds, verbose=1)
        print(f"Test Results: Loss={results[0]:.4f}, Accuracy={results[1]:.4f}, AUC={results[2]:.4f}")

        y_true = np.concatenate([y for x, y in test_ds], axis=0)
        y_pred = (self.model.predict(test_ds) > 0.5).astype(int)

        plot_confusion_matrix(y_true, y_pred, self.config.CLASS_NAMES, self.method_name, self.config.OUTPUT_DIR)
        plot_training_history(self.history, self.method_name, self.config.OUTPUT_DIR)