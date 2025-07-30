import os
import tensorflow as tf
import numpy as np
import random

class Config:
    """Central configuration class for the project."""
    # --- File Paths ---
    BASE_INPUT_DIR = '/kaggle/input/faceforencispp-extracted-frames'
    BASE_WORKING_DIR = '/kaggle/working'
    OUTPUT_DIR = os.path.join(BASE_WORKING_DIR, 'output')

    # --- Model & Data Parameters ---
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    MAX_FRAMES_PER_VIDEO = 32

    # --- Training Parameters ---
    EPOCHS_HEAD = 8
    EPOCHS_FINETUNE = 12
    LEARNING_RATE_HEAD = 1e-4
    LEARNING_RATE_FINETUNE = 1e-5

    # --- Data Splitting & Reproducibility ---
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1
    RANDOM_SEED = 123

    # --- Class & Method Definitions ---
    CLASS_NAMES = ['fake', 'real']
    METHODS_TO_TRAIN = ['NeuralTextures', 'FaceSwap', 'FaceShifter', 'Face2Face', 'Deepfakes']

def setup_environment(config):
    """Initializes output directory and sets random seeds."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)
    print("Configuration and environment set up.")
    print(f"Output directory: {config.OUTPUT_DIR}")