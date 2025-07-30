import os
import tensorflow as tf
import random


@tf.function
def _process_path(img_path, label):
    """Processes an image path into a tensor for the model."""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))  # Assuming IMG_SIZE is (224, 224)
    img = tf.keras.applications.xception.preprocess_input(img)
    return img, label


class DatasetProvider:
    """Handles data sourcing, splitting, and tf.data.Dataset creation."""

    def __init__(self, config, method):
        self.config = config
        self.method = method
        self.train_ds, self.val_ds, self.test_ds = self._prepare_datasets()

    def _prepare_datasets(self):
        """Main method to create and return all datasets."""
        print(f"\n{'=' * 40}\n PREPARING DATA FOR: {self.method}\n{'=' * 40}")

        # 1. Find real/fake video pairs
        source_pairs = self._find_source_pairs()
        if not source_pairs:
            print(f"No source video pairs found for method {self.method}. Skipping.")
            return None, None, None

        random.shuffle(source_pairs)

        # 2. Split pairs to prevent data leakage
        n = len(source_pairs)
        n_train = int(n * (1 - self.config.VAL_RATIO - self.config.TEST_RATIO))
        n_val = int(n * self.config.VAL_RATIO)

        train_pairs = source_pairs[:n_train]
        val_pairs = source_pairs[n_train: n_train + n_val]
        test_pairs = source_pairs[n_train + n_val:]
        print(
            f"Found {n} video pairs. Split into {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test pairs.")

        # 3. Create tf.data.Dataset objects
        train_ds = self._create_dataset_from_pairs(train_pairs, is_training=True)
        val_ds = self._create_dataset_from_pairs(val_pairs, is_training=False)
        test_ds = self._create_dataset_from_pairs(test_pairs, is_training=False)

        if train_ds is None or val_ds is None:
            print(f"Dataset creation failed for {self.method} (not enough samples).")
            return None, None, None

        return train_ds, val_ds, test_ds

    def _find_source_pairs(self):
        """Finds pairs of (real_video_path, fake_video_path)."""
        source_pairs = []
        real_dir = os.path.join(self.config.BASE_INPUT_DIR, 'real')
        fake_dir = os.path.join(self.config.BASE_INPUT_DIR, 'fake', self.method)
        if not os.path.isdir(fake_dir): return []

        for fake_folder in os.listdir(fake_dir):
            if not os.path.isdir(os.path.join(fake_dir, fake_folder)): continue
            source_id = fake_folder.split('_')[0]
            real_path = os.path.join(real_dir, source_id)
            if os.path.isdir(real_path):
                source_pairs.append({
                    'real_path': real_path,
                    'fake_path': os.path.join(fake_dir, fake_folder)
                })
        return source_pairs

    def _create_dataset_from_pairs(self, pairs, is_training):
        """Builds a tf.data.Dataset from a list of video pairs."""
        frame_paths, labels = [], []
        for pair in pairs:
            real_frames = sorted([f for f in os.listdir(pair['real_path']) if f.lower().endswith('.png')])[
                          :self.config.MAX_FRAMES_PER_VIDEO]
            fake_frames = sorted([f for f in os.listdir(pair['fake_path']) if f.lower().endswith('.png')])[
                          :self.config.MAX_FRAMES_PER_VIDEO]

            for frame in real_frames: frame_paths.append(os.path.join(pair['real_path'], frame)); labels.append(1)
            for frame in fake_frames: frame_paths.append(os.path.join(pair['fake_path'], frame)); labels.append(0)

        if not frame_paths or len(set(labels)) < 2: return None

        dataset = tf.data.Dataset.from_tensor_slices((frame_paths, labels))
        if is_training:
            dataset = dataset.shuffle(buffer_size=len(frame_paths), reshuffle_each_iteration=True)

        dataset = dataset.map(_process_path, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.config.BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset