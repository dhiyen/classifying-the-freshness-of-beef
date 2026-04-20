import os
import time
import math
import numpy as np
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ===== CONFIG =====
IMG_SIZE = 224
BATCH_SIZE = 16
K_FOLDS = 5
EPOCHS = 12
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

# ===== LOAD DATA =====
def load_dataset(data_dir, split="train"):
    X, y = [], []
    classes = ["fresh", "rotten"]

    for label, cls in enumerate(classes):
        folder = os.path.join(data_dir, split, cls)

        for file in os.listdir(folder):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                X.append(os.path.join(folder, file))
                y.append(label)

    return np.array(X), np.array(y)

# ===== DATA GENERATOR =====
class MeatDataSequence(tf.keras.utils.Sequence):
    def __init__(self, filepaths, labels, preprocess_input, augment=False, batch_size=16):
        self.filepaths = filepaths
        self.labels = labels
        self.preprocess_input = preprocess_input
        self.augment = augment
        self.batch_size = batch_size
        self.indices = np.arange(len(self.filepaths))

        # ✅ AUGMENT NHẸ (đã fix)
        self.aug = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )

    def __len__(self):
        return math.ceil(len(self.filepaths) / self.batch_size)

    def __getitem__(self, index):
        batch_idx = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        batch_x, batch_y = [], []

        for i in batch_idx:
            img = load_img(self.filepaths[i], target_size=(IMG_SIZE, IMG_SIZE))
            img = img_to_array(img)
            batch_x.append(img)
            batch_y.append(self.labels[i])

        batch_x = np.array(batch_x, dtype="float32")
        batch_y = np.array(batch_y, dtype="float32")

        batch_x = self.preprocess_input(batch_x)

        if self.augment:
            batch_x = next(self.aug.flow(batch_x, batch_size=len(batch_x), shuffle=False))

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# ===== TRAIN K-FOLD =====
def train_kfold(build_model_fn, preprocess_input, model_name, X, y, batch_size):

    total_start = time.time()
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    accs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- FOLD {fold}/{K_FOLDS} ---")

        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        train_gen = MeatDataSequence(X_tr, y_tr, preprocess_input, True, batch_size)
        val_gen   = MeatDataSequence(X_va, y_va, preprocess_input, False, batch_size)

        model = build_model_fn()

        fold_path = os.path.join(SAVE_DIR, f"{model_name}_fold{fold}.keras")

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2),
            ModelCheckpoint(fold_path, monitor='val_accuracy', save_best_only=True)
        ]

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )

        _, acc = model.evaluate(val_gen, verbose=0)
        accs.append(acc)

        tf.keras.backend.clear_session()

    return np.mean(accs)