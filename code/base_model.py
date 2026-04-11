import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 20
K_FOLDS = 5

UNFREEZE_PERCENT = 0.25
LR_PHASE2 = 1e-5

KFOLD_DIR = r'D:/Do_An/Data_Final/kfold_splits'
SAVE_DIR = 'saved_models'
os.makedirs(SAVE_DIR, exist_ok=True)

tf.keras.utils.set_random_seed(123)

def get_datagen(preprocess_input):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.25,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='reflect'
    )

    val_test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    return train_datagen, val_test_datagen

# UNFREEZE
def unfreeze_model(base_model, percent=UNFREEZE_PERCENT):
    total_layers = len(base_model.layers)
    unfreeze_from = int(total_layers * (1 - percent))

    for i, layer in enumerate(base_model.layers):
        layer.trainable = i >= unfreeze_from

    # Freeze BatchNorm (quan trọng)
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

# TRAIN ONE FOLD
def train_one_fold(fold, build_model_fn, preprocess_input, model_name):

    print(f"\nTRAIN FOLD {fold}/{K_FOLDS}")

    train_datagen, val_test_datagen = get_datagen(preprocess_input)

    fold_dir = os.path.join(KFOLD_DIR, f'fold_{fold}')

    train_gen = train_datagen.flow_from_directory(
        os.path.join(fold_dir, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(fold_dir, 'val'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(fold_dir, 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    model, base_model = build_model_fn()

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(SAVE_DIR, f'{model_name}_Fold_{fold}_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]

    # PHASE 1
    print("\nPHASE 1: Train Head")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    h1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE1,
        callbacks=callbacks,
        verbose=1
    )

    # PHASE 2
    print("\nPHASE 2: Fine-tuning")

    unfreeze_model(base_model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_PHASE2),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    h2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE2,
        callbacks=callbacks,
        verbose=1
    )

    # EVALUATE
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)

    print(f"Fold {fold} - Test Accuracy: {test_acc:.4f} | Loss: {test_loss:.4f}")

    total_epochs = EPOCHS_PHASE1 + EPOCHS_PHASE2
    augmented_size = train_gen.samples * total_epochs

    print("\nDATASET AFTER AUGMENT")
    print(f"Original train images: {train_gen.samples}")
    print(f"Epochs: {total_epochs}")
    print(f"Total augmented samples seen by model: {augmented_size}")

    # HISTORY 
    history = {
        'loss': h1.history['loss'] + h2.history['loss'],
        'val_loss': h1.history['val_loss'] + h2.history['val_loss'],
        'accuracy': h1.history['accuracy'] + h2.history['accuracy'],
        'val_accuracy': h1.history['val_accuracy'] + h2.history['val_accuracy']
    }

    final_val_loss = history['val_loss'][-1]

    return test_acc, history, test_gen, final_val_loss

# TRAIN K-FOLD
def train_kfold(build_model_fn, preprocess_input, model_name):

    start_time = time.time()

    accs, val_losses = [], []
    histories, test_gens = [], []

    for fold in range(1, K_FOLDS + 1):

        fold_start = time.time()

        acc, hist, test_gen, val_loss = train_one_fold(
            fold, build_model_fn, preprocess_input, model_name
        )

        accs.append(acc)
        val_losses.append(val_loss)
        histories.append(hist)
        test_gens.append(test_gen)

        print(f"Fold {fold} time: {time.time() - fold_start:.2f}s")

    # CHỌN BEST 
    max_acc = max(accs)
    candidates = [i for i, acc in enumerate(accs) if acc == max_acc]
    best_idx = min(candidates, key=lambda i: val_losses[i])

    best_fold = best_idx + 1

    # TIME
    total_time = time.time() - start_time
    h, rem = divmod(total_time, 3600)
    m, s = divmod(rem, 60)

    print("\nKẾT QUẢ")
    print(f"Mean Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Best Fold: {best_fold} (Acc = {accs[best_idx]:.4f}, Val_loss = {val_losses[best_idx]:.4f})")
    print(f"Total time: {int(h)}h {int(m)}m {int(s)}s")

    return {
        "accs": accs,
        "val_losses": val_losses,
        "histories": histories,
        "test_gens": test_gens,
        "best_idx": best_idx
    }