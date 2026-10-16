
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------- SETTINGS ----------------
DATASET_DIR = "dataset_auto"
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42


def main():
    if not os.path.isdir(DATASET_DIR):
        sys.exit(
            f"Dataset folder '{DATASET_DIR}' not found. "
            "Run prepare_dataset.py first."
        )

    print("Loading training images...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    if num_classes < 2:
        sys.exit("At least two class folders are required for training.")

    print("\nClasses being trained:")
    for i, name in enumerate(class_names):
        print(f"{i}: {name}")

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # ---------------- BUILD MODEL ----------------
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),

        # The app detects this layer and scales images by 1/255.
        layers.Rescaling(1.0 / 255.0),

        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    print("\nStarting training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # ---------------- SAVE MODEL ----------------
    model.save(MODEL_PATH)

    # Keep labels in exactly the same order as the model outputs.
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        for i, name in enumerate(class_names):
            f.write(f"{i} {name}\n")

    print("\nTraining completed successfully!")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Labels saved: {LABELS_PATH}")
    print(f"Model output classes: {num_classes}")
    print(f"Labels written: {len(class_names)}")


if __name__ == "__main__":
    main()
