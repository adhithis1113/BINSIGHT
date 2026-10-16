
import os
import sys

# Set thread limits before TensorFlow initializes.
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models

DATASET_DIR = "dataset_auto"
MODEL_PATH = "keras_model_new.h5"
LABELS_PATH = "labels_new.txt"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42


def main():
    print("STEP 1: Checking dataset...", flush=True)

    if not os.path.isdir(DATASET_DIR):
        sys.exit(f"Dataset folder not found: {DATASET_DIR}")

    print("STEP 2: Loading training images...", flush=True)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    print("STEP 3: Loading validation images...", flush=True)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    if num_classes < 2:
        sys.exit("At least two class folders are required.")

    print("Classes:", class_names, flush=True)

    # Keep loading predictable on Windows.
    train_ds = train_ds.prefetch(1)
    val_ds = val_ds.prefetch(1)

    print("STEP 4: Building model...", flush=True)

    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
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
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("STEP 5: Starting training...", flush=True)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
    )

    print("STEP 6: Saving model...", flush=True)

    model.save(MODEL_PATH, include_optimizer=False)

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        for i, name in enumerate(class_names):
            f.write(f"{i} {name}\n")

    print("STEP 7: Verifying saved model...", flush=True)

    reloaded_model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("SUCCESS: Model reload verification passed!", flush=True)
    print("Model:", MODEL_PATH, flush=True)
    print("Labels:", LABELS_PATH, flush=True)
    print("Output shape:", reloaded_model.output_shape, flush=True)


if __name__ == "__main__":
    main()
