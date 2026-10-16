
import os
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

MODEL_PATH = "binsight_saved_model"
LABELS_PATH = "labels_new.txt"

# Load the exported model
model = tf.saved_model.load(MODEL_PATH)
predict_fn = model.signatures["serving_default"]

# Load class labels
with open(LABELS_PATH, "r") as f:
    labels = [
        line.strip().split(" ", 1)[1].lower()
        for line in f if line.strip()
    ]

print("Model loaded successfully!")
print("Categories:", labels)

# Get an image to test
image_path = input(
    "\nPaste the full path to a waste image: "
).strip().strip('"')

if not os.path.isfile(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

# Prepare image: MobileNetV2 expects 224 x 224 RGB images.
image = ImageOps.fit(
    Image.open(image_path).convert("RGB"),
    (224, 224),
    Image.Resampling.LANCZOS
)

# Keep pixel values in the 0–255 range.
arr = np.asarray(image, dtype=np.float32)
batch = tf.convert_to_tensor(arr[np.newaxis, ...])

# Run prediction
outputs = predict_fn(batch)

print("\nOutput keys:", list(outputs.keys()))

# Find the model's output tensor
scores = next(iter(outputs.values())).numpy().reshape(-1)

if len(scores) != len(labels):
    raise ValueError(
        f"Model returned {len(scores)} scores, "
        f"but there are {len(labels)} labels."
    )

best = int(np.argmax(scores))

print("\n--- PREDICTION ---")
print("Category:", labels[best])
print(f"Confidence: {scores[best] * 100:.2f}%")

print("\nAll category scores:")
for i in np.argsort(scores)[::-1]:
    print(f"{labels[i]}: {scores[i] * 100:.2f}%")
