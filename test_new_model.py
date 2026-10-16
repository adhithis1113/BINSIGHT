import os
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

MODEL_PATH = "keras_model_new.h5"
LABELS_PATH = "labels_new.txt"

# Load the new model and its labels
model = load_model(MODEL_PATH, compile=False)

with open(LABELS_PATH, "r") as f:
    labels = [
        line.strip().split(" ", 1)[1].lower()
        for line in f
        if line.strip()
    ]

if model.output_shape[-1] != len(labels):
    raise ValueError("The model output count doesn't match the labels.")

print("New BINSIGHT model loaded successfully!")
print("Categories:", labels)

# Ask for a real image on your computer
image_path = input("\nPaste the full path to a waste image: ").strip().strip('"')

if not os.path.isfile(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

# Prepare the image; the model contains its own input rescaling
size = tuple(d or 224 for d in model.input_shape[1:3])

image = ImageOps.fit(
    Image.open(image_path).convert("RGB"),
    (size[1], size[0]),
    Image.Resampling.LANCZOS
)

arr = np.asarray(image, dtype=np.float32)
predictions = model.predict(arr[np.newaxis, ...], verbose=0)[0]

best = int(np.argmax(predictions))

print("\n--- PREDICTION ---")
print("Category:", labels[best])
print(f"Confidence: {predictions[best] * 100:.2f}%")

print("\nAll category scores:")
for i in np.argsort(predictions)[::-1]:
    print(f"{labels[i]}: {predictions[i] * 100:.2f}%")