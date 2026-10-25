import sys
from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = "wood_model_20k.keras"
THRESHOLD = 0.4

model = keras.models.load_model(MODEL_PATH)
IMG_H, IMG_W = model.input_shape[1:3]

def predict(path):
    img = tf.io.read_file(str(path))
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_H, IMG_W))
    p = float(model.predict(img[None], verbose=0)[0][0])
    return ("defect" if p > THRESHOLD else "good"), p

folder = Path(sys.argv[1])
rows = []
for f in sorted(folder.iterdir()):
    if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
        label, p = predict(f)
        rows.append({"file": f.name, "prediction": label, "score": round(p, 3)})
        print(f.name, label, round(p, 3))

pd.DataFrame(rows).to_csv("predictions.csv", index=False)