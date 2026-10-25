from pathlib import Path

import gradio as gr
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = Path(__file__).resolve().parent.parent / "wood_model_20k.keras"
THRESHOLD = 0.4

model = keras.models.load_model(MODEL_PATH)
IMG_H, IMG_W = model.input_shape[1:3]


def classify(path):
    if path is None:
        return "Please upload an image."
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_H, IMG_W))
    p = float(model.predict(img[None], verbose=0)[0][0])
    label = "defect" if p > THRESHOLD else "good"
    return f"{label} (score {p:.3f}, threshold {THRESHOLD})"


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="filepath", label="Wood board photo"),
    outputs=gr.Textbox(label="Result"),
    title="Wood Defect Detection",
    description="Upload a photo of a wood board to check whether it is good or defective.",
)

if __name__ == "__main__":
    demo.launch()