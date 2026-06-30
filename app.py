"""Minimal Gradio UI for the pretrained font classifier.

Run:
    python app.py
then open the printed local URL in your browser, upload an image with text,
and get the top matching Google Fonts (with links).
"""

import albumentations as A
import csv
import gradio as gr
import numpy as np
import onnxruntime as ort
import yaml

# Reuse everything already defined for CLI inference.
from infer_pretrained import CONFIG_PATH, MODEL_PATH, MAPPING_PATH, softmax
from train import CutMax, ResizeWithPad

# --- Load model + config + mapping once at startup ---
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
input_size = config["size"]

google_font_mapping = {}
with open(MAPPING_PATH, "r") as f:
    for i, row in enumerate(csv.reader(f, delimiter="\t")):
        if i > 0:
            filename, font_name, version = row
            google_font_mapping[filename] = (font_name, version)

session = ort.InferenceSession(MODEL_PATH)

transform = A.Compose(
    [
        A.Lambda(image=CutMax(1024)),
        A.Lambda(image=ResizeWithPad((input_size, input_size))),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def font_link(filename):
    name = google_font_mapping.get(filename, (filename.replace(".ttf", ""), None))[0]
    query = name.replace(" ", "+")
    return name, f"https://fonts.google.com/?query={query}"


def predict(image):
    if image is None:
        return {}, "Upload an image with text to classify."

    image = np.asarray(image)
    if image.ndim == 2:  # grayscale -> RGB
        image = np.stack([image] * 3, axis=-1)
    image = image[:, :, :3]  # drop alpha if present

    image = transform(image=image)["image"]
    image = np.transpose(image, (2, 0, 1))[np.newaxis, ...]

    logits = session.run(None, {"input": image})[0][0]
    probs = softmax(logits)

    # ponytail: skip the 181 classes not on Google Fonts (mapped to "NONE")
    top = [i for i in probs.argsort()[::-1]
           if google_font_mapping.get(config["classnames"][i], ("",))[0] != "NONE"][:5]
    labels = {font_link(config["classnames"][i])[0]: float(probs[i]) for i in top}

    best_name, best_url = font_link(config["classnames"][top[0]])
    markdown = f"### Best match: [{best_name}]({best_url})"
    return labels, markdown


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Image with text"),
    outputs=[
        gr.Label(num_top_classes=5, label="Top font matches"),
        gr.Markdown(),
    ],
    title="Google Font Classifier",
    description="Upload an image containing text to identify the closest Google Font.",
)

if __name__ == "__main__":
    demo.launch()
