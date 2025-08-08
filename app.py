import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU to avoid CUDA errors

import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the model (ensure this file is in the same folder)
model = load_model("best_cnn_model.h5")

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Prediction function
def predict_image(filepath):
    try:
        img = Image.open(filepath).convert("RGB").resize((32, 32))
        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)[0]
        return {class_names[i]: float(predictions[i]) for i in range(10)}
    except Exception as e:
        return f"Error: Unable to process the image. Details: {str(e)}"

# Gradio interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="filepath", label="Upload an image (JPG, PNG, IMG, etc.)"),
    outputs=gr.Label(num_top_classes=3, label="Top 3 Predictions"),
    title="CIFAR-10 Image Classifier",
    description=(
        "Upload an image of an object from the CIFAR-10 dataset. "
        "The model will predict which of the following classes it belongs to: "
        "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck."
    )
)

interface.launch()

