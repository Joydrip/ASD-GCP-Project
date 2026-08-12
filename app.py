from flask import Flask, render_template, request
import os

import torch
from torchvision import transforms
from PIL import Image

from model import model, device


app = Flask(__name__)


# Upload folder
UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Class names
class_names = [
    "ASD",
    "Non-ASD"
]


# Same preprocessing used during testing
transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return "No image uploaded"


    file = request.files["image"]


    if file.filename == "":

        return "No image selected"


    # Save uploaded image
    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)


    # Open image
    image = Image.open(
        filepath
    ).convert("RGB")


    # Preprocess
    image_tensor = transform(
        image
    )


    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(
        device
    )


    # Prediction
    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )


    prediction = class_names[
        predicted_class.item()
    ]

    confidence_percentage = (
        confidence.item() * 100
    )


    return render_template(
        "index.html",
        prediction=prediction,
        confidence=f"{confidence_percentage:.2f}%",
        filename=file.filename
    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )