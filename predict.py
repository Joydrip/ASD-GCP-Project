import sys

import torch
from torchvision import transforms
from PIL import Image

from model import model, device


# IMPORTANT:
# We will verify this mapping from your original
# training dataset before treating it as final.
class_names = [
    "ASD",
    "Non-ASD"
]


# Same preprocessing used by your TEST dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Check command-line argument
if len(sys.argv) < 2:

    print("Usage:")
    print("python predict.py image.jpg")

    sys.exit()


image_path = sys.argv[1]


# Open image
image = Image.open(image_path).convert("RGB")


# Apply preprocessing
image_tensor = transform(image)


# Add batch dimension
image_tensor = image_tensor.unsqueeze(0)


# Move to same device as model
image_tensor = image_tensor.to(device)


# Prediction
with torch.no_grad():

    outputs = model(image_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, predicted_class = torch.max(
        probabilities,
        dim=1
    )


# Get result
prediction = class_names[
    predicted_class.item()
]

confidence_percentage = (
    confidence.item() * 100
)


print()
print("==============================")
print("       ASD DETECTION")
print("==============================")

print("Image:", image_path)

print("Prediction:", prediction)

print(
    "Confidence:",
    f"{confidence_percentage:.2f}%"
)

print("==============================")