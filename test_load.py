import torch
import timm


print("Loading checkpoint...")

checkpoint = torch.load(
    "Full_Dataset_EfficientNetB0.pth",
    map_location="cpu"
)

print("Checkpoint loaded.")


print("Creating model...")

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=2
)

print("Model created.")


print("Loading weights...")

result = model.load_state_dict(
    checkpoint,
    strict=True
)

print("Weights loaded successfully!")

print()
print("Result:")
print(result)

print()
print("SUCCESS")