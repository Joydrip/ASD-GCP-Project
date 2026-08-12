import torch
import timm


MODEL_PATH = "Full_Dataset_EfficientNetB0.pth"


# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Create EfficientNet-B0
model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=2
)


# Load trained weights
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint,
    strict=True
)


# Move to device
model = model.to(device)


# Evaluation mode
model.eval()


print("================================")
print("MODEL LOADED SUCCESSFULLY")
print("================================")
print("Architecture : EfficientNet-B0")
print("Classes      : 2")
print("Parameters   : 4,052,175")
print("Device       :", device)
print("================================")