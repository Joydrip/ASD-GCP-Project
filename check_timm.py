import torch
import timm

CHECKPOINT_PATH = "Full_Dataset_EfficientNetB0.pth"

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=2
)

model_state = model.state_dict()

print("========================================")
print("CHECKING TENSOR SHAPES")
print("========================================")

mismatches = []

for key in checkpoint.keys():

    checkpoint_shape = checkpoint[key].shape
    model_shape = model_state[key].shape

    if checkpoint_shape != model_shape:

        mismatches.append(key)

        print()
        print("MISMATCH:")
        print("Key:", key)
        print("Checkpoint:", checkpoint_shape)
        print("TIMM model :", model_shape)

print()
print("========================================")
print("RESULT")
print("========================================")

print("Total checkpoint keys:", len(checkpoint))
print("Total model keys:", len(model_state))
print("Shape mismatches:", len(mismatches))

print("========================================")