import torch

path = "Full_Dataset_EfficientNetB0.pth"

checkpoint = torch.load(path, map_location="cpu")

print("Classifier weight:", checkpoint["classifier.weight"].shape)
print("Classifier bias:", checkpoint["classifier.bias"].shape)

print("Conv stem:", checkpoint["conv_stem.weight"].shape)
print("Conv head:", checkpoint["conv_head.weight"].shape)