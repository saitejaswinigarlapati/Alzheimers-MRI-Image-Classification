import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ======================================================
# Device
# ======================================================
device = torch.device("cpu")

# ======================================================
# Read preprocessing configuration
# ======================================================
with open("preprocessing.json", "r") as f:
    preprocessing = json.load(f)

image_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

for step in preprocessing:
    if step["name"] == "Resize":
        image_size = step["size"][0]

    elif step["name"] == "Normalize":
        mean = step["mean"]
        std = step["std"]

# ======================================================
# Class Labels
# ======================================================
classes = [
    "Mild Dementia",
    "Moderate Dementia",
    "Non Demented",
    "Very Mild Dementia"
]

# ======================================================
# Image Transform
# ======================================================
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# ======================================================
# Model
# ======================================================
model = models.mobilenet_v2(weights=None)

model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.last_channel, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, len(classes))
)

# ======================================================
# Load trained weights
# ======================================================
checkpoint = torch.load(
    "best_mobilenetv2_alzheimer.pth",
    map_location=device
)

# If checkpoint contains only state_dict
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

# Remove "module." prefix if model was trained using DataParallel
new_checkpoint = {}
for k, v in checkpoint.items():
    if k.startswith("module."):
        new_checkpoint[k[7:]] = v
    else:
        new_checkpoint[k] = v

model.load_state_dict(new_checkpoint)

model.to(device)
model.eval()

# ======================================================
# Prediction Function
# ======================================================
def predict(image: Image.Image):

    image = image.convert("RGB")

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probabilities = F.softmax(outputs, dim=1)[0]

    scores = {
        classes[i]: float(probabilities[i])
        for i in range(len(classes))
    }

    prediction = classes[torch.argmax(probabilities).item()]

    return scores, prediction