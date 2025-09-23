# pipeline.py
import torch
from torchvision import transforms
from PIL import Image
import timm
from collections import OrderedDict

# 1. Load your trained model (EfficientNet-B1 binary classifier)
class DFUClassifier(torch.nn.Module):
    def __init__(self, model_name="efficientnet_b1", num_classes=1):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = DFUClassifier()

# Load checkpoint with prefix fix
def fix_state_dict(state_dict, add_prefix="model."):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # If keys already have prefix, leave them
        if add_prefix and not k.startswith(add_prefix):
            new_key = add_prefix + k
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

checkpoint = torch.load("dfu_classifier.pth", map_location=device)
# Fix keys if needed
if not list(checkpoint.keys())[0].startswith("model."):
    checkpoint = fix_state_dict(checkpoint, add_prefix="model.")

model.load_state_dict(checkpoint, strict=False)
model.to(device)
model.eval()

# 2. Preprocessing for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 3. Prediction function
def classify_image(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
    prediction = "Abnormal (DFU Detected)" if prob > 0.5 else "Normal"
    return prediction, prob

# 4. Placeholder for Grad-CAM / segmentation
def explain_image(image: Image.Image):
    # TODO: integrate Grad-CAM / segmentation
    return image

# 5. Placeholder for Groq LLM report
def generate_report(prediction: str, prob: float):
    # TODO: Call Groq API
    if "Abnormal" in prediction:
        return f"The uploaded image shows signs of a diabetic foot ulcer with confidence {prob:.2f}."
    else:
        return f"No signs of DFU detected. Confidence {prob:.2f}."
