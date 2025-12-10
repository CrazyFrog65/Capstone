import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Configure device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['fake', 'real']

# Define FusionModel matching the notebook
class FusionModel(nn.Module):
    def __init__(self, modelA, modelB, num_classes=2):
        super(FusionModel, self).__init__()
        self.modelA = nn.Sequential(*list(modelA.children())[:-1])
        self.modelB = modelB.features
        self.classifier = nn.Sequential(
            nn.Linear(512 + 1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        outA = torch.flatten(self.modelA(x), 1)
        outB = torch.nn.functional.adaptive_avg_pool2d(self.modelB(x), 1).flatten(1)
        combined = torch.cat((outA, outB), dim=1)
        return self.classifier(combined)

# Preprocess transform (same as notebook)
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def _resolve_path(path):
    # Check path relative to current dir and workspace root
    if os.path.isabs(path):
        return path
    candidate_1 = os.path.join(os.getcwd(), path)
    candidate_2 = os.path.join(os.path.dirname(__file__), '..', path)
    if os.path.exists(candidate_1):
        return candidate_1
    if os.path.exists(candidate_2):
        return os.path.normpath(candidate_2)
    return path


def load_model(model_path_fusion='best_model_fusion2.pth',
               model_path_resnet='best_model.pth'):
    """Try to load the Fusion model first. If not present, fallback to ResNet single model.
    Returns: model, class_names
    """
    # Try to instantiate fusion model
    try:
        resnet = models.resnet18(weights=None)
        num_features_res = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features_res, 2)

        effnet = models.efficientnet_b0(weights='IMAGENET1K_V1')

        fusion_model = FusionModel(resnet, effnet, num_classes=2)
        fusion_model.to(DEVICE)

        fusion_path = _resolve_path(model_path_fusion)
        if os.path.exists(fusion_path):
            print(f"Loading fusion model state from {fusion_path}")
            fusion_model.load_state_dict(torch.load(fusion_path, map_location=DEVICE))
            fusion_model.eval()
            return fusion_model, CLASS_NAMES
        else:
            print("Fusion model weights not found; falling back to ResNet model if available.")
    except Exception as e:
        print("Error while preparing FusionModel: ", e)

    # Fallback: ResNet only
    try:
        print("Loading ResNet-only model...")
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
        model.to(DEVICE)

        resnet_path = _resolve_path(model_path_resnet)
        if os.path.exists(resnet_path):
            model.load_state_dict(torch.load(resnet_path, map_location=DEVICE))
            model.eval()
            return model, CLASS_NAMES
        else:
            print("ResNet model weights not found. Returning uninitialized ResNet model (random weights).")
            model.eval()
            return model, CLASS_NAMES
    except Exception as e:
        raise RuntimeError("Failed to load any supported model. Please ensure the weight files are present: " + str(e))


def predict_image(model, image: Image.Image, class_names=CLASS_NAMES):
    """Predict given a PIL Image using provided model. Returns JSON-like dict."""
    # Preprocess
    input_tensor = INFERENCE_TRANSFORM(image)
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_batch)
        # If outputs have shape [1,2]
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_idx = torch.max(probs, 0)

    predicted = class_names[top_idx.item()]
    confidence = top_prob.item() * 100

    # Build result
    probs_cpu = probs.cpu().numpy().tolist()
    result = {
        'class': predicted,
        'confidence': round(confidence, 4),
        'probabilities': {class_names[i]: round(float(probs_cpu[i]) * 100, 4) for i in range(len(class_names))}
    }
    return result
