import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import timm

# Define data transformations for inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the trained ConvNeXt-V2 model
model = timm.create_model("convnext_v2_164", pretrained=False, num_classes=2)  # Assumes 2 classes (Cat and Dog)
model.load_state_dict(torch.load('convnext_v2_classifier.pth'))  # Load the saved model weights
model.eval()

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load and preprocess an image for inference
from PIL import Image

# Replace 'test_image.jpg' with the path to your test image
image_path = 'test_image.jpg'
input_image = Image.open(image_path).convert("RGB")
input_tensor = transform(input_image)
input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

# Use GPU if available
input_batch = input_batch.to(device)

# Perform inference
with torch.no_grad():
    output = model(input_batch)

# Get predicted class probabilities and class index
probabilities = F.softmax(output, dim=1)[0]
predicted_class = torch.argmax(probabilities).item()

# Define class labels
class_labels = ['Cat', 'Dog']

# Display the result
print(f"Predicted class: {class_labels[predicted_class]} (Probability: {probabilities[predicted_class]:.2f})")
