import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.transforms as transforms
from torchvision.data import datasets
from torch.utils.data import Dataloader 
import timm 

#Define data augmentations and loading...

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder('train_data/',transform=transform)
train_loader = Dataloader(train_data, batch_size=64, shuffle=True)

#Model load 
model = timm.create_model("convnext_v2_164", pretrained=False, num_classes=2)  # Assumes 2 classes (Cat and Dog)

#Loss Function 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


#GPU Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs,labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")


torch.save(model.state_dict(),'convnext_v2_classifier.pth')
