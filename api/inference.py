from PIL import Image
import matplotlib.pyplot as plt
# import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
import numpy as np




class Earlyfusion(nn.Module):
    def __init__(self):
        super(Earlyfusion, self).__init__()
        
        in_channels = 4  # 3 (RGB) + 1 (IR) = 4 channels

        # Initial Conv Layer
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        
        # First Separable Conv Block
        self.depthwise1 = nn.Conv2d(8, 8, kernel_size=3, groups=8, padding=1)
        self.pointwise1 = nn.Conv2d(8, 8, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(8)
        
        # Second Separable Conv Block
        self.depthwise2 = nn.Conv2d(8, 8, kernel_size=3, groups=8, padding=1)
        self.pointwise2 = nn.Conv2d(8, 8, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(8)
        
        # MaxPooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Skip Connection Layer
        self.skip_conv = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        
        # Third Separable Conv Block
        self.depthwise3 = nn.Conv2d(8, 8, kernel_size=3, groups=8, padding=1)
        self.pointwise3 = nn.Conv2d(8, 8, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(8)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Fully Connected Layer (Output 2 logits for Fire & Smoke)
        self.fc = nn.Linear(8, 2)  # 2 independent outputs: [fire, smoke]

        # Store normalized values
        self.bn_outputs = {}


    def forward(self, rgbir):
        """
        Forward pass for multi-input (RGB + IR images)
        :param rgb: RGB input image (batch_size, 3, H, W)
        :param ir: Infrared (IR) input image (batch_size, 1, H, W)
        """
        # Concatenated RGB and IR images along the channel dimension
        # print(f"Concatenated RGB and IR image Input Shape: {rgbir.shape}")  # Debugging

        # Initial Conv
        x = F.relu(self.bn1(self.conv1(rgbir)))
        # print(f"After Conv1: {x.shape}")  # Debugging
        skip = x
        
        # First Separable Conv Block
        x = self.depthwise1(x)
        x = self.pointwise1(x)
        x = F.relu(self.bn2(x))
        # print(f"After Separable Conv Block 1: {x.shape}")  # Debugging
        
        # Second Separable Conv Block
        x = self.depthwise2(x)
        x = self.pointwise2(x)
        x = F.relu(self.bn3(x))
        # print(f"After Separable Conv Block 2: {x.shape}")  # Debugging
        
        # MaxPooling
        x = self.maxpool(x)
        # print(f"After MaxPooling: {x.shape}")  # Debugging
        
        # Skip Connection
        skip = self.skip_conv(skip)  # Apply convolution
        skip = self.maxpool(skip)    # Downsample to match x's dimensions
        x = x + skip  # Element-wise addition
        # print(f"After Skip Connection: {x.shape}")  # Debugging
        
        # Third Separable Conv Block
        x = self.depthwise3(x)
        x = self.pointwise3(x)
        x = F.relu(self.bn4(x))
        # print(f"After Separable Conv Block 3: {x.shape}")  # Debugging
        
        # Global Pooling
        x = self.global_pool(x)
        # print(f"After Global Pooling: {x.shape}")  # Debugging
        x = torch.flatten(x, 1)
        # print(f"After Flattening: {x.shape}")  # Debugging
        
        # Dropout
        x = self.dropout(x)
        # print(f"After Dropout: {x.shape}")  # Debugging

        # Fully Connected Layer
        x = self.fc(x)
        # print(f"Final Output Shape: {x.shape}")  # Debugging

        # No Softmax, since we use BCEWithLogitsLoss (it applies sigmoid internally)
        return x

    

# Define transformations for the images
rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

thermal_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Assuming single-channel normalization
])

# Initialize the model architecture
model = Earlyfusion()  # Ensure this matches your model's class definition

# Load the trained model weights
model_path = '//kaggle/input/fire_cnn/pytorch/default/1/best_wildfire_model.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Specify the paths to the RGB and thermal images
rgb_image_path = '/kaggle/input/flame2-test/rgb_1000.jpg'
thermal_image_path = '/kaggle/input/flame2-test/thermal_1000.jpg'

# Load and preprocess the images
rgb_image = Image.open(rgb_image_path).convert('RGB')
thermal_image = Image.open(thermal_image_path).convert('L')  # Convert to grayscale if not already

rgb_image = rgb_transform(rgb_image).unsqueeze(0)  # Add batch dimension
thermal_image = thermal_transform(thermal_image).unsqueeze(0)  # Add batch dimension

# Combine the images along the channel dimension
input_tensor = torch.cat((rgb_image, thermal_image), dim=1)

# Perform inference
with torch.no_grad():
    output = model(input_tensor)
    probabilities = (torch.sigmoid(output).squeeze(0)>0.5).int()  # Remove batch dimension
print(probabilities)
# Interpret the results
fire_present = probabilities[0]
smoke_present = probabilities[1]
print(probabilities)


# Display the results
print(f"Fire detected: {'Yes' if fire_present else 'No'} ")
print(f"Smoke detected: {'Yes' if smoke_present else 'No'} ")
