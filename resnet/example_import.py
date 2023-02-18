import torch
from model import resnet152

# Create an instance of the model
model = resnet152(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Create a random input tensor
input_tensor = torch.randn(1, 3, 224, 224)

# Use the model to classify the input tensor
output = model(input_tensor)

# Print the output tensor
print(output)
