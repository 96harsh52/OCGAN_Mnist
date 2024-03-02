import torch
from torchvision import transforms
from PIL import Image
import os

from option1 import Options
# from lib.data import dataloader
from ocgan.networks import Classfier, weights_init

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load options and data loader
opt = Options().parse()
# dataloader = create_dataloader(opt)

# Load the pre-trained classifier
Main = r'D:\Pycharm\OCGAN\output\ocgan\mnist-0\train\weights'
Classfier_model = os.path.join(Main, 'netc.pth')
pretrained_dict = torch.load(Classfier_model)['state_dict']

try:
    netc = Classfier(opt).to(device)
    netc.apply(weights_init)    
    netc.load_state_dict(pretrained_dict)
    netc.eval()
    print('Classifier weights loaded successfully.')
except IOError:
    raise IOError("Classifier weights not found")

# Define image path
image_path = 'D:\Pycharm\OCGAN\images.jpg'

# Load and preprocess the image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

image = Image.open(image_path)
input_image = transform(image).unsqueeze(0).to(device)

# Use the classifier for prediction
with torch.no_grad():
    output = netc(input_image)
    print("Classification Output:", output)

# Define threshold and make predictions
threshold = 0.5
predicted_class = 'Real' if output.item() > threshold else 'Spoof'

# Print additional information for diagnosis
print("Threshold:", threshold)
print("Predicted Probability:", output.item())
print("Predicted Class:", predicted_class)
