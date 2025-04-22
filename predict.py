import torch
from model import MedicalNet
from torchvision import transforms
from PIL import Image

def predict(image_path):
    model = MedicalNet()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    classes = ['Normal', 'Pneumonia']
    print(f"Prediction: {classes[predicted.item()]}")
