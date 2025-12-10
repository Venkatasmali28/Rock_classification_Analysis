import io
from PIL import Image
from torchvision import transforms

def load_image_from_bytes(img_bytes):
    return Image.open(io.BytesIO(img_bytes)).convert('RGB')

def preprocess_input(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(image)
