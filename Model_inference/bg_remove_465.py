# from huggingface_hub import login
# login(new_session=False)

from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import datetime

# Initialize model with dynamic device
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                      else "cpu")

model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
model.to(device)
model.eval()

# Data settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def remove_background(input_image_path):
    """
    Remove background from an image using RMBG-2.0 model.
    
    Args:
        input_image_path (str): Path to the input image
    
    Returns:
        PIL.Image: Image object with background removeddevice
    """
    # Load and preprocess image
    image = Image.open(input_image_path)
    input_images = transform_image(image).unsqueeze(0).to(device)
    
    # Prediction
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    
    return image


# Example usage
if __name__ == "__main__":
    input_image_path = "/Users/alimran/Desktop/CSE465/Split_Dataset/test/Tomato_Healthy/IMG_0001.jpg"  # Update with actual test image
    result_image = remove_background(input_image_path)
    print(f"Background removed successfully")
    
    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(result_image)
    plt.axis('off')
    plt.title("Background Removed - Plant Leaf")
    plt.tight_layout()
    plt.show()