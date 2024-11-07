# Open Source Model Licensed under the Apache License Version 2.0 and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved. 
# The below software and/or models in this distribution may have been 
# modified by THL A29 Limited ("Tencent Modifications"). 
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT 
# except for the third-party components listed below. 
# Hunyuan 3D does not impose any additional limitations beyond what is outlined 
# in the repsective licenses of these third-party components. 
# Users must comply with all terms and conditions of original licenses of these third-party 
# components and must ensure that the usage of the third party components adheres to 
# all relevant laws and regulations. 

# For avoidance of doubts, Hunyuan 3D means the large language models and 
# their software and algorithms, including trained model weights, parameters (including 
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code, 
# fine-tuning enabling code and other elements of the foregoing made publicly available 
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.
import os
import logging
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Constants
MODEL_PATH = 'model.pth'
IMAGE_FOLDER = 'images/'
OUTPUT_FOLDER = 'output/'

# Logging Setup
logging.basicConfig(level=logging.INFO)

# Device Setup (CUDA or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Architecture
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Load Pretrained Model
def load_model(model_path=MODEL_PATH):
    model = CustomModel().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise e
    return model
# Image Preprocessing (Resize, Normalize)
def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0).to(device)  # Add batch dimension
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        raise e

# Model Prediction
def predict(model, image_tensor):
    try:
        with torch.no_grad():
            output = model(image_tensor)
            return output
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        raise e
# Image Preprocessing (Resize, Normalize)
def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0).to(device)  # Add batch dimension
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        raise e

# Model Prediction
def predict(model, image_tensor):
    try:
        with torch.no_grad():
            output = model(image_tensor)
            return output
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        raise e
# Saving Output Results
def save_output(output, output_folder=OUTPUT_FOLDER):
    try:
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, 'prediction_results.txt')
        with open(output_file, 'a') as file:
            file.write(f"Prediction: {output}\n")
        logging.info("Output saved successfully.")
    except Exception as e:
        logging.error(f"Error saving output: {e}")
        raise e

# Process a Batch of Images from Folder
def process_images(image_folder=IMAGE_FOLDER):
    try:
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
        if not image_files:
            logging.warning("No images found in the folder.")
            return

        model = load_model()  # Load model once for all images
        for image_file in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(image_folder, image_file)
            try:
                image_tensor = preprocess_image(image_path)
                output = predict(model, image_tensor)
                save_output(output)
            except Exception as e:
                logging.error(f"Error processing image {image_file}: {e}")
    except Exception as e:
        logging.error(f"Error processing images: {e}")
        raise e
# Main Execution Function
def main():
    try:
        logging.info("Starting the image processing pipeline...")
        process_images()
        logging.info("Image processing completed successfully.")
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        raise e

# Run the program
if __name__ == "__main__":
    main()

