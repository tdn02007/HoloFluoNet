from PIL import Image
import os
import numpy as np

file_name = "HoloFluoNet"
data_type = ["input", "background", "live", "dead", "nuclei", "distance"]

def split_image(image_path, data_name, save_path):
    image = Image.open(image_path).convert("L")
    img_width, img_height = image.size
    
    single_width = img_width // 6
    
    # Split and save the images
    for i in range(6):
        left = i * single_width
        right = (i + 1) * single_width
        box = (left, 0, right, img_height)
        
        # Crop the image
        img_crop = image.crop(box)

        os.makedirs(f"{save_path}/{data_type[i]}", exist_ok=True)

        # Save the image
        img_crop.save(f"{save_path}/{data_type[i]}/{data_name}")

source_folder = f'../result_mask_{file_name}/model_result/fake/'
save_folder = f'../result_mask_{file_name}/split_data'

image_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

for i, img in enumerate(image_files):
    image_path = os.path.join(source_folder, img)
    split_image(image_path, img, save_folder)

