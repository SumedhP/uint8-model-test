import os
import shutil
import glob
from tqdm import tqdm

def move_images(src_dir, dest_dir, image_extensions=('*.jpg', '*.png', '*.jpeg', '*.gif', '*.bmp')):
    # Ensure destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Loop through each image extension
    for ext in image_extensions:
        # Recursively find all images with the extension
        image_paths = glob.glob(os.path.join(src_dir, '**', ext), recursive=True)
        
        for img_path in tqdm(image_paths):
            # Move image to destination directory
            shutil.move(img_path, os.path.join(dest_dir, os.path.basename(img_path)))

# Usage
src_directory = 'images 2'
dest_directory = 'image_store_raw'

move_images(src_directory, dest_directory)
