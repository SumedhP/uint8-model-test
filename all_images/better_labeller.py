from model import labelImage
import os
import cv2
from tqdm import tqdm

# Open all the files in the directory
FILE_DIRECTORY = 'images'
OUTPUT_DIRECTORY = 'better labelled'

# The top directory is just filled with subdirectory
# For every subdirectory, we will label the images in it and create a video for it
subdirectories = os.listdir(FILE_DIRECTORY)

for subdirectory in subdirectories:
  subdirectoryFilePath = os.path.join(FILE_DIRECTORY, subdirectory)
  
  files = os.listdir(subdirectoryFilePath)
  print("Found, ", len(files), " files")
  
  newOutputDirectory = os.path.join(OUTPUT_DIRECTORY, subdirectory)
  
  # Check if an output directory exists for this subdirectory
  if not os.path.exists(newOutputDirectory):
    os.makedirs(newOutputDirectory)
    
  for file in tqdm(files, mininterval=0.0000000000000000000001):
    # Check if the file is already labelled
    if os.path.exists(os.path.join(newOutputDirectory, file)):
      continue
    
    # If the file isn't jpg, skip it
    if not file.endswith(".jpg"):
      continue
    
    labelled_image = labelImage(os.path.join(FILE_DIRECTORY, subdirectory, file))
    cv2.imwrite(os.path.join(newOutputDirectory, file), labelled_image)
