import os
import cv2
from tqdm import tqdm

FILE_DIRECTORY = 'better labelled'

# Go through each subdirectory and create a video for it
subdirectories = os.listdir(FILE_DIRECTORY)

VIDEO_WIDTH = 960
VIDEO_HEIGHT = 540

for subdirectory in tqdm(subdirectories):
  output_video_name = str(f'{subdirectory}.mp4')
  if os.path.exists(output_video_name):
    print("Video already exists for ", subdirectory)
    continue
  
  subdirectoryFilePath = os.path.join(FILE_DIRECTORY, subdirectory)
  
  files = os.listdir(subdirectoryFilePath)
  print("Found, ", len(files), " files")
  
  images = []
  
  for file in files:
    # If the file isn't jpg, skip it
    if not file.endswith(".jpg"):
      continue
    
    image = cv2.imread(os.path.join(subdirectoryFilePath, file))
    
    # Make sure the image is the right size, if not skip it, and print
    if image.shape[0] != VIDEO_HEIGHT or image.shape[1] != VIDEO_WIDTH:
      continue
    
    images.append(image)
    
  if(len(images) == 0):
    print("No valid images found for ", subdirectory)
    continue
  
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 format
  videoWriter = cv2.VideoWriter(output_video_name, fourcc, 30, (VIDEO_WIDTH, VIDEO_HEIGHT))
  
  for image in images:
    videoWriter.write(image)
  videoWriter.release()

