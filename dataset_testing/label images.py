from model import labelImage
import os
import cv2
from tqdm import tqdm

# Open all the files in the directory
FILE_DIRECTORY = 'raw images'
OUTPUT_DIRECTORY = 'labelled images'

files = os.listdir(FILE_DIRECTORY)

print("Found, ", len(files), " files")

from time import time

for file in tqdm(files):
  start = time()
  labelled_image = labelImage(os.path.join(FILE_DIRECTORY, file))
  end = time()
  print("Time taken: ", end - start)
  exit()
  
  
  # cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, file), labelled_image)
  
# Now compile the images into a mp4 video 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 format
videoWriter = cv2.VideoWriter('output.mp4', fourcc, 30, (960, 540))
images = [cv2.imread(os.path.join(OUTPUT_DIRECTORY, file)) for file in files]
for image in images:
  videoWriter.write(image)
videoWriter.release()
