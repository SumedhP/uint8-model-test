from model import labelImage
import os
import cv2
from tqdm import tqdm

# Open all the files in the directory
FILE_DIRECTORY = 'temporar'
OUTPUT_DIRECTORY = 'temporar_output'

files = os.listdir(FILE_DIRECTORY)

print("Found, ", len(files), " files")

for file in tqdm(files):
  labelled_image = labelImage(os.path.join(FILE_DIRECTORY, file))
  cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, file), labelled_image)

print("Finished labelling images")
print("Now compiling images into a video")

# Now compile the images into a mp4 video 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 format
videoWriter = cv2.VideoWriter('output.mp4', fourcc, 30, (960, 540))
images = [cv2.imread(os.path.join(OUTPUT_DIRECTORY, file)) for file in files]
for image in tqdm(images):
  videoWriter.write(image)
videoWriter.release()
