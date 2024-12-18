from model import labelImage
import os
import cv2
from tqdm import tqdm
import signal

# Open all the files in the directory
FILE_DIRECTORY = 'image_store_raw'
OUTPUT_DIRECTORY = 'labelled'

def handler(sig, frame):
  makeVideo()
  exit(0)

signal.signal(signal.SIGINT, handler)

files = os.listdir(FILE_DIRECTORY)

print("Found, ", len(files), " files")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 format
def makeVideo():
  print("Making video")
  files = os.listdir(OUTPUT_DIRECTORY)
  output_video_name = str(f'{0}_{len(files)}.mp4')
  videoWriter = cv2.VideoWriter(output_video_name, fourcc, 30, (960, 540))
  for file in files:
    image = cv2.imread(os.path.join(OUTPUT_DIRECTORY, file))
    videoWriter.write(image)
  videoWriter.release()

count = 0
for file in tqdm(files, mininterval=0.0000000000000000000001):
  # Check if the file is already labelled
  if os.path.exists(os.path.join(OUTPUT_DIRECTORY, file)):
    continue
  
  labelled_image = labelImage(os.path.join(FILE_DIRECTORY, file))
  cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, file), labelled_image)
  count+=1
  
  if(count % 500 == 0):
    makeVideo()
    print("Made a video as we have labelled 500 images")





