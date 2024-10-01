from model import labelImage
import os
import cv2
from tqdm import tqdm

# Open all the files in the directory
FILE_DIRECTORY = 'image_store_raw'
OUTPUT_DIRECTORY = 'labelled'

files = os.listdir(FILE_DIRECTORY)

print("Found, ", len(files), " files")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 format

File_Start = 0
File_End = File_Start + 1000

output_video_name = str(f'{File_Start}_{File_End}.mp4')

videoWriter = cv2.VideoWriter(output_video_name, fourcc, 30, (960, 540))


for file in tqdm(files[File_Start:File_End]):
  labelled_image = labelImage(os.path.join(FILE_DIRECTORY, file))
  cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, file), labelled_image)
  image = cv2.imread(os.path.join(OUTPUT_DIRECTORY, file))
  videoWriter.write(image)
videoWriter.release()
