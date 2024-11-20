from model import labelImage, printAverageTime
import os
import cv2
from tqdm import tqdm

def main():
  # Open all the files in the directory
  FILE_DIRECTORY = '1_9'
  OUTPUT_DIRECTORY = 'random ' + FILE_DIRECTORY

  # Make output dir if it doesn't exist
  if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

  files = os.listdir(FILE_DIRECTORY)

  print("Found, ", len(files), " files")

  for file in files:
    labelled_image = labelImage(os.path.join(FILE_DIRECTORY, file))
    cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, file), labelled_image)
  
  printAverageTime()

  print("Finished labelling images")
  print("Now compiling images into a video")

  # Now compile the images into a mp4 video 
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 format
  output_file = 'output_' + FILE_DIRECTORY + ' random ' + '.mp4'
  videoWriter = cv2.VideoWriter(output_file, fourcc, 30, (960, 540))
  images = [cv2.imread(os.path.join(OUTPUT_DIRECTORY, file)) for file in files]
  for image in tqdm(images):
    videoWriter.write(image)
  videoWriter.release()


if __name__ == "__main__":
  main()
