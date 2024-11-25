import os
import cv2
from tqdm import tqdm
from cv2.typing import MatLike
from model import getBoxesForImg, mergeListOfMatches, putTextOnImage
from typing import List
from Match import Match
from time import time_ns


TARGET_WH = 416
def compressImageAndScaleOutput(img: MatLike) -> List[Match]:
    input_w = img.shape[1]
    input_h = img.shape[0]

    scalar_w = input_w / TARGET_WH
    scalar_h = input_h / TARGET_WH
    img = cv2.resize(img, (TARGET_WH, TARGET_WH))

    boxes = getBoxesForImg(img)
    
    for box in boxes:
        for i in range(4):
            box.points[i].x = box.points[i].x * scalar_w
            box.points[i].y = box.points[i].y * scalar_h
    return boxes


INPUT_SIZE_W = 960
INPUT_SIZE_H = 540


def splitImageRunTwice(img: MatLike) -> List[Match]:
    x_diff = INPUT_SIZE_W - INPUT_SIZE_H
    img1 = img[0:INPUT_SIZE_H, 0 : INPUT_SIZE_W - x_diff]
    img2 = img[0:INPUT_SIZE_H, x_diff:]

    boxes1 = compressImageAndScaleOutput(img1)
    boxes2 = compressImageAndScaleOutput(img2)
    for box in boxes2:
        for i in range(4):
            box.points[i].x += x_diff

    return boxes1 + boxes2

def processImage(img: MatLike) -> List[Match]:
    boxes = splitImageRunTwice(img)
    # boxes = compressImageAndScaleOutput(img)
    merged = mergeListOfMatches(boxes)
    return merged


def main():
    INPUT_NAME = "2023_comp_footage/2_17"
    OUTPUT_FOLDER = INPUT_NAME + "_labelled/"
    INPUT_FOLDER = INPUT_NAME + "/"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    time_running = 0

    for filename in tqdm(os.listdir(INPUT_FOLDER)):
        img = cv2.imread(INPUT_FOLDER + filename)
        start_time = time_ns()
        boxes = processImage(img)
        end_time = time_ns()
        time_running += end_time - start_time
        img = putTextOnImage(img, boxes)
        cv2.imwrite(OUTPUT_FOLDER + filename, img)
    time_running /= len(os.listdir(INPUT_FOLDER))
    print(f"Time taken on avg: {(time_running) / 1e6} ms")

    print("Images processed and saved in output_dataset folder")

    #  Now compile the images into a mp4 video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # For MP4 format
    output_file = INPUT_NAME + "_output.mp4"
    videoWriter = cv2.VideoWriter(output_file, fourcc, 30, (960, 540))
    images = [cv2.imread(OUTPUT_FOLDER + file) for file in os.listdir(OUTPUT_FOLDER)]
    for image in images:
        videoWriter.write(image)
    videoWriter.release()


if __name__ == "__main__":
    main()
