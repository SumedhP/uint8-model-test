import cv2
import numpy as np
from dataclasses import dataclass
from cv2.typing import MatLike
from time import perf_counter_ns as time_ns
from typing import List


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Match:
    points: list
    color: str
    tag: str
    confidence: float

    # Print in a nice format with color, tag, points, and confidence
    def __str__(self):
        return f"Color: {self.color}, Tag: {self.tag}, Points: {self.points}, Confidence: {self.confidence}"

    # Default sort by confidence
    def __lt__(self, other):
        return self.confidence < other.confidence

@dataclass 
class GridAndStride:
    grid0: int
    grid1: int
    stride: int


def generateGridsAndStride() -> List[GridAndStride]:
    INPUT_W = 416
    INPUT_H = 416
    STRIDES = [8, 16, 32]
    output = []
    for stride in STRIDES:
        grid_h = INPUT_H // stride
        grid_w = INPUT_W // stride
        for i in range (grid_h):
            for j in range(grid_w):
                output.append(GridAndStride(i, j, stride))
    return output

def is_overlap(rect1: Match, rect2: Match) -> bool:
    # Calculate the bounding box of rect1
    r1_x_coords = [point.x for point in rect1.points]
    r1_y_coords = [point.y for point in rect1.points]
    r1_min_x, r1_max_x = min(r1_x_coords), max(r1_x_coords)
    r1_min_y, r1_max_y = min(r1_y_coords), max(r1_y_coords)

    # Calculate the bounding box of rect2
    r2_x_coords = [point.x for point in rect2.points]
    r2_y_coords = [point.y for point in rect2.points]
    r2_min_x, r2_max_x = min(r2_x_coords), max(r2_x_coords)
    r2_min_y, r2_max_y = min(r2_y_coords), max(r2_y_coords)

    # Check if the bounding boxes overlap
    return (r1_max_x >= r2_min_x and r2_max_x >= r1_min_x) and (
        r1_max_y >= r2_min_y and r2_max_y >= r1_min_y
    )


def merge_rectangles(rect1: Match, rect2: Match) -> Match:
    # If they are not same tag and color, throw an error
    if rect1.color != rect2.color or rect1.tag != rect2.tag:
        raise ValueError("Rectangles must have same color and tag to merge")
    
    # Just return the rectangle with higher confidence if they are the same
    return rect1 if rect1.confidence > rect2.confidence else rect2 

color_to_word = [
    "Blue",
    "Filler 1",
    "Red",
    "Filler 2",
    "Neutral",
    "Filler 3",
    "Purple",
    "Filler 4",
]
tag_to_word = ["Sentry", "1", "2", "3", "4", "5", "Outpost", "Base"]
INPUT_SIZE = 416


import onnxruntime as ort

# Provider options for CPU
model = ort.InferenceSession("model.onnx")

# Print model input and shape
print("Model input shape: ", model.get_inputs()[0].shape)
print("Model input type: ", model.get_inputs()[0].type)
print("Model input name: ", model.get_inputs()[0].name)

BBOX_CONFIDENCE_THRESHOLD = 0.85


def getBoxesFromOutput(output) -> List[Match]:
    boxes = []
    values = output[0]
    values = np.array(values)

    max_confidence = 0.0

    NUM_COLORS = 8
    NUM_TAGS = 8

    grid_strides = generateGridsAndStride()

    for i in range(len(values)):
        element = values[i]
        
        grid0 = grid_strides[i].grid0
        grid1 = grid_strides[i].grid1
        stride = grid_strides[i].stride
        
        
        x_1 = (element[0] + grid0) * stride
        y_1 = (element[1] + grid1) * stride
        x_2 = (element[2] + grid0) * stride
        y_2 = (element[3] + grid1) * stride
        x_3 = (element[4] + grid0) * stride
        y_3 = (element[5] + grid1) * stride
        x_4 = (element[6] + grid0) * stride
        y_4 = (element[7] + grid1) * stride

        confidence = element[8]
        color = np.argmax(element[9 : 9 + NUM_COLORS])
        tag = np.argmax(element[9 + NUM_COLORS : 9 + NUM_COLORS + NUM_TAGS])

        bottomLeft = Point(x_1, y_1)
        topLeft = Point(x_2, y_2)
        topRight = Point(x_3, y_3)
        bottomRight = Point(x_4, y_4)

        if confidence > max_confidence:
            max_confidence = confidence
        
        if confidence < BBOX_CONFIDENCE_THRESHOLD:
            continue

        box = Match(
            [bottomLeft, topLeft, topRight, bottomRight],
            color_to_word[int(color / 2)],
            tag_to_word[tag],
            confidence,
        )
        boxes.append(box)

    print("Max confidence: ", max_confidence)
    print("Number of boxes: ", len(boxes))

    # Sort the boxes by confidence
    boxes.sort(reverse=True)

    return boxes


def getBoxesForImg(img: MatLike) -> List[Match]:
    # Convert to float
    img = img.astype(np.float32)
    print("Image shape: ", img.shape)

    # Model expects input of 1,3,416,416
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    print("Image shape after transpose: ", img.shape)

    onnx_input = {"images": img}

    start = time_ns()
    output = model.run(None, onnx_input)
    end = time_ns()
    print(f"Time taken to run model: {(end - start) / 1e6} ms")

    output = np.array(output)
    print(output.shape)
    output = output[0]

    # Now evaluate the output, only making a Match object if the confidence is above 0.5
    boxes = getBoxesFromOutput(output)

    return boxes


def mergeBoxes(boxes: List[Match]) -> List[Match]:
    merged_boxes = []
    for box in boxes:
        merged = False
        for j in range(len(merged_boxes)):
            # Merge the boxes if they overlap
            if (
                is_overlap(box, merged_boxes[j])
                and (box.color == merged_boxes[j].color)
                and (box.tag == merged_boxes[j].tag)
            ):
                merged_boxes[j] = merge_rectangles(box, merged_boxes[j])
                merged = True
                break
        if not merged:
            merged_boxes.append(box)

    return merged_boxes


def getBoxesForCroppedImg(img: MatLike) -> List[Match]:
    # Expects a 960 x 540 image
    # We take the center 540 x 540 image
    img = img[0:540, 210:750]
    # Resize to 416 x 416
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))

    # show this image
    cv2.imshow("Image", img)
    cv2.waitKey(0)

    boxes = getBoxesForImg(img)

    # Scale the points back to the original image
    x_scalar = 540 / INPUT_SIZE
    y_scalar = 540 / INPUT_SIZE

    for box in boxes:
        for i in range(4):
            box.points[i].x = box.points[i].x * x_scalar
            box.points[i].y = box.points[i].y * y_scalar

    return boxes


def scaleBoxes(boxes: List[Match]) -> List[Match]:
    X_OFFSET = 40
    Y_OFFSET = 168
    X_SCALAR = 8.3
    Y_SCALAR = 12.4

    for box in boxes:
        for i in range(4):
            box.points[i].x = box.points[i].x * X_SCALAR + X_OFFSET + 210
            box.points[i].y = box.points[i].y * Y_SCALAR + Y_OFFSET

    return boxes


def labelImage(filename: str):
    img = cv2.imread(filename)

    boxes = getBoxesForCroppedImg(img)
    print()
    print("Number of boxes: ", len(boxes))
    for box in boxes:
        print(box)
    print()

    merged_boxes = mergeBoxes(boxes)

    print("Number of boxes: ", len(merged_boxes))
    print("Detected this: \n ")
    for box in merged_boxes:
        print(box)
        print()
    print()

    # merged_boxes = scaleBoxes(merged_boxes)

    # Now add the labels to the image
    for i in range(len(merged_boxes)):
        box = merged_boxes[i]
        for j in range(4):
            cv2.line(
                img,
                (int(box.points[j].x), int(box.points[j].y)),
                (int(box.points[(j + 1) % 4].x), int(box.points[(j + 1) % 4].y)),
                (0, 255, 0),
                2,
            )
        cv2.putText(
            img,
            f"{box.color} {box.tag} {box.confidence : .02f}",
            (int(box.points[0].x), int(box.points[0].y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return img


def main():
    filename = "test_images/multirobot.jpg"
    # filename = "test_images/sentry_up.jpg"
    img = labelImage(filename)
    cv2.imshow("Image", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()

"""
0: -3.98410
1: -0.40040
2: -3.83973
3: 1.85531
4: 4.81911
5: 1.54798
6: 4.65551
7: -0.66024
8: 0.96329
9: 0.01173
10: 0.79137
11: 0.00000
12: 0.00000
13: 0.00000
14: 0.00001
15: 0.00001
16: 0.00403
17: 0.53521
18: 0.02295
19: 0.00033
20: 0.00010
21: 0.00653
22: 0.00037
23: 0.00029
24: 0.22932
"""

"""
0 - 7: Bounding box coordinates
8: Confidence
9 - 16: Color
17 - 24: Tag
"""
