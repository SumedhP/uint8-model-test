import cv2
import numpy as np
from dataclasses import dataclass
from cv2.typing import MatLike
from time import perf_counter_ns as time_ns


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
    # Extract all x and y coordinates from both rectangles in one go
    x_coords = [p.x for p in rect1.points] + [p.x for p in rect2.points]
    y_coords = [p.y for p in rect1.points] + [p.y for p in rect2.points]

    # Compute new boundary coordinates (min/max x and y)
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    tag = rect1.tag if rect1.confidence > rect2.confidence else rect2.tag

    # Return the merged rectangle
    return Match(
        [
            Point(min_x, min_y),  # Bottom-left
            Point(min_x, max_y),  # Top-left
            Point(max_x, max_y),  # Top-right
            Point(max_x, min_y),  # Bottom-right
        ],
        rect1.color,
        tag,
        max(rect1.confidence, rect2.confidence),
    )


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

from typing import List

BBOX_CONFIDENCE_THRESHOLD = 0.85

def getBoxesFromOutput(output) -> List[Match]:
    boxes = []
    values = output[0]
    values = np.array(values)

    max_confidence = 0.0

    NUM_COLORS = 8
    NUM_TAGS = 8
    
    CONFIDENCE_INDEX = 8
    confidence_values = values[:, CONFIDENCE_INDEX]
    # Get the indices of the values that are above the threshold
    indices = np.where(confidence_values > BBOX_CONFIDENCE_THRESHOLD)
    values = values[indices]
    
    POINT_X_OFFSET = 204
    POINT_Y_OFFSET = 142.5
    X_SCALAR = 8.3
    Y_SCALAR = 16.1

    for element in values:
        x_1 = element[0] * X_SCALAR + POINT_X_OFFSET
        y_1 = element[1] * Y_SCALAR + POINT_Y_OFFSET
        x_2 = element[2] * X_SCALAR + POINT_X_OFFSET
        y_2 = element[3] * Y_SCALAR + POINT_Y_OFFSET
        x_3 = element[4] * X_SCALAR + POINT_X_OFFSET
        y_3 = element[5] * Y_SCALAR + POINT_Y_OFFSET
        x_4 = element[6] * X_SCALAR + POINT_X_OFFSET
        y_4 = element[7] * Y_SCALAR + POINT_Y_OFFSET

        confidence = element[8]
        color = np.argmax(element[9 : 9 + NUM_COLORS])
        tag = np.argmax(
            element[9 + NUM_COLORS : 9 + NUM_COLORS + NUM_TAGS]
        )

        bottomLeft = Point(x_1, y_1)
        topLeft = Point(x_2, y_2)
        topRight = Point(x_3, y_3)
        bottomRight = Point(x_4, y_4)
        
        if confidence > max_confidence:
            max_confidence = confidence
        
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
            if is_overlap(box, merged_boxes[j]):
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
    
    boxes = getBoxesForImg(img)
    
    # Scale the points back to the original image
    x_scalar = 540 / INPUT_SIZE
    y_scalar = 540 / INPUT_SIZE
    
    X_OFFSET = 210
    Y_OFFSET = 0
    
    for box in boxes:
        for i in range(4):
            box.points[i].x = box.points[i].x * x_scalar + 210
            box.points[i].y = box.points[i].y * y_scalar
    
    return boxes

def labelImage(filename: str):
    # open up new image
    # Since it is 960x540, split it into two 540x540 images
    img = cv2.imread(filename)

    # Take a INPUT_SIZExINPUT_SIZE image
    # # These are for sentry image
    # X_OFFSET = 500
    # Y_OFFSET = 100
    # img = img[Y_OFFSET : Y_INPUT_SIZE, X_OFFSET : X_INPUT_SIZE]

    # Show the image
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)

    boxes = getBoxesForCroppedImg(img)
    merged_boxes = mergeBoxes(boxes)
    
    print("Number of boxes: ", len(merged_boxes))
    print("Detected this: \n ", merged_boxes[0])
    
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
    filename = "test_images/sentry2.jpg"
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
