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


def inverseSigmoid(x: float) -> float:
    return -np.log(1 / x - 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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


color_to_word = ["Blue", "Red", "Neutral", "Purple", "Color 5", "Color 6", "Color 7", "Color 8"]
tag_to_word = ["Sentry", "1", "2", "3", "4", "5", "Outpost", "Base", "Base big armor"]

# from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType

import onnxruntime as ort

model_path = "model-infer.onnx"
model_quant = "model_quant.onnx"
# print("Quantizing model")
# quantize_dynamic(model_path, model_quant, weight_type=QuantType.QUInt8)
# print("Model quantized")

sessionOptions = ort.SessionOptions()
sessionOptions.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)  # May ot may not be faster
sessionOptions.optimized_model_filepath = "really_optimized_model.onnx"

# Provider options for GPU
providers = [
    (
        "TensorrtExecutionProvider",
        {
            "device_id": 0,  # Select GPU to execute
            "trt_max_workspace_size": 2
            * 1024
            * 1024
            * 1024,  # Set GPU memory usage limit (2GB)
            "trt_int8_enable": True,  # Enable INT8 precision for faster inference
        },
    ),
    (
        "CUDAExecutionProvider",
        {
            "device_id": 0,  # Select GPU to execute
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2 GB
            "cudnn_conv_algo_search": "DEFAULT",  # The exhuastive one might be slower
            "do_copy_in_default_stream": True,  # False leads to race conditions and maybe faster performance
        },
    ),
]

# Provider options for CPU
model = ort.InferenceSession("best_06_02 (1).onnx", sess_options=sessionOptions)

from typing import List


def makeBoxesFromOutput(output) -> List[Match]:
    boxes = []
    values = output[0]

    # Filter using NumPy vectorization for confidence threshold
    values = np.array(values)
    confidence_scores = values[:, 8]  # Extract the confidence column
    filtered_indices = confidence_scores >= inverseSigmoid(
        0.5
    )  # Boolean mask for confidence >= 0.5
    filtered_values = values[filtered_indices]

    # Loop over the filtered values
    for element in filtered_values:
        # Extract most likely color and tag using np.argmax
        most_likely_color = np.argmax(element[9:13])  # Slice directly for color indices
        most_likely_tag = np.argmax(element[13:22])  # Slice directly for tag indices
        confidence = sigmoid(element[8])  # Convert confidence score using sigmoid

        # Define corners using Point
        bottom_left = Point(element[0], element[1])
        top_left = Point(element[2], element[3])
        top_right = Point(element[4], element[5])
        bottom_right = Point(element[6], element[7])

        # Create the Match object
        box = Match(
            [bottom_left, top_left, top_right, bottom_right],
            color_to_word[most_likely_color],
            tag_to_word[most_likely_tag],
            confidence,
        )
        boxes.append(box)

    return boxes


def makeBoxesFromNewModelOutput(output) -> List[Match]:
    boxes = []
    values = output[0]
    values = np.array(values)
    
    max_confidence = 0.0

    NUM_COLORS = 8
    NUM_TAGS = 8
    OFFSET = 0

    for element in values:
        x_1 = element[OFFSET + 0] * 416
        y_1 = element[OFFSET + 1] * 416
        x_2 = element[OFFSET + 2] * 416
        y_2 = element[OFFSET + 3] * 416
        x_3 = element[OFFSET + 4] * 416
        y_3 = element[OFFSET + 5] * 416
        x_4 = element[OFFSET + 6] * 416
        y_4 = element[OFFSET + 7] * 416

        confidence = element[OFFSET + 8]
        color = np.argmax(element[OFFSET + 9 : OFFSET + 9 + NUM_COLORS])
        tag = np.argmax(
            element[OFFSET + 9 + NUM_COLORS : OFFSET + 9 + NUM_COLORS + NUM_TAGS]
        )
        
        bottomLeft = Point(x_1, y_1)
        topLeft = Point(x_2, y_2)
        topRight = Point(x_3, y_3)
        bottomRight = Point(x_4, y_4)
     
        max_confidence = max(max_confidence, confidence)
        
        CONFIDENCE = 0.5
        if confidence < CONFIDENCE:
            continue
        
     
        
        box = Match(
            [bottomLeft, topLeft, topRight, bottomRight],
            color_to_word[color],
            tag_to_word[tag],
            confidence,
        )
        boxes.append(box)
        
    
    print("Max confidence: ", max_confidence)
  
    return boxes
        

avg_time = 0
count = 0


def getBoxesForImg(img: MatLike) -> List[Match]:
    global avg_time
    global count

    # create a black image and paste the resized image on it
    input_img = np.full((416, 416, 3), 127, dtype=np.uint8)
    input_img[0 : img.shape[0], 0 : img.shape[1]] = img

    # Manually convert BGR to RGB by swapping the color channels
    input_img = input_img[..., ::-1]

    # The following bit is done to replace the cv2.dnn.blobFromImage function
    # Normalize the image to [0, 1] range by dividing by 255
    input_img = input_img.astype(np.float32) / 255.0

    # Transpose to the format (channels, height, width) expected by ONNX models
    input_img = np.transpose(input_img, (2, 0, 1))

    # Add batch dimension, making the shape (1, channels, height, width)
    input_img = np.expand_dims(input_img, axis=0)

    onnx_input = {"images": input_img}

    start = time_ns()
    output = model.run(None, onnx_input)
    output = np.array(output)
    print(output.shape)

    end = time_ns()
    print(f"Time taken to run model: {(end - start) / 1e6} ms")
    avg_time += (end - start) / 1e6
    count += 1

    output = output[0]

    # Now evaluate the output, only making a Match object if the confidence is above 0.5
    boxes = makeBoxesFromNewModelOutput(output)
    print("Found ", len(boxes), " boxes with good confidence")
    
    return boxes


# Unused function
def getMergedBoxesForImg(img: MatLike) -> List[Match]:
    offset = 960 - 540

    img1 = img[0:540, 0:540]
    img1_boxes = getBoxesForImg(img1)

    img2 = img[0:540, offset : offset + 540]
    img2_boxes = getBoxesForImg(img2)

    # Offset the x values of the second image
    for box in img2_boxes:
        for point in box.points:
            point.x += offset

    # Merge the two lists
    img1_boxes.extend(img2_boxes)
    # Now merge overlapping boxes
    merged_boxes = []
    for i in range(len(img1_boxes)):
        overlaps_with = False
        # Check if the current box overlaps with any of the merged boxes
        for j in range(len(merged_boxes)):
            # It overlaps, so merge the two boxes
            if is_overlap(img1_boxes[i], merged_boxes[j]):
                overlaps_with = True
                merged_boxes[j] = merge_rectangles(img1_boxes[i], merged_boxes[j])
                break
        # It doesn't overlap with any of the merged boxes, so add it to the list
        if not overlaps_with:
            merged_boxes.append(img1_boxes[i])

    return merged_boxes


# Not as good as getMergedBoxesForImg
def getBoxesForScaledImg(img: MatLike) -> List[Match]:
    # Resize from a 960x540 image to a 640x640 image
    resized_img = cv2.resize(img, (416, 416))

    boxes = getBoxesForImg(resized_img)

    x_scalar = 960 / 416
    y_scalar = 540 / 416

    for box in boxes:
        for point in box.points:
            point.x *= x_scalar
            point.y *= y_scalar

    # merge the boxes
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


def labelImage(filename: str):
    # open up new image
    # Since it is 960x540, split it into two 540x540 images
    img = cv2.imread(filename)

    # Flip vertically
    # img = cv2.flip(img, 0)

    # # Greyscale image but keep dimensions
    # grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

    start = time_ns()
    merged_boxes = getBoxesForScaledImg(img)
    end = time_ns()
    # print(f"Time taken to label image: {(end - start) / 1e6} ms")

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


def printAverageTime():
    global avg_time
    global count
    avg_time /= count
    print(f"Average time taken to run model: {avg_time} ms")
