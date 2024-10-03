import cv2
import cv2.dnn as dnn
import numpy as np
from dataclasses import dataclass
from cv2.typing import MatLike

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
    # Extract the corners of both rectangles
    r1_bottom_left = rect1.points[0]
    r1_top_right = rect1.points[2]
    
    r2_bottom_left = rect2.points[0]
    r2_top_right = rect2.points[2]
    
    # Check for non-overlapping conditions
    if (r1_bottom_left.x > r2_top_right.x or r2_bottom_left.x > r1_top_right.x):
        return False  # One rectangle is to the left of the other
    
    if (r1_bottom_left.y > r2_top_right.y or r2_bottom_left.y > r1_top_right.y):
        return False  # One rectangle is above the other
    
    # If none of the non-overlapping conditions are met, the rectangles overlap
    return True
  
def merge_rectangles(rect1: Match, rect2: Match) -> Match:
    # Extract all x and y coordinates from both rectangles
    x_coords = [p.x for p in rect1.points] + [p.x for p in rect2.points]
    y_coords = [p.y for p in rect1.points] + [p.y for p in rect2.points]
    
    # Find the min and max x and y values
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    
    # Create the merged rectangle with new boundaries
    merged_rectangle = Match([
        Point(min_x, min_y),  # Bottom-left
        Point(min_x, max_y),  # Top-left
        Point(max_x, max_y),  # Top-right
        Point(max_x, min_y)   # Bottom-right
    ], rect1.color, rect1.tag, max(rect1.confidence, rect2.confidence))
    
    return merged_rectangle

color_to_word = ["Blue", "Red", "Neutral", "Purple"]
tag_to_word = ["Sentry", "1", "2", "3", "4", "5", "Outpost", "Base", "Base big armor"]

model = dnn.readNetFromONNX('model-opt.onnx')
model.setPreferableBackend(dnn.DNN_BACKEND_DEFAULT)
model.setPreferableTarget(dnn.DNN_TARGET_CPU)

from time import time

from typing import List
def makeBoxesFromOutput(output) -> List[Match]:
  boxes = []
  values = output[0]
  # Filter out the values that have a confidence below 0.5
  filtered_values = [element for element in values if element[8] >= inverseSigmoid(0.5)]
  
  for i in range(len(filtered_values)):
    element = filtered_values[i]
    colors = element[9:9+4]
    most_likely_color = np.argmax(colors)
    tags = element[13:13+9]
    most_likely_tag = np.argmax(tags)
    confidence = sigmoid(element[8])
    
    bottom_left = Point(element[0], element[1])
    top_left = Point(element[2], element[3])
    top_right = Point(element[4], element[5])
    bottom_right = Point(element[6], element[7])
    box = Match([bottom_left, top_left, top_right, bottom_right], color_to_word[most_likely_color], tag_to_word[most_likely_tag], confidence)
    boxes.append(box)
  
  return boxes

def getBoxesForImg(img: MatLike) -> List[Match]:
  # create a black image and paste the resized image on it
  input_img = np.full((640, 640, 3), 127, dtype=np.uint8)
  input_img[0:img.shape[0], 0:img.shape[1]] = img
  input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
  
  # Make input to system
  x = cv2.dnn.blobFromImage(input_img) / 255.0
  
  model.setInput(x)
  output = np.array(model.forward())
  
  # Now evaluate the output, only making a Match object if the confidence is above 0.5
  boxes = makeBoxesFromOutput(output)
  
  # Now do it again but flip the image vertically
  input_img = cv2.flip(input_img, 0)
  x = cv2.dnn.blobFromImage(input_img) / 255.0
  model.setInput(x)
  output = np.array(model.forward())
  
  # Now evaluate the output, only making a Match object if the confidence is above 0.5
  flipped_boxes = makeBoxesFromOutput(output)
  
  return boxes

def getMergedBoxesForImg(img: MatLike) -> List[Match]:
  offset = 960 - 540
  
  img1 = img[0:540, 0:540]
  img1_boxes = getBoxesForImg(img1)
  
  img2 = img[0:540, offset:offset+540]
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
    if(not overlaps_with):
      merged_boxes.append(img1_boxes[i])
      
  return merged_boxes

def labelImage(filename: str):
  # open up new image
  # Since it is 960x540, split it into two 540x540 images
  img = cv2.imread(filename)
  
  merged_boxes = getMergedBoxesForImg(img)
  
  # Now rotate the image 180 degrees clockwise
  img = cv2.rotate(img, cv2.ROTATE_180)
  
  
  
  # Now add the labels to the image
  for i in range(len(merged_boxes)):
    box = merged_boxes[i]
    for j in range(4):
      cv2.line(img, (int(box.points[j].x), int(box.points[j].y)), (int(box.points[(j + 1) % 4].x), int(box.points[(j + 1) % 4].y)), (0, 255, 0), 2)
    cv2.putText(img, f'{box.color} {box.tag}', (int(box.points[0].x), int(box.points[0].y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
  
  return img
  
  
  
  
   
  
  
  
