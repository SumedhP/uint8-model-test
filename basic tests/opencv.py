import cv2
import cv2.dnn as dnn
import numpy as np

model = dnn.readNetFromONNX('model-opt.onnx')
print(model)
model.setPreferableBackend(dnn.DNN_BACKEND_DEFAULT)
model.setPreferableTarget(dnn.DNN_TARGET_CPU)

# read in match screenshot.png
img = cv2.imread('match screenshot 2 sentry crop.png')
scale = 640 / max(img.shape[1], img.shape[0])
img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

# Make a mat from the image
input_img = np.full((640, 640, 3), 127, dtype=np.uint8)
input_img[0:img.shape[0], 0:img.shape[1]] = img
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
x = cv2.dnn.blobFromImage(input_img) / 255.0

# display x
x_display = x[0].transpose(1, 2, 0)
x_display = np.clip(x_display * 255, 0, 255).astype(np.uint8)
# cv2.imshow('Output Image', x_display)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('color shifted.png', input_img)

model.setInput(x)
output = model.forward()
print(output.shape)

def inverseSigmoid(x: float) -> float:
  return -np.log(1 / x - 1)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

output_array = np.array(output.data)

values = output_array[0]
print("Length of values:", len(values))

color_to_word = ["Blue", "Red", "Neutral", "Purple"]
tag_to_word = ["Sentry", "1", "2", "3", "4", "5", "Outpost", "Base", "Base big armor"]

from dataclasses import dataclass
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
  


print()
print("--------------------")
boxes = []


for i in range(len(values)):
  element = values[i]
  if (element[8] < inverseSigmoid(0.5)):
    continue
  print('Found match at', i)
  colors = element[9:9+4]
  most_likely_color = np.argmax(colors)
  tags = element[13:13+9]
  most_likely_tag = np.argmax(tags)
  confidence = sigmoid(element[8])
  # print('Most likely color:', color_to_word[most_likely_color])
  # print('Most likely tag:', tag_to_word[most_likely_tag])
  # print('Confidence:', confidence)
  # print('')
  
  # Now make 4 points
  bottom_left = Point(element[0], element[1])
  top_left = Point(element[2], element[3])
  top_right = Point(element[4], element[5])
  bottom_right = Point(element[6], element[7])
  box = Match([bottom_left, top_left, top_right, bottom_right], color_to_word[most_likely_color], tag_to_word[most_likely_tag], confidence)
  boxes.append(box)



# Sort the boxes on their confidence
boxes.sort(key=lambda x: x.confidence, reverse=True)

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
  
good_boxes = []
for i in range(len(boxes)):
  good = True
  for j in range(i):
    if is_overlap(boxes[i], boxes[j]):
      good = False
      break
  if good:
    good_boxes.append(boxes[i])
    print('Good box:', i)

print("Remaining boxes:", len(good_boxes))
for i in range(len(good_boxes)):
  print(good_boxes[i])
  
temp = input_img.copy()
  
# Now draw the boxes, don't scale values
for i in range(len(good_boxes)):
  box = good_boxes[i]
  for j in range(4):
    cv2.line(input_img, (int(box.points[j].x), int(box.points[j].y)), (int(box.points[(j + 1) % 4].x), int(box.points[(j + 1) % 4].y)), (0, 255, 0), 2)
  cv2.putText(input_img, f'{box.color} {box.tag}', (int(box.points[0].x), int(box.points[0].y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image
cv2.imshow('Output Image', input_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('regular boxing.png', input_img)


# now instead merged the rectangles into the largest possible rectangles
merged_boxes = []
print("Length of boxes:", len(boxes))
for i in range(len(boxes)):
  overlaps_with = False
  for j in range(len(merged_boxes)):
    if is_overlap(boxes[i], merged_boxes[j]):
      overlaps_with = True
      print('Merging boxes:', i, j, " with label:", boxes[i].tag, merged_boxes[j].tag)
      boxes[i] = merge_rectangles(boxes[i], merged_boxes[j])
      merged_boxes[j] = boxes[i]
      break
    
  if(not overlaps_with):
    print('Adding box:', i, " with label:", boxes[i].tag)
    merged_boxes.append(boxes[i])

print("Remaining boxes:", len(merged_boxes))
for i in range(len(merged_boxes)):
  print(merged_boxes[i])
  
# Now draw the boxes, don't scale values

for i in range(len(merged_boxes)):
  box = merged_boxes[i]
  for j in range(4):
    cv2.line(temp, (int(box.points[j].x), int(box.points[j].y)), (int(box.points[(j + 1) % 4].x), int(box.points[(j + 1) % 4].y)), (0, 255, 0), 2)
  cv2.putText(temp, f'{box.color} {box.tag}', (int(box.points[0].x), int(box.points[0].y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image
cv2.imshow('Output Image', temp)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('merged boxing.png', temp)
