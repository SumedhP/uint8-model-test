x_input =[-0.23, -0.15, 2.77, 2.69]
y_input = [0.75, 1.86, 1.58, 0.49]

# My annotations:
x = [559, 559, 584, 584]
y = [298, 281, 282, 299]

# Draw these points on the image
filepath = "test_images/multirobot.jpg"

CENTER_X = (960 - 540) / 2 + (540 / 2)
CENTER_Y = (540 / 2)

x_input = [x + CENTER_X for x in x_input]
y_input = [y + CENTER_Y for y in y_input]


import numpy as np

# Now try and scale and shift the points till they match my annotations
# Compute the scaling factors for x and y based on the distances between points
def compute_scaling_factor(input_points, annotation_points):
    # Calculate the distance between points in both input and annotation sets
    input_distance = np.linalg.norm(np.array(input_points[1]) - np.array(input_points[0]))
    annotation_distance = np.linalg.norm(np.array(annotation_points[1]) - np.array(annotation_points[0]))

    # Scaling factor is the ratio of these distances
    return annotation_distance / input_distance

# Compute the scaling factors
scale_x = compute_scaling_factor([(x_input[0], y_input[0]), (x_input[1], y_input[1])],
                                 [(x[0], y[0]), (x[1], y[1])])
scale_y = compute_scaling_factor([(x_input[0], y_input[0]), (x_input[2], y_input[2])],
                                 [(x[0], y[0]), (x[2], y[2])])

# Now apply the scaling and shifting to all the input points
x_transformed = [scale_x * (xi - CENTER_X) + x[0] for xi in x_input]
y_transformed = [scale_y * (yi - CENTER_Y) + y[0] for yi in y_input]

print("Transformed x points:", x_transformed)
print("Transformed y points:", y_transformed)

x_input = x_transformed
y_input = y_transformed

import cv2
img = cv2.imread(filepath)

# Convert the points to integers
x_input = [int(i) for i in x_input]
y_input = [int(i) for i in y_input]

# Draw the points on the image
for i in range(4):
    cv2.circle(img, (x[i], y[i]), 5, (0, 0, 255))
    cv2.circle(img, (x_input[i], y_input[i]), 5, (255, 5, 250), -1)

# Draw the lines
for i in range(4):
    cv2.line(img, (x[i], y[i]), (x[(i + 1) % 4], y[(i + 1) % 4]), (0, 0, 255), 2)
    cv2.line(img, (x_input[i], y_input[i]), (x_input[(i + 1) % 4], y_input[(i + 1) % 4]), (255, 0, 250), 2)

cv2.imshow("Image", img)
cv2.waitKey(0)

