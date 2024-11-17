import onnxruntime as ort
import os
import cv2
from tqdm import tqdm


model_path = "model.onnx"
output_path = "optimized.onnx"

# Go through all the photos in data folder and resize to 640x640
data_folder = "data"
data_output = "data_formatted"

files = os.listdir(data_folder)

if not os.path.exists(data_output):
    os.makedirs(data_output)

for file in tqdm(files):
    image = cv2.imread(os.path.join(data_folder, file))
    image = cv2.resize(image, (640, 640))
    cv2.imwrite(os.path.join(data_output, file), image)


# Apply a static quantization to the model
