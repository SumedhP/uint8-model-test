import numpy as np
import onnxruntime
import time
import os
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static, quantize_dynamic
from onnxruntime.quantization import CalibrationDataReader
import cv2

from cv2.typing import MatLike


def makeImageFitFormat(img: MatLike):
  output = np.full((640, 640, 3), 127, dtype=np.uint8)
  output[0:img.shape[0], 0:img.shape[1]] = img
  
  # Manually convert BGR to RGB by swapping the color channels
  output = output[..., ::-1]
  
  # Make it grey scale but keep the 3 channels
  
  # The following bit is done to replace the cv2.dnn.blobFromImage function
  # Normalize the image to [0, 1] range by dividing by 255
  output = output.astype(np.float32) / 255.0

  # Transpose to the format (channels, height, width) expected by ONNX models
  output = np.transpose(output, (2, 0, 1))

  # Add batch dimension, making the shape (1, channels, height, width)
  output = np.expand_dims(output, axis=0)
  
  return output


class DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)

        self.images = []
        files = os.listdir(calibration_image_folder)

        for file in files:
            image = cv2.imread(os.path.join(calibration_image_folder, file))
            image = cv2.resize(image, (640, 640))
            
            image = makeImageFitFormat(image)
            
            self.images.append(image)

        print(f"Loaded {len(self.images)} images for calibration.")

        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.images)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([{self.input_name: image} for image in self.images])
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def benchmark(model_path):
    session = onnxruntime.InferenceSession(model_path)
    input_name = "images"

    total = 0.0
    runs = 10
    input_data = np.zeros((1, 3, 640, 640), np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


def main():
    input_model_path = "model-preprocess.onnx"
    output_model_path = "model-quantized.onnx"

    calibration_dataset_path = "data_formatted"

    dr = DataReader(calibration_dataset_path, input_model_path)
    
    print("DataReader created.")

    # Calibrate and quantize model
    # Turn off model optimization during quantization
    quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=QuantType.QInt8,
    )
    
    print("Model quantized.")
    
    # Dynamic quantize model
    dynamic_output_model_path = "model-quantized-dynamic.onnx"
    quantize_dynamic(input_model_path, dynamic_output_model_path)
    
    
    print("Calibrated and quantized model saved.")

    print("benchmarking fp32 model...")
    benchmark(input_model_path)

    print("benchmarking int8 model...")
    benchmark(output_model_path)
    
    print("benchmarking dynamic model...")
    benchmark(dynamic_output_model_path)


if __name__ == "__main__":
    main()
