import cv2
import numpy as np
from onnxruntime import InferenceSession
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class GridAndStride:
    grid0: int
    grid1: int
    stride: int


@dataclass
class ArmorObject:
    apex: List[Tuple[float, float]]
    pts: List[Tuple[float, float]]
    cls: int
    color: int
    prob: float


class ArmorDetector:
    INPUT_W = 416
    INPUT_H = 416
    NUM_CLASSES = 8
    NUM_COLORS = 8
    TOPK = 128
    NMS_THRESH = 0.3
    BBOX_CONF_THRESH = 0.85
    MERGE_CONF_ERROR = 0.15
    MERGE_MIN_IOU = 0.9

    def __init__(self, model_path: str):
        self.model = InferenceSession(model_path)
        self.transform_matrix = None

    def scaled_resize(self, img: np.ndarray) -> np.ndarray:
        """Resize image with letterbox scaling and compute the transform matrix."""
        r = min(self.INPUT_W / img.shape[1], self.INPUT_H / img.shape[0])
        unpad_w = int(r * img.shape[1])
        unpad_h = int(r * img.shape[0])
        dw = (self.INPUT_W - unpad_w) // 2
        dh = (self.INPUT_H - unpad_h) // 2

        self.transform_matrix = np.array([
            [1.0 / r, 0, -dw / r],
            [0, 1.0 / r, -dh / r],
            [0, 0, 1]
        ])

        resized_img = cv2.resize(img, (unpad_w, unpad_h))
        padded_img = cv2.copyMakeBorder(resized_img, dh, dh, dw, dw, cv2.BORDER_CONSTANT)
        return padded_img

    def generate_grids_and_stride(self, strides: List[int]) -> List[GridAndStride]:
        """Generate grid strides for YOLOX."""
        grid_strides = []
        for stride in strides:
            num_grid_w = self.INPUT_W // stride
            num_grid_h = self.INPUT_H // stride
            for g1 in range(num_grid_h):
                for g0 in range(num_grid_w):
                    grid_strides.append(GridAndStride(g0, g1, stride))
        return grid_strides

    def decode_outputs(self, prob: np.ndarray) -> List[ArmorObject]:
        """Decode the model's output into bounding boxes."""
        grid_strides = self.generate_grids_and_stride([8, 16, 32])
        proposals = []

        for idx, grid in enumerate(grid_strides):
            stride = grid.stride
            base_pos = idx * (9 + self.NUM_CLASSES + self.NUM_COLORS)

            x1, y1 = prob[base_pos + 0] + grid.grid0, prob[base_pos + 1] + grid.grid1
            x2, y2 = prob[base_pos + 2] + grid.grid0, prob[base_pos + 3] + grid.grid1
            x3, y3 = prob[base_pos + 4] + grid.grid0, prob[base_pos + 5] + grid.grid1
            x4, y4 = prob[base_pos + 6] + grid.grid0, prob[base_pos + 7] + grid.grid1

            x1, x2, x3, x4 = [x * stride for x in [x1, x2, x3, x4]]
            y1, y2, y3, y4 = [y * stride for y in [y1, y2, y3, y4]]

            box_objectness = prob[base_pos + 8]
            if box_objectness < self.BBOX_CONF_THRESH:
                continue

            apex_norm = np.array([[x1, x2, x3, x4], [y1, y2, y3, y4], [1, 1, 1, 1]])
            apex_dst = self.transform_matrix @ apex_norm
            pts = [(apex_dst[0, i], apex_dst[1, i]) for i in range(4)]

            proposals.append(ArmorObject(
                apex=pts,
                pts=pts,
                cls=int(np.argmax(prob[base_pos + 9 + self.NUM_COLORS: base_pos + 9 + self.NUM_COLORS + self.NUM_CLASSES])),
                color=int(np.argmax(prob[base_pos + 9: base_pos + 9 + self.NUM_COLORS])),
                prob=box_objectness
            ))

        # Non-maximum suppression
        proposals.sort(key=lambda x: x.prob, reverse=True)
        if len(proposals) > self.TOPK:
            proposals = proposals[:self.TOPK]

        return proposals

    def detect(self, img: np.ndarray) -> List[ArmorObject]:
        """Perform detection on the input image."""
        resized_img = self.scaled_resize(img)
        img_input = resized_img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]

        # Perform inference
        inputs = {self.model.get_inputs()[0].name: img_input}
        outputs = self.model.run(None, inputs)
        prob = outputs[0].squeeze()

        return self.decode_outputs(prob)


def main():
    detector = ArmorDetector("model.onnx")
    img = cv2.imread("cropped.jpg")
    objects = detector.detect(img)

    # Draw bounding boxes on the image
    for obj in objects:
        for i in range(4):
            pt1 = (int(obj.apex[i][0]), int(obj.apex[i][1]))
            pt2 = (int(obj.apex[(i + 1) % 4][0]), int(obj.apex[(i + 1) % 4][1]))
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)

    cv2.imshow("Detections", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
