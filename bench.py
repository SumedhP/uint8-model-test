import cv2
import numpy as np
import time


def fancy_benchmark_func(num_runs=1000, function=None):
    start_time = time.time_ns()

    count = 0
    for _ in range(num_runs):
        count += 1
        function()
        current_time = time.time_ns()
        print(
            f"\r Current average: {1e9 / (current_time - start_time) * count:.2f} FPS",
            end="",
        )

    end_time = time.time_ns()

    elapsed_time = (end_time - start_time) / 1e6
    avg_time_per_run = elapsed_time / num_runs

    print()
    print(f"Benchmark complete: {num_runs} runs.")
    print(f"Total time: {elapsed_time:.5f} milliseconds")
    print(f"Average time per run: {avg_time_per_run:.5f} milliseconds")
    print()


def benchmark_func(num_runs=1000, function=None):
    # Time the function calls
    start_time = time.time_ns()

    for _ in range(num_runs):
        function()

    end_time = time.time_ns()

    # Calculate average time per run
    elapsed_time = (end_time - start_time) / 1e6  # Convert to seconds
    avg_time_per_run = elapsed_time / num_runs

    print(f"Benchmark complete: {num_runs} runs.")
    print(f"Total time: {elapsed_time:.5f} milliseconds")
    print(f"Average time per run: {avg_time_per_run:.5f} milliseconds")
    print()


def benchmark_solve_pnp(num_runs=1000):
    # Define random 3D object points (3D points in the object space)
    object_points = np.random.rand(10, 3).astype(np.float32)  # 10 points, 3D

    # Define random 2D image points (2D projections of the 3D points)
    image_points = np.random.rand(10, 2).astype(np.float32)  # 10 points, 2D

    # Define a random camera matrix (intrinsics), usually 3x3
    camera_matrix = np.array(
        [
            [1000, 0, 320],  # fx, 0, cx
            [0, 1000, 240],  # 0, fy, cy
            [0, 0, 1],
        ],  # 0, 0, 1
        dtype=np.float32,
    )

    # Random distortion coefficients (k1, k2, p1, p2, k3)
    dist_coeffs = np.random.rand(5).astype(np.float32)  # k1, k2, p1, p2, k3

    # Time the solvePnP calls
    benchmark_func(
        num_runs,
        lambda: cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs),
    )


def benchmark_resize_img(num_runs=1000):
    # Make fake image
    img = np.random.rand(480, 640, 3).astype(np.float32)

    TARGET_SIZE = (416, 416)

    POSSIBLE_INTERPOLATIONS = [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_LANCZOS4,
        cv2.INTER_LINEAR_EXACT,
        cv2.INTER_NEAREST_EXACT,
    ]

    NAME_LUT = {
        cv2.INTER_NEAREST: "INTER_NEAREST",
        cv2.INTER_LINEAR: "INTER_LINEAR",
        cv2.INTER_CUBIC: "INTER_CUBIC",
        cv2.INTER_AREA: "INTER_AREA",
        cv2.INTER_LANCZOS4: "INTER_LANCZOS4",
        cv2.INTER_LINEAR_EXACT: "INTER_LINEAR_EXACT",
        cv2.INTER_NEAREST_EXACT: "INTER_NEAREST_EXACT",
    }

    for interpolation in POSSIBLE_INTERPOLATIONS:
        print(f"Interpolation: {NAME_LUT[interpolation]}")
        # Time the resize calls
        benchmark_func(
            num_runs,
            lambda: cv2.resize(img, TARGET_SIZE, interpolation=interpolation),
        )


def benchmark_resize_pil(num_runs=1000):
    from PIL import Image

    # Make fake image
    img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)  # Convert to uint8

    TARGET_SIZE = (416, 416)

    # Time the resize calls
    benchmark_func(num_runs, lambda: Image.fromarray(img).resize(TARGET_SIZE))


def benchmark_onnx_model(model_path: str, num_runs=1000):
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(model_path)
    # Get input name and shape
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    input_data = np.random.rand(*input_shape).astype(np.float32)

    benchmark_func(num_runs, lambda: session.run(None, {input_name: input_data}))


# benchmark_solve_pnp()
# benchmark_resize_img()
# benchmark_resize_pil()
benchmark_onnx_model("HUST_model.onnx")
benchmark_onnx_model("Generic_model.onnx", 100)

