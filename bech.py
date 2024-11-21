import cv2
import numpy as np
import time

def benchmark_solve_pnp(num_runs=10000):
    # Define random 3D object points (3D points in the object space)
    object_points = np.random.rand(10, 3).astype(np.float32)  # 10 points, 3D
    
    # Define random 2D image points (2D projections of the 3D points)
    image_points = np.random.rand(10, 2).astype(np.float32)  # 10 points, 2D
    
    # Define a random camera matrix (intrinsics), usually 3x3
    camera_matrix = np.array([[1000, 0, 320],   # fx, 0, cx
                              [0, 1000, 240],   # 0, fy, cy
                              [0, 0, 1]],       # 0, 0, 1
                             dtype=np.float32)
    
    # Random distortion coefficients (k1, k2, p1, p2, k3)
    dist_coeffs = np.random.rand(5).astype(np.float32)  # k1, k2, p1, p2, k3
    
    # Time the solvePnP calls
    start_time = time.time_ns()
    
    for _ in range(num_runs):
        # Call solvePnP and get rotation and translation vectors
        retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    
    end_time = time.time_ns()
    
    # Calculate average time per run
    elapsed_time = (end_time - start_time) / 1e9  # Convert to seconds
    avg_time_per_run = elapsed_time / num_runs
    
    print(f"Benchmark complete: {num_runs} runs.")
    print(f"Total time: {elapsed_time:.10f} seconds")
    print(f"Average time per run: {avg_time_per_run:.5f} seconds")

# Example usage
benchmark_solve_pnp()
