import os
import cv2
from tqdm import tqdm

def combine_videos():
    # Get the current directory
    current_dir = os.getcwd()

    # Find all MP4 files in the current directory
    mp4_files = [f for f in os.listdir(current_dir) if f.endswith('.mp4')]
    
    # If no MP4 files found, exit
    if not mp4_files:
        print("No MP4 files found in the current directory.")
        return

    # Initialize variables
    video_writer = None

    for mp4_file in tqdm(mp4_files):
        # Skip the combined video file
        if mp4_file == "combined_video.mp4":
            continue
      
        # Open the video file
        cap = cv2.VideoCapture(mp4_file)
        
        if not cap.isOpened():
            print(f"Error opening video file {mp4_file}. Skipping.")
            continue
        
        # Get the video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize the VideoWriter if it's not initialized
        if video_writer is None:
            output_file = os.path.join(current_dir, "combined_video.mp4")
            video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # Read and write frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_writer.write(frame)
        
        # Release the video capture object
        cap.release()

    # Release the VideoWriter
    if video_writer is not None:
        video_writer.release()

    print(f"Combined video saved as {output_file}")

if __name__ == "__main__":
    combine_videos()
