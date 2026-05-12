import cv2
import numpy as np
import os

output_path = "test_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_path, fourcc, 30, (640, 480))

for i in range(90):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    x = int((i / 90) * 600) + 20
    cv2.circle(frame, (x, 240), 40, (0, 255, 0), -1)
    cv2.putText(frame, f"Frame {i+1}/90", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    writer.write(frame)

writer.release()
print("Video created")

cap = cv2.VideoCapture(output_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Total frames : {total_frames}")
print(f"FPS          : {fps}")
print(f"Resolution   : {width}x{height}")
print(f"Duration     : {total_frames/fps:.2f} seconds")

os.makedirs("frames/raw", exist_ok=True)

for i in range(5):
    ret, frame = cap.read()
    if ret:
        path = f"frames/raw/frame_{i+1:04d}.png"
        cv2.imwrite(path, frame)
        print(f"Saved: {path} | shape: {frame.shape}")

cap.release()
print(f"Total numbers per frame = {height * width * 3:,}")