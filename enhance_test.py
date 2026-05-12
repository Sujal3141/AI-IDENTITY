import cv2
import numpy as np
import os
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

print("=" * 50)
print("PHASE B — FRAME ENHANCEMENT")
print("=" * 50)

FRAMES_DIR = r"C:\Sujal Workspace\Projects\AI IDENTITY\frames\raw"
ENHANCED_DIR = r"C:\Sujal Workspace\Projects\AI IDENTITY\frames\enhanced"

os.makedirs(ENHANCED_DIR, exist_ok=True)

# Load Real-ESRGAN model
print("Loading Real-ESRGAN model...")
model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=6,
    num_grow_ch=32,
    scale=4
)

upsampler = RealESRGANer(
    scale=4,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animatesr-x4.pth',
    model=model,
    tile=256,        # process in tiles → saves VRAM
    tile_pad=10,
    pre_pad=0,
    half=True        # fp16 → saves VRAM
)
print("Model loaded\n")

# Test on single frame first
print("Testing on frame_00000.png...")
test_frame = cv2.imread(
    os.path.join(FRAMES_DIR, "frame_00000.png")
)

# Enhance
output, _ = upsampler.enhance(test_frame, outscale=2)

cv2.imwrite("enhanced_test.jpg", output)
print(f"Original size : {test_frame.shape[:2]}")
print(f"Enhanced size : {output.shape[:2]}")
print("Saved: enhanced_test.jpg")
print("Open and check quality before continuing\n")

confirm = input("Does enhancement look good? (yes/no): ")
if confirm.lower() != 'yes':
    print("Adjust settings and try again")
    exit()

# Process all frames
print("Enhancing all frames...")
frame_files = sorted(os.listdir(FRAMES_DIR))
failed = 0
success = 0

for frame_file in tqdm(frame_files):
    frame_path = os.path.join(FRAMES_DIR, frame_file)
    output_path = os.path.join(ENHANCED_DIR, frame_file)

    frame = cv2.imread(frame_path)
    if frame is None:
        failed += 1
        continue

    try:
        enhanced, _ = upsampler.enhance(frame, outscale=2)
        cv2.imwrite(output_path, enhanced)
        success += 1
    except Exception as e:
        # If enhancement fails, save original
        cv2.imwrite(output_path, frame)
        failed += 1

print(f"\nResults:")
print(f"  Enhanced : {success} frames")
print(f"  Failed   : {failed} frames")
print(f"  Saved to : {ENHANCED_DIR}")
print(f"\n✅ Enhancement complete")
print("Now update FRAMES_DIR in phase5_face_swap.py")
print(f"to: {ENHANCED_DIR}")