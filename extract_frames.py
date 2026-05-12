import cv2
import os
from tqdm import tqdm

# Config
VIDEO_PATH = r"C:\Sujal Workspace\Projects\AI IDENTITY\target_video.mp4"
FRAMES_DIR = r"C:\Sujal Workspace\Projects\AI IDENTITY\frames\raw"
AUDIO_PATH = r"C:\Sujal Workspace\Projects\AI IDENTITY\input\audio\audio.wav"
FFMPEG_PATH = r"C:\ProgramData\anaconda3\envs\ai-identity\Library\bin\ffmpeg.exe"

os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(AUDIO_PATH), exist_ok=True)

# Step 1 — Get video properties
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("=" * 50)
print("PHASE 4 — FRAME EXTRACTION")
print("=" * 50)
print(f"Video    : {VIDEO_PATH}")
print(f"FPS      : {fps}")
print(f"Frames   : {total_frames}")
print(f"Duration : {total_frames/fps:.2f} seconds")
print(f"Resolution: {width}x{height}")

# Step 2 — Extract frames
print(f"\nExtracting frames to {FRAMES_DIR}...")

frame_idx = 0
success = 0

with tqdm(total=total_frames) as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save as PNG — lossless, better for face swap
        path = os.path.join(FRAMES_DIR, f"frame_{frame_idx:05d}.png")
        cv2.imwrite(path, frame)

        frame_idx += 1
        success += 1
        pbar.update(1)

cap.release()
print(f"Extracted {success} frames successfully")

# Step 3 — Extract audio track
print(f"\nExtracting audio to {AUDIO_PATH}...")
import subprocess

result = subprocess.run([
    FFMPEG_PATH,
    "-i", VIDEO_PATH,
    "-vn",              # no video
    "-acodec", "pcm_s16le",  # WAV format
    "-ar", "44100",     # sample rate
    "-ac", "2",         # stereo
    "-y",               # overwrite
    AUDIO_PATH
], capture_output=True, text=True)

if os.path.exists(AUDIO_PATH):
    size = os.path.getsize(AUDIO_PATH)
    print(f"Audio extracted: {size/1024:.1f} KB")
else:
    print("Audio extraction failed — video may have no audio")
    print(result.stderr[:200])

# Step 4 — Verify frames
frame_files = os.listdir(FRAMES_DIR)
print(f"\nVerification:")
print(f"  Frames saved : {len(frame_files)}")
print(f"  First frame  : {frame_files[0]}")
print(f"  Last frame   : {sorted(frame_files)[-1]}")

# Step 5 — Check first frame
first_frame = cv2.imread(os.path.join(FRAMES_DIR, "frame_00000.png"))
print(f"  Frame shape  : {first_frame.shape}")
print(f"\n✅ Phase 4 complete")
print(f"Ready for face swap pipeline")