import cv2
import os
import subprocess
import torch
from fairseq.data.dictionary import Dictionary
from infer_rvc_python import BaseLoader

# Allow fairseq Dictionary to be loaded safely
torch.serialization.add_safe_globals([Dictionary])

# ── PATHS ──────────────────────────────────────────────────────────────────
SWAPPED_FRAMES_DIR = r"C:\Sujal Workspace\Projects\AI IDENTITY\frames\swapped_pro"
AUDIO_INPUT        = r"C:\Sujal Workspace\Projects\AI IDENTITY\input\audio\audio.wav"
MODEL_PATH         = r"C:\Sujal Workspace\Projects\AI IDENTITY\Yun Seong - Weights Model\model.pth"
INDEX_PATH         = r"C:\Sujal Workspace\Projects\AI IDENTITY\Yun Seong - Weights Model\model.index"
TARGET_VIDEO       = r"C:\Sujal Workspace\Projects\AI IDENTITY\target_video.mp4"
OUTPUT_DIR         = r"C:\Sujal Workspace\Projects\AI IDENTITY\output"
AUDIO_CONVERTED    = r"C:\Sujal Workspace\Projects\AI IDENTITY\input\audio\audio.wav"
VIDEO_NO_AUDIO     = os.path.join(OUTPUT_DIR, "video_no_audio.mp4")
FINAL_OUTPUT       = os.path.join(OUTPUT_DIR, "final_output.mp4")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── STEP 1: GET FPS FROM ORIGINAL VIDEO ───────────────────────────────────
print("\n[1/4] Reading video info...")
cap = cv2.VideoCapture(TARGET_VIDEO)
FPS = cap.get(cv2.CAP_PROP_FPS)
cap.release()
print(f"      FPS: {FPS}")

# ── STEP 2: VOICE CONVERSION ───────────────────────────────────────────────
print("\n[2/4] Converting voice with RVC...")

converter = BaseLoader(only_cpu=False, hubert_path=None, rmvpe_path=None)

# Configure the model
converter.apply_conf(
    tag="yun_seong",
    file_model=MODEL_PATH,
    pitch_algo="rmvpe",
    pitch_lvl=0,
    file_index=INDEX_PATH,
    index_influence=0.75,
    respiration_median_filtering=3,
    envelope_ratio=0.25,
    consonant_breath_protection=0.33,
    resample_sr=0
)

# Run inference — audio_files is a list, tag_list maps each file to a model tag
converter(
    audio_files=[AUDIO_INPUT],
    tag_list=["yun_seong"],
    overwrite=True,
    parallel_workers=1,
)

print(f"      Saved: {AUDIO_CONVERTED}")

# ── STEP 3: ASSEMBLE FRAMES INTO VIDEO (no audio) ─────────────────────────
print("\n[3/4] Assembling frames into video...")

first_frame = cv2.imread(os.path.join(SWAPPED_FRAMES_DIR, "frame_00000.png"))
h, w = first_frame.shape[:2]
print(f"      Frame size: {w}x{h}")

frames = sorted([
    f for f in os.listdir(SWAPPED_FRAMES_DIR)
    if f.endswith(".png") or f.endswith(".jpg")
])
print(f"      Total frames: {len(frames)}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(VIDEO_NO_AUDIO, fourcc, FPS, (w, h))

for i, fname in enumerate(frames):
    frame = cv2.imread(os.path.join(SWAPPED_FRAMES_DIR, fname))
    if frame is None:
        print(f"      Warning: could not read {fname}, skipping")
        continue
    writer.write(frame)
    if i % 100 == 0:
        print(f"      Written {i}/{len(frames)} frames...")

writer.release()
print(f"      Saved: {VIDEO_NO_AUDIO}")

# ── STEP 4: MERGE VIDEO + CONVERTED AUDIO ─────────────────────────────────
# ── STEP 4: MERGE VIDEO + CONVERTED AUDIO ─────────────────────────────────
print("\n[4/4] Merging video and audio with ffmpeg...")

# Find ffmpeg — try common locations
import shutil
ffmpeg_path = shutil.which("ffmpeg")

if ffmpeg_path is None:
    # Common conda/windows locations
    candidates = [
        r"C:\ProgramData\anaconda3\envs\ai-identity\Library\bin\ffmpeg.exe",
        r"C:\ProgramData\anaconda3\Library\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            ffmpeg_path = c
            break

if ffmpeg_path is None:
    print("❌ ffmpeg not found. Install with: conda install ffmpeg -c conda-forge")
else:
    print(f"      Using ffmpeg: {ffmpeg_path}")
    cmd = [
        ffmpeg_path, "-y",
        "-i", VIDEO_NO_AUDIO,
        "-i", AUDIO_CONVERTED,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        FINAL_OUTPUT
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("ffmpeg error:")
        print(result.stderr)
    else:
        print(f"      Saved: {FINAL_OUTPUT}")
        print("\n✅ Done! Final video: " + FINAL_OUTPUT)