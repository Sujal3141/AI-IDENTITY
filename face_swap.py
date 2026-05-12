import cv2
import numpy as np
import os
import torch
from torchvision.transforms import Normalize
from tqdm import tqdm

# Face Models
import insightface
from insightface.app import FaceAnalysis

# Optional GFPGAN
try:
    from gfpgan import GFPGANer
    HAS_GFPGAN = True
except ImportError:
    HAS_GFPGAN = False
    print("WARNING: GFPGAN not installed. Output will remain low resolution.")

from facexlib.parsing import init_parsing_model

print("=" * 70)
print("PHASE 7 — THE GOD-TIER VIDEO PIPELINE (All Fixes Applied)")
print("=" * 70)

# ── 1. CONFIGURATION (CHANGE THESE PATHS TO YOUR FOLDERS) ────
FRAMES_DIR = r"C:\Sujal Workspace\Projects\AI IDENTITY\frames\raw"
OUTPUT_DIR = r"C:\Sujal Workspace\Projects\AI IDENTITY\frames\swapped_pro"
SOURCE_IMAGE_PATH = r"C:\Sujal Workspace\Projects\AI IDENTITY\source2.png"
MODEL_ROOT = r"C:\Users\n6989\.insightface"
SWAPPER_MODEL = os.path.join(MODEL_ROOT, "models", "inswapper_128.onnx")

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Executing mathematical operations on: {DEVICE.upper()}")

# ── 2. CAMERA NOISE INJECTOR ─────────────────────────────────
def match_camera_noise(original_crop, fake_crop, intensity=0.8):
    orig_gray = cv2.cvtColor(original_crop, cv2.COLOR_BGR2GRAY)
    orig_blur = cv2.GaussianBlur(orig_gray, (5, 5), 0)
    
    # Extract high-frequency noise
    noise_map = cv2.subtract(orig_gray, orig_blur)
    noise_std = np.std(noise_map)
    
    # Generate synthetic noise
    synthetic_noise = np.random.normal(0, noise_std * intensity, fake_crop.shape).astype(np.float32)
    noisy_fake = fake_crop.astype(np.float32) + synthetic_noise
    
    return np.clip(noisy_fake, 0, 255).astype(np.uint8)

# ── 3. THE BISENET SEMANTIC MASK ─────────────────────────────
def generate_bisenet_mask(orig_crop, parse_net):
    h, w = orig_crop.shape[:2]
    
    input_img = cv2.resize(orig_crop, (512, 512))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = input_img.transpose((2, 0, 1)).astype(np.float32) / 255.0
    
    input_tensor = torch.from_numpy(input_img).unsqueeze(0).to(DEVICE)
    normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    input_tensor = normalize(input_tensor)
    
    with torch.no_grad():
        out = parse_net(input_tensor)[0]
        out = out.squeeze(0).cpu().numpy().argmax(0)
        
    mask = np.zeros_like(out, dtype=np.float32)
    
    # 1:Skin, 2/3:Brows, 4/5:Eyes, 7/8:Ears, 10:Nose, 11-13:Mouth, 14:Neck
    swap_classes = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]
    
    for c in swap_classes:
        mask[out == c] = 1.0
        
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask = cv2.resize(mask, (w, h))
    return np.repeat(mask[:, :, np.newaxis], 3, axis=2)

# ── 4. THE ULTIMATE BLEND (OCCLUSION + NOISE + COLOR) ────────
def professional_blend(original_frame, fake_frame, bbox, parse_net, target_face):
    img_h, img_w = original_frame.shape[:2]
    
    # --- A. EXPAND THE BOUNDING BOX ---
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    
    pad_x = int(w * 0.25)
    pad_top = int(h * 0.40)
    pad_bottom = int(h * 0.15)
    
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_top))
    x2 = min(img_w, int(x2 + pad_x))
    y2 = min(img_h, int(y2 + pad_bottom))
    
    orig_crop = original_frame[y1:y2, x1:x2]
    fake_crop = fake_frame[y1:y2, x1:x2]
    
    # --- B. GENERATE MASKS & INTERSECTION ---
    bisenet_mask = generate_bisenet_mask(orig_crop, parse_net)
    
    landmark_mask_full = np.zeros((img_h, img_w), dtype=np.float32)
    hull = cv2.convexHull(target_face.landmark_2d_106.astype(np.int32))
    cv2.fillConvexPoly(landmark_mask_full, hull, 1.0)
    
    landmark_crop = landmark_mask_full[y1:y2, x1:x2]
    landmark_crop = np.repeat(landmark_crop[:, :, np.newaxis], 3, axis=2)
    
    kernel = np.ones((15, 15), np.uint8)
    landmark_crop = cv2.erode(landmark_crop, kernel, iterations=1)
    
    combined_mask = bisenet_mask * landmark_crop
    combined_mask = cv2.GaussianBlur(combined_mask, (21, 21), 0)
    
    # --- C. BOOLEAN LAB COLOR TRANSFER ---
    orig_lab = cv2.cvtColor(orig_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    fake_lab = cv2.cvtColor(fake_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    skin_bool = combined_mask[:,:,0] > 0.5 
    corrected_lab = fake_lab.copy()
    
    if np.any(skin_bool):
        for c in range(3):
            o_mean = np.mean(orig_lab[:,:,c][skin_bool])
            o_std  = np.std(orig_lab[:,:,c][skin_bool])
            f_mean = np.mean(fake_lab[:,:,c][skin_bool])
            f_std  = np.std(fake_lab[:,:,c][skin_bool])
            
            if f_std > 0:
                corrected_lab[:,:,c] = ((fake_lab[:,:,c] - f_mean) * (o_std / f_std) + o_mean)
                
    corrected_lab = np.clip(corrected_lab, 0, 255)
    fake_crop_colored = cv2.cvtColor(corrected_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # --- D. INJECT CAMERA NOISE ---
    fake_crop_colored = match_camera_noise(orig_crop, fake_crop_colored, intensity=0.8)
    
    # --- E. FINAL ALPHA BLEND ---
    orig_crop_float = orig_crop.astype(np.float32)
    fake_crop_float = fake_crop_colored.astype(np.float32)
    
    final_crop = (fake_crop_float * combined_mask) + (orig_crop_float * (1.0 - combined_mask))
    
    final_frame = original_frame.copy()
    final_frame[y1:y2, x1:x2] = final_crop.astype(np.uint8)
    
    return final_frame

# ── 5. BOOT UP NEURAL NETWORKS ───────────────────────────────
print("\nBooting up models...")
app = FaceAnalysis(name='buffalo_l', root=MODEL_ROOT)
app.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model(SWAPPER_MODEL, download=False, download_zip=False)

if HAS_GFPGAN:
    upscaler = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None
    )

parse_net = init_parsing_model(model_name='bisenet', device=DEVICE)

# ── 6. PROCESS SOURCE IDENTITY ───────────────────────────────
source_img = cv2.imread(SOURCE_IMAGE_PATH)
if source_img is None:
    print(f"ERROR: Cannot load {SOURCE_IMAGE_PATH}")
    exit()
source_faces = app.get(source_img)
if not source_faces:
    print("ERROR: No face detected in source image.")
    exit()
source_face = source_faces[0]

# ── 7. THE MAIN VIDEO LOOP ───────────────────────────────────
print("\nInitiating massive frame processing loop...")
frame_files = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])

oom_warnings = 0

for frame_file in tqdm(frame_files, desc="Rendering Frames"):
    frame_path  = os.path.join(FRAMES_DIR, frame_file)
    output_path = os.path.join(OUTPUT_DIR, frame_file)

    original_frame = cv2.imread(frame_path)
    if original_frame is None:
        continue

    # Alignment
    faces = app.get(original_frame)
    if len(faces) == 0:
        cv2.imwrite(output_path, original_frame)
        continue
    target_face = max(faces, key=lambda f: f.bbox[2] - f.bbox[0])

    # Latent Swap
    swapped_frame = swapper.get(original_frame, target_face, source_face, paste_back=True)

    # GFPGAN Upscale with OOM Protection
    hi_res_swapped_frame = swapped_frame.copy()
    if HAS_GFPGAN:
        try:
            _, _, hi_res_swapped_frame = upscaler.enhance(
                swapped_frame, has_aligned=False, only_center_face=False, paste_back=True
            )
        except RuntimeError:
            hi_res_swapped_frame = swapped_frame.copy()
            torch.cuda.empty_cache()
            if oom_warnings < 3:
                print(f"\n[WARNING] VRAM OOM on {frame_file}. Falling back to 128x128.")
                oom_warnings += 1

    # Professional Matrix Blend
    final_result = professional_blend(original_frame, hi_res_swapped_frame, target_face.bbox, parse_net, target_face)

    # Save
    cv2.imwrite(output_path, final_result)

print(f"\n✅ Render Complete! All frames processed and saved to:\n{OUTPUT_DIR}")
print("Run the FFmpeg command to compile these frames back into a video.")