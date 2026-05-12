# 🎭 AI-Identity

> **Complete AI-powered identity transformation pipeline** — face swap + voice conversion on video, frame by frame.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![InsightFace](https://img.shields.io/badge/InsightFace-inswapper__128-blueviolet?style=flat-square)](https://github.com/deepinsight/insightface)
[![GFPGAN](https://img.shields.io/badge/GFPGAN-v1.3-brightgreen?style=flat-square)](https://github.com/TencentARC/GFPGAN)
[![RVC](https://img.shields.io/badge/RVC-Voice%20Conversion-orange?style=flat-square)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

---

## 📌 What is AI-Identity?

**AI-Identity** is a research-grade, end-to-end video identity transformation system that:

1. **Swaps faces** in every frame of a target video using a source identity (real or AI-generated)
2. **Enhances the swap** with semantic segmentation masks, LAB color transfer, and camera noise injection for photorealism
3. **Converts the voice** in the audio track to match a trained RVC model
4. **Reassembles** everything into a final output video using FFmpeg

This project was built as a deep-dive into the internals of modern deepfake pipelines — understanding how face detection, latent-space swapping, perceptual blending, and voice synthesis interact in a production context.

---

## 🧠 Pipeline Architecture

```
Source Image (AI-generated face)
        │
        ▼
┌───────────────────────┐
│  InsightFace buffalo_l │  ← Face Detection + Embedding
│  + inswapper_128.onnx  │  ← Latent Face Swap
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  BiSeNet Segmentation  │  ← Semantic face mask (skin, eyes, nose, mouth)
│  Landmark Hull Mask    │  ← 106-point convex hull
│  Intersection + Blur   │  ← Combined soft mask
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  LAB Color Transfer    │  ← Match skin tone statistics
│  Camera Noise Injection│  ← Add realistic sensor noise
│  Alpha Blend (final)   │  ← Seamless composite
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  GFPGAN v1.3 Upscaler │  ← Face restoration + super-resolution
│  (with OOM fallback)   │
└───────────┬───────────┘
            │
            ▼
┌─────────────────────────────────────┐
│  RVC Voice Conversion (RMVPE pitch) │  ← Voice identity swap on audio
└───────────────────────┬─────────────┘
                         │
                         ▼
              FFmpeg: frames + audio → final_output.mp4
```

---

## ✨ Key Technical Features

| Feature | Detail |
|--------|--------|
| **Face Swap Engine** | `inswapper_128.onnx` via InsightFace — latent space identity injection |
| **Semantic Masking** | BiSeNet parses 14 face classes; intersection with 106-point landmark hull |
| **Color Correction** | Per-channel LAB mean/std transfer on skin-masked region only |
| **Noise Matching** | High-frequency noise extracted from original frame and re-injected into swap |
| **Upscaling** | GFPGAN v1.3 face restoration with CUDA OOM guard and graceful fallback |
| **Voice Conversion** | RVC with RMVPE pitch algorithm + fairseq HuBERT feature extraction |
| **Assembly** | OpenCV `VideoWriter` → FFmpeg AAC mux → final MP4 |

---

## 🗂️ Project Structure

```
AI-Identity/
├── phase7_face_swap.py       # Main frame-by-frame face swap pipeline
├── voice_convert.py          # RVC voice conversion + FFmpeg assembly
├── frames/
│   ├── raw/                  # Input extracted frames
│   └── swapped_pro/          # Output blended frames
├── input/
│   └── audio/
│       └── audio.wav         # Source audio for voice conversion
├── output/
│   ├── video_no_audio.mp4    # Intermediate: assembled frames
│   └── final_output.mp4      # Final output: video + converted voice
├── source2.png               # Source identity image (AI-generated face)
├── target_video.mp4          # Original target video
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- FFmpeg installed and on PATH (`conda install ffmpeg -c conda-forge`)

### Install Dependencies

```bash
git clone https://github.com/YOUR_USERNAME/AI-Identity.git
cd AI-Identity
pip install -r requirements.txt
```

### Download Models

| Model | Source | Place at |
|-------|--------|----------|
| `inswapper_128.onnx` | [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo) | `~/.insightface/models/` |
| `buffalo_l` face pack | Auto-downloaded by InsightFace | `~/.insightface/models/buffalo_l/` |
| `GFPGANv1.3.pth` | [GFPGAN Releases](https://github.com/TencentARC/GFPGAN/releases/tag/v1.3.0) | Auto-downloaded at runtime |
| RVC model `.pth` + `.index` | Your trained RVC weights | Set path in `voice_convert.py` |

---

## 🚀 Usage

### Step 1 — Extract frames from your target video

```bash
ffmpeg -i target_video.mp4 frames/raw/frame_%05d.png
```

### Step 2 — Run face swap pipeline

Edit paths at the top of `phase7_face_swap.py`:

```python
FRAMES_DIR        = r"path/to/frames/raw"
OUTPUT_DIR        = r"path/to/frames/swapped_pro"
SOURCE_IMAGE_PATH = r"path/to/source.png"
```

Then run:

```bash
python phase7_face_swap.py
```

### Step 3 — Run voice conversion + final assembly

Edit paths in `voice_convert.py`:

```python
AUDIO_INPUT  = r"path/to/audio.wav"
MODEL_PATH   = r"path/to/model.pth"
INDEX_PATH   = r"path/to/model.index"
```

Then run:

```bash
python voice_convert.py
```

Output: `output/final_output.mp4`

---

## 🔬 How the Blending Works (Phase 7 Details)

The core challenge in face swapping is making the swap **invisible**. This pipeline uses a 5-stage blend:

1. **Expanded bounding box** — the face crop extends 25% on sides and 40% above to capture hairline and forehead
2. **BiSeNet semantic mask** — 14 facial classes (skin, brows, eyes, ears, nose, mouth, neck) are parsed at 512×512 and resized back
3. **Landmark convex hull** — InsightFace's 106 2D landmarks are eroded and intersected with the BiSeNet mask
4. **LAB color transfer** — only pixels inside the mask participate in mean/std matching per channel (L, A, B), preserving global lighting
5. **Camera noise injection** — high-freq noise from the original frame is extracted via Gaussian difference and added back to the swap, eliminating the "too clean" CGI look

---

## ⚠️ Ethical Use & Disclaimer

This project was created **for research and educational purposes only** to understand the technical internals of modern face-swap and voice-conversion pipelines.

- **Do not** use this to create non-consensual deepfakes of real people
- **Do not** use this to spread misinformation or impersonate individuals
- All test footage used in development was **self-recorded** by the author
- The author takes **no responsibility** for misuse of this codebase

Deepfake technology raises serious ethical questions. Understanding how it works is the first step toward building detection systems and media authentication tools.

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Face detection | `insightface` (buffalo_l) |
| Face swap | `inswapper_128.onnx` |
| Segmentation | `facexlib` BiSeNet |
| Enhancement | `GFPGAN v1.3` |
| Voice features | `fairseq` HuBERT |
| Voice conversion | `infer-rvc-python` (RMVPE) |
| Video I/O | `OpenCV`, `FFmpeg` |
| Deep learning | `PyTorch` (CUDA) |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Sujal** — B.E. CSE @ SLIET Punjab | NLP & ML Researcher  
[![GitHub](https://img.shields.io/badge/GitHub-Sujal3141-181717?style=flat-square&logo=github)](https://github.com/YOUR_USERNAME)

---

> *"The best deepfake is the one you can't detect."*
