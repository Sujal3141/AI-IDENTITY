import torch
from diffusers import StableDiffusionPipeline
import os

print("=" * 50)
print("CONCEPT 3 — GENERATING AI FEMALE IDENTITY")
print("=" * 50)

# Local model path — no internet needed
MODEL_PATH = r"C:\Sujal Workspace\Projects\AI IDENTITY\models\v1-5-pruned-emaonly.safetensors"


# Verify file exists before loading
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Make sure you moved the downloaded file to models/ folder")
    exit()

print(f"Loading model from: {MODEL_PATH}")
print("This takes ~30 seconds...\n")

# Load from local single file
pipe = StableDiffusionPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
)

# Move to GPU
pipe = pipe.to("cuda")
pipe.vae = pipe.vae.to(torch.float32)
# VRAM optimizations
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

print("Model loaded")
print(f"VRAM used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Customize these to your preference
PROMPT = (
    "portrait of a young woman in her mid 20s, "
    "front facing, neutral expression, "
    "blue eyes, blonde hair, fair skin, "
    "soft lighting, photorealistic, high detail, "
    "clean background, face centered, "
    "sharp focus, natural makeup, "
    "perfect symmetric eyes, detailed pupils, "
    "realistic iris texture, white sclera, "
    "eyes looking straight at camera, "
    "full head portrait, hair fully visible above forehead, "
    "top of head visible, medium shot, "
    "Canon 85mm portrait photography"
)

NEGATIVE_PROMPT = (
    "blurry, distorted, cartoon, anime, "
    "side view, turned head, sunglasses, "
    "multiple faces, bad anatomy, watermark, "
    "dark skin, brown eyes, dark hair, "
    "old, wrinkles, child, "
    "crossed eyes, lazy eye, asymmetric eyes, "
    "deformed eyes, extra eyes, closed eyes, "
    "blurry eyes, cartoonish eyes, "
    "cropped head, cropped hair, forehead cut off"
)

# Generate 4 variations to choose from
os.makedirs("face/reference", exist_ok=True)

seeds = [1024, 2048, 3072, 4096]

for seed in seeds:
    print(f"Generating seed {seed}...")

    generator = torch.Generator(device="cuda").manual_seed(seed)

    with torch.autocast("cuda"):
        result = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=40,
            guidance_scale=8.5,
            width=512,
            height=768,
            generator=generator
        )

    path = f"face/reference/ai_female_seed{seed}.png"
    result.images[0].save(path)
    print(f"Saved: {path}")

print(f"\nVRAM peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print("\nDone. Open face/reference/ folder and pick your favorite face.")
print("Note the seed number — that is your AI female identity.")