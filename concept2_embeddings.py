import torch
import numpy as np
from PIL import Image
import cv2


from facenet_pytorch import InceptionResnetV1, MTCNN
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")

detector = MTCNN(device=device) #detects faces and returns aligned face tensors
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device) #generates 512-dim embeddings from aligned face tensors
print("Models loaded")

def get_embedding(image_path):
    # Load image
    img = Image.open(image_path).convert('RGB')
    print(f"Loaded: {image_path} | size: {img.size}")

    # Detect and crop face automatically
    face_tensor = detector(img)

    if face_tensor is None:
        print("No face detected in image")
        return None

    print(f"Face detected | tensor shape: {face_tensor.shape}")

    # Add batch dimension and send to GPU
    face_tensor = face_tensor.unsqueeze(0).to(device)

    # Extract embedding
    with torch.no_grad():
        embedding = embedder(face_tensor)

    embedding = embedding.cpu().numpy()[0]
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 8 values: {embedding[:8].round(4)}")
    return embedding
emb1 = get_embedding('image1.jpg')

if emb1 is not None:
    # L2 norm should be close to 1.0
    norm = np.linalg.norm(emb1)
    print(f"\nEmbedding L2 norm: {norm:.4f}")
    print("(close to 1.0 means properly normalized)")

    # Self similarity — same image compared to itself
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim = cosine_similarity(emb1, emb1)
    print(f"\nSelf similarity score: {sim:.4f}")
    print("(should be exactly 1.0 — same face vs itself)")

print("\nDone")

emb2 = get_embedding('image2.jpg')

if emb1 is not None and emb2 is not None:
    sim_different = cosine_similarity(emb1, emb2)
    print(f"\nSimilarity score:")
    print(f"  image1 vs image1 (same person): {cosine_similarity(emb1, emb1):.4f}")
    print(f"  image1 vs image2 (diff person): {sim_different:.4f}")
    print(f"\nDifference: {1.0 - sim_different:.4f}")