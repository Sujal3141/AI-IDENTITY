import cv2
import numpy as np
from mediapipe import solutions

mp_face_mesh = solutions.face_mesh

def extract_landmarks(image_path):
    """
    Extract 478 landmarks from a single image.
    Returns landmark array of shape (478, 2) in pixel coordinates.
    """
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not load: {image_path}")
        return None

    h, w = frame.shape[:2]
    print(f"Image size: {w}x{h}")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print("No face detected in source image.")
        return None

    landmarks = results.multi_face_landmarks[0]

    # Convert normalized coords → pixel coords
    points = np.array([
        [lm.x * w, lm.y * h]
        for lm in landmarks.landmark
    ], dtype=np.float32)

    print(f"Extracted {len(points)} landmarks")
    print(f"Sample point [0]: {points[0]}")
    print(f"Sample point [10]: {points[10]}")

    face_mesh.close()
    return points


source_landmarks = extract_landmarks(
    r"C:\Sujal Workspace\Projects\AI IDENTITY\face\reference\ai_female_seed1024.png"
)