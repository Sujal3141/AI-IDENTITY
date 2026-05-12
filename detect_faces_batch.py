import cv2
import numpy as np
from mediapipe import solutions

mp_face_mesh = solutions.face_mesh
mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles

def detect_faces_batch(frames_dir, output_dir, num_frames=10):
    """
    Run face detection on first N frames and save annotated results.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Get first N frames sorted by name
    all_frames = sorted([
        f for f in os.listdir(frames_dir) if f.endswith(".jpg")
    ])[:num_frames]

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    results_log = []

    for filename in all_frames:
        path = os.path.join(frames_dir, filename)
        frame = cv2.imread(path)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            print(f"{filename}: NO FACE DETECTED")
            results_log.append((filename, False, None))
            continue

        landmarks = results.multi_face_landmarks[0]
        print(f"{filename}: face detected, {len(landmarks.landmark)} landmarks")
        results_log.append((filename, True, landmarks))

        # Draw and save
        annotated = frame.copy()
        mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

        cv2.imwrite(os.path.join(output_dir, filename), annotated)

    face_mesh.close()

    # Summary
    detected = sum(1 for _, found, _ in results_log if found)
    print(f"\nSummary: {detected}/{len(all_frames)} frames had a face detected")


detect_faces_batch(
    frames_dir="frames",
    output_dir="frames_annotated",
    num_frames=10
)