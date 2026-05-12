import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import os

mp_face_mesh = solutions.face_mesh
mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles

def detect_face(image_path):
    """
    Detect face and extract 468 landmarks from a single image.
    """
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,    # True = single image (not video stream)
        max_num_faces=1,
        refine_landmarks=True,     # gives 478 points incl. eyes/lips detail
        min_detection_confidence=0.5
    )

    # Load frame
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not load: {image_path}")
        return

    h, w = frame.shape[:2]

    # MediaPipe expects RGB, OpenCV loads BGR
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print("No face detected.")
        return

    print(f"Face detected! Landmarks found: {len(results.multi_face_landmarks[0].landmark)}")

    # Draw landmarks on a copy of the frame
    annotated = frame.copy()
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

    cv2.imwrite("face_detected.jpg", annotated)
    print(f"Saved to face_detected.jpg")

    face_mesh.close()


# Test on first frame
detect_face("frames/frame_00000.jpg")