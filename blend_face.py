import cv2
import numpy as np
from mediapipe import solutions

mp_face_mesh = solutions.face_mesh


def get_landmarks(image, face_mesh):
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    return np.array([
        [lm.x * w, lm.y * h]
        for lm in results.multi_face_landmarks[0].landmark
    ], dtype=np.float32)


def warp_triangle(src_img, dst_img, src_tri, dst_tri):
    src_rect = cv2.boundingRect(np.float32([src_tri]))
    dst_rect = cv2.boundingRect(np.float32([dst_tri]))

    src_cropped = src_img[
        src_rect[1]:src_rect[1]+src_rect[3],
        src_rect[0]:src_rect[0]+src_rect[2]
    ]

    src_tri_offset = [(p[0]-src_rect[0], p[1]-src_rect[1]) for p in src_tri]
    dst_tri_offset = [(p[0]-dst_rect[0], p[1]-dst_rect[1]) for p in dst_tri]

    M = cv2.getAffineTransform(
        np.float32(src_tri_offset),
        np.float32(dst_tri_offset)
    )

    warped = cv2.warpAffine(
        src_cropped, M,
        (dst_rect[2], dst_rect[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    mask = np.zeros((dst_rect[3], dst_rect[2]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_tri_offset), 255)

    dst_patch = dst_img[
        dst_rect[1]:dst_rect[1]+dst_rect[3],
        dst_rect[0]:dst_rect[0]+dst_rect[2]
    ]
    dst_patch = cv2.bitwise_and(dst_patch, dst_patch, mask=cv2.bitwise_not(mask))
    dst_patch = dst_patch + cv2.bitwise_and(warped, warped, mask=mask)
    dst_img[
        dst_rect[1]:dst_rect[1]+dst_rect[3],
        dst_rect[0]:dst_rect[0]+dst_rect[2]
    ] = dst_patch


def color_correct_face(source_face, target_region):
    """Shift AI face colors to match target skin tone using LAB color space."""
    source_lab = cv2.cvtColor(source_face, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB).astype(np.float32)

    for i in range(3):
        src_mean, src_std = source_lab[:,:,i].mean(), source_lab[:,:,i].std()
        tgt_mean, tgt_std = target_lab[:,:,i].mean(), target_lab[:,:,i].std()
        if src_std > 0:
            source_lab[:,:,i] = (source_lab[:,:,i] - src_mean) * (tgt_std / src_std) + tgt_mean

    source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)


def blend_face_into_frame(warped_face, frame, tgt_landmarks):
    """
    After triangle warping, seamlessly blend the warped region into the frame.
    warped_face : the output image from warp_triangle steps (same size as frame)
    frame       : the original unmodified target frame
    tgt_landmarks: (N,2) float32 array from MediaPipe
    """
    # convexHull needs shape (N,1,2) int32
    pts = tgt_landmarks.astype(np.int32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts)

    # --- Color correction: compare warped face region vs original frame region ---
    rough_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(rough_mask, hull, 255)

    target_region = cv2.bitwise_and(frame, frame, mask=rough_mask)
    warped_region = cv2.bitwise_and(warped_face, warped_face, mask=rough_mask)
    corrected = color_correct_face(warped_region, target_region)

    # --- Soft mask: blur the hull boundary for feathered edges ---
    soft_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(soft_mask, hull, 255)
    soft_mask = cv2.GaussianBlur(soft_mask, (21, 21), 0)

    # --- Seamless clone: Poisson blending into original frame ---
    x, y, w, h = cv2.boundingRect(hull)
    center = (x + w // 2, y + h // 2)

    result = cv2.seamlessClone(
        corrected,       # color-corrected warped face (full frame size)
        frame,           # original untouched frame
        soft_mask,       # soft hull mask
        center,
        cv2.NORMAL_CLONE
    )
    return result


def warp_face(source_path, target_path, output_path):
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    source = cv2.imread(source_path)
    target = cv2.imread(target_path)

    print("Extracting source landmarks...")
    src_landmarks = get_landmarks(source, face_mesh)
    if src_landmarks is None:
        print("No face in source image.")
        return

    print("Extracting target landmarks...")
    tgt_landmarks = get_landmarks(target, face_mesh)
    if tgt_landmarks is None:
        print("No face in target frame.")
        return

    # Delaunay triangulation on source landmarks
    src_h, src_w = source.shape[:2]
    subdiv = cv2.Subdiv2D((0, 0, src_w, src_h))
    for point in src_landmarks:
        subdiv.insert((float(point[0]), float(point[1])))

    triangles = subdiv.getTriangleList().astype(np.float32)
    src_points_list = [tuple(p) for p in src_landmarks]

    def find_index(pt, points_list):
        for i, p in enumerate(points_list):
            if abs(p[0]-pt[0]) < 1.0 and abs(p[1]-pt[1]) < 1.0:
                return i
        return -1

    # --- Step 1: Triangle warp into output (hard edges, no blending yet) ---
    output = target.copy()
    skipped = 0

    for tri in triangles:
        pts_src = [(tri[0],tri[1]), (tri[2],tri[3]), (tri[4],tri[5])]
        indices = [find_index(pt, src_points_list) for pt in pts_src]
        if -1 in indices:
            skipped += 1
            continue
        pts_tgt = [tuple(tgt_landmarks[i]) for i in indices]
        warp_triangle(source, output, pts_src, pts_tgt)

    print(f"Skipped {skipped} triangles (out of bounds)")

    # --- Step 2: Blend warped result seamlessly into original frame ---
    print("Blending warped face into frame...")
    blended = blend_face_into_frame(output, target, tgt_landmarks)

    cv2.imwrite(output_path, blended)
    print(f"Saved to {output_path}")

    face_mesh.close()


# Test on first frame
warp_face(
    source_path=r"C:\Sujal Workspace\Projects\AI IDENTITY\face\reference\ai_female_seed1024.png",
    target_path="frames/frame_00000.jpg",
    output_path="warp_test.jpg"
)