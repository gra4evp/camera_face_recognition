import torch
from facenet_pytorch import MTCNN


def detect_faces(frame, landmarks=False):
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = Image.fromarray(frame)

    # Если landmarks=True, то вернется кортеж длиной 3
    boxes, probs = mtcnn.detect(frame, landmarks=landmarks)
    return boxes, probs


def process_frame(frame, frame_count):
    boxes, probs = detect_faces(frame)
    detected_faces = {'boxes': [], 'images': [], 'filenames': []}
    if boxes is not None:
        for person_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(b) for b in box]
            face_img = frame[y1:y2, x1:x2]

            detected_faces['boxes'].append((x1, y1, x2, y2))
            detected_faces['images'].append(face_img)
            detected_faces['filenames'].append(f"face_frame{frame_count:04}_person{person_idx:02}.jpg")
    return detected_faces


DEVICE = device='cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=DEVICE)
