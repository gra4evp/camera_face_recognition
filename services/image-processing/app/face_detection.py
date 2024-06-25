import torch
from facenet_pytorch import MTCNN
from typing import Tuple, List, Optional, Dict, Any


class BaseFaceDetector:
    device: str | None
    model: Any

    def __init__(self, identify_device=True):
        self.device = None
        if identify_device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = None

    def detect_faces(self, frame, landmarks=False):
        raise NotImplementedError("This method should be overridden in subclasses")

    def process_frame(self, frame, frame_count):
        boxes, probs = self.detect_faces(frame)
        detected_faces = {'boxes': [], 'images': [], 'filenames': []}
        if boxes is not None:
            for person_idx, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(b) for b in box]
                face_img = frame[y1:y2, x1:x2]

                detected_faces['boxes'].append((x1, y1, x2, y2))
                detected_faces['images'].append(face_img)
                detected_faces['filenames'].append(f"face_frame{frame_count:04}_person{person_idx:02}.jpg")
        return detected_faces


class MTCNNFaceDetector(BaseFaceDetector):
    def __init__(self, identify_device=False):
        super().__init__(identify_device)
        self.model = MTCNN(keep_all=True, device=self.device)

    def detect_faces(self, frame, landmarks=False):
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = Image.fromarray(frame)

        # Если landmarks=True, то вернется кортеж длиной 3
        boxes, probs = self.model.detect(frame, landmarks=landmarks)
        return boxes, probs
