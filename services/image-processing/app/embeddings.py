# embeddings.py
from abc import ABC, abstractmethod
import numpy as np
import torch


class FaceEmbedder(ABC):
    @abstractmethod
    def get_embeddings(self, face_images: list) -> np.ndarray:
        pass


class InceptionResnetV1Embedder(FaceEmbedder):
    def __init__(self, pretrained='vggface2'):
        from facenet_pytorch import InceptionResnetV1
        self.model = InceptionResnetV1(pretrained=pretrained).eval()

    def get_embeddings(self, face_images: list) -> np.ndarray:
        embeddings = []
        for img in face_images:
            with torch.no_grad():
                embedding = self.model(img.unsqueeze(0))
            embeddings.append(embedding.numpy())
        return np.array(embeddings)
