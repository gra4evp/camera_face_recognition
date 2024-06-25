import cv2
import os
import numpy as np
from logger import collogger

from typing import Tuple, List, Callable


class ScaleFrame:
    """
    Класс для масштабирования изображения.

    Parameters:
    - scale (float): Коэффициент масштабирования.
    """
    scale: float

    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """
        Масштабирует изображение до указанного коэффициента.

        Parameters:
        - frame (numpy.ndarray): Входное изображение.

        Returns:
        - numpy.ndarray: Масштабированное изображение.
        """
        height, width = frame.shape[:2]
        dim = (int(width * self.scale), int(height * self.scale))
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


class CropFrame:
    """
    Класс для обрезки изображения.

    Parameters:
    - bbox (tuple[int, int, int, int]): Координаты обрезки (h1, h2, w1, w2).
    """
    def __init__(self, bbox: Tuple[int, int, int, int]):
        self.bbox = bbox

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """
        Обрезает изображение по заданным координатам.

        Parameters:
        - frame (numpy.ndarray): Входное изображение.

        Returns:
        - numpy.ndarray: Обрезанное изображение.
        """
        h1, h2, w1, w2 = self.bbox
        return frame[h1:h2, w1:w2]


class Canny:
    """
    Класс для применения Canny edge detection к изображению.

    Parameters:
    - threshold1 (int): Нижний порог для гистерезиса.
    - threshold2 (int): Верхний порог для гистерезиса.
    """

    def __init__(self, threshold1: int = 100, threshold2: int = 200):
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """
        Применяет Canny edge detection к изображению.

        Parameters:
        - frame (numpy.ndarray): Входное изображение.

        Returns:
        - numpy.ndarray: Изображение с применением Canny edge detection.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, self.threshold1, self.threshold2)
        return edges


class GrayToRGB:
    """
    Класс для преобразования одноканального изображения в трехканальное (RGB).
    """

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """
        Преобразует одноканальное изображение в трехканальное (RGB).

        Parameters:
        - frame (numpy.ndarray): Входное изображение.

        Returns:
        - numpy.ndarray: Трехканальное изображение (RGB).
        """
        if len(frame.shape) == 2:  # если изображение одноканальное
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        return frame


class Compose:
    """
    Класс для последовательного применения нескольких трансформаций.

    Parameters:
    - transforms (list[callable]): Список трансформаций для применения.
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """
        Последовательно применяет трансформации к изображению.

        Parameters:
        - frame (numpy.ndarray): Входное изображение.

        Returns:
        - numpy.ndarray: Преобразованное изображение.
        """
        for transform in self.transforms:
            frame = transform(frame)
        return frame


def scale_frame(frame, scale):
    """
    Масштабирует изображение (frame) до указанного коэффициента (scale).

    Parameters:
    - frame: numpy.ndarray, входное изображение.
    - scale: float, коэффициент масштабирования.

    Returns:
    - numpy.ndarray: масштабированное изображение.
    """
    height, width = frame.shape[:2]
    dim = (int(width * scale), int(height * scale))
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def crop_frame(frame, bbox):
    """
    Обрезает изображение по заданным координатам (bbox).

    Parameters:
    - frame: numpy.ndarray, входное изображение.
    - bbox: tuple[int], координаты обрезки (h1, h2, w1, w2).

    Returns:
    - numpy.ndarray: обрезанное изображение.
    """
    h1, h2, w1, w2 = bbox
    return frame[h1:h2, w1:w2]


def canny(frame, threshold1=100, threshold2=200):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, threshold1, threshold2)
    return edges


def transform_frame(frame, transforms):
    """
    Применяет последовательность преобразований к изображению.

    Parameters:
    - frame: numpy.ndarray, входное изображение.
    - transforms: list[tuple[callable, dict]], список преобразований и их аргументов.

    Returns:
    - numpy.ndarray: преобразованное изображение.
    """
    for transform, args in transforms:
        frame = transform(frame, **args)
    return frame


def save_face_images(imgs, filenames, dirpath):
    """
    Сохраняет изображения лиц в указанную директорию.

    Parameters:
    - imgs: list[numpy.ndarray], список изображений лиц.
    - filenames: list[str], список имен файлов.
    - dirpath: str, путь к директории для сохранения изображений.

    Returns:
    - None
    """
    for img, filename in zip(imgs, filenames):
        filepath = os.path.join(dirpath, filename)
        cv2.imwrite(filename=filepath, img=img)
        collogger.info(f"Saved cropped face image: {filepath}")


def draw_boxes(frame, boxes):
    """
    Рисует зеленые прямоугольники на изображении для каждой координаты в boxes.

    Parameters:
    - frame: исходное изображение
    - boxes: [list[tuple[int]]] список координат прямоугольников [(x1, y1, x2, y2), ...]

    Returns:
    - frame_with_boxes: изображение с нарисованными прямоугольниками
    """
    frame_with_boxes = frame.copy()  # Копируем изображение, чтобы сохранить оригинал

    for box in boxes:
        x1, y1, x2, y2 = box
        # Зеленый цвет (0, 255, 0), толщина линии 2 пикселя
        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame_with_boxes
