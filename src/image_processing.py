import cv2
import os
from logger import logger


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
        logger.info(f"Saved cropped face image: {filepath}")


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
