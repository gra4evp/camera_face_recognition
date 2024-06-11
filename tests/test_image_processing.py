import os
import sys
import numpy as np
import pytest

# Добавляем путь к корню проекта в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.image_processing import scale_frame, crop_frame, transform_frame, save_face_images, draw_boxes


@pytest.fixture
def sample_image():
    # Создаем простое изображение 100x100 с белым фоном
    return np.ones((100, 100, 3), dtype=np.uint8) * 255


def test_scale_frame(sample_image):
    scaled_image = scale_frame(sample_image, 0.5)
    assert scaled_image.shape == (50, 50, 3)


def test_crop_frame(sample_image):
    cropped_image = crop_frame(sample_image, (10, 90, 20, 80))
    assert cropped_image.shape == (80, 60, 3)


def test_transform_frame(sample_image):
    transforms = [(crop_frame, {'bbox': (10, 90, 20, 80)}), (scale_frame, {'scale': 0.5})]
    transformed_image = transform_frame(sample_image, transforms)
    assert transformed_image.shape == (40, 30, 3)


def test_save_face_images(sample_image, tmp_path):
    dirpath = tmp_path / "faces"
    os.mkdir(dirpath)
    filenames = ["face1.jpg", "face2.jpg"]
    save_face_images([sample_image, sample_image], filenames, dirpath)
    assert (dirpath / "face1.jpg").exists()
    assert (dirpath / "face2.jpg").exists()


def test_draw_boxes(sample_image):
    boxes = [(10, 10, 50, 50), (20, 20, 40, 40)]
    image_with_boxes = draw_boxes(sample_image, boxes)
    assert image_with_boxes[10, 10].tolist() == [0, 255, 0]  # Проверка, что пиксель окрашен в зеленый
    assert image_with_boxes[20, 20].tolist() == [0, 255, 0]
