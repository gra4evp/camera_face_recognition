import time
import os
import cv2
from config import RTSP_URL
from logger import logger
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


def connect_to_stream(video_src, max_attempts=5, delay=5):
    '''
    Функция для подключения к видеопотоку с повторными попытками
    :param video_src:
    :param max_attempts:
    :param delay:
    :return:
    '''
    attempts = 0
    logger.info('Началось подключение')
    while attempts < max_attempts:
        cap = cv2.VideoCapture(video_src)
        if cap.isOpened():
            return cap
        else:
            logger.warning(f"Попытка {attempts + 1} не удалась. Повторная попытка через {delay} секунд.")
            attempts += 1
            time.sleep(delay)
    logger.error("Не удалось подключиться к видеопотоку после нескольких попыток")
    return None


def scale_frame(frame, scale):
    height, width = frame.shape[:2]
    dim = (int(width * scale), int(height * scale))
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def crop_frame(frame, bbox):
    h1, h2, w1, w2 = bbox
    return frame[h1:h2, w1:w2]


def detect_faces(frame, landmarks=False):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # img = Image.fromarray(img)

    # Если landmarks=True, то вернется кортеж длиной 3
    boxes, probs = mtcnn.detect(img, landmarks=landmarks)
    return boxes, probs


def process_frame(frame, frame_count, scale):
    processed_frame = scale_frame(frame=crop_frame(frame, CAMERA_ROI), scale=scale)

    boxes, probs = detect_faces(processed_frame)
    detected_faces = {'boxes': [], 'images': [], 'filenames': []}
    if boxes is not None:
        for person_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(b) for b in box]
            face_img = processed_frame[y1:y2, x1:x2]

            detected_faces['boxes'].append((x1, y1, x2, y2))
            detected_faces['images'].append(face_img)
            detected_faces['filenames'].append(f"face_frame{frame_count:04}_person{person_idx:02}.jpg")
    return processed_frame, detected_faces


def save_face_images(imgs, filenames, dirpath):
    for img, filename in zip(imgs, filenames):
        filepath = os.path.join(dirpath, filename)
        cv2.imwrite(filename=filepath, img=img)
        logger.info(f"Saved cropped face image: {filepath}")


def process_stream(cap, lag, frame_scale, need_draw=False, faces_dirpath=None):
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            logger.warning("Не удалось получить кадр. Прерывание...")
            break

        frame_count += 1
        frame, detected_faces = process_frame(frame=frame, frame_count=frame_count, scale=frame_scale)

        if need_draw:
            draw_boxes(frame, boxes=detected_faces['boxes'])

        if faces_dirpath is not None:
            save_face_images(
                imgs=detected_faces['images'],
                filenames=detected_faces['filenames'],
                dirpath=faces_dirpath
            )

        cv2.imshow('RTSP Stream', frame)

        time.sleep(lag)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()  # Закрывает все окна, созданные OpenCV


def draw_boxes(frame, boxes):
    """
    Рисует зеленые прямоугольники на изображении для каждой координаты в boxes.

    Parameters:
    - frame: исходное изображение
    - boxes: [list[tuple[int]]] список координат прямоугольников [(x1, y1, x2, y2), ...]

    Returns:
    - frame_with_boxes: изображение с нарисованными прямоугольниками
    """
    # Копируем изображение, чтобы сохранить оригинал
    frame_with_boxes = frame.copy()

    for box in boxes:
        x1, y1, x2, y2 = box
        # Зеленый цвет (0, 255, 0), толщина линии 2 пикселя
        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame_with_boxes


def calculate_fps_of_stream(cap, num_frames):
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Frames per second using cap.get(cv2.CAP_PROP_FPS) : {fps}")

    start = time.time()
    for i in range(0, num_frames):
        ret, frame = cap.read()
    end = time.time()

    # Time elapsed
    seconds = end - start
    logger.info(f"Time taken : {round(seconds, 2)} seconds")

    # Calculate frames per second
    fps = num_frames / seconds
    logger.info(f"Estimated frames per second : {round(fps, 2)}")

    cap.release()


if __name__ == "__main__":
    flag_testing = False
    if not flag_testing:
        video_source = RTSP_URL
    else:
        video_source = 0

    CAMERA_ROI = (1000, 1800, 250, 2350)  # Область на камере для обнаружения людей (region_of_interest)
    DEVICE = device='cuda' if torch.cuda.is_available() else 'cpu'

    FRAMES_DIRNAME = 'frames'
    frames_dirpath = os.path.join(os.getcwd(), FRAMES_DIRNAME)
    if not os.path.exists(frames_dirpath):
        os.mkdir(frames_dirpath)

    FRAME_SCALE = 0.5

    # расрешение камеры # 2592 x 1920
    screen_width = 2520  # 2520
    screen_height = 1680  # 1680

    EVERY_Nth_FRAME = 5  # Считываем каждый n-ый кадр
    mtcnn = MTCNN(keep_all=True, device=DEVICE)

    cap = connect_to_stream(video_src=video_source)
    if cap is not None:
        logger.info("Успешно подключились к видеопотоку")
        # calculate_fps_of_stream(cap, num_frames=120)
        stream_fps = cap.get(cv2.CAP_PROP_FPS)
        lag = 1 / (stream_fps // EVERY_Nth_FRAME)
        logger.info(f"Stream FPS: {stream_fps}, LAG = {lag}")
        process_stream(cap, lag=lag, frame_scale=FRAME_SCALE, faces_dirpath=frames_dirpath)
    else:
        logger.error("Завершение программы из-за невозможности подключиться к видеопотоку")
    logger.info("Завершено")
