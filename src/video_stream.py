from logger import collogger
import cv2
import time


def connect_to_stream(video_src, max_attempts=5, delay=5):
    '''
    Функция для подключения к видеопотоку с повторными попытками
    :param video_src:
    :param max_attempts:
    :param delay:
    :return:
    '''
    attempts = 0
    collogger.info('Началось подключение')
    while attempts < max_attempts:
        cap = cv2.VideoCapture(video_src)
        if cap.isOpened():
            return cap
        else:
            collogger.warning(f"Попытка {attempts + 1} не удалась. Повторная попытка через {delay} секунд.")
            attempts += 1
            time.sleep(delay)
    collogger.error("Не удалось подключиться к видеопотоку после нескольких попыток")
    return None


def calculate_fps_of_stream(cap, num_frames):
    fps = cap.get(cv2.CAP_PROP_FPS)
    collogger.info(f"Frames per second using cap.get(cv2.CAP_PROP_FPS) : {fps}")

    start = time.time()
    for i in range(0, num_frames):
        ret, frame = cap.read()
    end = time.time()

    # Time elapsed
    seconds = end - start
    collogger.info(f"Time taken : {round(seconds, 2)} seconds")

    # Calculate frames per second
    fps = num_frames / seconds
    collogger.info(f"Estimated frames per second : {round(fps, 2)}")

    cap.release()
