import os
import cv2
from config import RTSP_URL, CAMERA_ROI
from logger import collogger
from stream_utils import connect_to_stream
from face_detection import MTCNNFaceDetector
from image_processing import scale_frame, crop_frame, canny, transform_frame, draw_boxes, save_face_images


face_detector = MTCNNFaceDetector(identify_device=True)


def watch_stream(cap, process_every_n_frame, frame_scale, need_draw=False, faces_dirpath=None):
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            collogger.warning("Не удалось получить кадр. Прерывание...")
            break

        frame = transform_frame(
            frame,
            transforms=[
                (crop_frame, dict(bbox=CAMERA_ROI)),
                (scale_frame, dict(scale=frame_scale)),
                (canny, dict(threshold1=100, threshold2=200))
            ]
        )

        # Преобразование одноканального изображения в трехканальное
        if len(frame.shape) == 2:  # если изображение одноканальное
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        frame_count += 1
        if frame_count % process_every_n_frame == 0:
            detected_faces = face_detector.process_frame(frame=frame, frame_count=frame_count)

            if need_draw:
                frame = draw_boxes(frame, boxes=detected_faces['boxes'])

            if faces_dirpath is not None:
                save_face_images(
                    imgs=detected_faces['images'],
                    filenames=detected_faces['filenames'],
                    dirpath=faces_dirpath
                )

            # Если нажата клавиша 's', сохранить изображение с разметкой
            if cv2.waitKey(1) & 0xFF == ord('s'):
                filepath = os.path.join(frames_dirpath, f"detected_faces_{frame_count}.jpg")
                cv2.imwrite(filename=filepath, img=frame)
                collogger.info(f"Frame with face markings has been saved: {filepath}")

        cv2.imshow('RTSP Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Выход из стрима по клавише q
            break

    cap.release()
    cv2.destroyAllWindows()  # Закрывает все окна, созданные OpenCV


if __name__ == "__main__":
    flag_testing = False
    if not flag_testing:
        video_source = RTSP_URL
    else:
        video_source = 0

    FRAMES_DIRNAME = 'frames'
    frames_dirpath = os.path.join(os.getcwd(), FRAMES_DIRNAME)
    if not os.path.exists(frames_dirpath):
        os.mkdir(frames_dirpath)

    FRAME_SCALE = 0.5

    # разрешение экрана
    # screen_width = 2520
    # screen_height = 1680

    EVERY_Nth_FRAME = 5  # Считываем каждый n-ый кадр

    cap = connect_to_stream(video_src=video_source)
    if cap is not None:
        collogger.info("Успешно подключились к видеопотоку")
        stream_fps = cap.get(cv2.CAP_PROP_FPS)
        collogger.info(f"Stream FPS: {stream_fps}")
        watch_stream(cap, process_every_n_frame=EVERY_Nth_FRAME, frame_scale=FRAME_SCALE, need_draw=True, faces_dirpath=None)
    else:
        collogger.error("Завершение программы из-за невозможности подключиться к видеопотоку")
    collogger.info("Завершено")

