import os
import cv2
from config import RTSP_URL, CAMERA_ROI
from logger import logger


def process_stream(cap, process_every_n_frame, frame_scale, need_draw=False, faces_dirpath=None):
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            logger.warning("Не удалось получить кадр. Прерывание...")
            break

        frame = transform_frame(
            frame,
            transforms=[
                (crop_frame, {'bbox': CAMERA_ROI}),
                # (scale_frame, {'scale': frame_scale})
            ]
        )
        frame_count += 1
        if frame_count % process_every_n_frame == 0:
            detected_faces = process_frame(frame=frame, frame_count=frame_count)

            if need_draw:
                frame = draw_boxes(frame, boxes=detected_faces['boxes'])

            if faces_dirpath is not None:
                save_face_images(
                    imgs=detected_faces['images'],
                    filenames=detected_faces['filenames'],
                    dirpath=faces_dirpath
                )

        cv2.imshow('RTSP Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
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

    # расрешение камеры # 2592 x 1920
    screen_width = 2520  # 2520
    screen_height = 1680  # 1680

    EVERY_Nth_FRAME = 5  # Считываем каждый n-ый кадр

    cap = connect_to_stream(video_src=video_source)
    if cap is not None:
        logger.info("Успешно подключились к видеопотоку")
        # _, frame = cap.read()
        # frame = crop_frame(frame, CAMERA_ROI)
        # cv2.imwrite('camera_croped.jpg', frame)

        # calculate_fps_of_stream(cap, num_frames=120)
        stream_fps = cap.get(cv2.CAP_PROP_FPS)
        lag = 1 / (stream_fps // EVERY_Nth_FRAME)
        logger.info(f"Stream FPS: {stream_fps}, LAG = {lag}")
        process_stream(cap, process_every_n_frame=EVERY_Nth_FRAME, frame_scale=FRAME_SCALE, need_draw=True, faces_dirpath=frames_dirpath)
    else:
        logger.error("Завершение программы из-за невозможности подключиться к видеопотоку")
    logger.info("Завершено")
