import os
from threading import Thread

import cv2
from flask import Flask, request, jsonify

from image_processing import Compose, CropFrame, ScaleFrame, Canny, GrayToRGB
from face_detection import MTCNNFaceDetector
from embeddings import InceptionResnetV1Embedder
from stream_utils import connect_to_stream
from config import RTSP_URL, CAMERA_ROI
from logger import collogger
from db.models import save_faces_to_db


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://username:password@db:5432/mydatabase')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

face_detector = MTCNNFaceDetector(identify_device=True)
embedder = InceptionResnetV1Embedder(pretrained='vggface2')


def detect_stream(cap, process_every_n_frame, transforms):
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            collogger.warning("Не удалось получить кадр. Прерывание...")
            break

        frame = transforms(frame)

        frame_count += 1
        if frame_count % process_every_n_frame == 0:
            detected_faces = face_detector.process_frame(frame=frame, frame_count=frame_count)
            face_embeddings = embedder.get_embeddings(detected_faces['images'])
            save_faces_to_db(imgs=detected_faces['images'], filenames=detected_faces['filenames'], embeddings=face_embeddings)

    cap.release()


@app.route('/start_detection', methods=['POST'])
def start_detection():
    data = request.json

    if 'every_n_frame' not in data:
        return jsonify({"error": "Parameter 'every_n_frame' is required"}), 400
    every_n_frame = data['every_n_frame']

    try:
        every_n_frame = int(every_n_frame)
        if every_n_frame <= 0:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Parameter 'every_n_frame' must be a positive integer"}), 400

    cap = connect_to_stream(video_src=RTSP_URL)
    if cap is not None:
        collogger.info("Успешно подключились к видеопотоку")
        stream_fps = cap.get(cv2.CAP_PROP_FPS)
        collogger.info(f"Stream FPS: {stream_fps}")

        frame_transforms = Compose([
            CropFrame(bbox=CAMERA_ROI),
            ScaleFrame(scale=0.5),
            # Canny(threshold1=100, threshold2=200),
            # GrayToRGB()
        ])

        thread = Thread(target=detect_stream, args=(cap, every_n_frame, frame_transforms))
        thread.start()

        collogger.info(f"Начато обнаружение лиц для потока: {RTSP_URL} с каждым {every_n_frame}-ым кадром")
        return jsonify({"message": "Face detection started"}), 200

    collogger.error("Ошибка подключения к видеопотоку")
    return jsonify({"error": "Error connecting to the video stream"}), 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
