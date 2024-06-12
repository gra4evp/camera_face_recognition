import os
import cv2
from image_processing import transform_frame, crop_frame, scale_frame
from face_detection import process_frame
from stream_utils import connect_to_stream
from config import RTSP_URL, CAMERA_ROI
from logger import collogger
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, request, jsonify
from threading import Thread

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://username:password@db:5432/mydatabase')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class DetectedFace(db.Model):
    __tablename__ = 'detected_faces'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    filename = db.Column(db.String, nullable=False)
    image = db.Column(db.LargeBinary, nullable=False)

    def __repr__(self):
        return f"<DetectedFace(id={self.id}, filename='{self.filename}')>"


@app.before_first_request
def setup():
    db.create_all()


def save_faces_to_db(imgs, filenames):
    for img, filename in zip(imgs, filenames):
        success, face_img_encoded = cv2.imencode('.jpg', img)
        if success:
            face_img_bytes = face_img_encoded.tobytes()
            db_face = DetectedFace(filename=filename, image=face_img_bytes)
            db.session.add(db_face)
            db.session.commit()
            db.session.refresh(db_face)
        else:
            collogger.warning(f'cv2.imencode: {filename}')


def detect_stream(cap, process_every_n_frame, frame_scale=0.5):
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            collogger.warning("Не удалось получить кадр. Прерывание...")
            break

        frame = transform_frame(
            frame,
            transforms=[
                (crop_frame, {'bbox': CAMERA_ROI}),
                (scale_frame, {'scale': frame_scale})
            ]
        )
        frame_count += 1
        if frame_count % process_every_n_frame == 0:
            detected_faces = process_frame(frame=frame, frame_count=frame_count)
            save_faces_to_db(imgs=detected_faces['images'], filenames=detected_faces['filenames'])

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

        thread = Thread(target=detect_stream, args=(every_n_frame,))
        thread.start()

        collogger.info(f"Начато обнаружение лиц для потока: {RTSP_URL} с каждым {every_n_frame}-ым кадром")
        return jsonify({"message": "Face detection started"}), 200

    collogger.error("Ошибка подключения к видеопотоку")
    return jsonify({"error": "Error connecting to the video stream"}), 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
