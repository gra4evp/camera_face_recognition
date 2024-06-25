from flask_sqlalchemy import SQLAlchemy
import cv2

from services.logger import collogger

db = SQLAlchemy()


class DetectedFace(db.Model):
    __tablename__ = 'detected_faces'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    filename = db.Column(db.String, nullable=False)
    image = db.Column(db.LargeBinary, nullable=False)
    embedding = db.Column(db.LargeBinary, nullable=False)

    def __repr__(self):
        return f"<DetectedFace(id={self.id}, filename='{self.filename}')>"


def save_faces_to_db(imgs, filenames, embeddings):
    for img, filename, embedding in zip(imgs, filenames, embeddings):
        success, face_img_encoded = cv2.imencode('.jpg', img)
        if success:
            face_img_bytes = face_img_encoded.tobytes()
            embedding_bytes = embedding.tobytes()
            db_face = DetectedFace(filename=filename, image=face_img_bytes, embedding=embedding_bytes)
            db.session.add(db_face)
            db.session.commit()
            db.session.refresh(db_face)
        else:
            collogger.warning(f'cv2.imencode: {filename}')
