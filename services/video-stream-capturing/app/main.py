# =============================
# import standard libraries
# =============================
import os
from threading import Thread

# =============================
# import additional installed libraries
# =============================
import cv2
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import httpx

# =============================
# import local modules
# =============================
from image_processing import Compose, CropFrame, ScaleFrame
from stream_utils import connect_to_stream
from config import RTSP_URL, CAMERA_ROI
from services.logger import collogger

# =============================
# Database configuration (if needed)
# =============================
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://username:password@db:5432/mydatabase')
# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


app = FastAPI()


class CaptureRequest(BaseModel):
    process_every_n_frame: int


async def process_frame(frame, transforms, url):
    transformed_frame = transforms(frame)
    _, buffer = cv2.imencode('.jpg', transformed_frame)
    frame_bytes = buffer.tobytes()

    async with httpx.AsyncClient() as client:
        response = await client.post(url, files={'file': ('frame.jpg', frame_bytes, 'image/jpeg')})
        if response.status_code != 200:
            collogger.error(f"Ошибка при отправке кадра: {response.text}")


async def process_stream(cap, process_every_n_frame, transforms, url):
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            collogger.warning("Не удалось получить кадр. Прерывание...")
            break

        frame_count += 1
        if frame_count % process_every_n_frame == 0:
            await process_frame(frame, transforms, url)

    cap.release()


@app.post('/start_capturing')
async def start_capturing(capture_request: CaptureRequest, background_tasks: BackgroundTasks):
    # тут нужно обработать запрос в нем передается количество кадров для обработки и если все, хорошо
    # Нужно начать посылать картинки в контейнер image-processing
    process_every_n_frame = capture_request.process_every_n_frame

    cap = connect_to_stream(video_src=RTSP_URL)
    if cap is not None:
        collogger.info("Успешно подключились к видеопотоку")
        stream_fps = cap.get(cv2.CAP_PROP_FPS)
        collogger.info(f"Stream FPS: {stream_fps}")

        frame_transforms = Compose([CropFrame(bbox=CAMERA_ROI), ScaleFrame(scale=0.5)])
        processing_url = "http://image-processing-container-url/process_frame"

        background_tasks.add_task(process_stream, cap, process_every_n_frame, frame_transforms, processing_url)

        collogger.info(f"Начато обнаружение лиц для потока: {RTSP_URL} с каждым {process_every_n_frame}-ым кадром")
        return {"message": "Face detection started"}

    collogger.error("Ошибка подключения к видеопотоку")
    raise HTTPException(status_code=400, detail="Error connecting to the video stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)