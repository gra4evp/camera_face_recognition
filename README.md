# Camera Face Recognition

Этот проект реализует распознавание лиц в реальном времени с использованием камеры и библиотеки OpenCV. Он считывает видеопоток с RTSP-камеры, обрабатывает каждый N-ый кадр, масштабирует и обрезает его, а затем распознает лица и сохраняет их изображения.

## Оглавление
- [Установка](#установка)
- [Использование](#использование)
- [Настройка](#настройка)
- [Примеры](#примеры)

## Установка

1. Склонируйте репозиторий:
    ```sh
    git clone https://github.com/gra4evp/camera_face_recognition.git
    cd camera-face-recognition
    ```

2. Установите необходимые зависимости:
    ```sh
    pip install -r requirements.txt
    ```

## Использование

Запустите скрипт `watch_stream.py`, чтобы посмотреть обработку видеопотока с использованием модели распознавания:
```sh
python ./src/watch_stream.py
```

Запустите контейнеры Docker для старта обработки стрима и детекции лиц
```sh
docker-compose up -d
```

## Что видит камера
![alt text](https://github.com/gra4evp/camera_face_recognition/blob/main/camera.jpg?raw=true)

После обрезки
![alt text](https://github.com/gra4evp/camera_face_recognition/blob/main/camera_croped.jpg?raw=true)

Пример разметки 
![alt text](https://github.com/gra4evp/camera_face_recognition/blob/main/detected_faces_example.jpg?raw=true)