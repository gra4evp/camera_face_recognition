# Используйте официальный образ Python от Docker Hub
FROM python:3.10

# Установите рабочую директорию внутри контейнера
WORKDIR /app

# Скопируйте зависимости проекта и установите их через pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скопируйте все файлы из текущего контекста сборки внутрь контейнера
COPY . .
