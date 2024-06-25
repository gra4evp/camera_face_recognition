FROM python:3.12

# Установить рабочую директорию внутри контейнера
WORKDIR /app

# Скопировать зависимости проекта и установить
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скопировать все файлы из текущего контекста сборки внутрь контейнера
COPY . .