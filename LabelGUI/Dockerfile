FROM python:3.11.1

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgtk2.0-dev \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir opencv-python yt-dlp flask

EXPOSE 5000

CMD ["python", "app.py"]