FROM ultralytics/ultralytics

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgtk2.0-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=

WORKDIR /app

COPY . /app

RUN pip install yt-dlp