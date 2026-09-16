FROM python:3.10-slim

LABEL maintainer="your.email@example.com"
LABEL description="Autonomous Driving Perception System — YOLOv8 + OpenCV"

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY detect_image.py detect_video_v2.py evaluate.py train.py ./
COPY model_data/ ./model_data/
COPY font/ ./font/

# Create output directories
RUN mkdir -p outputs/images outputs/videos outputs/logs outputs/reports models

# Default: image detection
ENTRYPOINT ["python", "detect_image.py"]
