FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2 + FULL audio support (ffmpeg for ffprobe/pydub/Gradio)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    ffmpeg \
    libsndfile1 \
    libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]