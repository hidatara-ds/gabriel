FROM python:3.11-slim
WORKDIR /app

# Install ffmpeg for pydub to process webm audio
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
ENV PORT=8080
EXPOSE 8080

RUN useradd -m -u 1001 gabriel
USER gabriel

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
