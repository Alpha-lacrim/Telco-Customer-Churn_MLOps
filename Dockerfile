FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    CONFIG_PATH=/app/config.yaml \
    MODEL_URI=/app/models/best_model \
    MLFLOW_TRACKING_URI=file:///app/mlruns

WORKDIR /app

COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml run_pipeline.py ./
COPY src ./src

RUN mkdir -p /app/artifacts /app/models /app/mlruns

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
