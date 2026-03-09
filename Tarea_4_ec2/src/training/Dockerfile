FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
COPY src/training/ ./src/training/
RUN pip install --no-cache-dir -r requirements.txt
COPY src/training/ ./src/training/
RUN mkdir -p data/prep artifacts/model artifacts/logs
ENTRYPOINT ["python", "src/training/train.py"]