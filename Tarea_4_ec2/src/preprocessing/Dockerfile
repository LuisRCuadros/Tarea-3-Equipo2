FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
COPY src/preprocessing/ ./src/preprocessing/
RUN pip install --no-cache-dir -r requirements.txt
COPY src/preprocessing/ ./src/preprocessing/
RUN mkdir -p data/raw data/prep artifacts/logs
ENTRYPOINT ["python", "src/preprocessing/prep.py"]