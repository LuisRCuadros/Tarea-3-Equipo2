FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
COPY src/inference/ ./src/inference/
RUN pip install --no-cache-dir -r requirements.txt
COPY src/inference/ ./src/inference/
RUN mkdir -p data/inference data/prep data/predictions artifacts/logs
ENTRYPOINT ["python", "src/inference/inference.py"]