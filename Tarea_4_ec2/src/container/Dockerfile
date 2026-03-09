# Container de entrenamiento e inferencia para SageMaker Equipo 2
FROM python:3.12-slim

MAINTAINER Amazon AI <sage-learner@amazon.com>

# --------------------------------------------------------------------------
# Dependencias del sistema
# Al final se limpia el cache de apt para reducir el tamaño de la imagen.
# --------------------------------------------------------------------------
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Se agrega xgboost joblib
RUN pip --no-cache-dir install numpy scipy scikit-learn pandas flask gunicorn xgboost joblib

# --------------------------------------------------------------------------
# Variables de entorno
# PYTHONUNBUFFERED=TRUE: los logs de Python se escriben en tiempo real a stdout,
#   lo que permite verlos inmediatamente en CloudWatch o en la terminal local.
# PYTHONDONTWRITEBYTECODE=TRUE: evita generar archivos .pyc innecesarios.
# PATH: agrega /opt/program al PATH para que los comandos 'train' y 'serve'
#   sean encontrados directamente cuando SageMaker invoca el container.
# --------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# --------------------------------------------------------------------------
# Código de la aplicación
# Copia los scripts de entrenamiento e inferencia al directorio /opt/program.
# SageMaker invoca 'train' o 'serve' dependiendo del tipo de job.
# --------------------------------------------------------------------------
#cambia carpeta
COPY scripts-src /opt/program
WORKDIR /opt/program

RUN chmod +x /opt/program/train /opt/program/serve
