# Tarea 05: MLOps en Práctica — AWS Sagemaker
## Diana Arroyo / Luis Cuadros

## Descripción del proyecto
Diseño, entrenamiento y despliegue en la nube de un pipeline de Machine Learning de extremo a extremo para pronosticar las ventas mensuales de productos por tienda. El proyecto implementa una arquitectura Bring Your Own Container (BYOC) en Amazon SageMaker, garantizando un entorno escalable, aislado y listo para producción.

Esta tarea pone a prueba los conceptos revisados de MLOps, Docker, Git Workflow y AWS Sagemaker.

---

## Estructura del repositorio
```text
.
├── artifacts
│   ├── logs
│   │   ├── prep_20260302_043537.log
│   │   ├── prep_20260302_044034.log
│   │   ├── prep_20260302_044117.log
│   │   └── prep_20260302_044526.log
│   └── model.joblib
├── data
│   ├── images
│   ├── inference
│   │   └── test.csv
│   ├── predictions
│   │   └── Prediccion_Equipo2.csv
│   ├── prep
│   │   └── monthly_sales.csv
│   └── raw
│       ├── sales_train.csv
│       ├── sample_submission.csv
│       └── test.csv
├── requirements.txt
└── src
    ├── container   #NUEVA CARPETA que sigue la estructura del notebook
    │   ├── build_and_push.sh
    │   ├── Dockerfile
    │   └── scripts-src
    │       ├── nginx.conf
    │       ├── predictor.py
    │       ├── serve
    │       ├── train
    │       └── wsgi.py
    ├── data        #NUEVA CARPETA que sigue la estructura del notebook
    │   └── monthly_sales.csv    
    ├── inference
    │   ├── Dockerfile
    │   ├── inference.py
    │   └── test
    │       └── test_inference.py
    ├── preprocessing
    │   ├── Dockerfile
    │   ├── prep.py
    │   └── test
    │       └── test_preprocessing.py
    ├── training
    │   ├── Dockerfile
    │   ├── test
    │   │   └── test_train.py
    │   └── train.py
    └── sm_train_build_your_own_container.ipynb

```
---

## Git Workflow
La presente tarea se encuentra en el repositorio llamado Tarea-3-Equipo2 en la rama `feature/sagemaker-training-byoc`. Para subir los cambios se usaron los siguientes comandos:

```sh
git checkout feature/sagemaker-training-byoc
git add .
git commit -m "AWS Sagemaker"
git push origin feature/sagemaker-training-byoc
```

La presente rama se creó a partir de la rama `development`.


## AWS

#### ECR

Se creó un repositorio en Amazon ECR llamado **supermarket** a partir del script denominado `build_and_push.sh`.

```sh
bash build_and_push.sh supermarket
```
![ECR](data/images/ecr.png)

#### Buckets

El bucket en donde se introdujo el csv de entrenamiento se llamó **sales_supermarket** y se creó a partir del script llamado `build_and_push.sh`.

![bucket1](data/images/bucket_input.png)

Adicionalmente, se guardó el modelo en el bucket llamado **output**.

![bucket2](data/images/bucket_output.png)

#### Entrenamiento
Se entrenó el script de entrenamiento `train` contenido en la carpeta `src/container/scripts-src` con el siguiente código:

```python
tree = sage.estimator.Estimator(
    image,
    role,
    1,
    "ml.c4.2xlarge",
    output_path=s3_output_path,
    sagemaker_session=sess,
)

tree.fit(data_location)
```
A continuación se muestra una captura de pantalla del **output** del entrenamiento del modelo.

![training](data/images/training.png)

#### Deploy del modelo
Se desplegó el modelo usando el siguiente código.

```python
from sagemaker.serializers import CSVSerializer
predictor = tree.deploy(1, "ml.m4.xlarge", serializer=CSVSerializer())
```
A continuación se muestra una captura de pantalla del **output** del deploy del modelo.

![deploy](data/images/deploy.png)

Adicionalmente, se muestra una captura de pantalla de una predicción en tiempo real.

![prediction](data/images/prediction.png)

#### Endpoint
Al desplegar el modelo con el código mostrado anteriormente, se creó un endpoint que permite recibir datos en tiempo real y devolver predicciones del modelo entrenado.

![endpoint](data/images/endpoint.png)