# Servidor de inferencia Flask para SageMaker BYOC.
#
# SageMaker invoca el serving ejecutando: docker run <image> serve
# El script 'serve' arranca nginx + gunicorn, que a su vez carga esta app Flask.
#
# Implementa el contrato de inferencia de SageMaker:
#   GET  /ping         -> health check. 200 si el modelo cargo, 404 si no.
#   POST /invocations  -> recibe features en CSV, devuelve predicciones en CSV.
#
# Input esperado en el filesystem del container:
#   /opt/ml/
#   └── model/
#       └── decision-tree-model.pkl   # Generado por el script 'train'.
#                                     # En produccion, SageMaker lo descarga de S3 al arrancar.
#
# Formato del request POST /invocations:
#   Content-Type: text/csv
#   Body: filas CSV sin header, solo features. Ejemplo: 5.1,3.5,1.4,0.2
#
# Formato del response:
#   Content-Type: text/csv — una prediccion por linea

from __future__ import print_function

import io
import json
import os
import pickle
import signal
import sys
import traceback
import joblib

import flask
import numpy as np
import pandas as pd

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# ScoringService implementa el patron singleton para el modelo.
# El modelo se carga desde disco la primera vez que se necesita y queda en memoria
# para todas las requests siguientes — evita releerlo en cada invocacion.
class ScoringService(object):
    model = None

    @classmethod
    def get_model(cls):
        """Carga el modelo desde disco si no esta en memoria y lo retorna."""
        if cls.model == None:
            with open(os.path.join(model_path, "model.joblib"), "rb") as inp:
                cls.model = pickle.load(inp)
        return cls.model

    @classmethod
    def predict(cls, input):
        """Ejecuta la prediccion sobre el DataFrame recibido.

        Args:
            input (pd.DataFrame): features de entrada, una fila por observacion.

        Returns:
            numpy array con una prediccion por fila.
        """
        clf = cls.get_model()
        return clf.predict(input)


# Aplicacion Flask que expone los endpoints de inferencia.
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Health check del container.

    SageMaker llama a este endpoint periodicamente para verificar que el servidor
    esta listo. Se considera saludable si el modelo cargo correctamente.

    Returns:
        200 si el modelo esta disponible, 404 si no pudo cargar.
    """
    health = ScoringService.get_model() is not None

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Endpoint de inferencia.

    Recibe un batch de observaciones en CSV, ejecuta la prediccion
    y devuelve los resultados en CSV (una prediccion por linea).

    Solo acepta Content-Type: text/csv — devuelve 415 para cualquier otro formato.
    """
    data = None

    # Parsear el body CSV a DataFrame. Sin header — solo features.
    if flask.request.content_type == "text/csv":
        data = flask.request.data.decode("utf-8")
        s = io.StringIO(data)
        data = pd.read_csv(s, header=None)
        # 6 columnas en el nombre exacto
        data.columns = [
            'date_block_num', 'shop_id', 'item_id', 
            'item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3'
        ]
    else:
        return flask.Response(
            response="This predictor only supports CSV data", status=415, mimetype="text/plain"
        )

    print("Invoked with {} records".format(data.shape[0]))

    # Ejecutar la prediccion.
    predictions = ScoringService.predict(data)
    predictions = np.clip(predictions, 0, 20) #Clipping entre 0 y 20

    # Convertir el array de predicciones a CSV y devolverlo como response.
    out = io.StringIO()
    pd.DataFrame({"results": predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype="text/csv")
