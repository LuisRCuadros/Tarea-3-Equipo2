"""
Módulo de entrenamiento para la predicción de demanda mensual mediante XGBoost.

Este script carga datos procesados de ventas, realiza una partición temporal para validación,
entrena un modelo de regresión Gradient Boosting y persiste el artefacto resultante.
Está diseñado para integrarse en pipelines de CI/CD o flujos de orquestación de datos.

Argumentos de línea de comandos:
    --data (str): Ruta al archivo CSV con las métricas mensuales.
    --model_out (str): Ruta de destino para el modelo serializado (.joblib).
    --val_block (int): Número del bloque temporal asignado para validación.

Ejemplo de uso:
    $ python train.py --data data/sales.csv --model_out models/xgb_v1.joblib
"""
import argparse
from pathlib import Path
import logging
import time
from datetime import datetime
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from utils.functions import split_train_val


# Configuración de logging
LOG_DIR = 'artifacts/logs'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOG_DIR}/prep_{timestamp}.log'),
        logging.StreamHandler()  # También imprime en consola
    ]
)

logger = logging.getLogger(__name__)

def main():
    """
    Punto de entrada principal para el pipeline de entrenamiento.
    
    Carga la configuración de hiperparámetros de XGBoost, ejecuta el ciclo de 
    entrenamiento con early stopping y exporta el modelo tras calcular el RMSE.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/prep/monthly_sales.csv")
    parser.add_argument("--model_out", default="artifacts/model.joblib")
    parser.add_argument("--val_block", type=int, default=33)
    args = parser.parse_args()

    logger.info("Iniciando Carga de Datos...")
    monthly_sales = pd.read_csv(args.data)
    logger.info("Dividiendo el set de para entrenamiento y validación...")

    x_train, y_train, x_val, y_val = split_train_val(monthly_sales, val_block=args.val_block)

    # Parámetros
    model = xgb.XGBRegressor(
        max_depth=8, # profundidad máxima de cada árbol
        n_estimators=100, # número de boosting rounds (árboles)
        min_child_weight=300, # observaciones en un nodo de hoja
        colsample_bytree=0.8, # fracción de features por árbol
        subsample=0.8, #  fracción de datos por árbol
        learning_rate=0.3,
        seed=53,
        eval_metric="rmse",
        early_stopping_rounds=10,
    )

    start_time = time.time()
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    duration = time.time() - start_time
    logger.info(f"Tiempo de ejecución: {duration:.2f} segundos") # pylint: disable=logging-fstring-interpolation

    preds_val = model.predict(x_val).clip(0, 20)
    rmse = float(np.sqrt(mean_squared_error(y_val, preds_val)))
    print(f"[RMSE val] {rmse:.4f}")
    logger.info(f"Modelo entrenado - RMSE: {rmse:.4f}") # pylint: disable=logging-fstring-interpolation

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)

    print(f"Modelo guardado en: {model_out}")

if __name__ == "__main__":
    main()
