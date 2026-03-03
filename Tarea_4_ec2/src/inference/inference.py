"""
Este script carga un modelo XGBoost previamente entrenado y lo aplica sobre el
conjunto de datos de prueba. Realiza el post-procesamiento necesario (clipping)
y genera un archivo CSV con el formato requerido para la competencia o entrega.

Argumentos de línea de comandos:
    --test (str): Ruta al archivo CSV con los datos de prueba (ID, shop_id, item_id).
    --sample (str): Ruta al archivo de ejemplo de submission para extraer los IDs.
    --model (str): Ruta al modelo serializado (.joblib).
    --out (str): Ruta de destino para el archivo de predicciones finales.
    --date_block_num (int): Identificador del bloque mensual para la predicción (ej. 34).
"""
## este es un cambio para el pull request
import argparse
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

# Configuración de logging
LOG_DIR = "artifacts/logs"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/prep_{timestamp}.log"),
        logging.StreamHandler(),  # También imprime en consola
    ],
)

logger = logging.getLogger(__name__)

def create_lags(df, lags=[1, 2, 3]):
    """
    Crea columnas de lag para la variable 'item_cnt_month'.
    """
    tmp = df[['date_block_num', 'shop_id', 'item_id', 'item_cnt_month']]
    
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num', 'shop_id', 'item_id', f'item_cnt_month_lag_{i}']
        # Se mueve el bloque temporal hacia adelante
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    
    # Llenar con 0 los meses sin ventas
    return df.fillna(0)

def split_train_val_with_lags(df, val_block=33):
    """
    Aplica la ingeniería de variables y divide el dataset.
    """
    # Crea lags
    df_with_lags = create_lags(df)
    # Filtrado de los primeros 3 meses, para poder obtener lags
    df_with_lags = df_with_lags[df_with_lags.date_block_num > 3]
    # Features de lags
    features = ['date_block_num', 'shop_id', 'item_id', 
                'item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3']
    target = 'item_cnt_month'
    # División temporal
    train = df_with_lags[df_with_lags.date_block_num < val_block]
    val = df_with_lags[df_with_lags.date_block_num == val_block]
    return train[features], train[target], val[features], val[target]

def main():
    """
    1. Carga de datos de test y el modelo persistido.
    2. Preparación de las características (features) para el bloque temporal objetivo.
    3. Generación de predicciones y aplicación de límites [0, 20].
    4. Mapeo de resultados a los IDs originales y exportación a CSV.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="data/inference/test.csv")
    parser.add_argument("--sample", default="data/raw/sample_submission.csv")
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--out", default="data/predictions/Prediccion_Equipo2.csv")
    args = parser.parse_args()

    # Carga de recursos
    logger.info("Iniciando Carga de Datos...")
    test = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample)
    model = joblib.load(args.model)
    history = pd.read_csv("data/prep/monthly_sales.csv")

    # Preparación de features
    # Se asigna el bloque temporal 34 (el mes siguiente al último del entrenamiento)
    logger.info("Inicializacion de variables..")
    test["date_block_num"] = 34
    #x_test = test[["date_block_num", "shop_id", "item_id"]]
    test["item_cnt_month"] = 0 # Valor temporal para que la función funcione

    # Unimos el historial con el test para poder calcular los lags del bloque 34
    combined = pd.concat([history, test], axis=0).reset_index(drop=True)
    combined = create_lags(combined) # Esto genera las 6 columnas necesarias

    # Filtramos solo las filas que pertenecen al bloque de inferencia (34)
    x_test = combined[combined.date_block_num == 34]

    # Seleccionamos las 6 columnas exactas en el orden que espera el modelo
    features = ['date_block_num', 'shop_id', 'item_id', 
            'item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3']
    x_test = x_test[features]

    # Inferencia
    logger.info("Realizando las predicciones")
    preds = model.predict(x_test)
    # Aplicar clipping para mantener coherencia con el entrenamiento y reglas de negocio
    preds = np.clip(preds, 0, 20)
    logger.info("Guardadndo las predicciones")
    submission = pd.DataFrame({"ID": sample["ID"], "item_cnt_month": preds})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"Predicciones guardadas en: {out_path}  (n={len(submission)})")

if __name__ == "__main__":
    main()
