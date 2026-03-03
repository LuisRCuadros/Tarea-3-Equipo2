"""
Este script transforma los registros de ventas diarios en un formato agregado mensual,
realizando limpieza de valores atípicos (clipping) y estructurando el dataset para
su posterior uso en modelos de Machine Learning.

Argumentos de línea de comandos:
    --sales (str): Ruta al archivo CSV con los datos de ventas crudos (daily).
    --out (str): Ruta de destino para el CSV procesado con ventas mensuales.
"""
import argparse
import logging
import time
from datetime import datetime
import pandas as pd
import numpy as np

def load_data(path: str) -> pd.DataFrame:
    """Carga inicial de datos."""
    return pd.read_csv(path)

def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega ventas por bloque, tienda y producto."""
    return (df.groupby(["date_block_num", "shop_id", "item_id"])["item_cnt_day"]
            .sum()
            .reset_index())

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ajusta nombres de columnas tras agregación."""
    df.columns = ["date_block_num", "shop_id", "item_id", "item_cnt_month"]
    return df

def clip_values(df: pd.DataFrame, lower: int = 0, upper: int = 20) -> pd.DataFrame:
    """Aplica clipping al target."""
    df["item_cnt_month"] = df["item_cnt_month"].clip(lower, upper)
    return df

# Configuración de logging
LOG_DIR = "artifacts/logs" # Directorio en donde se guardarán logs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Formato en cómo se guardarán los logs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/prep_{timestamp}.log"),
        logging.StreamHandler(),  # Imprime los logs en la consola al ejecutar el script
    ],
)

logger = logging.getLogger(__name__)

def main():
    """
    Gestiona la entrada de argumentos, asegura la creación de directorios de
    salida y coordina la persistencia de los datos procesados.
    """

    # Configuración de argumentos
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales", default="data/raw/sales_train.csv")
    parser.add_argument("--out", default="data/prep/monthly_sales.csv")
    args = parser.parse_args()

    # Method Chaining
    logger.info("Iniciando Carga de Datos...")
    start_time = time.time()
    monthly = (
        load_data(args.sales)
        .pipe(aggregate_monthly)
        .pipe(rename_columns)
        .pipe(clip_values, lower=0, upper=20)  # Pasamos argumentos extra aquí
    )
    duration = time.time() - start_time
    logger.info(f"Tiempo de ejecución: {duration:.2f} segundos") # pylint: disable=logging-fstring-interpolation

    logger.info("Guardando las ventas mensuales...")
    monthly.to_csv(args.out, index=False)
    print(f"Guardado: {args.out} (shape={monthly.shape})")

if __name__ == "__main__":
    main()