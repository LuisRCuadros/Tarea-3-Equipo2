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
from utils.functions import load_data, aggregate_monthly, rename_columns, clip_values

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

def main():
    """
    Gestiona la entrada de argumentos, asegura la creación de directorios de
    salida y coordina la persistencia de los datos procesados.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales", default="data/raw/sales_train.csv")
    parser.add_argument("--out", default="data/prep/monthly_sales.csv")
    args = parser.parse_args()

    # se sigue el Method Chaining
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
