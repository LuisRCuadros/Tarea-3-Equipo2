"""
Funciones realizadas para prep.py y train.py
"""

import pandas as pd

def split_train_val(monthly_sales: pd.DataFrame, val_block: int = 33):
    """
    Realiza una partición temporal de los datos basada en bloques mensuales.
    Args:
        monthly_sales (pd.DataFrame): Conjunto de datos completo que contiene 
        la columna 'date_block_num'.
        val_block (int): El identificador del bloque que se utilizará exclusivamente 
            para validación. Los bloques menores se usarán para entrenamiento.

    Returns:
        tuple: Contiene (X_train, y_train, X_val, y_val), donde X son las 
            características y y es el target 'item_cnt_month'.
    """
    sales_for_train = monthly_sales[monthly_sales["date_block_num"] < val_block].copy()
    sales_for_validation = monthly_sales[monthly_sales["date_block_num"] == val_block].copy()

    x_train = sales_for_train.drop(columns=["item_cnt_month"])
    y_train = sales_for_train["item_cnt_month"]

    x_val = sales_for_validation.drop(columns=["item_cnt_month"])
    y_val = sales_for_validation["item_cnt_month"]

    return x_train, y_train, x_val, y_val

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
