import pytest
import pandas as pd
import numpy as np
from src.inference.inference import create_lags, split_train_val_with_lags

def test_split_train_val_with_lags_structure():
    """
    Verifica que la función divida correctamente y mantenga las 6 columnas esperadas.
    """
    # Dataframe de prueba
    data = []
    for month in range(35):
        data.append({
            'date_block_num': month,
            'shop_id': 1,
            'item_id': 10,
            'item_cnt_month': float(month) # Ventas ascendentes para rastrear lags
        })
    df = pd.DataFrame(data)
    val_block = 33
    x_train, y_train, x_val, y_val = split_train_val_with_lags(df, val_block=val_block)
    assert x_train['date_block_num'].min() == 4 # Verificación de que se filtraron los primeros 3 meses, porque aquí no hay datos
    assert (x_val['date_block_num'] == val_block).all() # x_val solo tiene el bloque solicitado
    assert x_train.shape[1] == 6 # Verificación de las 6 columnas que espera XGBoost
    assert list(x_train.columns) == [
        'date_block_num', 'shop_id', 'item_id', 
        'item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3'
    ]
    sample_lag = x_train.loc[x_train['date_block_num'] == 10, 'item_cnt_month_lag_1'].values[0] # Verificación lógica del lag. Ejemplo: el lag_1 del mes 10debe ser el mes 9
    assert sample_lag == 9.0

def test_lags_fillna_is_zero():
    """
    Verifica que no queden valores NaN en las columnas de lags
    después de procesar los datos.
    """
    # Dataframe de prueba
    data = pd.DataFrame({
        'date_block_num': [4, 5],
        'shop_id': [1, 1],
        'item_id': [10, 10],
        'item_cnt_month': [100.0, 200.0]
    })
    x_train, y_train, x_val, y_val = split_train_val_with_lags(data, val_block=5) # Bloque 5 para validación

    # Verificación que no existan NaNs
    assert x_train.isna().sum().sum() == 0, "Se encontraron NaNs en x_train" 
    assert x_val.isna().sum().sum() == 0, "Se encontraron NaNs en x_val"
    
    # Verificación que los lags sin datos sean 0.0 y no NaN.
    assert (x_train.loc[x_train['date_block_num'] == 4, 
           ['item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3']] == 0.0).all().all()