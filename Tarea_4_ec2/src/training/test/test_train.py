import pytest
import pandas as pd
import numpy as np
from src.training.train import create_lags

def test_create_lags_logic():
    """
    Verifica que la función create_lags genere las columnas correctas.
    """
    # Dataframe de prueba
    data = pd.DataFrame({
        'date_block_num': [0, 1, 2],
        'shop_id': [1, 1, 1],
        'item_id': [10, 10, 10],
        'item_cnt_month': [5, 10, 15]
    })
    result = create_lags(data, lags=[1])
    assert 'item_cnt_month_lag_1' in result.columns # Verifica que se haya creado la columna de lag
    assert result.loc[result['date_block_num'] == 1, 'item_cnt_month_lag_1'].values[0] == 5 #Lag del bloque 1 debe ser igual al valor del bloque 0, en este caso 5
