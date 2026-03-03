import pytest
import pandas as pd
import numpy as np
from src.preprocessing.prep import rename_columns, clip_values

def test_rename_columns():
    """Verifica que la columna de conteo cambie su nombre a 'item_cnt_month'."""
    df = pd.DataFrame({
        'date_block_num': [1], 'shop_id': [1], 'item_id': [1], 'item_cnt_day': [10]
    })
    result = rename_columns(df)
    assert 'item_cnt_month' in result.columns
    assert 'item_cnt_day' not in result.columns

def test_clip_values():
    """Verifica que los valores se mantengan dentro del rango [0, 20]."""
    df = pd.DataFrame({'item_cnt_month': [-5, 10, 50]})
    result = clip_values(df, lower=0, upper=20)
    assert result['item_cnt_month'].min() == 0
    assert result['item_cnt_month'].max() == 20