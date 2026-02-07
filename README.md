# Tarea 3. Optimización de inventarios

## Descripción del proyecto

Este repositorio contiene un pipeline end-to-end de Machine Learning para pronosticar demanda/ventas mensuales a nivel tienda–producto, desarrollado para 1C Company, una firma de software con operación retail a gran escala. El objetivo del proyecto es mejorar la planeación de inventarios y la toma de decisiones operativas mediante predicciones granulares (shop_id, item_id, date_block_num) que permitan anticipar sobrestock y quiebres de stock.

El trabajo está diseñado para ser reproducible y desplegable, separando claramente las etapas de preparación de datos, entrenamiento del modelo y generación de predicciones batch. El modelo base utiliza XGBoost sobre agregaciones mensuales y controles de valores extremos (clipping 0–20), generando un artefacto de modelo versionable y una salida de predicciones lista para integración con procesos de negocios.


---

## Estructura del repositorio

```text
.
├── artifacts/                 # Modelos entrenados y outputs del pipeline
│   ├── model.joblib
│   ├── modelo_xgboost.pkl
│   └── logs/                  # Logs del pipeline
├── data/
│   ├── raw/                   # Datos originales (entrada de prep)
│   │   ├── sales_train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv                   
│   ├── prep/                  # Datos preparados (salida de prep)
│   ├── inference/             # Datos para inferencia batch
│   ├── predictions/           # Predicciones generadas
|   └── images/
├── src/                       # Scripts del pipeline
│   ├── 
│   ├── prep.py
│   ├── train.py
│   ├── inference.py
│   └── utils/
│       ├── __init__.py
│       └── functions.py 
├── main.py                    # Entrypoint general
├── pyproject.toml             # Dependencias / configuración del proyecto
└── uv.lock                    # Lockfile de dependencias (reproducibilidad)

```
---

## Instalación y setup
Este repo usa uv para instalar dependencias y ejecutar scripts de forma reproducible.

**Requisitos**:
- Versión de Python >= 3.12
- uv instalado

---

## Cómo ejecutar el pipeline

Se debe de ejecutar estos comandos desde la raíz del repositorio:
- uv run python -m src.prep
- uv run python -m src.train
- uv run python -m src.inference

Outputs esperados:
- `src.prep`
  - **Input:** `data/raw/*.csv`
  - **Output:** `data/prep/monthly_sales.csv`

- `src.train`
  - **Input:** `data/prep/monthly_sales.csv`
  - **Output:** `artifacts/model.joblib` (modelo entrenado)

- `src.inference`
  - **Inputs:**
    - `data/inference/test.csv`
    - `data/raw/sample_submission.csv`
    - `artifacts/model.joblib`
  - **Output:** `data/predictions/Prediccion_Equipo2.csv`


---

### Descripción de cada script 

#### `src/prep.py`
- Lee datos crudos de `data/raw/`
- Genera agregación mensual y limpieza básica (por ejemplo, clipping de valores extremos)
- Guarda dataset preparado en `data/prep/monthly_sales.csv`

#### `src/train.py`
- Carga `data/prep/monthly_sales.csv`
- Separa train/validación por `date_block_num`
- Entrena `XGBRegressor` y reporta RMSE de validación
- Guarda el modelo entrenado en `artifacts/model.joblib`

#### `src/inference.py`
- Carga modelo (`artifacts/model.joblib`)
- Aplica inferencia batch sobre `data/inference/test.csv`
- Genera archivo de predicción en formato Kaggle: `data/predictions/Prediccion_Equipo2.csv`


#### Logging
- Se creó el directorio de logs: mkdir artifacts/logs


---

### Métricas del modelo

- RMSE (validación): ≈ 2.37
- RMSE reportado (resumen): ≈ 2.36
- Kaggle Public Score: 1.98078
     - Link: https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/leaderboard

---

### Dependencias principales

- pandas
- numpy
- scikit-learn
- xgboost
- joblib

### Calidad del Código
![Pylint](data/images/pylint.png)