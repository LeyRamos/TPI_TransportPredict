# Databricks notebook source
# MAGIC %pip install pmdarima
# MAGIC %pip install arch
# MAGIC %pip install keras
# MAGIC %pip install optree
# MAGIC %pip install tensorflow
# MAGIC %pip install keras-tuner

# COMMAND ----------

# DBTITLE 1,Restart kernel
dbutils.library.restartPython()

# COMMAND ----------

import pyspark

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DateType, TimestampType
from pyspark.sql.functions import explode, posexplode

import pmdarima as pm
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# COMMAND ----------

df_bel_stream_15m_complete = spark.table("default.df_bel_stream_15m_complete")

df_bel_stream_15m_sub = df_bel_stream_15m_complete.filter((df_bel_stream_15m_complete['street']== '1593'))

#df_bel_stream_15m_sub = df_bel_stream_15m_complete.filter((df_bel_stream_15m_complete['street']== '1593') & (df_bel_stream_15m_complete['datetime_full'] >= '2019-01-25 00:00:00.000+00:00'))


#df_bel_stream_15m_sub = df_bel_stream_15m_complete.filter((df_bel_stream_15m_complete['street']== '1593') | (df_bel_stream_15m_complete['street']== '2613') | (df_bel_stream_15m_complete['street']== '1845'))

# COMMAND ----------

df_bel_stream_15m_sub = df_bel_stream_15m_sub.withColumn("datetime", col("datetime_full"))

df_bel_stream_15m_sub = df_bel_stream_15m_sub.orderBy("datetime_full")

pdf = df_bel_stream_15m_sub.toPandas()

# Asegúrate de que la columna de fecha esté en el formato de fecha correcto
pdf['datetime_full'] = pd.to_datetime(pdf['datetime_full'])

# Establece la columna de fecha como índice
pdf.set_index('datetime_full', inplace=True)


# COMMAND ----------

# MAGIC %md
# MAGIC ### ARIMA con training-testing ok

# COMMAND ----------


# Crear listas para almacenar predicciones
predictions = []

# Definir la ventana de una semana y el intervalo de predicción (1 hora)
window_size = 7 * 24 * 4  # Una semana de datos con intervalos de 15 minutos
prediction_interval = 4  # 1 hora de datos con intervalos de 15 minutos

# Recorrer los datos en ventanas deslizantes
for end in range(window_size, len(pdf), prediction_interval):
    start = end - window_size
    train_data = pdf.iloc[start:end]["count"]
    
    # Ajustar el modelo auto_arima
    model = pm.auto_arima(train_data, seasonal=False, stepwise=True)
    
    # Hacer predicción para el siguiente intervalo de 1 hora
    pred = model.predict(n_periods=prediction_interval)
    
    # Agregar predicciones a la lista
    predictions.extend(pred)

# Convertir predicciones a DataFrame de pandas
pred_df = pd.DataFrame(predictions, index=pdf.index[window_size:window_size+len(predictions)], columns=["predictions"])

# Unir las predicciones con los datos originales
result_df = pdf.join(pred_df, how="left")


# COMMAND ----------

display(result_df)

# COMMAND ----------

# Resumen del modelo
print(model.summary())

# COMMAND ----------

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Extraer valores reales y predicciones
y_true = result_df.dropna(subset=['predictions'])['count']
y_pred = result_df.dropna(subset=['predictions'])['predictions']

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    #msle = mean_squared_log_error(y_true, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R2): {r2}")
    #print(f"Mean Squared Log Error (MSLE): {msle}")

# Llamar a la función para evaluar
evaluate_model(y_true, y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ARIMA con exogenous

# COMMAND ----------

from pmdarima.arima.utils import nsdiffs

# estimate number of seasonal differences using a Canova-Hansen test
ch = nsdiffs(pdf['count'],
            m=672,  # commonly requires knowledge of dataset
            max_D=5000,
            test='ch')  # -> 0

# or use the OCSB test (by default)
ocsb = nsdiffs(pdf['count'],
        m=672,
        max_D=5000,
        test='ocsb')  # -> 0

# COMMAND ----------

ch

# COMMAND ----------

ocsb

# COMMAND ----------

# Crear listas para almacenar predicciones
predictions = []

# Definir la ventana de una semana y el intervalo de predicción (1 hora)
window_size = 7 * 24 * 4  # Una semana de datos con intervalos de 15 minutos
prediction_interval = 4  # 1 hora de datos con intervalos de 15 minutos

# Recorrer los datos en ventanas deslizantes
for end in range(window_size, len(pdf), prediction_interval):
    start = end - window_size
    train_data = pdf.iloc[start:end]["count"]
    
    # Ajustar el modelo auto_arima
    model = pm.auto_arima(train_data, seasonal=True, stepwise=True) #exogenous , m= 3
    
    # Hacer predicción para el siguiente intervalo de 1 hora
    pred = model.predict(n_periods=prediction_interval)
    
    # Agregar predicciones a la lista
    predictions.extend(pred)

# Convertir predicciones a DataFrame de pandas
pred_df = pd.DataFrame(predictions, index=pdf.index[window_size:window_size+len(predictions)], columns=["predictions"])

# Unir las predicciones con los datos originales
result_df = pdf.join(pred_df, how="left")


# COMMAND ----------

# Resumen del modelo
print(model.summary())

# COMMAND ----------

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Extraer valores reales y predicciones
y_true = result_df.dropna(subset=['predictions'])['count']
y_pred = result_df.dropna(subset=['predictions'])['predictions']

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    #msle = mean_squared_log_error(y_true, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R2): {r2}")
    #print(f"Mean Squared Log Error (MSLE): {msle}")

# Llamar a la función para evaluar
evaluate_model(y_true, y_pred)

# COMMAND ----------

# Crear listas para almacenar predicciones
predictions = []

# Definir la ventana de una semana y el intervalo de predicción (1 hora)
window_size = 7 * 24 * 4  # Una semana de datos con intervalos de 15 minutos
prediction_interval = 4  # 1 hora de datos con intervalos de 15 minutos

# Recorrer los datos en ventanas deslizantes
for end in range(window_size, len(pdf), prediction_interval):
    start = end - window_size
    train_data = pdf.iloc[start:end]["count"]
    
    # Ajustar el modelo auto_arima
    model = pm.auto_arima(train_data, seasonal=True, stepwise=True, m= 3) #exogenous , m= 3
    
    # Hacer predicción para el siguiente intervalo de 1 hora
    pred = model.predict(n_periods=prediction_interval)
    
    # Agregar predicciones a la lista
    predictions.extend(pred)

# Convertir predicciones a DataFrame de pandas
pred_df = pd.DataFrame(predictions, index=pdf.index[window_size:window_size+len(predictions)], columns=["predictions"])

# Unir las predicciones con los datos originales
result_df = pdf.join(pred_df, how="left")

# COMMAND ----------

# Resumen del modelo
print(model.summary())

# COMMAND ----------

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Extraer valores reales y predicciones
y_true = result_df.dropna(subset=['predictions'])['count']
y_pred = result_df.dropna(subset=['predictions'])['predictions']

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    #msle = mean_squared_log_error(y_true, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R2): {r2}")
    #print(f"Mean Squared Log Error (MSLE): {msle}")

# Llamar a la función para evaluar
evaluate_model(y_true, y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ARIMA con transformacion box cox (no termina de correr)

# COMMAND ----------


import numpy as np
from scipy.stats import boxcox
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Crear listas para almacenar predicciones
predictions = []

# Definir la ventana de una semana y el intervalo de predicción (1 hora)
window_size = 7 * 24 * 4  # Una semana de datos con intervalos de 15 minutos
prediction_interval = 4  # 1 hora de datos con intervalos de 15 minutos

# Recorrer los datos en ventanas deslizantes
for end in range(window_size, len(pdf), prediction_interval):
    start = end - window_size
    train_data = pdf.iloc[start:end]["count"]

    # Verificar que los datos sean estrictamente positivos
    if (train_data <= 0).any():
        offset = 1e-6
        train_data_adjusted = train_data + offset
    else:
        train_data_adjusted = train_data

    # Transformar la serie de tiempo
    train_data_transformed, lambda_value = boxcox(train_data_adjusted)
    
    # Ajustar el modelo auto_arima
    model = pm.auto_arima(train_data_transformed, seasonal=False, stepwise=True)
    
    # Hacer predicción para el siguiente intervalo de 1 hora
    pred = model.predict(n_periods=prediction_interval)

    # Invertir la transformación Box-Cox
    if lambda_value == 0:
        pred_inv = np.exp(pred)
    else:
        pred_inv = (pred * lambda_value + 1) ** (1 / lambda_value) - 1
    
    # Agregar predicciones a la lista
    predictions.extend(pred_inv)

# Convertir predicciones a DataFrame de pandas
pred_df = pd.DataFrame(predictions, index=pdf.index[window_size:window_size+len(predictions)], columns=["predictions"])

# Unir las predicciones con los datos originales
result_df = pdf.join(pred_df, how="left")


# COMMAND ----------

# Evaluar los residuos
residuals = model.resid()
plt.plot(residuals)
plt.show()

# Verificar la normalidad y homocedasticidad de los residuos
from statsmodels.graphics.gofplots import qqplot
qqplot(residuals, line='s')
plt.show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### MODELO GARCH

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optener p y q de GARCH

# COMMAND ----------

# DBTITLE 1,calculo de retornos
offset = 0.1
pdf['count'] = pdf['count']+ offset

# Supongamos que pdf contiene nuestra serie de tiempo
pdf['count'] = pdf['count'].replace(0, np.nan)  # Reemplaza ceros por NaN

# Calcular retornos logarítmicos (si no están ya en este formato)
returns = np.log(pdf['count']).diff().dropna()

# Escalar los retornos
scaling_factor = 10
scaled_returns = returns * scaling_factor

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Graficar ACF y PACF de retornos al cuadrado para seleccionar p y q

# Generar el correlograma
fig, ax = plt.subplots(figsize=(15, 5))
plot_acf(scaled_returns**2, lags=20, ax=ax)

ax.set_title('Función de Autocorrelación (ACF) de Retornos cuadrados', fontsize=16)
ax.set_xlabel('Lags', fontsize=10)
ax.set_ylabel('Autocorrelación', fontsize=10)

# COMMAND ----------

# Generar el correlograma
fig, ax = plt.subplots(figsize=(15, 5))
plot_pacf(scaled_returns**2, lags=20, ax=ax)

ax.set_title('Función de Autocorrelación Parcial (PACF) de Retornos cuadrados', fontsize=16)
ax.set_xlabel('Lags', fontsize=10)
ax.set_ylabel('Autocorrelación Parcial', fontsize=10)

# COMMAND ----------

from statsmodels.tsa.stattools import acf, pacf

acf_values = acf(returns**2, nlags=30)
pacf_values = pacf(returns**2, nlags=30)
acf_values


# COMMAND ----------

pacf_values

# COMMAND ----------

import numpy as np
from arch import arch_model

def select_best_garch_model(returns, max_p=5, max_q=5):
    best_aic = np.inf
    best_bic = np.inf
    best_order = None
    best_model = None
    
    aic_values = []
    bic_values = []
    orders = []

    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:               
                model = arch_model(returns, vol='Garch', mean='ARX', p=p, q=q)
                model_fit = model.fit(disp="off")
                aic = model_fit.aic
                bic = model_fit.bic
                aic_values.append(aic)
                bic_values.append(bic)
                orders.append((p, q))
                
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, q)
                    best_model = model_fit
                if bic < best_bic:
                    best_bic = bic
            except Exception as e:
                continue
    
    return best_order, best_model, best_aic, best_bic, orders, aic_values, bic_values

# Suponiendo que tienes una serie temporal de retornos escalados 'scaled_returns'
# Seleccionar el mejor modelo GARCH
best_order, best_model, best_aic, best_bic, orders, aic_values, bic_values = select_best_garch_model(scaled_returns)

print(f"Mejores parámetros (p, q): {best_order}")
print(f"Mejor AIC: {best_aic}")
print(f"Mejor BIC: {best_bic}")
print(best_model.summary())

# Imprimir los valores de AIC y BIC para cada combinación de p y q
for order, aic, bic in zip(orders, aic_values, bic_values):
    print(f"Orden (p, q): {order}, AIC: {aic}, BIC: {bic}")


# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# Graficar los valores de AIC y BIC
fig, ax = plt.subplots(figsize=(12, 6))

order_labels = [f"({p},{q})" for (p, q) in orders]

ax.plot(order_labels, aic_values, marker='o', label='AIC')
ax.plot(order_labels, bic_values, marker='s', label='BIC')

ax.set_xlabel('Parámetros(p, q)')
ax.set_ylabel('Valor')
ax.set_title('Valores AIC y BIC para diferentes modelos GARCH(p, q)')
ax.legend()
ax.grid(True)

# Rotar las etiquetas del eje x para que se vean mejor
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,seleccionar parametros de GARCH
def select_best_garch_model(returns, max_p=5, max_q=5):
    best_aic = np.inf
    best_bic = np.inf
    best_order = None
    best_model = None

    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:               
                model = arch_model(returns, vol='GARCH', mean='ARX', p=p, q=q)
                model_fit = model.fit(disp="off")
                aic = model_fit.aic
                bic = model_fit.bic
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, q)
                    best_model = model_fit
                if bic < best_bic:
                    best_bic = bic
            except Exception as e:
                continue
    
    return best_order, best_model, best_aic, best_bic

# Seleccionar el mejor modelo GARCH
best_order, best_model, best_aic, best_bic = select_best_garch_model(scaled_returns)

print(f"Mejores parámetros (p, q): {best_order}")
print(f"Mejor AIC: {best_aic}")
print(f"Mejor BIC: {best_bic}")
print(best_model.summary())

# COMMAND ----------

# Evaluar residuos del mejor modelo
standardized_residuals = best_model.resid / best_model.conditional_volatility

# COMMAND ----------

# Graficar ACF y PACF de residuos estandarizados
plot_acf(standardized_residuals, lags=40)
plt.title('ACF of Standardized Residuals')
plt.show()

# COMMAND ----------

plot_pacf(standardized_residuals, lags=40)
plt.title('PACF of Standardized Residuals')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Parámetros Principales
# MAGIC - **y**: La serie temporal a la que se ajustará el modelo ARIMA. Este es un parámetro obligatorio.
# MAGIC - **exogenous**: Un array opcional de variables exógenas (predictoras) para el modelo. Debe tener la misma longitud que y. 
# MAGIC - **seasonal**: (booleano, por defecto True) Si se debe ajustar un modelo ARIMA estacional.
# MAGIC - **m**: (entero, por defecto 1) El número de períodos en una temporada completa. Por ejemplo, m=12 para datos mensuales si esperas un patrón anual.
# MAGIC - **d**: (entero o None) El orden de diferenciación. Si es None, se determinará automáticamente usando pruebas estadísticas.
# MAGIC - **D**: (entero o None) El orden de diferenciación estacional. Si es None, se determinará automáticamente usando pruebas estadísticas.
# MAGIC - start_p: (entero, por defecto 2) El valor inicial de p en la búsqueda grid (AR).
# MAGIC - start_q: (entero, por defecto 2) El valor inicial de q en la búsqueda grid (MA).
# MAGIC - max_p: (entero, por defecto 5) El valor máximo de p en la búsqueda grid.
# MAGIC - max_q: (entero, por defecto 5) El valor máximo de q en la búsqueda grid.
# MAGIC - start_P: (entero, por defecto 1) El valor inicial de P en la búsqueda grid (SAR).
# MAGIC - start_Q: (entero, por defecto 1) El valor inicial de Q en la búsqueda grid (SMA).
# MAGIC - max_P: (entero, por defecto 2) El valor máximo de P en la búsqueda grid.
# MAGIC - max_Q: (entero, por defecto 2) El valor máximo de Q en la búsqueda grid.
# MAGIC - max_d: (entero, por defecto 2) El valor máximo de d en la búsqueda grid.
# MAGIC - max_D: (entero, por defecto 1) El valor máximo de D en la búsqueda grid.
# MAGIC - **seasonal_test**: (string, por defecto 'ocsb') El nombre de la prueba estadística que se utilizará para determinar la estacionalidad. Puede ser 'ocsb' o 'ch'.
# MAGIC - stepwise: (booleano, por defecto True) Si se debe utilizar un procedimiento stepwise (paso a paso) para la búsqueda de parámetros. Esto puede hacer que la búsqueda sea más rápida.
# MAGIC - n_jobs: (entero, por defecto 1) El número de trabajos (hilos) a ejecutar en paralelo para la búsqueda de parámetros. Utilizar -1 para usar todos los procesadores disponibles.
# MAGIC - max_order: (entero, por defecto 5) La suma máxima de p, d, q, P, D y Q. Este parámetro se usa para limitar la complejidad del modelo.
# MAGIC - information_criterion: (string, por defecto 'aic') El criterio de información que se utilizará para comparar modelos. Puede ser 'aic', 'bic', 'hqic', etc.
# MAGIC - trace: (booleano, por defecto False) Si se debe imprimir la salida de la búsqueda de parámetros.
# MAGIC - error_action: (string, por defecto 'warn') Qué hacer si hay un error al ajustar un modelo. Las opciones incluyen 'warn', 'ignore', 'raise' y 'trace'.
# MAGIC - suppress_warnings: (booleano, por defecto False) Si se deben suprimir las advertencias de pmdarima.
# MAGIC - random: (booleano, por defecto False) Si se debe realizar una búsqueda aleatoria en lugar de una búsqueda grid completa.
# MAGIC - **random_state**: (entero o None) La semilla para la búsqueda aleatoria.
# MAGIC - n_fits: (entero, por defecto 10) El número de modelos que se ajustarán si random=True.

# COMMAND ----------

# Extrae la serie temporal
ts = pdf['count']

# Aplica auto_arima para encontrar los mejores parámetros
model = pm.auto_arima(ts, seasonal=False, stepwise=True)

# Resumen del modelo
print(model.summary())


# COMMAND ----------

# Realiza un pronóstico de los próximos 10 períodos
forecast, conf_int = model.predict(n_periods=4, return_conf_int=True)

# Crear un DataFrame de pandas con los resultados
future_dates = pd.date_range(start=ts.index[-1], periods=4, freq='D')
forecast_df = pd.DataFrame({'date': future_dates, 'forecast': forecast, 'lower_ci': conf_int[:, 0], 'upper_ci': conf_int[:, 1]})

# Convertir de nuevo a un DataFrame de PySpark
forecast_spark_df = spark.createDataFrame(forecast_df)

# Mostrar el DataFrame de PySpark con los resultados del pronóstico
forecast_spark_df.show()


# COMMAND ----------

display(forecast_spark_df)

# COMMAND ----------

import pmdarima as pm
from scipy.stats import boxcox
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot

# Supongamos que 'data' es tu serie temporal original
# data = pd.Series(...)  # Tu serie temporal aquí

# Aplicar la transformación de Box-Cox
data_boxcox, lambda_ = boxcox(data)
print(f'Transformación de Box-Cox aplicada con lambda = {lambda_}')

# Ajustar el modelo auto_arima a los datos transformados
model = pm.auto_arima(data_boxcox, seasonal=False, stepwise=True, suppress_warnings=True)

# Residuos del modelo
residuals = model.resid()

# Realizar la prueba de Shapiro-Wilk en los residuos transformados
from scipy.stats import shapiro

shapiro_test_stat, shapiro_p_value = shapiro(residuals)
print(f'Shapiro-Wilk Test (Residuos Transformados): Statistic={shapiro_test_stat}, p-value={shapiro_p_value}')

# Plot Q-Q para los residuos transformados
qqplot(residuals, line='s')
plt.title('Q-Q Plot de los Residuos Transformados')
plt.show()

# Realizar la prueba de Ljung-Box nuevamente en los residuos transformados
from statsmodels.stats.diagnostic import acorr_ljungbox
import pandas as pd

residuals_series = pd.Series(residuals)
ljung_box_result = acorr_ljungbox(residuals_series, lags=[10], return_df=True)
print(ljung_box_result)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Modelo GARCH prueba 1

# COMMAND ----------


import numpy as np
from arch import arch_model

# Crear listas para almacenar predicciones
predictions = []

# Definir la ventana de una semana y el intervalo de predicción (1 hora)
window_size = 7 * 24 * 4  # Una semana de datos con intervalos de 15 minutos
prediction_interval = 4  # 1 hora de datos con intervalos de 15 minutos

# Recorrer los datos en ventanas deslizantes
for end in range(window_size, len(pdf), prediction_interval):
    start = end - window_size
    train_data = pdf.iloc[start:end]["count"]
    
    # Ajustar el modelo GARCH(1,1)
    model = arch_model(train_data, vol='Garch', p=1, q=2) #rescale=False
    model_fit = model.fit(disp='off')
    
    # Hacer predicción para el siguiente intervalo de 1 hora
    forecast = model_fit.forecast(horizon=prediction_interval)
    pred = forecast.mean.iloc[-1].values
    
    # Agregar predicciones a la lista
    predictions.extend(pred)

# Convertir predicciones a DataFrame de pandas
pred_df = pd.DataFrame(predictions, index=pdf.index[window_size:window_size+len(predictions)], columns=["predictions"])

# Unir las predicciones con los datos originales
result_df = pdf.join(pred_df, how="left")




# COMMAND ----------

# DBTITLE 1,MODELO GARCH CORRECTO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
import numpy as np
from arch import arch_model

# Crear listas para almacenar predicciones
predictions = []

# Definir la ventana de una semana y el intervalo de predicción (1 hora)
window_size = 7 * 24 * 4  # Una semana de datos con intervalos de 15 minutos
prediction_interval = 4  # 1 hora de datos con intervalos de 15 minutos

# Recorrer los datos en ventanas deslizantes
for end in range(window_size, len(pdf), prediction_interval):
    start = end - window_size
    train_data = pdf.iloc[start:end]["count"]

    offset = 0.1
    train_data_adjusted = train_data + offset
    
    # Calcular retornos logarítmicos
    log_returns = np.log(train_data_adjusted).diff().dropna()

    # Asegurarse de que no hay NaNs
    #log_returns = log_returns.dropna()
    
    # Escalar los retornos
    scaling_factor = 10
    scaled_returns = log_returns * scaling_factor
    
    # Ajustar el modelo GARCH(1,2) a los retornos escalados
    model = arch_model(scaled_returns, vol='Garch', p=1, q=2)
    model_fit = model.fit(disp='off')
    
    # Hacer predicción para el siguiente intervalo de 1 hora
    forecast = model_fit.forecast(horizon=prediction_interval)
    pred_scaled = forecast.mean.iloc[-1].values / scaling_factor  # Desescalar la predicción
    
    # Convertir los retornos predichos de regreso a count
    last_count = train_data_adjusted.iloc[-1]
    pred_count = last_count * np.exp(pred_scaled) - offset  # Ajustar por el offset
    
    # Agregar predicciones a la lista
    predictions.extend(pred_count)

# Convertir predicciones a DataFrame de pandas
pred_df = pd.DataFrame(predictions, index=pdf.index[window_size:window_size+len(predictions)], columns=["predictions"])

# Unir las predicciones con los datos originales
result_df = pdf.join(pred_df, how="left")


# COMMAND ----------

display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Agregar el Offset Antes de Convertir a Retornos Logarítmicos:
# MAGIC El offset se suma a los datos para evitar valores no positivos. Asegúrate de que el offset también se maneje correctamente cuando realices las predicciones.
# MAGIC
# MAGIC Convertir los Retornos Logarítmicos Predichos de Regreso a los Valores Originales:
# MAGIC Para convertir los retornos logarítmicos de regreso a los valores originales, necesitas tener en cuenta el offset que agregaste al principio.
# MAGIC
# MAGIC Escalado y Desescalado:
# MAGIC Escalas los retornos logarítmicos antes de ajustar el modelo y desescala las predicciones después del pronóstico.
# MAGIC
# MAGIC Conversión a count:
# MAGIC Después de obtener las predicciones en términos de retornos logarítmicos escalados, conviértelas de vuelta a los valores originales usando el último valor del count ajustado y ajustando por el offset.
# MAGIC Índice de Predicciones:
# MAGIC
# MAGIC Asegúrate de que el índice de pred_df coincida con el índice del DataFrame original para que la unión sea correcta.

# COMMAND ----------

# Resumen del ajuste del modelo
print(model_fit.summary())


# COMMAND ----------

from sklearn.metrics import mean_squared_error

# Calcular RMSE
rmse = np.sqrt(mean_squared_error(pdf["count"].iloc[window_size:window_size+len(predictions)], predictions))
print(f'Root Mean Squared Error: {rmse}')


# COMMAND ----------

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Obtener residuos estandarizados
standardized_residuals = model_fit.resid / model_fit.conditional_volatility

# Histograma de residuos estandarizados
plt.figure(figsize=(10, 6))
plt.hist(standardized_residuals, bins=50)
plt.title('Histogram of Standardized Residuals')
plt.show()




# COMMAND ----------

# ACF y PACF de residuos estandarizados
plot_acf(standardized_residuals, lags=30)
plt.title('ACF of Standardized Residuals')
plt.show()



# COMMAND ----------

plot_pacf(standardized_residuals, lags=30)
plt.title('PACF of Standardized Residuals')
plt.show()

# COMMAND ----------

from statsmodels.stats.diagnostic import het_arch

# Prueba de Engle
arch_test = het_arch(standardized_residuals)
print(f'Engle test p-value: {arch_test[1]}')


# COMMAND ----------

# MAGIC %md
# MAGIC ### GARCH con mean ARX

# COMMAND ----------

import numpy as np
from arch import arch_model

# Crear listas para almacenar predicciones
predictions = []

# Definir la ventana de una semana y el intervalo de predicción (1 hora)
window_size = 7 * 24 * 4  # Una semana de datos con intervalos de 15 minutos
prediction_interval = 4  # 1 hora de datos con intervalos de 15 minutos

# Recorrer los datos en ventanas deslizantes
for end in range(window_size, len(pdf), prediction_interval):
    start = end - window_size
    train_data = pdf.iloc[start:end]["count"]

    offset = 0.1
    train_data_adjusted = train_data + offset
    
    # Calcular retornos logarítmicos
    log_returns = np.log(train_data_adjusted).diff().dropna()

    # Asegurarse de que no hay NaNs
    #log_returns = log_returns.dropna()
    
    # Escalar los retornos
    scaling_factor = 10
    scaled_returns = log_returns * scaling_factor
    
    # Ajustar el modelo GARCH(1,1) a los retornos escalados
    model = arch_model(scaled_returns, vol='Garch', mean='ARX', p=1, q=2)
    model_fit = model.fit(disp='off')
    
    # Hacer predicción para el siguiente intervalo de 1 hora
    forecast = model_fit.forecast(horizon=prediction_interval)
    pred_scaled = forecast.mean.iloc[-1].values / scaling_factor  # Desescalar la predicción
    
    # Convertir los retornos predichos de regreso a count
    last_count = train_data_adjusted.iloc[-1]
    pred_count = last_count * np.exp(pred_scaled) - offset  # Ajustar por el offset
    
    # Agregar predicciones a la lista
    predictions.extend(pred_count)

# Convertir predicciones a DataFrame de pandas
pred_garch = pd.DataFrame(predictions, index=pdf.index[window_size:window_size+len(predictions)], columns=["predictions"])

# Unir las predicciones con los datos originales
result_garch = pdf.join(pred_garch, how="left")


# COMMAND ----------

# Resumen del ajuste del modelo
print(model_fit.summary())

# COMMAND ----------

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Calcular RMSE
mse = np.sqrt(mean_squared_error(
    pdf["count"].iloc[window_size:window_size + len(predictions)], 
    predictions
))
mae = mean_absolute_error(
    pdf["count"].iloc[window_size:window_size + len(predictions)], 
    predictions
)
r2 = r2_score(
    pdf["count"].iloc[window_size:window_size + len(predictions)], 
    predictions
)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Obtener residuos estandarizados
standardized_residuals = model_fit.resid / model_fit.conditional_volatility

# Histograma de residuos estandarizados
# Configurar el estilo de Seaborn
sns.set(style="whitegrid")

# Crear la figura y el eje
plt.figure(figsize=(15, 6))

# Crear el histograma con la curva KDE
sns.histplot(standardized_residuals, bins=50, color='#077A9D', kde=False, stat='density')
sns.kdeplot(standardized_residuals, color='#C20114')

# Limitar los rangos de X
plt.xlim(-5, 5)

# Agregar títulos y etiquetas
plt.title('Histograma de Residuos estandarizados', fontsize=16)
plt.xlabel('Residuos estandarizados', fontsize=14)
plt.ylabel('Frecuencia', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Mostrar la gráfica
plt.show()

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Graficar ACF y PACF de retornos al cuadrado para seleccionar p y q

# Generar el correlograma
fig, ax = plt.subplots(figsize=(15, 5))
plot_acf(standardized_residuals, lags=10, ax=ax)

ax.set_title('Función de Autocorrelación (ACF) de Residuos estandarizados', fontsize=16)
ax.set_xlabel('Lags', fontsize=10)
ax.set_ylabel('Autocorrelación', fontsize=10)

# COMMAND ----------

# Generar el correlograma
fig, ax = plt.subplots(figsize=(15, 5))
plot_pacf(standardized_residuals, lags=10, ax=ax)

ax.set_title('Función de Autocorrelación Parcial (PACF) de Residuos estandarizados', fontsize=16)
ax.set_xlabel('Lags', fontsize=10)
ax.set_ylabel('Autocorrelación Parcial', fontsize=10)

# COMMAND ----------

from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(standardized_residuals, model_fit.conditional_volatility)

# COMMAND ----------

# MAGIC %md
# MAGIC ### GARCH con mean ARX y exogenous = speed

# COMMAND ----------

import numpy as np
import pandas as pd
from arch import arch_model

# Crear listas para almacenar predicciones
predictions = []

# Definir la ventana de una semana y el intervalo de predicción (1 hora)
window_size = 7 * 24 * 4  # Una semana de datos con intervalos de 15 minutos
prediction_interval = 4  # 1 hora de datos con intervalos de 15 minutos

# Recorrer los datos en ventanas deslizantes
for end in range(window_size, len(pdf), prediction_interval):
    start = end - window_size
    train_data = pdf.iloc[start:end]["count"]
    train_data_speed = pdf.iloc[start:end]["speed"]

    offset = 0.1
    train_data_adjusted = train_data + offset
    
    # Calcular retornos logarítmicos
    log_returns = np.log(train_data_adjusted).diff().dropna()
    train_data_speed = train_data_speed[1:]  # Ajustar tamaño de train_data_speed

    # Sincronizar los índices de log_returns y train_data_speed
    log_returns, train_data_speed = log_returns.align(train_data_speed, join='inner')

    # Escalar los retornos
    scaling_factor = 10
    scaled_returns = log_returns * scaling_factor
    
    # Ajustar el modelo GARCH(1,2) a los retornos escalados con la variable externa
    model = arch_model(scaled_returns, vol='Garch', mean='ARX', p=1, q=2, x=train_data_speed.values.reshape(-1, 1))
    model_fit = model.fit(disp='off')
    
    # Obtener las últimas observaciones de speed para el horizonte de predicción
    future_speed = pdf.iloc[end:end + prediction_interval]["speed"].values.reshape(-1, 1)
    future_speed_list = future_speed.flatten().tolist()

    # Hacer predicción para el siguiente intervalo de 1 hora
    forecast = model_fit.forecast(horizon=prediction_interval, x=future_speed_list)
    
    # Ajustar predicciones con datos exógenos futuros
    mean_forecast = forecast.mean.iloc[-prediction_interval:].values.flatten()
    pred_scaled = mean_forecast / scaling_factor  # Desescalar la predicción
    
    # Convertir los retornos predichos de regreso a count
    last_count = train_data_adjusted.iloc[-1]
    pred_count = last_count * np.exp(pred_scaled) - offset  # Ajustar por el offset
    
    # Agregar predicciones a la lista
    predictions.extend(pred_count)

# Convertir predicciones a DataFrame de pandas
pred_df = pd.DataFrame(predictions, index=pdf.index[window_size:window_size + len(predictions)], columns=["predictions"])

# Unir las predicciones con los datos originales
result_df = pdf.join(pred_df, how="left")



# COMMAND ----------

# Resumen del ajuste del modelo
print(model_fit.summary())

# COMMAND ----------

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Calcular RMSE
mse = np.sqrt(mean_squared_error(
    pdf["count"].iloc[window_size:window_size + len(predictions)], 
    predictions
))
mae = mean_absolute_error(
    pdf["count"].iloc[window_size:window_size + len(predictions)], 
    predictions
)
r2 = r2_score(
    pdf["count"].iloc[window_size:window_size + len(predictions)], 
    predictions
)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Obtener residuos estandarizados
standardized_residuals = model_fit.resid / model_fit.conditional_volatility

# Histograma de residuos estandarizados
# Configurar el estilo de Seaborn
sns.set(style="whitegrid")

# Crear la figura y el eje
plt.figure(figsize=(15, 6))

# Crear el histograma con la curva KDE
sns.histplot(standardized_residuals, bins=50, color='#077A9D', kde=False, stat='density')
sns.kdeplot(standardized_residuals, color='#C20114')

# Limitar los rangos de X
plt.xlim(-5, 5)

# Agregar títulos y etiquetas
plt.title('Histograma de Residuos estandarizados', fontsize=16)
plt.xlabel('Residuos estandarizados', fontsize=14)
plt.ylabel('Frecuencia', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Mostrar la gráfica
plt.show()

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Graficar ACF y PACF de retornos al cuadrado para seleccionar p y q

# Generar el correlograma
fig, ax = plt.subplots(figsize=(15, 5))
plot_acf(standardized_residuals, lags=10, ax=ax)

ax.set_title('Función de Autocorrelación (ACF) de Residuos estandarizados', fontsize=16)
ax.set_xlabel('Lags', fontsize=10)
ax.set_ylabel('Autocorrelación', fontsize=10)

# COMMAND ----------

# Generar el correlograma
fig, ax = plt.subplots(figsize=(15, 5))
plot_pacf(standardized_residuals, lags=10, ax=ax)

ax.set_title('Función de Autocorrelación Parcial (PACF) de Residuos estandarizados', fontsize=16)
ax.set_xlabel('Lags', fontsize=10)
ax.set_ylabel('Autocorrelación Parcial', fontsize=10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Redes LSTM

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Definir la función create_dataset
def create_dataset(series, timesteps=1, prediction_interval=4):
    X, y = [], []
    for i in range(len(series) - timesteps - prediction_interval + 1):
        X.append(series[i:(i + timesteps), 0])
        y.append(series[i + timesteps + prediction_interval - 1, 0])
    return np.array(X), np.array(y)

# Normalizar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(pdf["count"].values.reshape(-1, 1))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Redes LSTM - Optimización Bayesiana

# COMMAND ----------

from sklearn.model_selection import train_test_split
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

# Paso 1: Preparar los datos
timesteps = 672  # Ventana de 2 semanas
prediction_interval = 4  # Predicción de 1 hora

# Crear el dataset para la LSTM
X, y = create_dataset(scaled_data, timesteps, prediction_interval)

# Reshape de los datos para LSTM [samples, timesteps, features]
X = np.reshape(X, (X.shape[0], timesteps, 1))

# Dividir los datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 2: Definir el modelo para la búsqueda
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units_1', min_value=32, max_value=128, step=16),
                   return_sequences=True, 
                   input_shape=(timesteps, 1)))
    
    model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=128, step=16)))
    
    model.add(Dense(1))
    
    model.compile(
        optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='mean_squared_error')
    
    return model

# Paso 3: Configurar la optimización bayesiana
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_loss',  # Minimiza el error de validación
    max_trials=10,  # Número de configuraciones de hiperparámetros a probar
    directory='bayesian_opt',
    project_name='lstm_optimization'
)

# COMMAND ----------

# DBTITLE 1,Best Model
# Paso 4: Realizar la búsqueda
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Paso 5: Obtener los mejores hiperparámetros y entrenar el modelo final
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
#model = tuner.hypermodel.build(best_hps)
#model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))

print(f"Best number of units in first LSTM layer: {best_hps.get('units_1')}")
print(f"Best number of units in second LSTM layer: {best_hps.get('units_2')}")
print(f"Best learning rate: {best_hps.get('learning_rate')}")

# COMMAND ----------

# DBTITLE 1,Recolectar hiperparámetros para graficar
import matplotlib.pyplot as plt
import pandas as pd

# Obtén el historial completo de los resultados de las pruebas
all_trials = tuner.oracle.get_best_trials(num_trials=10)

# Listas para almacenar los datos
units_1 = []
units_2 = []
learning_rates = []
val_losses = []

# Extraer datos de cada prueba
for trial in all_trials:
    # Obtener los hiperparámetros de la prueba
    hyperparameters = trial.hyperparameters
    # Obtener las métricas de la prueba
    metrics = trial.metrics

    # Extraer los valores de los hiperparámetros
    units_1.append(hyperparameters.get('units_1'))
    units_2.append(hyperparameters.get('units_2'))
    learning_rates.append(hyperparameters.get('learning_rate'))
    
    # Extraer el valor de la métrica val_loss
    val_loss = trial.metrics.get_best_value('val_loss')
    val_losses.append(val_loss)

# Crear un DataFrame para los resultados
results_df = pd.DataFrame({
    'units_1': units_1,
    'units_2': units_2,
    'learning_rate': learning_rates,
    'val_loss': val_losses
})


# COMMAND ----------

# Graficar los resultados
plt.figure(figsize=(15, 5))

# Gráfico de 'units_1' vs 'val_loss'
plt.subplot(1, 3, 1)
plt.scatter(results_df['units_1'], results_df['val_loss'])
plt.xlabel('Unidades en primera capa LSTM')
plt.ylabel('MSE')
plt.title('Unidades en primera capa LSTM vs MSE')

# Gráfico de 'units_2' vs 'val_loss'
plt.subplot(1, 3, 2)
plt.scatter(results_df['units_2'], results_df['val_loss'])
plt.xlabel('Unidades en segunda capa LSTM')
plt.ylabel('MSE')
plt.title('Unidades en segunda capa LSTM vs MSE')

# Gráfico de 'learning_rate' vs 'val_loss'
plt.subplot(1, 3, 3)
plt.scatter(results_df['learning_rate'], results_df['val_loss'])
plt.xlabel('Tasa de aprendizaje')
plt.ylabel('MSE')
plt.title('Tasa de aprendizaje vs MSE')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Modelo LSTM con best model

# COMMAND ----------

from keras.optimizers import Adam

# 1) Definir dataset
timesteps = 672
prediction_interval = 4
X, y = create_dataset(scaled_data, timesteps, prediction_interval)

# Reshape de los datos para LSTM [samples, timesteps, features]
X = np.reshape(X, (X.shape[0], timesteps, 1))

# 2) definir el modelo LSTM
# Definir el optimizador Adam con el learning rate óptimo
optimizer = Adam(learning_rate=0.00534)

# Definir el modelo LSTM
model = Sequential()
model.add(LSTM(96, return_sequences=True, input_shape=(timesteps, 1)))
model.add(LSTM(48))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=optimizer)

# 3) Entrenar el modelo
model_lstm = model.fit(X, y, epochs=10, batch_size=4, validation_split=0.2, verbose=1)

# 4) Hacer predicciones
predictions = model.predict(X)

# Desescalar las predicciones
predictions = scaler.inverse_transform(predictions)

# Crear un índice para las predicciones
pred_index = pdf.index[timesteps + prediction_interval - 1: timesteps + prediction_interval - 1 + len(predictions)]

# Convertir las predicciones a DataFrame de pandas
pred_lstm = pd.DataFrame(predictions, index=pred_index, columns=["predictions"])

# Unir las predicciones con los datos originales
result_lstm = pdf.join(pred_lstm, how="left")

# COMMAND ----------

# DBTITLE 1,Evaluar Convergencia: Epoch vs Loss
# Entrenar el modelo
#model_lstm = model.fit(X_train, y_train, epochs=10, batch_size=4, validation_split=0.2, verbose=1)

# Graficar la pérdida de entrenamiento y validación
plt.figure(figsize=(15, 5))
plt.plot(model_lstm.history['loss'], label='Pérdida de MSE Entrenamiento')
if 'val_loss' in model_lstm.history:
    plt.plot(model_lstm.history['val_loss'], label='Pérdida de MSE Validación')
plt.title('Epoch vs Pérdida de MSE')
plt.xlabel('Epochs')
plt.ylabel('Pérdida de MSE')
plt.legend()
plt.show()



# COMMAND ----------

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Elimina filas con NaN
result_lstm_clean = result_lstm.dropna(subset=['count', 'predictions'])

# Calcular métricas con los datos limpios
mse = mean_squared_error(result_lstm_clean['count'], result_lstm_clean['predictions'])
mae = mean_absolute_error(result_lstm_clean['count'], result_lstm_clean['predictions'])
r2 = r2_score(result_lstm_clean['count'], result_lstm_clean['predictions'])

# Imprimir resultados
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

# COMMAND ----------

import matplotlib.pyplot as plt


# Graficar los resultados
plt.figure(figsize=(12, 6))
plt.plot(result_lstm['count'], label='Original')
plt.plot(result_lstm['predictions'], label='Predicciones', color='red')
plt.legend()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Combinación de predicciones: AR-GARCH + LSTM

# COMMAND ----------

#Rename columns
result_garch = result_garch.rename(columns={
    "predictions": "pred_garch"
})

result_lstm = result_lstm.rename(columns={
    "street": "street_lstm",
    "datetime": "datetime_lstm",
    "predictions": "pred_lstm"
})

result_lstm = result_lstm[["street_lstm", "datetime_lstm", "pred_lstm"]]

result_comb = result_garch.merge(
    result_lstm,
    left_on=['street', 'datetime_full'],
    right_on=['street_lstm', 'datetime_lstm'],
    how='left'
)

#Merge results
result_comb = result_comb.drop(columns=['street_lstm', 'datetime_lstm'])

#Calculate combination
# Simple avg
result_comb['pred_comb'] = (result_comb['pred_garch'] + result_comb['pred_lstm']) / 2

# Pond average by r2
r2_garch= 0.895569813769839
r2_lstm= 0.9503031268205974
total_r2 = r2_garch + r2_lstm
result_comb['pred_comb_pond_r2'] = ((result_comb['pred_garch'] * r2_garch) + (result_comb['pred_lstm'] * r2_lstm)) / total_r2

# Pond average by mse
mse_garch= 1/41.019582090951495
mse_lstm= 1/800.6696096855311
total_mse = mse_garch + mse_lstm
result_comb['pred_comb_pond_mse'] = ((result_comb['pred_garch'] * mse_garch) + (result_comb['pred_lstm'] * mse_lstm)) / total_mse

# Pond average by mae
mae_garch= 1/26.686690044921303
mae_lstm= 1/18.616731509042925
total_mae = mae_garch + mae_lstm
result_comb['pred_comb_pond_mae'] = ((result_comb['pred_garch'] * mae_garch) + (result_comb['pred_lstm'] * mae_lstm)) / total_mae

# Diferencias absolutas
result_comb['pred_garch_diff'] = result_comb['pred_garch'] - result_comb['count']
result_comb['pred_lstm_diff'] = result_comb['pred_lstm'] - result_comb['count']
result_comb['pred_comb_diff'] = result_comb['pred_comb'] - result_comb['count']
result_comb['pred_comb_r2_diff'] = result_comb['pred_comb_pond_r2'] - result_comb['count']
result_comb['pred_comb_mse_diff'] = result_comb['pred_comb_pond_mse'] - result_comb['count']
result_comb['pred_comb_mae_diff'] = result_comb['pred_comb_pond_mae'] - result_comb['count']

# COMMAND ----------

display(result_comb)

# COMMAND ----------

# DBTITLE 1,Save as table
result_comb_sp = spark.createDataFrame(result_comb)

result_comb_sp.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("default.result_comb")

# COMMAND ----------

from pyspark.sql.functions import col, to_timestamp

result_comb_sp = result_comb_sp.withColumn(
    'datetime', 
    to_timestamp(col('datetime'))
)

display(
    result_comb_sp.filter(
        (col('datetime') >= '2019-02-03T21:00:00.000+00:00') & 
        (col('datetime') <= '2019-02-10T23:45:00.000+00:00')
    )
)

# COMMAND ----------

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Elimina filas con NaN
result_comb_clean = result_comb.dropna(subset=['count', 'pred_comb'])

# Calcular métricas con los datos limpios
mse = mean_squared_error(result_comb_clean['count'], result_comb_clean['pred_comb'])
mae = mean_absolute_error(result_comb_clean['count'], result_comb_clean['pred_comb'])
r2 = r2_score(result_comb_clean['count'], result_comb_clean['pred_comb'])

# Imprimir resultados
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

# COMMAND ----------

# Elimina filas con NaN
result_comb_clean = result_comb.dropna(subset=['count', 'pred_comb_pond_r2'])

# Calcular métricas con los datos limpios
mse = mean_squared_error(result_comb_clean['count'], result_comb_clean['pred_comb_pond_r2'])
mae = mean_absolute_error(result_comb_clean['count'], result_comb_clean['pred_comb_pond_r2'])
r2 = r2_score(result_comb_clean['count'], result_comb_clean['pred_comb_pond_r2'])

# Imprimir resultados
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

# COMMAND ----------

# Elimina filas con NaN
result_comb_clean = result_comb.dropna(subset=['count', 'pred_comb_pond_mse'])

# Calcular métricas con los datos limpios
mse = mean_squared_error(result_comb_clean['count'], result_comb_clean['pred_comb_pond_mse'])
mae = mean_absolute_error(result_comb_clean['count'], result_comb_clean['pred_comb_pond_mse'])
r2 = r2_score(result_comb_clean['count'], result_comb_clean['pred_comb_pond_mse'])

# Imprimir resultados
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

# COMMAND ----------

# Elimina filas con NaN
result_comb_clean = result_comb.dropna(subset=['count', 'pred_comb_pond_mae'])

# Calcular métricas con los datos limpios
mse = mean_squared_error(result_comb_clean['count'], result_comb_clean['pred_comb_pond_mae'])
mae = mean_absolute_error(result_comb_clean['count'], result_comb_clean['pred_comb_pond_mae'])
r2 = r2_score(result_comb_clean['count'], result_comb_clean['pred_comb_pond_mae'])

# Imprimir resultados
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")
