# Databricks notebook source
# MAGIC %pip install arch
# MAGIC %pip install keras
# MAGIC %pip install tensorflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pyspark

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DateType, TimestampType
from pyspark.sql.functions import explode, posexplode

from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

# COMMAND ----------

df_bel_stream_15m_complete = spark.table("default.df_bel_stream_15m_complete")

df_bel_stream_15m_complete = df_bel_stream_15m_complete.withColumn("datetime", col("datetime_full"))
df_bel_stream_15m_complete = df_bel_stream_15m_complete.orderBy("datetime_full")

df = df_bel_stream_15m_complete

# COMMAND ----------

pdf = df.toPandas()

# Asegúrate de que la columna de fecha esté en el formato de fecha correcto
pdf['datetime_full'] = pd.to_datetime(pdf['datetime_full'])

# Establece la columna de fecha como índice
pdf.set_index('datetime_full', inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Modelo ARX-GARCH - Pipeline

# COMMAND ----------

# Predict over each street
def predict_for_segment(segment, window_size, prediction_interval):
    predictions = []
    
    for end in range(window_size, len(segment), prediction_interval):
        start = end - window_size
        train_data = segment.iloc[start:end]["count"]

        offset = 0.1
        train_data_adjusted = train_data + offset
        
        # Calcular retornos logarítmicos
        log_returns = np.log(train_data_adjusted).diff().dropna()
        
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
    
    # Crear DataFrame con las predicciones
    pred_df = pd.DataFrame(predictions, index=segment.index[window_size:window_size+len(predictions)], columns=["predictions"])
    return pred_df


# COMMAND ----------

# Función para aplicar el pipeline a todos los segmentos
def apply_pipeline_to_all_segments(df_spark, id_column, window_size, prediction_interval):
    result_df = pd.DataFrame()

    # Obtener la lista de valores únicos de ID
    unique_ids = df_spark.select(id_column).distinct().rdd.flatMap(lambda x: x).collect()

    for id_value in unique_ids:
        # Filtrar el DataFrame de Spark para el segmento específico
        df_segment_spark = df_spark.filter(df_spark[id_column] == id_value)
        
        # Convertir el DataFrame de Spark a Pandas
        pdf = df_segment_spark.toPandas()

        # Asegurarte de que la columna de fecha esté en el formato correcto
        pdf['datetime_full'] = pd.to_datetime(pdf['datetime_full'])
        
        # Establecer la columna de fecha como índice
        pdf.set_index('datetime_full', inplace=True)

        # Aplicar la función de predicción
        pred_df = predict_for_segment(pdf, window_size, prediction_interval)
        
        # Agregar la columna ID a las predicciones para mantener la referencia
        pred_df[id_column] = id_value
        
        # Hacer append de las predicciones al resultado final
        result_df = pd.concat([result_df, pred_df], axis=0)
    
    return result_df

# COMMAND ----------

# Definir los parámetros
window_size = 7 * 24 * 4  # Una semana de datos con intervalos de 15 minutos
prediction_interval = 4  # 1 hora de datos con intervalos de 15 minutos

pred_garch = apply_pipeline_to_all_segments(df, 'street', window_size, prediction_interval)

# COMMAND ----------

display(pred_garch)

# COMMAND ----------

pred_garch_spark = spark.createDataFrame(pred_garch.reset_index())

# COMMAND ----------

result_garch = df.join(
    pred_garch_spark,
    (df["street"] == pred_garch_spark["ID"]) & (df["datetime_full"] == pred_garch_spark["index"]),
    how="left"
)

# Opcional: Si quieres renombrar las columnas después del join
result_comb = result_garch.withColumnRenamed("pred_garch", "pred_garch_spark")

# COMMAND ----------

# Unir las predicciones con los datos originales
pdf_with_pred_garch = pdf.join(pred_garch.set_index(['ID', pred_garch.index]), how="left", on=['ID', pdf.index])


# COMMAND ----------

def apply_pipeline_to_all_segments(pdf, id_column, window_size, prediction_interval):
    result_df = pd.DataFrame()

    # Agrupar el dataframe por ID y aplicar la función de predicción
    for id_value, segment in pdf.groupby(id_column):
        pred_df = predict_for_segment(segment, window_size, prediction_interval)
        
        # Agregar la columna ID a las predicciones para mantener la referencia
        pred_df[id_column] = id_value
        
        # Hacer append de las predicciones al resultado final
        result_df = pd.concat([result_df, pred_df], axis=0)
    
    return result_df

# Definir los parámetros
window_size = 7 * 24 * 4  # Una semana de datos con intervalos de 15 minutos
prediction_interval = 4  # 1 hora de datos con intervalos de 15 minutos

# Aplicar el pipeline al dataframe segmentado por el ID
pred_garch = apply_pipeline_to_all_segments(pdf, 'ID', window_size, prediction_interval)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Modelo LSTM - Pipeline

# COMMAND ----------

# Definir la función create_dataset
def create_dataset(series, timesteps=1, prediction_interval=4):
    X, y = [], []
    for i in range(len(series) - timesteps - prediction_interval + 1):
        X.append(series[i:(i + timesteps), 0])
        y.append(series[i + timesteps + prediction_interval - 1, 0])
    return np.array(X), np.array(y)

# Parámetros del modelo
timesteps = 672
prediction_interval = 4
learning_rate = 0.00534
epochs = 9
batch_size = 4

# Crear una lista para almacenar los resultados
all_predictions = []

# Asumimos que pdf es un DataFrame que contiene una columna 'ID' para la segmentación
for id_value in pdf['ID'].unique():
    # Filtrar los datos para el ID actual
    data_segment = pdf[pdf['ID'] == id_value]
    
    # Normalizar los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_segment["count"].values.reshape(-1, 1))

    # Crear el dataset para la LSTM
    X, y = create_dataset(scaled_data, timesteps, prediction_interval)
    
    # Reshape de los datos para LSTM [samples, timesteps, features]
    X = np.reshape(X, (X.shape[0], timesteps, 1))
    
    # Definir el optimizador Adam con el learning rate óptimo
    optimizer = Adam(learning_rate=learning_rate)

    # Definir el modelo LSTM
    model = Sequential()
    model.add(LSTM(96, return_sequences=True, input_shape=(timesteps, 1)))
    model.add(LSTM(48))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    # Entrenar el modelo
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    
    # Hacer predicciones
    predictions = model.predict(X)
    
    # Desescalar las predicciones
    predictions = scaler.inverse_transform(predictions)
    
    # Crear un índice para las predicciones
    pred_index = data_segment.index[timesteps + prediction_interval - 1: timesteps + prediction_interval - 1 + len(predictions)]
    
    # Convertir las predicciones a DataFrame de pandas
    pred_df = pd.DataFrame(predictions, index=pred_index, columns=["predictions"])
    
    # Agregar una columna con el ID para la fusión posterior
    pred_df['ID'] = id_value
    
    # Añadir las predicciones a la lista de resultados
    all_predictions.append(pred_df)

# Concatenar todas las predicciones en un único DataFrame
pred_lstm = pd.concat(all_predictions)

# Unir las predicciones con los datos originales
pdf_with_pred_lstm = pdf.merge(pred_lstm, on=['ID'], how='left')

