# Databricks notebook source
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


from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

# Asume que ya tienes una sesión de Spark activa
# spark = SparkSession.builder.appName("YourAppName").getOrCreate()

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

# Asumimos que df_spark es un DataFrame de Spark que contiene una columna 'ID' para la segmentación
for id_value in df.select("street").distinct().rdd.flatMap(lambda x: x).collect():
    # Filtrar los datos para el ID actual en Spark
    df_segment_spark = df.filter(col("street") == id_value)
    
    # Convertir el DataFrame de Spark a Pandas
    pdf = df_segment_spark.toPandas()

    # Asegurarte de que la columna de fecha esté en el formato correcto
    pdf['datetime_full'] = pd.to_datetime(pdf['datetime_full'])
    
    # Establecer la columna de fecha como índice
    pdf.set_index('datetime_full', inplace=True)

    # Normalizar los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(pdf["count"].values.reshape(-1, 1))

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
    pred_index = pdf.index[timesteps + prediction_interval - 1: timesteps + prediction_interval - 1 + len(predictions)]
    
    # Convertir las predicciones a DataFrame de pandas
    pred_df = pd.DataFrame(predictions, index=pred_index, columns=["predictions"])
    
    # Agregar una columna con el ID para la fusión posterior
    pred_df['street'] = id_value
    
    # Añadir las predicciones a la lista de resultados
    all_predictions.append(pred_df)

# Concatenar todas las predicciones en un único DataFrame de Pandas
pred_lstm = pd.concat(all_predictions)

# Convertir el DataFrame de Pandas de nuevo a Spark para unirlo con el DataFrame original
pred_lstm_spark = spark.createDataFrame(pred_lstm.reset_index())

# Unir las predicciones con los datos originales en Spark
df_spark_with_pred = df.join(pred_lstm_spark, on=['street', 'datetime_full'], how='left')


# COMMAND ----------

all_predictions

# COMMAND ----------

# Concatenar todas las predicciones en un único DataFrame de Pandas
pred_lstm = pd.concat(all_predictions)

# Convertir el DataFrame de Pandas de nuevo a Spark para unirlo con el DataFrame original
pred_lstm_spark = spark.createDataFrame(pred_lstm.reset_index())

# Unir las predicciones con los datos originales en Spark
df_spark_with_pred = df.join(pred_lstm_spark, on=['street', 'datetime_full'], how='left')


# COMMAND ----------

display(
    df_spark_with_pred.filter(
        df_spark_with_pred['street'].isin(
            ['2904', '4032', '3959', '3414', '296', '2162', '829', '2294', '1512', '467', '2088', '1159', '1436', '2136', '3210', '1090', '2069', '691', '675', '4937', '3606', '1572', '4821', '2756', '3517', '5149', '2110', '5023', '3441', '3281', '800', '5067', '4975', '944', '2275', '2464', '853', '3858', '1669', '1394', '3015', '451', '4838', '2393', '1372', '3650']
        )
    )
)


# COMMAND ----------

df_spark_with_pred.createOrReplaceTempView('df_spark_with_pred')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC street,
# MAGIC COUNT(DISTINCT(predictions)) as preds
# MAGIC FROM df_spark_with_pred
# MAGIC GROUP BY street
# MAGIC having COUNT(DISTINCT(predictions)) > 1000

# COMMAND ----------

df_spark_with_pred_filter = df_spark_with_pred.filter(
        df_spark_with_pred['street'].isin(
            ['2904', '4032', '3959', '3414', '296', '2162', '829', '2294', '1512', '467', '2088', '1159', '1436', '2136', '3210', '1090', '2069', '691', '675', '4937', '3606', '1572', '4821', '2756', '3517', '5149', '2110', '5023', '3441', '3281', '800', '5067', '4975', '944', '2275', '2464', '853', '3858', '1669', '1394', '3015', '451', '4838', '2393', '1372', '3650']
        )
    )

df_spark_with_pred_filter = df_spark_with_pred_filter.drop('datetime_full')

# COMMAND ----------

df_spark_with_pred1593 = spark.table("default.result_comb")

# COMMAND ----------

df_spark_with_pred1593 = df_spark_with_pred1593.drop(
'pred_garch',
'pred_comb',
'pred_comb_pond_r2',
'pred_comb_pond_mse',
'pred_comb_pond_mae',
'pred_garch_diff',
'pred_lstm_diff',
'pred_comb_diff',
'pred_comb_r2_diff',
'pred_comb_mse_diff',
'pred_comb_mae_diff'
)

df_spark_with_pred1593 = df_spark_with_pred1593.withColumnRenamed("pred_lstm", "predictions")

# COMMAND ----------

df_spark_with_pred_filter = df_spark_with_pred1593.unionByName(df_spark_with_pred_filter)
display(df_spark_with_pred_filter)

# COMMAND ----------

df_spark_with_pred_filter.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("default.result_lstm_47streets")

# COMMAND ----------

# Configurate access to Blob Storage
storage_account_name = 'stgaclnrmlab'
storage_account_access_key = 'rf/4ogYc/eG+oqVD8K9xjsVamosf1qO1s0Kab+ujHsTt0GjaGY2XHfXFNVdti4iaUndCJjNSqizi+ASt8IWqHw=='
spark.conf.set(f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net", storage_account_access_key)

# Blob Storage directory
blob_container = 'data'

# Save df_spark_with_pred_filter as CSV in Blob Storage
df_spark_with_pred_filter.write \
    .format("csv") \
    .mode("overwrite") \
    .option("header", "true") \
    .save(f"wasbs://{blob_container}@{storage_account_name}.blob.core.windows.net/df_spark_with_pred_filter.csv")
