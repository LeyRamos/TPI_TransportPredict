# Databricks notebook source
import pyspark
from pyspark.sql import SparkSession

from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DateType, TimestampType

from pyspark.sql.functions import explode
from pyspark.sql.functions import posexplode

# COMMAND ----------

df_bel_stream_15m_complete = spark.table("default.df_bel_stream_15m_complete")

# COMMAND ----------

df_bel_stream_15m_complete.createOrReplaceTempView('df_bel_stream_15m_complete_vw')

# COMMAND ----------

result_topstreets = spark.sql("""
SELECT
    street,
    count(*) as reg,
    sum(count) as countn
FROM df_bel_stream_15m_complete_vw
GROUP BY street
ORDER BY countn DESC
LIMIT 100
""")

# Convertir la columna 'street' en una lista
Top_street_list = [row['street'] for row in result_topstreets.collect()]

# COMMAND ----------

#df = df_bel_stream_15m_complete # .filter((df_bel_stream_15m_complete['street']== '1593'))

#pdf = df_bel_stream_15m_complete.filter((df_bel_stream_15m_complete['street']== '1593') | (df_bel_stream_15m_complete['street']== '2613') | (df_bel_stream_15m_complete['street']== '1845')).toPandas()

#pdf = df_bel_stream_15m_complete.filter((df_bel_stream_15m_complete['street']== '1593')).toPandas()

#pdf = df_bel_stream_15m_complete.toPandas()

pdf = df_bel_stream_15m_complete.filter(df_bel_stream_15m_complete['street'].isin(Top_street_list)).toPandas()

# COMMAND ----------

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import numpy as np

pdf_ = pdf

pdf_['count'] = pdf_['count'].replace(0, np.nan)  # Reemplaza ceros por NaN

# Asegurarse de que no hay NaNs
pdf_ = pdf_.dropna()


# Calcular la ACF
acf_values = acf(pdf_['count'], nlags=10)
pacf_values = pacf(pdf_['count'], nlags=10)

# COMMAND ----------

# Generar el correlograma
fig, ax = plt.subplots(figsize=(15, 5))
plot_acf(pdf_['count'], lags=10, ax=ax)

ax.set_title('Función de Autocorrelación (ACF)', fontsize=16)
ax.set_xlabel('Lags', fontsize=10)
ax.set_ylabel('Autocorrelación', fontsize=10)

plt.show()

# COMMAND ----------

acf_values

# COMMAND ----------


# Generar el correlograma
fig, ax = plt.subplots(figsize=(15, 5))
plot_pacf(pdf['count'], lags=10, ax=ax)

ax.set_title('Función de Autocorrelación Parcial (PACF)', fontsize=16)
ax.set_xlabel('Lags', fontsize=10)
ax.set_ylabel('Autocorrelación Parcial', fontsize=10)

plt.show()

# COMMAND ----------

pacf_values

# COMMAND ----------

df_decomposed = spark.read.table("default.df_bel_15m_serie1593_decomposed")

# COMMAND ----------

pdf_decomposed = df_decomposed.toPandas()
