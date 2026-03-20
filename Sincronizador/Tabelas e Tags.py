# Databricks notebook source
# MAGIC %md
# MAGIC # Projeto Sincronizador - README
# MAGIC
# MAGIC Este notebook tem como objetivo mapear todas as tabelas necessárias, features e tags, organizando as etapas do projeto sincronizador.
# MAGIC
# MAGIC ## Estrutura do Notebook
# MAGIC
# MAGIC 1. **Mapeamento de Tabelas**
# MAGIC    - Listagem das tabelas envolvidas no projeto
# MAGIC    - Descrição de cada tabela e sua finalidade
# MAGIC
# MAGIC 2. **Definição de Features**
# MAGIC    - Identificação das principais features utilizadas
# MAGIC    - Documentação das transformações e cálculos aplicados
# MAGIC
# MAGIC 3. **Tags e Classificações**
# MAGIC    - Mapeamento de tags relevantes para o projeto
# MAGIC    - Critérios de classificação e agrupamento
# MAGIC
# MAGIC 4. **Etapas do Projeto**
# MAGIC    - Levantamento dos passos necessários para sincronização
# MAGIC    - Cronograma e checkpoints de desenvolvimento
# MAGIC
# MAGIC ## Como Utilizar
# MAGIC
# MAGIC - Utilize este notebook para consultar o mapeamento das tabelas e features.
# MAGIC - Atualize as seções conforme novas tabelas ou features forem identificadas.
# MAGIC - Registre o progresso das etapas do projeto para manter o time sincronizado.
# MAGIC
# MAGIC ## Contato
# MAGIC
# MAGIC Em caso de dúvidas ou sugestões, entre em contato com o responsável pelo projeto.

# COMMAND ----------

# DBTITLE 1,IMPORTS
import pyspark.sql.functions as F
from pyspark.sql.window import Window



# COMMAND ----------

# MAGIC %md
# MAGIC # tb_aciaria_vd

# COMMAND ----------

# DBTITLE 1,tb_aciaria_vd
# Display the table
df_vd = spark.read.table("industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_vd")
display(df_vd)

# Print all columns in alphabetical order
print(sorted(df_vd.columns))

# Show the number of rows
print(f"Total rows: {df_vd.count()}")

# Show the earliest and latest dates in the 'timestamp' column

min_max = df_vd.select(
    F.min("timestamp").alias("min_timestamp"),
    F.max("timestamp").alias("max_timestamp")
).collect()[0]
print(f"timestamp: starts at {min_max['min_timestamp']}, ends at {min_max['max_timestamp']}")

# Verificar se a coluna surrogate_key é sempre nula ou tem valores
null_count = df_vd.filter(F.col("surrogate_key").isNull()).count()
notnull_count = df_vd.filter(F.col("surrogate_key").isNotNull()).count()
print(f"surrogate_key: nula em {null_count} linhas, não nula em {notnull_count} linhas")

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

# Definir o intervalo de tempo desejado
start_time = "2025-12-02T12:00:00.000+00:00"
end_time = "2025-12-02T18:00:00.000+00:00"

# Filtrar o DataFrame para o intervalo de 6 horas
df_vd_filtered = df_vd.filter(
    (F.col("timestamp") >= F.lit(start_time)) &
    (F.col("timestamp") <= F.lit(end_time))
).orderBy("timestamp")

# Converter para Pandas para plotar (apenas para o intervalo filtrado)
pdf = df_vd_filtered.toPandas()

# Remover surrogate_key e transformar todas as colunas (exceto timestamp) em numéricas
pdf = pdf.drop(columns=["surrogate_key"], errors="ignore")
for col in pdf.columns:
    if col != "timestamp":
        pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

# Adicionar coluna com hora:minuto para o eixo x
pdf["hora_minuto"] = pd.to_datetime(pdf["timestamp"]).dt.strftime("%H:%M")

# Selecionar colunas numéricas para plotar
cols_to_plot = [col for col in pdf.columns if col not in ["timestamp", "hora_minuto"] and pd.api.types.is_numeric_dtype(pdf[col])]

# Plotar cada coluna em um gráfico separado usando matplotlib
for col in cols_to_plot:
    plt.figure()
    plt.plot(pdf["hora_minuto"], pdf[col])
    plt.title(f"{col} ao longo do tempo (HH:MM)")
    plt.xlabel("Hora:Minuto")
    plt.ylabel(col)
    # Definir ticks do eixo x a cada 30 minutos
    xticks = pdf["hora_minuto"].unique()
    xticks_30min = [tick for i, tick in enumerate(xticks) if i % 30 == 0]
    plt.xticks(xticks_30min, rotation=45)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # tb_aciaria_fp

# COMMAND ----------

# DBTITLE 1,tb_aciaria_fp

# Display the table
df_fp = spark.read.table("industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_fp")
display(df_fp)

# Print all columns in alphabetical order
print(sorted(df_fp.columns))

# Show the number of rows
print(f"Total rows: {df_fp.count()}")

# Show the earliest and latest dates in the 'timestamp' column

min_max = df_fp.select(
    F.min("timestamp").alias("min_timestamp"),
    F.max("timestamp").alias("max_timestamp")
).collect()[0]
print(f"timestamp: starts at {min_max['min_timestamp']}, ends at {min_max['max_timestamp']}")

# Verificar se a coluna surrogate_key é sempre nula ou tem valores
null_count = df_fp.filter(F.col("surrogate_key").isNull()).count()
notnull_count = df_fp.filter(F.col("surrogate_key").isNotNull()).count()
print(f"surrogate_key: nula em {null_count} linhas, não nula em {notnull_count} linhas")

# COMMAND ----------

# MAGIC %md
# MAGIC # tb_corridatempos

# COMMAND ----------

# DBTITLE 1,tb_corridatempos
df_corridatempos = spark.read.table("industrial_iatemperaturascha_refined_prd.computed_features.tb_corridatempos")
display(df_corridatempos)

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

# Definir o intervalo de tempo desejado
start_time = "2025-12-02T12:00:00.000+00:00"
end_time = "2025-12-02T18:00:00.000+00:00"

# Filtrar o DataFrame para o intervalo de 6 horas
df_fp_filtered = df_fp.filter(
    (F.col("timestamp") >= F.lit(start_time)) &
    (F.col("timestamp") <= F.lit(end_time))
).orderBy("timestamp")

# Converter para Pandas para plotar (apenas para o intervalo filtrado)
pdf = df_fp_filtered.toPandas()

# Remover surrogate_key e transformar todas as colunas (exceto timestamp) em numéricas
pdf = pdf.drop(columns=["surrogate_key"], errors="ignore")
for col in pdf.columns:
    if col != "timestamp":
        pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

# Selecionar colunas numéricas para plotar
cols_to_plot = [col for col in pdf.columns if col != "timestamp" and pd.api.types.is_numeric_dtype(pdf[col])]

# Plotar cada coluna em um gráfico separado usando matplotlib
for col in cols_to_plot:
    plt.figure()
    plt.plot(pdf["timestamp"], pdf[col])
    plt.title(f"{col} ao longo do tempo (segundo a segundo)")
    plt.xlabel("timestamp")
    plt.ylabel(col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # tb_admaci_lc_corridalingotamento

# COMMAND ----------

# DBTITLE 1,tb_admaci_lc_corridalingotamento


# Display the table
df_lc = spark.read.table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_lc_corridalingotamento").withColumnRenamed("headers_timestamp", "timestamp")
display(df_lc)

# Print all columns in alphabetical order
print(sorted(df_lc.columns))

# Show the number of rows
print(f"Total rows: {df_lc.count()}")

# Show the earliest and latest dates in the 'timestamp' column

min_max = df_lc.select(
    F.min("timestamp").alias("min_timestamp"),
    F.max("timestamp").alias("max_timestamp")
).collect()[0]
print(f"timestamp: starts at {min_max['min_timestamp']}, ends at {min_max['max_timestamp']}")


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

# Definir o intervalo de tempo desejado
start_time = "2025-12-02T12:00:00.000+00:00"
end_time = "2025-12-02T18:00:00.000+00:00"

# Filtrar o DataFrame para o intervalo de 6 horas
df_lc_filtered = df_lc.filter(
    (F.col("timestamp") >= F.lit(start_time)) &
    (F.col("timestamp") <= F.lit(end_time))
).orderBy("timestamp")

# Converter para Pandas para plotar (apenas para o intervalo filtrado)
pdf = df_lc_filtered.toPandas()

# Remover surrogate_key e transformar todas as colunas (exceto timestamp) em numéricas
pdf = pdf.drop(columns=["surrogate_key"], errors="ignore")
for col in pdf.columns:
    if col not in ["timestamp", "INICIOLINGOTAMENTO", "FINALLINGOTAMENTO"]:
        pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

# Converter INICIOLINGOTAMENTO e FINALLINGOTAMENTO para hora:minuto
if "INICIOLINGOTAMENTO" in pdf.columns:
    pdf["INICIOLINGOTAMENTO_HM"] = pd.to_datetime(pdf["INICIOLINGOTAMENTO"]).dt.strftime("%H:%M")
if "FINALLINGOTAMENTO" in pdf.columns:
    pdf["FINALLINGOTAMENTO_HM"] = pd.to_datetime(pdf["FINALLINGOTAMENTO"]).dt.strftime("%H:%M")

# Plotar valores de INICIOLINGOTAMENTO no eixo x (hora:minuto) e valores das CORRIDAs no eixo y
if "INICIOLINGOTAMENTO_HM" in pdf.columns:
    plt.figure()
    plt.scatter(pdf["INICIOLINGOTAMENTO_HM"], pdf["CORRIDA"], alpha=0.7)
    plt.title("Valores das CORRIDAs vs Início Lingotamento (Hora:Minuto)")
    plt.xlabel("INICIOLINGOTAMENTO (HH:MM)")
    plt.ylabel("CORRIDA")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plotar valores de FINALLINGOTAMENTO no eixo x (hora:minuto) e valores das CORRIDAs no eixo y
if "FINALLINGOTAMENTO_HM" in pdf.columns:
    plt.figure()
    plt.scatter(pdf["FINALLINGOTAMENTO_HM"], pdf["CORRIDA"], alpha=0.7)
    plt.title("Valores das CORRIDAs vs Final Lingotamento (Hora:Minuto)")
    plt.xlabel("FINALLINGOTAMENTO (HH:MM)")
    plt.ylabel("CORRIDA")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # tb_admaci_fea_corridas

# COMMAND ----------

# DBTITLE 1,tb_admaci_fea_corridas
# Display the table
df_fea = spark.read.table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fea_corridas").orderBy(F.col("CORRIDA").desc())
display(df_fea)

# Print all columns in alphabetical order
print(sorted(df_fea.columns))

# Show the number of rows
print(f"Total rows: {df_fea.count()}")

# Show the earliest and latest dates in the 'timestamp' column

# min_max = df_fea.select(
#     F.min("timestamp").alias("min_timestamp"),
#     F.max("timestamp").alias("max_timestamp")
# ).collect()[0]
# print(f"timestamp: starts at {min_max['min_timestamp']}, ends at {min_max['max_timestamp']}")


# COMMAND ----------





# Display the table
df_lc = spark.read.table("industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_lc")
display(df_lc)

# Print all columns in alphabetical order
print(sorted(df_lc.columns))

# Show the number of rows
print(f"Total rows: {df_lc.count()}")

# Show the earliest and latest dates in the 'timestamp' column

min_max = df_lc.select(
    F.min("timestamp").alias("min_timestamp"),
    F.max("timestamp").alias("max_timestamp")
).collect()[0]
print(f"timestamp: starts at {min_max['min_timestamp']}, ends at {min_max['max_timestamp']}")


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

# Definir o intervalo de tempo desejado
start_time = "2025-12-09T15:00:00.000+00:00"
end_time = "2025-12-09T23:00:00.000+00:00"

# Filtrar o DataFrame para o intervalo de 6 horas
df_lc_filtered = df_lc.filter(
    (F.col("timestamp") >= F.lit(start_time)) &
    (F.col("timestamp") <= F.lit(end_time))
).orderBy("timestamp")

# Converter para Pandas para plotar (apenas para o intervalo filtrado)
pdf = df_lc_filtered.toPandas()

# Remover surrogate_key e transformar todas as colunas (exceto timestamp) em numéricas
pdf = pdf.drop(columns=["surrogate_key"], errors="ignore")
for col in pdf.columns:
    if col != "timestamp":
        pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

# Converter timestamp para datetime em UTC-3 e criar coluna 'hora_minuto'
pdf["timestamp_utc3"] = pd.to_datetime(pdf["timestamp"]).dt.tz_localize("UTC").dt.tz_convert("America/Sao_Paulo")
pdf["hora_minuto"] = pdf["timestamp_utc3"].dt.strftime("%H:%M")

# Definir os horários das linhas verticais
ilc_hora = "13:33"
flc_hora = "14:26"

# Selecionar colunas numéricas para plotar
cols_to_plot = [col for col in pdf.columns if col not in ["timestamp", "timestamp_utc3", "hora_minuto"] and pd.api.types.is_numeric_dtype(pdf[col])]

# Plotar cada coluna em um gráfico separado usando matplotlib
for col in cols_to_plot:
    plt.figure()
    plt.plot(pdf["hora_minuto"], pdf[col])
    plt.title(f"{col}")
    plt.xlabel("Hora:Minuto")
    plt.ylabel(col)
    # Definir ticks do eixo x a cada 30 minutos
    xticks = pdf["hora_minuto"].unique()
    xticks_30min = [tick for i, tick in enumerate(xticks) if i % 30 == 0]
    plt.xticks(xticks_30min, rotation=45)
    # Adicionar linhas verticais pontilhadas
    plt.axvline(x=ilc_hora, color="red", linestyle="--", linewidth=1)
    plt.axvline(x=flc_hora, color="blue", linestyle="--", linewidth=1)
    # Adicionar texto nas linhas (sem fundo)
    plt.text(ilc_hora, plt.ylim()[1], "ILC - 130595", color="red", ha="center", va="top", fontsize=9, rotation=90)
    plt.text(flc_hora, plt.ylim()[1], "FLC - 130595", color="blue", ha="center", va="top", fontsize=9, rotation=90)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # tb_admaci_fp_corrida

# COMMAND ----------

# DBTITLE 1,tb_admaci_fp_corrida
# Local timezone
tb_admaci_fp_corrida = (
    spark.table(f"industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fp_corrida")
    .withColumn("corrida", F.col("corrida").cast("int"))
    .dropDuplicates(["corrida"])
    .dropna(subset=["horavazamento"])
    .orderBy("corrida", ascending=False)
)
 
 
tb_admaci_fp_corrida = (
    tb_admaci_fp_corrida
    .withColumn(
        "data_hora_vazamento",
        F.to_timestamp(
            F.concat(
                F.date_format(F.col("datavazamento"), "yyyy-MM-dd"), F.lit(" "), F.col("horavazamento")
            ),
            "yyyy-MM-dd HH:mm"
        )
    )
    .withColumn(
        "data_hora_chegadafp",
        F.to_timestamp(
            F.concat(
                F.date_format(F.col("datachegadafp"), "yyyy-MM-dd"), F.lit(" "), F.col("horachegadafp")
            ),
            "yyyy-MM-dd HH:mm"
        )
    )
    .withColumn(
        "data_hora_saidafp",
        F.to_timestamp(
            F.concat(
                F.date_format(F.col("datasaidafp"), "yyyy-MM-dd"), F.lit(" "), F.col("horasaidafp")
            ),
            "yyyy-MM-dd HH:mm"
        )
    )
)
 
display(tb_admaci_fp_corrida)

# COMMAND ----------

# =========================================================
# 0. IMPORTS
# =========================================================
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# =========================================================
# 1. TABELA DELTA
# =========================================================
table_name = "industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fp_corrida"

# =========================================================
# 2. LEITURA DO CHANGE DATA FEED (CDF)
# =========================================================
df_cdf = (
    spark.read
    .format("delta")
    .option("readChangeFeed", "true")
    .option("startingVersion", 202703)   # ajuste se necessário
    .table(table_name)
    # .filter(F.col("CORRIDA").between(130604, 130639))
    
)

# COMMAND ----------

display(
    df_cdf
    # .filter(F.col("CORRIDA").between(130604, 130639))
    .orderBy(F.desc("CORRIDA"))
)


# COMMAND ----------



# =========================================================
# 3. COLUNAS DE TEMPO A ANALISAR
# =========================================================
time_columns = [
    "HORAVAZAMENTO",
    "HORACHEGADAFP",
    "HORASAIDAFP",
    "HORASAIDALINGOTAMENTO"
]

# =========================================================
# 4. CDF → FORMATO LONGO (corrida, variavel, valor)
# =========================================================
df_cdf_long = (
    df_cdf
    .select(
        "CORRIDA",
        "_commit_timestamp",
        *time_columns
    )
    .selectExpr(
        "CORRIDA",
        "_commit_timestamp",
        f"""
        stack(
            {len(time_columns)},
            {", ".join([f"'{c}', {c}" for c in time_columns])}
        ) as (variavel, valor)
        """
    )
)

# =========================================================
# 5. PRIMEIRA VEZ QUE O VALOR APARECE (NULL → NOT NULL)
# =========================================================
w_first = (
    Window
    .partitionBy("CORRIDA", "variavel")
    .orderBy("_commit_timestamp")
)

df_first_arrival = (
    df_cdf_long
    .filter(F.col("valor").isNotNull())
    .withColumn("rn", F.row_number().over(w_first))
    .filter(F.col("rn") == 1)
    .drop("rn")
)

# =========================================================
# 6. FUNÇÃO PARA NORMALIZAR HORA (HHmm | HH:mm → HH:mm)
# =========================================================
def normalize_hora(col):
    return F.when(
        F.col(col).rlike("^[0-9]{4}$"),
        F.format_string(
            "%02d:%02d",
            F.substring(col, 1, 2).cast("int"),
            F.substring(col, 3, 2).cast("int")
        )
    ).otherwise(F.col(col))

# =========================================================
# 7. CRIAR TIMESTAMPS REAIS (DATA + HORA)
# =========================================================
df_eventos = (
    df_cdf
    .select(
        "CORRIDA",
        "DATAVAZAMENTO", "HORAVAZAMENTO",
        "DATACHEGADAFP", "HORACHEGADAFP",
        "DATASAIDAFP", "HORASAIDAFP",
        "DATASAIDALINGOTAMENTO", "HORASAIDALINGOTAMENTO"
    )
    .withColumn(
        "ts_vazamento",
        F.to_timestamp(
            F.concat(
                F.date_format("DATAVAZAMENTO", "yyyy-MM-dd"), F.lit(" "),
                normalize_hora("HORAVAZAMENTO")
            ),
            "yyyy-MM-dd HH:mm"
        )
    )
    .withColumn(
        "ts_chegadafp",
        F.to_timestamp(
            F.concat(
                F.date_format("DATACHEGADAFP", "yyyy-MM-dd"), F.lit(" "),
                normalize_hora("HORACHEGADAFP")
            ),
            "yyyy-MM-dd HH:mm"
        )
    )
    .withColumn(
        "ts_saidafp",
        F.to_timestamp(
            F.concat(
                F.date_format("DATASAIDAFP", "yyyy-MM-dd"), F.lit(" "),
                normalize_hora("HORASAIDAFP")
            ),
            "yyyy-MM-dd HH:mm"
        )
    )
    .withColumn(
        "ts_saidalingotamento",
        F.to_timestamp(
            F.concat(
                F.date_format("DATASAIDALINGOTAMENTO", "yyyy-MM-dd"), F.lit(" "),
                normalize_hora("HORASAIDALINGOTAMENTO")
            ),
            "yyyy-MM-dd HH:mm"
        )
    )
)

# =========================================================
# 8. EVENTOS → FORMATO LONGO
# =========================================================
df_eventos_long = (
    df_eventos
    .select(
        "CORRIDA",
        F.expr("""
            stack(
                4,
                'HORAVAZAMENTO', ts_vazamento,
                'HORACHEGADAFP', ts_chegadafp,
                'HORASAIDAFP', ts_saidafp,
                'HORASAIDALINGOTAMENTO', ts_saidalingotamento
            ) as (variavel, ts_evento)
        """)
    )
    .filter(F.col("ts_evento").isNotNull())
)

# =========================================================
# 9. JOIN: HORA REAL x HORA QUE CHEGOU
# =========================================================
df_delay = (
    df_first_arrival
    .join(
        df_eventos_long,
        on=["CORRIDA", "variavel"],
        how="left"
    )
    .withColumn(
        "commit_brt",
        F.col("_commit_timestamp") - F.expr("INTERVAL 3 HOURS")
    )
    .withColumn(
        "delay_minutos",
        (F.unix_timestamp("commit_brt") - F.unix_timestamp("ts_evento")) / 60
    )
)

# =========================================================
# 10. DISPLAY FINAL
# =========================================================



# COMMAND ----------

display(
    df_delay
    .select(
        "CORRIDA",
        "variavel",
        "ts_evento",
        "commit_brt",
        F.round("delay_minutos", 2).alias("delay_minutos")
    )
    .orderBy(F.desc("CORRIDA"), "variavel")
)

# COMMAND ----------

df_delay_avg = (
    df_delay
    .groupBy("CORRIDA", "variavel")
    .agg(
        F.first("ts_evento").alias("ts_evento"),
        F.first("commit_brt").alias("commit_brt"),
        F.avg("delay_minutos").alias("delay_minutos")
    )
    .orderBy(F.desc("CORRIDA"), "variavel")
)

display(df_delay_avg)

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Seleciona as variáveis de delay (renomeando HORASAIDALINGOTAMENTO para HORASAIDAVD)
variaveis = ["HORAVAZAMENTO", "HORACHEGADAFP", "HORASAIDAFP", "HORASAIDAVD"]

# Coleta os dados para pandas, renomeando a coluna se necessário
pdf_delay = df_delay_avg.select("variavel", "delay_minutos").toPandas()
pdf_delay["variavel"] = pdf_delay["variavel"].replace({"HORASAIDALINGOTAMENTO": "HORASAIDAVD"})

# Plota 4 histogramas, um para cada variável, com curva normal (KDE)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, var in enumerate(variaveis):
    dados = pdf_delay[pdf_delay["variavel"] == var]["delay_minutos"].dropna()
    q05 = dados.quantile(0.10)
    q95 = dados.quantile(0.90)
    dados_filtrados = dados[(dados >= q05) & (dados <= q95)]
    sns.histplot(
        dados_filtrados,
        bins=30,
        kde=True,
        ax=axes[i],
        color="skyblue"
    )
    axes[i].set_title(f"Delay - {var} ")
    axes[i].set_xlabel("Delay (minutos)")
    axes[i].set_ylabel("Frequência")

plt.tight_layout()
plt.show()

# COMMAND ----------

# =========================================================
# 0. IMPORTS
# =========================================================
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# =========================================================
# 1. TABELA DELTA
# =========================================================
table_name_lc = "industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_lc_corridalingotamento"

# =========================================================
# 2. LEITURA DO CHANGE DATA FEED (CDF)
# =========================================================
df_cdf_lc = (
    spark.read
    .format("delta")
    .option("readChangeFeed", "true")
    .option("startingVersion", 18690)   # ajuste se necessário
    .table(table_name_lc)
    .filter(F.col("CORRIDA").between(130604, 130639))
    
)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# =========================================================
# 1. COLUNAS DE TEMPO (LC)
# =========================================================
time_columns_lc = [
    "INICIOLINGOTAMENTO",
    "FINALLINGOTAMENTO",
]

# =========================================================
# 2. CDF → FORMATO LONGO
# =========================================================
df_cdf_long_lc = (
    df_cdf_lc
    .select(
        "CORRIDA",
        "_commit_timestamp",
        *time_columns_lc
    )
    .selectExpr(
        "CORRIDA",
        "_commit_timestamp",
        f"""
        stack(
            {len(time_columns_lc)},
            {", ".join([f"'{c}', {c}" for c in time_columns_lc])}
        ) as (variavel, valor)
        """
    )
)

# =========================================================
# 3. PRIMEIRA VEZ QUE O VALOR APARECE (NULL → NOT NULL)
# =========================================================
w_first = (
    Window
    .partitionBy("CORRIDA", "variavel")
    .orderBy("_commit_timestamp")
)

df_first_arrival_lc = (
    df_cdf_long_lc
    .filter(F.col("valor").isNotNull())
    .withColumn("rn", F.row_number().over(w_first))
    .filter(F.col("rn") == 1)
    .drop("rn")
)

# =========================================================
# 4. EVENTOS REAIS → FORMATO LONGO
#    (aqui o valor já É o timestamp do evento)
# =========================================================
df_eventos_long_lc = (
    df_cdf_lc
    .select(
        "CORRIDA",
        F.expr("""
            stack(
                2,
                'INICIOLINGOTAMENTO', INICIOLINGOTAMENTO,
                'FINALLINGOTAMENTO', FINALLINGOTAMENTO
            ) as (variavel, ts_evento)
        """)
    )
    .filter(F.col("ts_evento").isNotNull())
)

# =========================================================
# 5. JOIN: HORA REAL x HORA QUE CHEGOU
# =========================================================
df_delay_lc = (
    df_first_arrival_lc
    .join(
        df_eventos_long_lc,
        on=["CORRIDA", "variavel"],
        how="left"
    )
    .withColumn(
        "commit_brt",
        F.col("_commit_timestamp") - F.expr("INTERVAL 3 HOURS")
    )
    .withColumn(
        "delay_minutos",
        (F.unix_timestamp("commit_brt") - F.unix_timestamp("ts_evento")) / 60
    )
)

# =========================================================
# 6. DISPLAY FINAL
# =========================================================
display(
    df_delay_lc
    .select(
        "CORRIDA",
        "variavel",
        "ts_evento",
        "commit_brt",
        F.round("delay_minutos", 2).alias("delay_minutos")
    )
    .orderBy("CORRIDA", "variavel")
)


# COMMAND ----------

df_delay_avg_lc = (
    df_delay_lc
    .groupBy("CORRIDA", "variavel")
    .agg(
        F.first("ts_evento").alias("ts_evento"),
        F.first("commit_brt").alias("commit_brt"),
        F.avg("delay_minutos").alias("delay_minutos")
    )
    .orderBy(F.desc("CORRIDA"), "variavel")
)



# COMMAND ----------

display(df_delay_avg_lc)

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Seleciona as variáveis de delay (renomeando HORASAIDALINGOTAMENTO para HORASAIDAVD)
variaveis = ["INICIOLINGOTAMENTO", "FINALLINGOTAMENTO"]

# Coleta os dados para pandas, renomeando a coluna se necessário
pdf_delay_lc = df_delay_avg_lc.select("variavel", "delay_minutos").toPandas()


# Plota 4 histogramas, um para cada variável, com curva normal (KDE)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, var in enumerate(variaveis):
    dados = pdf_delay_lc[pdf_delay_lc["variavel"] == var]["delay_minutos"].dropna()
    q05 = dados.quantile(0.10)
    q95 = dados.quantile(0.90)
    dados_filtrados = dados[(dados >= q05) & (dados <= q95)]
    sns.histplot(
        dados_filtrados,
        bins=30,
        kde=True,
        ax=axes[i],
        color="skyblue"
    )
    axes[i].set_title(f"Delay - {var} ")
    axes[i].set_xlabel("Delay (minutos)")
    axes[i].set_ylabel("Frequência")

plt.tight_layout()
plt.show()

# COMMAND ----------

def normalize_hora(col):
    """
    Converte HHmm ou HH:mm em HH:mm
    """
    return F.when(
        F.col(col).rlike("^[0-9]{4}$"),
        F.format_string(
            "%02d:%02d",
            F.substring(col, 1, 2).cast("int"),
            F.substring(col, 3, 2).cast("int")
        )
    ).otherwise(F.col(col))


# COMMAND ----------

# MAGIC %md
# MAGIC # STREAMING DAS TABELAS PARA VERIFICAR LATENCIA

# COMMAND ----------

# DBTITLE 1,tb_admaci_fea_corridas
from pyspark.sql import functions as F

df_stream_fea = (
    spark.readStream
        .format("delta")
        .option("readChangeFeed", "true")
        .option("startingVersion", "latest")
        .table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fea_corridas")
)

# Filtra apenas inserts e updates finais
df_filtered_fea = df_stream_fea.filter(F.col("_change_type").isin("insert", "update_postimage"))

display(df_filtered_fea, streamName="corrida_stream_cdf_filtered_fea")

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,tb_admaci_lc_corridalingotamento
from pyspark.sql import functions as F

df_stream_lc = (
    spark.readStream
        .format("delta")
        .option("readChangeFeed", "true")
        .option("startingVersion", "latest")
        .table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_lc_corridalingotamento")
)

# Filtra apenas inserts e updates finais
df_filtered_fea = df_stream_lc.filter(F.col("_change_type").isin("insert", "update_postimage"))

display(df_filtered_fea, streamName="corrida_stream_cdf_filtered_lc")

# COMMAND ----------

# DBTITLE 1,tb_admaci_fp_corrida
from pyspark.sql import functions as F

df_stream = (
    spark.readStream
        .format("delta")
        .option("readChangeFeed", "true")
        .option("startingVersion", "latest")
        .table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fp_corrida")
)

# Filtra apenas inserts e updates finais
df_filtered = df_stream.filter(F.col("_change_type").isin("insert", "update_postimage"))

# Suas transformações
df_transformed = (
    df_filtered
        .select(
            F.col("_commit_timestamp"),
            F.col("CORRIDA").cast("int").alias("corrida"),
            F.col("TANQUEVD"),
            F.col("DATAVAZAMENTO"),
            F.col("HORAVAZAMENTO"),
            F.col("DATACHEGADAFP"),
            F.col("HORACHEGADAFP"),
            F.col("DATASAIDAFP"),
            F.col("HORASAIDAFP"),
            F.col("DATASAIDALINGOTAMENTO"),
            F.col("HORASAIDALINGOTAMENTO")
        )
        .withColumn(
            "data_hora_vazamento",
            F.to_timestamp(
                F.concat(
                    F.date_format("DATAVAZAMENTO", "yyyy-MM-dd"), F.lit(" "),
                    F.format_string("%02d:%02d",
                        F.substring("HORAVAZAMENTO", 1, 2).cast("int"),
                        F.substring("HORAVAZAMENTO", 3, 2).cast("int")
                    )
                ),
                "yyyy-MM-dd HH:mm"
            )
        )
        .withColumn(
            "data_hora_chegadafp",
            F.to_timestamp(
                F.concat(
                    F.date_format("DATACHEGADAFP", "yyyy-MM-dd"), F.lit(" "),
                    F.format_string("%02d:%02d",
                        F.substring("HORACHEGADAFP", 1, 2).cast("int"),
                        F.substring("HORACHEGADAFP", 3, 2).cast("int")
                    )
                ),
                "yyyy-MM-dd HH:mm"
            )
        )
        .withColumn(
            "data_hora_saidafp",
            F.to_timestamp(
                F.concat(
                    F.date_format("DATASAIDAFP", "yyyy-MM-dd"), F.lit(" "),
                    F.format_string("%02d:%02d",
                        F.substring("HORASAIDAFP", 1, 2).cast("int"),
                        F.substring("HORASAIDAFP", 3, 2).cast("int")
                    )
                ),
                "yyyy-MM-dd HH:mm"
            )
        )
        .withColumn(
            "data_hora_saidalingotamento",
            F.to_timestamp(
                F.concat(
                    F.date_format("DATASAIDALINGOTAMENTO", "yyyy-MM-dd"), F.lit(" "),
                    F.format_string("%02d:%02d",
                        F.substring("HORASAIDALINGOTAMENTO", 1, 2).cast("int"),
                        F.substring("HORASAIDALINGOTAMENTO", 3, 2).cast("int")
                    )
                ),
                "yyyy-MM-dd HH:mm"
            )
        )
)

# Agora sim, display em tempo real
display(df_transformed, streamName="corrida_stream_cdf_transformed")


# COMMAND ----------

# MAGIC %md
# MAGIC # tb_corridatempos

# COMMAND ----------

# DBTITLE 1,tb_corridatempos
# Local timezone
tb_corridatempos = (
    spark.read.table("industrial_iatemperaturascha_refined_prd.computed_features.tb_corridatempos")
    .orderBy("corrida", ascending=False)
)
 
display(tb_corridatempos)
 


# COMMAND ----------

# MAGIC %md
# MAGIC # tb_admaci_fp_corridasaidavd

# COMMAND ----------

# DBTITLE 1,tb_admaci_fp_corridasaidavd
df_fp_corridasaidavd = (
    spark.read.table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fp_corridasaidavd")
    .orderBy(F.col("CORRIDA").desc())
)
display(df_fp_corridasaidavd)

# COMMAND ----------

# MAGIC %md
# MAGIC # VÁRIAVEIS PELOS OPERADORES

# COMMAND ----------

# MAGIC %md
# MAGIC # 📘 Resumo das Variáveis Construídas
# MAGIC
# MAGIC A seguir estão as variáveis finais geradas, com a explicação resumida de sua origem e lógica de cálculo.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔵 **FEA**
# MAGIC
# MAGIC **Início FEA**  
# MAGIC `HRVAZAMENTO - TTT`  
# MAGIC Fonte: `industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fea_corrida`  
# MAGIC Informação REAL (não operador).
# MAGIC
# MAGIC **Final FEA**  
# MAGIC `HRVAZAMENTO`  
# MAGIC Fonte: `industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fea_corrida`  
# MAGIC Informação REAL (não operador).
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🟣 **FP — Forno Panela**
# MAGIC
# MAGIC **Início FP**  
# MAGIC `HORACHEGADAFP`  
# MAGIC Fonte: `industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fp_corrida`  
# MAGIC Informação OPERADOR.
# MAGIC
# MAGIC **Final FP**  
# MAGIC `HORASAIDAFP`  
# MAGIC Fonte: `industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fp_corrida`  
# MAGIC Informação OPERADOR.
# MAGIC
# MAGIC **Início FP considerando TTTFP**  
# MAGIC `saida_fp - TTTFP`  
# MAGIC Fonte: `industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fp_corrida`  
# MAGIC Informação OPERADOR.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🟠 **VD — Vaso Desgaseificação**
# MAGIC
# MAGIC **Início VD**  
# MAGIC `datahorasaidavd - TTTVD`  
# MAGIC Fonte:  
# MAGIC - `industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fp_corrida` (TTTVD)  
# MAGIC - `industrial_iatemperaturascha_refined_prd.computed_features.tb_corridatempos` (datahorasaidavd)  
# MAGIC Informação OPERADOR.
# MAGIC
# MAGIC **Final VD**  
# MAGIC `datahorasaidavd`  
# MAGIC Fonte: `industrial_iatemperaturascha_refined_prd.computed_features.tb_corridatempos`  
# MAGIC Informação OPERADOR.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🟢 **LC — Lingotamento Contínuo**
# MAGIC
# MAGIC **Início LC**  
# MAGIC `INICIOLINGOTAMENTO`  
# MAGIC Fonte: `industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_lc_corridalingotamento`  
# MAGIC Informação REAL.
# MAGIC
# MAGIC **Final LC**  
# MAGIC `FINALLINGOTAMENTO`  
# MAGIC Fonte: `industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_lc_corridalingotamento`  
# MAGIC Informação REAL.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC

# COMMAND ----------

def convert_datetime(df, date_col, hour_col, new_col):
    return (
        df.withColumn(
            "date_clean",
            F.to_date(F.col(date_col))   # extrai YYYY-MM-dd
        )
        .withColumn(
            "hour_clean",
            F.when(
                F.length(F.col(hour_col)) == 4,   # caso HHmm (ex: 0830)
                F.concat_ws(
                    ":",
                    F.col(hour_col).substr(1, 2),
                    F.col(hour_col).substr(3, 2)
                )
            ).otherwise(
                F.col(hour_col)                   # caso HH:mm (ex: 09:28)
            )
        )
        .withColumn(
            new_col,
            F.to_timestamp(
                F.concat_ws(" ", F.col("date_clean"), F.col("hour_clean")),
                "yyyy-MM-dd HH:mm"
            )
        )
        .drop("date_clean", "hour_clean")
    )


# COMMAND ----------

# DBTITLE 1,df_final
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ======================================================
# 1. LEITURA DAS TABELAS
# ======================================================
fea_df = spark.table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fea_corridas")
fp_df  = spark.table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fp_corrida")
vd_df  = spark.table("industrial_iatemperaturascha_refined_prd.computed_features.tb_corridatempos")
lc_df  = spark.table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_lc_corridalingotamento")

# ======================================================
# 2. TRATAMENTO FEA (corrigido conforme você avisou)
# ======================================================
# HRVAZAMENTO já vem como HH:mm
# DATA vem como timestamp ("YYYY-MM-dd 00:00:00")

# Monta timestamp de vazamento no FEA
fea_df = fea_df.withColumn(
    "hr_vaz_fea",
    F.to_timestamp(
        F.concat_ws(" ",
            F.to_date("DATA"),
            F.col("HRVAZAMENTO")
        ),
        "yyyy-MM-dd HH:mm"
    )
)

# Converte TTT de string "58.0000000000" → double → segundos
fea_df = fea_df.withColumn(
    "TTT_min",
    F.col("TTT").cast("double")
).withColumn(
    "TTT_seconds",
    (F.col("TTT_min") * 60).cast("long")
)

# final_fea = hr_vaz_fea
# inicio_fea = hr_vaz_fea - (TTT_min * 60)
fea_df = fea_df.withColumn("final_fea", F.col("hr_vaz_fea")) \
               .withColumn(
                    "inicio_fea",
                    F.from_unixtime(
                        F.col("hr_vaz_fea").cast("long") - F.col("TTT_seconds")
                    ).cast("timestamp")
               )


# ======================================================
# 3. TRATAMENTO FP (estes sim vêm com HHmm)
# ======================================================
fp_df = convert_datetime(fp_df, "DATACHEGADAFP", "HORACHEGADAFP", "chegada_fp")
fp_df = convert_datetime(fp_df, "DATASAIDAFP", "HORASAIDAFP", "saida_fp")

fp_df = fp_df.withColumn("inicio_fp", F.col("chegada_fp")) \
             .withColumn("final_fp", F.col("saida_fp"))

# Converter TTTFP (string) para double → minutos → segundos
fp_df = fp_df.withColumn("TTTFP_min", F.col("TTTFP").cast("double")) \
             .withColumn("TTTFP_seconds", (F.col("TTTFP_min") * 60).cast("long"))

# inicio_fp_tttfp = saida_fp - TTTFP em segundos
fp_df = fp_df.withColumn(
    "inicio_fp_tttfp",
    F.from_unixtime(
        F.col("saida_fp").cast("long") - F.col("TTTFP_seconds")
    ).cast("timestamp")
)


# ======================================================
# 4. TRATAMENTO VD
# ======================================================

# 1. Primeiro juntamos FP + VD para trazer o TTTVD
vd_joined = (
    vd_df.alias("vd")
    .join(fp_df.select("corrida", "TTTVD"), "corrida", "left")  # <-- TTTVD vem do FP
)

# 2. Convertemos TTTVD e montamos inicio_vd
vd_joined = (
    vd_joined
    .withColumn("final_vd", F.to_timestamp("datahorasaidavd"))
    .withColumn("TTTVD_min", F.col("TTTVD").cast("double"))
    .withColumn("TTTVD_seconds", (F.col("TTTVD_min") * 60).cast("long"))
    .withColumn(
        "inicio_vd",
        F.from_unixtime(
            F.col("datahorasaidavd").cast("long") - F.col("TTTVD_seconds")
        ).cast("timestamp")
    )
)


# ======================================================
# 5. TRATAMENTO LC (tempo real)
# ======================================================

lc_df = lc_df.withColumn("inicio_lc", F.to_timestamp("INICIOLINGOTAMENTO")) \
             .withColumn("final_lc", F.to_timestamp("FINALLINGOTAMENTO"))

# ======================================================
# 6. JUNÇÃO FINAL (ajuste corrida se o nome for diferente)
# ======================================================
df_final = (
    fea_df.alias("fea")
    .join(fp_df.alias("fp"), "corrida", "left")
    .join(vd_joined.alias("vd"), "corrida", "left")
    .join(lc_df.alias("lc"), "corrida", "left")
    .select(
        "corrida",
        # FEA
        "inicio_fea", "final_fea",
        # FP
        "inicio_fp_tttfp", "inicio_fp", "final_fp", 
        # VD
        "inicio_vd", "final_vd",
        # LC
        "inicio_lc", "final_lc", "ADICIONALTTTFP", "ADICIONALTTTVD"
    )
)

# ======================================================
# 7. EXIBE O RESULTADO FINAL
# ======================================================
display(df_final.orderBy(F.desc("corrida")))


# COMMAND ----------

from pyspark.sql import functions as F

def registrar_evento(micro_batch, batch_id, origem):
    eventos = (
        micro_batch
        .filter(F.col("_change_type").isin("insert", "update_postimage"))
        .select("corrida")
        .withColumn("origem", F.lit(origem))
        .withColumn("updated_at", F.current_timestamp())
    )

    eventos.write.format("delta")\
           .mode("append")\
           .saveAsTable("industrial_iatemperaturascha_refined_dev.computed_features.tb_eventos_tempos")


# COMMAND ----------

# DBTITLE 1,stream_fea
stream_fea = (
    spark.readStream
        .format("delta")
        .option("readChangeFeed", "true")
        .table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fea_corridas")
        .writeStream
        .foreachBatch(lambda mb, bid: registrar_evento(mb, bid, origem="FEA"))
        # .option("checkpointLocation", f"{CHECKPOINT_BASE}/eventos_fea")
        .start()
)


# COMMAND ----------

# DBTITLE 1,stream_fp
stream_fp = (
    spark.readStream
        .format("delta")
        .option("readChangeFeed", "true")
        .table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fp_corrida")
        .writeStream
        .foreachBatch(lambda mb, bid: registrar_evento(mb, bid, origem="FP"))
        # .option("checkpointLocation", f"{CHECKPOINT_BASE}/eventos_fp")
        .start()
)


# COMMAND ----------

# DBTITLE 1,stream_vd
stream_vd = (
    spark.readStream
        .format("delta")
        .option("readChangeFeed", "true")
        .table("industrial_iatemperaturascha_refined_prd.computed_features.tb_corridatempos")
        .writeStream
        .foreachBatch(lambda mb, bid: registrar_evento(mb, bid, origem="VD"))
        # .option("checkpointLocation", f"{CHECKPOINT_BASE}/eventos_vd")
        .start()
)


# COMMAND ----------

# DBTITLE 1,stream_lc
stream_lc = (
    spark.readStream
        .format("delta")
        .option("readChangeFeed", "true")
        .table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_lc_corridalingotamento")
        .writeStream
        .foreachBatch(lambda mb, bid: registrar_evento(mb, bid, origem="LC"))
        # .option("checkpointLocation", f"{CHECKPOINT_BASE}/eventos_lc")
        .start()
)


# COMMAND ----------

# DBTITLE 1,upsert_to_tempos
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from delta.tables import DeltaTable

# ======================================================
# upsert_to_tempos FINAL (PRODUÇÃO)
# ======================================================

def upsert_to_tempos(micro_batch, batch_id, step):

    # ======================================================
    # 0. TRATAMENTO DO MICRO-BATCH
    # ======================================================

    # Caso venha da tabela de eventos → não tem _change_type
    if "_change_type" not in micro_batch.columns:
        micro_batch = micro_batch.select("corrida").dropDuplicates()

    else:
        # Deduplicação para tabelas CDF (FEA, FP, VD, LC)
        windowSpecFirst = Window.partitionBy("corrida", "_change_type")\
                                .orderBy("_commit_version")
        windowSpecLast  = Window.partitionBy("corrida", "_change_type")\
                                .orderBy(F.desc("_commit_version"))

        micro_batch = (
            micro_batch
            .withColumn(
                "rank",
                F.when(F.col("_change_type") == "update_preimage",
                       F.row_number().over(windowSpecFirst)
                ).when(
                    F.col("_change_type") == "update_postimage",
                    F.row_number().over(windowSpecLast)
                )
            )
            .filter(F.col("rank") == 1)
            .drop("rank")
        )

        micro_batch = (
            micro_batch
            .withColumn("_change_type_not_preimage",
                F.when(F.col("_change_type") != "update_preimage", True)
            )
            .groupBy("corrida")
            .agg(
                F.count("corrida").alias("count"),
                F.last("_change_type_not_preimage",
                       ignorenulls=True).alias("_change_type_not_preimage")
            )
            .filter(F.col("count") == 1)
            .filter(F.col("_change_type_not_preimage") == True)
            .select("corrida")
        )

    # Se não há nada para processar
    if micro_batch.count() == 0:
        return

    # ======================================================
    # 1. CARREGAR TABELAS ORIGINAIS (FEA, FP, VD, LC)
    # ======================================================

    fea_df = spark.table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fea_corridas")\
                  .join(micro_batch, "corrida")
    fp_df  = spark.table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fp_corrida")\
                  .join(micro_batch, "corrida")
    vd_df  = spark.table("industrial_iatemperaturascha_refined_prd.computed_features.tb_corridatempos")\
                  .join(micro_batch, "corrida")
    lc_df  = spark.table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_lc_corridalingotamento")\
                  .join(micro_batch, "corrida")

    # ======================================================
    # 2. PROCESSAMENTO FEA
    # ======================================================

    fea_df = (
        fea_df
        .withColumn(
            "hr_vaz_fea",
            F.to_timestamp(
                F.concat_ws(" ", F.to_date("DATA"), F.col("HRVAZAMENTO")),
                "yyyy-MM-dd HH:mm"
            )
        )
        .withColumn("TTT_min", F.col("TTT").cast("double"))
        .withColumn("TTT_seconds", (F.col("TTT_min") * 60).cast("long"))
        .withColumn("final_fea", F.col("hr_vaz_fea"))
        .withColumn(
            "inicio_fea",
            F.from_unixtime(F.col("hr_vaz_fea").cast("long") - F.col("TTT_seconds"))\
             .cast("timestamp")
        )
    )

    # ======================================================
    # 3. PROCESSAMENTO FP
    # ======================================================

    fp_df = convert_datetime(fp_df, "DATACHEGADAFP", "HORACHEGADAFP", "chegada_fp")
    fp_df = convert_datetime(fp_df, "DATASAIDAFP", "HORASAIDAFP", "saida_fp")

    fp_df = (
        fp_df
        .withColumn("inicio_fp", F.col("chegada_fp"))
        .withColumn("final_fp", F.col("saida_fp"))
        .withColumn("TTTFP_min", F.col("TTTFP").cast("double"))
        .withColumn("TTTFP_seconds", (F.col("TTTFP_min") * 60).cast("long"))
        .withColumn(
            "inicio_fp_tttfp",
            F.from_unixtime(F.col("saida_fp").cast("long") - F.col("TTTFP_seconds"))\
             .cast("timestamp")
        )
    )

    # ======================================================
    # 4. PROCESSAMENTO VD
    # ======================================================

    vd_joined = (
        vd_df.alias("vd")
        .join(fp_df.select("corrida", "TTTVD"), "corrida", "left")
        .withColumn("final_vd", F.to_timestamp("datahorasaidavd"))
        .withColumn("TTTVD_min", F.col("TTTVD").cast("double"))
        .withColumn("TTTVD_seconds", (F.col("TTTVD_min") * 60).cast("long"))
        .withColumn(
            "inicio_vd",
            F.from_unixtime(F.col("datahorasaidavd").cast("long") - F.col("TTTVD_seconds"))\
             .cast("timestamp")
        )
    )

    # ======================================================
    # 5. PROCESSAMENTO LC
    # ======================================================

    lc_df = (
        lc_df
        .withColumn("inicio_lc", F.to_timestamp("INICIOLINGOTAMENTO"))
        .withColumn("final_lc", F.to_timestamp("FINALLINGOTAMENTO"))
    )

    # ======================================================
    # 6. MONTAR DF FINAL (já vindo de todos os joins)
    # ======================================================

    df_final = (
        fea_df.alias("fea")
        .join(fp_df.alias("fp"), "corrida", "left")
        .join(vd_joined.alias("vd"), "corrida", "left")
        .join(lc_df.alias("lc"), "corrida", "left")
        .select(
            "corrida",
            "inicio_fea", "final_fea",
            "inicio_fp_tttfp", "inicio_fp", "final_fp",
            "inicio_vd", "final_vd",
            "inicio_lc", "final_lc",
            "ADICIONALTTTFP", "ADICIONALTTTVD"
        )
    )

    # ======================================================
    # 6.1 GARANTIR QUE CADA CORRIDA TENHA UMA ÚNICA LINHA
    # (fundamental para evitar erro no MERGE)
    # ======================================================

    df_final = df_final.groupBy("corrida").agg(
        *[
            F.last(c, ignorenulls=True).alias(c)
            for c in df_final.columns
            if c != "corrida"
        ]
    )

    # ======================================================
    # 7. MERGE NA TABELA FINAL tb_tempos
    # ======================================================

    target_table = "industrial_iatemperaturascha_refined_dev.computed_features.tb_tempos"

    if not spark.catalog.tableExists(target_table):
        df_final.write.format("delta").mode("overwrite").saveAsTable(target_table)

    else:
        delta_target = DeltaTable.forName(spark, target_table)

        update_expr = {c: f"source.{c}" for c in df_final.columns if c != "corrida"}
        insert_expr = {c: f"source.{c}" for c in df_final.columns}

        (
            delta_target.alias("t")
            .merge(df_final.alias("source"), "t.corrida = source.corrida")
            .whenMatchedUpdate(set=update_expr)
            .whenNotMatchedInsert(values=insert_expr)
            .execute()
        )


    # ======================================================
    # 8. REGISTRAR HISTÓRICO (STAMP)
    # ======================================================

    stamp_df = (
        df_final
        .withColumn("change_type", F.lit("update"))
        .withColumn("origem", F.lit(step))
        .withColumn("updated_at", F.current_timestamp())
        .withColumn(
            "payload",
            F.struct(
                "inicio_fea", "final_fea",
                "inicio_fp_tttfp", "inicio_fp", "final_fp",
                "inicio_vd", "final_vd",
                "inicio_lc", "final_lc",
                "ADICIONALTTTFP", "ADICIONALTTTVD"
            )
        )
        .withColumn("ingestion_time", F.current_timestamp())
    )

    stamp_df.write.format("delta")\
        .mode("append")\
        .saveAsTable("industrial_iatemperaturascha_refined_dev.computed_features.tb_tempos_stamp")


# COMMAND ----------

from delta.tables import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from datetime import datetime
from pyspark.sql.window import Window

# COMMAND ----------

# DBTITLE 1,streaming_writer_tempos
streaming_writer_tempos = (
    spark.readStream
        .format("delta")
        .table("industrial_iatemperaturascha_refined_dev.computed_features.tb_eventos_tempos")
        .writeStream
        .foreachBatch(lambda mb, bid: upsert_to_tempos(mb, bid, step="tempos"))
        # .option("checkpointLocation", f"{CHECKPOINT_BASE}/tempos_master")
        .start()
)


# COMMAND ----------

# DBTITLE 1,df_stream_tempos
from pyspark.sql import functions as F

df_stream_tempos = (
    spark.readStream
        .format("delta")
        .option("readChangeFeed", "true")
        .option("startingVersion", "latest")
        .table("industrial_iatemperaturascha_refined_dev.computed_features.tb_tempos")
)

df_filtered_tempos = df_stream_tempos.filter(
    F.col("_change_type").isin("insert", "update_postimage")
)

display(df_filtered_tempos.filter(F.col("corrida") >= 130590), streamName="stream_tb_tempos")


# COMMAND ----------

df_stream_stamp = (
    spark.readStream
        .format("delta")
        .table("industrial_iatemperaturascha_refined_dev.computed_features.tb_tempos_stamp")
)

display(df_stream_stamp, streamName="stream_tb_tempos_stamp")


# COMMAND ----------

spark.table("industrial_iatemperaturascha_refined_prd.data_preparation.tb_corridainfo_temp").filter(F.col('corrida')==130595).dropna(subset=['hora']).display()

# COMMAND ----------

display(spark.table("industrial_iatemperaturascha_refined_dev.computed_features.tb_tempos_stamp").filter(F.col("corrida") == 130595))

# COMMAND ----------

display(spark.table("industrial_iatemperaturascha_refined_dev.computed_features.tb_tempos").filter(F.col("corrida") == 130595))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Gráfico dos tempos para uma corrida

# COMMAND ----------



# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

# Escolha as corridas específicas
corridas_escolhidas = [130590 ,130591,130592, 130593, 130594, 130595]  # substitua pelos números desejados

# Filtra o DataFrame final para as corridas escolhidas
df_corridas = df_final.filter(F.col("corrida").isin(corridas_escolhidas)).toPandas()

# Monta os pontos de evolução (x = etapa, y = timestamp)
etapas = [
    ("inicio_fea", "Início FEA"),
    ("final_fea", "Final FEA"),
    # ("inicio_fp_tttfp", "Início FP TTTFP"),
    ("inicio_fp", "Início FP"),
    ("final_fp", "Final FP"),
    ("inicio_vd", "Início VD"),
    ("final_vd", "Final VD"),
    ("inicio_lc", "Início LC"),
    ("final_lc", "Final LC")
]

plt.figure(figsize=(10, 6))

for corrida in corridas_escolhidas:
    df_corrida = df_corridas[df_corridas["corrida"] == corrida]
    x_labels = []
    y_times = []
    hm_labels = []
    for col, label in etapas:
        if col in df_corrida.columns and pd.notnull(df_corrida.iloc[0][col]):
            x_labels.append(label)
            y_time = pd.to_datetime(df_corrida.iloc[0][col])
            y_times.append(y_time)
            hm_labels.append(y_time.strftime("%H:%M"))
    plt.plot(x_labels, y_times, marker='o', label=f"Corrida {corrida}")
    for i, (x, y, hm) in enumerate(zip(x_labels, y_times, hm_labels)):
        plt.annotate(hm, (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)

plt.title(f"Evolução das Corridas: {', '.join(map(str, corridas_escolhidas))}")
plt.xlabel("Etapa")
plt.ylabel("Timestamp")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

tb_aci_retorno = (
    spark.read.table(
        f"industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_aci_retorno"
    )
    .withColumn("corrida", F.col("corrida").cast("int"))

)
 
returned_heats = list(
    tb_aci_retorno.toPandas().apply(pd.to_numeric, errors="ignore")["corrida"]
)

# COMMAND ----------

from pyspark.sql import functions as F

df_final = (
    df_final
    # Tempo em cada etapa (em minutos)
    .withColumn("tempo_no_fea", (F.col("final_fea").cast("long") - F.col("inicio_fea").cast("long")) / 60)
    .withColumn("tempo_no_fp", (F.col("final_fp").cast("long") - F.col("inicio_fp").cast("long")) / 60)
    .withColumn("tempo_no_vd", (F.col("final_vd").cast("long") - F.col("inicio_vd").cast("long")) / 60)
    .withColumn("tempo_no_lc", (F.col("final_lc").cast("long") - F.col("inicio_lc").cast("long")) / 60)
    # Tempo de transporte entre etapas (em minutos)
    .withColumn("transporte_fea_fp", (F.col("inicio_fp").cast("long") - F.col("final_fea").cast("long")) / 60)
    .withColumn("transporte_fp_vd", (F.col("inicio_vd").cast("long") - F.col("final_fp").cast("long")) / 60)
    .withColumn("transporte_vd_lc", (F.col("inicio_lc").cast("long") - F.col("final_vd").cast("long")) / 60)
)

display(df_final)

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Converte todo o df_final para pandas
df_corridas = df_final.toPandas()

# Monta DataFrame para plotagem
etapas = [
    ("tempo_no_fea", "Tempo no FEA"),
    ("transporte_fea_fp", "Transporte FEA-FP"),
    ("tempo_no_fp", "Tempo no FP"),
    ("transporte_fp_vd", "Transporte FP-VD"),
    ("tempo_no_vd", "Tempo no VD"),
    ("transporte_vd_lc", "Transporte VD-LC"),
    ("tempo_no_lc", "Tempo no LC")
]

data_plot = []
for idx, row in df_corridas.iterrows():
    corrida = row["corrida"]
    for col, label in etapas:
        if col in row and pd.notnull(row[col]):
            data_plot.append({
                "corrida": corrida,
                "etapa": label,
                "tempo": row[col]
            })

df_plot = pd.DataFrame(data_plot)

# Calcula média e desvio padrão entre percentis 5% e 95% para cada etapa
etapa_labels = [label for _, label in etapas]
means = []
stds = []

for label in etapa_labels:
    tempos = df_plot[df_plot["etapa"] == label]["tempo"].dropna()
    if len(tempos) > 0:
        p5, p95 = np.percentile(tempos, [5, 95])
        tempos_filtered = tempos[(tempos >= p5) & (tempos <= p95)]
        means.append(tempos_filtered.mean())
        stds.append(tempos_filtered.std())
    else:
        means.append(np.nan)
        stds.append(np.nan)

plt.figure(figsize=(10, 6))
plt.bar(etapa_labels, means, yerr=stds, capsize=5, alpha=0.7)
plt.title("Média dos Tempos das Etapas (todas as corridas)")
plt.xlabel("Etapa")
plt.ylabel("Tempo médio (minutos)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

# Escolha as corridas específicas
corridas_escolhidas = [130593, 130594, 130595]  # substitua pelos números desejados

# Filtra o DataFrame final para as corridas escolhidas
df_corridas = df_final.filter(F.col("corrida").isin(corridas_escolhidas)).toPandas()

# Monta os pontos de evolução (x = etapa, y = tempo acumulado em minutos)
etapas = [
    ("tempo_no_fea", "Tempo no FEA"),
    ("transporte_fea_fp", "Transporte FEA-FP"),
    ("tempo_no_fp", "Tempo no FP"),
    ("transporte_fp_vd", "Transporte FP-VD"),
    ("tempo_no_vd", "Tempo no VD"),
    ("transporte_vd_lc", "Transporte VD-LC"),
    ("tempo_no_lc", "Tempo no LC")
]

plt.figure(figsize=(10, 6))

for corrida in corridas_escolhidas:
    df_corrida = df_corridas[df_corridas["corrida"] == corrida]
    x_labels = []
    y_times = []
    acumulado = 0
    for col, label in etapas:
        if col in df_corrida.columns and pd.notnull(df_corrida.iloc[0][col]):
            acumulado += df_corrida.iloc[0][col]
            x_labels.append(label)
            y_times.append(acumulado)
    plt.plot(x_labels, y_times, marker='o', label=f"Corrida {corrida}")

plt.title(f"Tempo Acumulado das Etapas das Corridas: {', '.join(map(str, corridas_escolhidas))}")
plt.xlabel("Etapa")
plt.ylabel("Tempo acumulado (minutos)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# import pandas as pd
# import matplotlib.pyplot as plt

# # Qualidades a serem analisadas
# qualidades_escolhidas = ["A", "B", "C"]  # substitua pelas qualidades desejadas

# # Adapte o nome da coluna de qualidade conforme seu DataFrame
# col_qualidade = "qualidade"  # ajuste se necessário

# # Filtra apenas as corridas das qualidades escolhidas
# df_qualidades = df_final.filter(F.col(col_qualidade).isin(qualidades_escolhidas)).toPandas()

# etapas = [
#     ("tempo_no_fea", "Tempo no FEA"),
#     ("transporte_fea_fp", "Transporte FEA-FP"),
#     ("tempo_no_fp", "Tempo no FP"),
#     ("transporte_fp_vd", "Transporte FP-VD"),
#     ("tempo_no_vd", "Tempo no VD"),
#     ("transporte_vd_lc", "Transporte VD-LC"),
#     ("tempo_no_lc", "Tempo no LC")
# ]

# plt.figure(figsize=(10, 6))

# for qualidade in qualidades_escolhidas:
#     df_q = df_qualidades[df_qualidades[col_qualidade] == qualidade]
#     x_labels = []
#     y_times = []
#     acumulado = 0
#     for col, label in etapas:
#         if col in df_q.columns:
#             media_etapa = df_q[col].dropna().mean()
#             if pd.notnull(media_etapa):
#                 acumulado += media_etapa
#                 x_labels.append(label)
#                 y_times.append(acumulado)
#     plt.plot(x_labels, y_times, marker='o', label=f"Qualidade {qualidade}")
#     for x, y in zip(x_labels, y_times):
#         plt.annotate(f"{y:.1f} min", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)

# plt.title(f"Tempo Acumulado Médio por Etapa para Qualidades: {', '.join(qualidades_escolhidas)}")
# plt.xlabel("Etapa")
# plt.ylabel("Tempo acumulado médio (minutos)")
# plt.xticks(rotation=45)
# plt.legend()
# plt.tight_layout()
# plt.show()

# COMMAND ----------

# Análise do período dos dados em df_final usando inicio_fea
df_final_clean = df_final.dropna()

min_max = df_final_clean.select(
    F.min("inicio_fea").alias("data_inicial"),
    F.max("inicio_fea").alias("data_final"),
    F.min("corrida").alias("corrida_inicial"),
    F.max("corrida").alias("corrida_final"),
    F.count("corrida").alias("quantidade_corridas")
).collect()[0]

print(f"Período analisado:")
print(f"Data inicial: {min_max['data_inicial']}")
print(f"Data final:   {min_max['data_final']}")
print(f"Corrida inicial: {min_max['corrida_inicial']}")
print(f"Corrida final:   {min_max['corrida_final']}")
print(f"Quantidade de corridas: {min_max['quantidade_corridas']}")

# COMMAND ----------

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns

# # Seleciona apenas as colunas de tempo e transporte
# cols_hist = [
#     "tempo_no_fea", "tempo_no_fp", "tempo_no_vd", "tempo_no_lc",
#     "transporte_fea_fp", "transporte_fp_vd", "transporte_vd_lc"
# ]

# # Filtra para início de setembro em diante e remove linhas com pelo menos um null
# df_filtrado = (
#     df_final
#     .filter(
#         (F.col("inicio_fea") >= "2025-09-01") &
#         (F.col("ADICIONALTTTFP") == 0) &
#         (F.col("ADICIONALTTTVD") == 0) &
#         (~F.col("corrida").isin(returned_heats))
#     )
#     .dropna(subset=cols_hist)
# )

# # Converte para pandas
# pdf = df_filtrado.select(cols_hist).toPandas()

# # Plota histograma com KDE para cada coluna, apenas entre os percentis 5% e 95%
# for col in cols_hist:
#     data = pdf[col].dropna()
#     p5, p95 = np.percentile(data, [5, 95])
#     data_filtered = data[(data >= p5) & (data <= p95)]
#     binwidth = max((p95 - p5) / 30, 1)
#     bins = max(int((p95 - p5) / binwidth), 1)
#     plt.figure()
#     sns.histplot(
#         data_filtered,
#         stat='count',
#         bins=bins,
#         alpha=0.7,
#         kde=True,
#         kde_kws={'cut': 0}
#     )
#     plt.title(f"{col} - Geral")
#     plt.xlabel("Tempo (min)")
#     plt.ylabel("Frequência")
#     plt.tight_layout()
#     plt.show()

# COMMAND ----------

pdf = df_filtrado.toPandas()
display(pdf.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).reset_index().rename(columns={"index": "estatistica"}))

# COMMAND ----------

import pandas as pd

numeric_cols = [c for c, t in df_filtrado.dtypes if t in ("int", "bigint", "double", "float")]

percentis = {
    col: df_filtrado.approxQuantile(col, [0.25, 0.5, 0.75], 0.01)
    for col in numeric_cols
}

pdf = pd.DataFrame(percentis, index=["25%", "50%", "75%"])
display(pdf)
