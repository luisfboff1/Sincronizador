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
df_fp = spark.read.table("industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_lc")
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
        "inicio_fp", "final_fp", 
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

# ============================================================
# LC – DETECÇÃO DE CICLOS DE LINGOTAMENTO
# Baseado em ACI@LC_TORRE_PESOREAL
# ============================================================

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import Optional


# ============================================================
# 1. REPARO DE GLITCHES (MESMA LÓGICA USADA ANTES)
# ============================================================

def repair_lc_weight_glitches(
    df,
    column: str,
    active_threshold: float = 1.0,
    lookahead_steps: int = 2
):
    """
    Remove quedas rápidas para zero que retornam rapidamente (ruído de OPC)
    """
    w_lag = Window.orderBy("timestamp")
    w_lookahead = Window.orderBy("timestamp").rowsBetween(1, lookahead_steps)
    w_ffill = Window.orderBy("timestamp").rowsBetween(
        Window.unboundedPreceding, Window.currentRow
    )

    is_glitch = (
        (F.col(column) <= active_threshold) &
        (F.lag(column, 1).over(w_lag) > active_threshold) &
        (F.max(column).over(w_lookahead) > active_threshold)
    )

    df_clean = (
        df
        .withColumn(
            f"{column}_clean",
            F.when(is_glitch, F.lit(None)).otherwise(F.col(column))
        )
        .withColumn(
            column,
            F.last(f"{column}_clean", ignorenulls=True).over(w_ffill)
        )
        .drop(f"{column}_clean")
    )

    return df_clean


# ============================================================
# 2. DETECÇÃO DE EVENTOS DE LINGOTAMENTO (LC)
# ============================================================

def detect_lc_lingotamento_events(
    df_lc,
    weight_column: str = "ACI@LC_TORRE_PESOREAL",
    initial_corrida: int = 0,
    initial_timestamp: Optional[str] = None,
    minimal_duration_seconds: int = 60,
    active_threshold: float = 1.0,
    debug: bool = False
):
    """
    Detecta ciclos de lingotamento usando peso real da torre LC.

    Retorna:
    corrida,
    start_time,
    end_time,
    duration_seconds,
    max_weight
    """

    # --------------------------------------------------------
    # 1. Filtro temporal opcional
    # --------------------------------------------------------
    df = df_lc
    if initial_timestamp:
        df = df.filter(F.col("timestamp") >= F.lit(initial_timestamp))

    df = df.select(
        "timestamp",
        F.col(weight_column).alias("peso")
    ).fillna(0)

    # --------------------------------------------------------
    # 2. Reparar glitches
    # --------------------------------------------------------
    df = repair_lc_weight_glitches(
        df,
        column="peso",
        active_threshold=active_threshold,
        lookahead_steps=2
    )

    # --------------------------------------------------------
    # 3. Janelas temporais
    # --------------------------------------------------------
    w = Window.orderBy("timestamp")

    df_edges = df.select(
        "*",
        F.lag("peso", 1, 0).over(w).alias("peso_prev"),
        F.lead("peso", 1, 0).over(w).alias("peso_next"),
    )

    # --------------------------------------------------------
    # 4. Flags de estado
    # --------------------------------------------------------
    df_flagged = df_edges.select(
        "*",
        (
            (F.col("peso_prev") <= active_threshold) &
            (F.col("peso") > active_threshold)
        ).alias("is_start"),

        (
            F.col("peso") > active_threshold
        ).alias("is_active"),

        (
            (F.col("peso") > active_threshold) &
            (F.col("peso_next") <= active_threshold)
        ).alias("is_end")
    )

    # --------------------------------------------------------
    # 5. Criar sessões (lingotamentos)
    # --------------------------------------------------------
    df_sessions = (
        df_flagged
        .filter(F.col("is_active"))
        .withColumn(
            "session_id",
            F.sum(F.col("is_start").cast("long")).over(w)
        )
        .filter(F.col("session_id") > 0)
    )

    # --------------------------------------------------------
    # 6. Agregar eventos
    # --------------------------------------------------------
    events = (
        df_sessions
        .groupBy("session_id")
        .agg(
            F.min("timestamp").alias("start_time"),
            F.max("timestamp").alias("last_seen_time"),
            F.max("peso").alias("max_weight"),
            F.max("is_end").alias("has_clean_end")
        )
        .withColumn(
            "end_time",
            F.when(F.col("has_clean_end"), F.col("last_seen_time"))
        )
        .withColumn(
            "duration_seconds",
            F.col("last_seen_time").cast("long") -
            F.col("start_time").cast("long")
        )
        .filter(F.col("duration_seconds") >= minimal_duration_seconds)
    )

    # --------------------------------------------------------
    # 7. Gerar corrida incremental
    # --------------------------------------------------------
    w_order = Window.orderBy("start_time")

    events = (
        events
        .withColumn(
            "corrida",
            F.lit(initial_corrida) +
            F.row_number().over(w_order) - 1
        )
        .select(
            "corrida",
            "start_time",
            "end_time",
            "duration_seconds",
            "max_weight"
        )
        .orderBy("start_time")
    )

    if debug:
        display(events)

    return events


# ============================================================
# 3. VALIDAÇÃO COM INÍCIO / FIM DE LINGOTAMENTO (LC OFICIAL)
# ============================================================

def validate_lc_events(
    events_df,
    lc_reference_df
):
    """
    Compara eventos detectados vs. timestamps oficiais
    INICIOLINGOTAMENTO / FINALLINGOTAMENTO
    """

    ref = lc_reference_df.select(
        "CORRIDA",
        F.col("INICIOLINGOTAMENTO").alias("start_ref"),
        F.col("FINALLINGOTAMENTO").alias("end_ref"),
    )

    validation = (
        events_df.alias("e")
        .join(ref.alias("r"), "corrida", "left")
        .select(
            "corrida",

            "start_time",
            "start_ref",
            (
                F.col("start_time").cast("long") -
                F.col("start_ref").cast("long")
            ).alias("delta_start_seconds"),

            "end_time",
            "end_ref",
            (
                F.col("end_time").cast("long") -
                F.col("end_ref").cast("long")
            ).alias("delta_end_seconds"),

            "duration_seconds",
            "max_weight"
        )
        .orderBy("corrida")
    )

    return validation


# ============================================================
# 4. USO FINAL
# ============================================================




# COMMAND ----------

df_lc = spark.read.table(
    "industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_lc"
)

events_lc = detect_lc_lingotamento_events(
    df_lc=df_lc,
    weight_column="ACI@LC_TORRE_PESOREAL",
    initial_corrida=130590,
    initial_timestamp="2025-12-09 08:55:00",
    minimal_duration_seconds=120,
)

df_lc_ref = spark.read.table(
    "industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_lc_corridalingotamento"
)

validation = validate_lc_events(events_lc, df_lc_ref)

display(validation)

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import re
from typing import Optional, Dict


def plot_lc_peso_real(
    df: pd.DataFrame,
    timestamp_column: str,
    peso_column: str,
    timestamp_markers: Optional[Dict[str, pd.Timestamp]] = None,
    title: str = "LC – Peso Real da Torre ao longo do tempo",
):
    """
    Plota o peso real da torre do LC com marcadores verticais
    (INICIO / FINAL de lingotamento), mantendo o estilo do gráfico original.
    """

    df = df.copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    min_value = df[peso_column].min()
    max_value = df[peso_column].max()

    # ---------------------------------------------------
    # Figura
    # ---------------------------------------------------
    plt.figure(figsize=(14, 6.2))

    plt.plot(
        df[timestamp_column],
        df[peso_column],
        label="Peso Real – Torre LC",
        linewidth=1.8
    )

    ax = plt.gca()
    trans = ax.get_xaxis_transform()

    # ---------------------------------------------------
    # Marcadores verticais (linhas pontilhadas)
    # ---------------------------------------------------
    if timestamp_markers:
        BASE_COLORS = [
            "darkgreen",
            "darkred",
            "purple",
            "midnightblue",
            "saddlebrown",
        ]

        for i, (label, ts) in enumerate(timestamp_markers.items()):
            if ts is None:
                continue

            ts = pd.to_datetime(ts)

            if ts < df[timestamp_column].min() or ts > df[timestamp_column].max():
                continue

            color = BASE_COLORS[i % len(BASE_COLORS)]

            plt.axvline(x=ts, linestyle="--", color=color, alpha=0.9)

            y_height = 0.25 + (i % 4) * 0.18
            plt.text(
                ts + pd.Timedelta(minutes=1),
                y=y_height,
                s=label,
                transform=trans,
                fontsize="small",
                rotation=90,
                verticalalignment="top",
                color=color
            )

    # ---------------------------------------------------
    # Eixo Y
    # ---------------------------------------------------
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=20))
    plt.ylabel("Peso (kg)")

    # ---------------------------------------------------
    # Eixo X inteligente (igual ao original)
    # ---------------------------------------------------
    min_ts = df[timestamp_column].min()
    max_ts = df[timestamp_column].max()
    total_duration = max_ts - min_ts

    byminute_range_stop = 61
    byminute_range_minor_increment = 15

    if total_duration < pd.Timedelta(hours=6):
        minute_interval = 15
    elif total_duration < pd.Timedelta(hours=24):
        minute_interval = 30
    elif total_duration < pd.Timedelta(days=2):
        minute_interval = 60
    elif total_duration < pd.Timedelta(days=4):
        minute_interval = 4 * 60
        ax.tick_params(axis='x', labelrotation=90)
    else:
        minute_interval = 24 * 60
        ax.tick_params(axis='x', labelrotation=90)

    ax.xaxis.set_major_locator(
        mdates.MinuteLocator(byminute=range(0, byminute_range_stop, minute_interval))
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    ax.xaxis.set_minor_locator(
        mdates.MinuteLocator(byminute=range(0, byminute_range_stop, byminute_range_minor_increment))
    )
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # Grid
    plt.grid(which="major", alpha=0.5, linestyle="-")
    plt.grid(which="minor", alpha=0.15, linestyle=":")

    # ---------------------------------------------------
    # Eixo secundário (Data)
    # ---------------------------------------------------
    sec_xaxis = ax.secondary_xaxis("bottom")
    sec_xaxis.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    sec_xaxis.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
    sec_xaxis.spines["bottom"].set_position(("outward", 30))
    sec_xaxis.spines["bottom"].set_visible(False)

    # ---------------------------------------------------
    # Ajustes finais
    # ---------------------------------------------------
    plt.xlabel("Data / Hora")
    plt.title(title)
    plt.legend(loc="upper left")

    plt.xlim(
        min_ts - pd.Timedelta(minutes=4),
        max_ts + pd.Timedelta(minutes=4)
    )
    plt.ylim(min_value, max_value * 1.10)

    plt.tight_layout()
    plt.show()


# COMMAND ----------

def build_lc_timestamp_markers(df_lc_ref: pd.DataFrame, corrida: int):
    row = df_lc_ref[df_lc_ref["CORRIDA"] == corrida].iloc[0]

    return {
        f"Início Lingotamento {corrida}": row["INICIOLINGOTAMENTO"],
        f"Fim Lingotamento {corrida}": row["FINALLINGOTAMENTO"],
    }


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # NOVA LÓGICA
# MAGIC
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd

df_lc_raw = (
    spark.read.table(
        "industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_lc"
    )
    # 🔥 AJUSTE DE FUSO: UTC → UTC-3 (São Paulo)
    .withColumn(
        "timestamp",
        F.col("timestamp") - F.expr("INTERVAL 3 HOURS")
    )
)

df_lc = (
    df_lc_raw
    .withColumn(
        "peso_raw",
        F.regexp_replace("ACI@LC_TORRE_PESOREAL", ",", ".").cast("double")
    )
)

w = Window.orderBy("timestamp").rowsBetween(Window.unboundedPreceding, 0)

df_lc = df_lc.withColumn(
    "peso",
    F.last("peso_raw", ignorenulls=True).over(w)
)






# COMMAND ----------

START = "2025-12-09 13:00:00"
END   = "2025-12-09 17:00:00"

df_lc_pd = (
    df_lc
    .filter(
        (F.col("timestamp") >= START) &
        (F.col("timestamp") <= END)
    )
    .select("timestamp", "peso")
    .orderBy("timestamp")
    .toPandas()
)

df_lc_pd["timestamp"] = pd.to_datetime(df_lc_pd["timestamp"])


# COMMAND ----------

import matplotlib.pyplot as plt

plt.figure(figsize=(14,5))
plt.plot(df_lc_pd["timestamp"], df_lc_pd["peso"])
plt.title("LC – Peso Real da Torre (PIMS)")
plt.ylabel("Peso")
plt.xlabel("Tempo")
plt.grid(True)
plt.show()


# COMMAND ----------

df_ref_pd = (
    spark.read.table(
        "industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_lc_corridalingotamento"
    )
    .select("CORRIDA", "INICIOLINGOTAMENTO", "FINALLINGOTAMENTO")
    .toPandas()
)

df_ref_pd["INICIOLINGOTAMENTO"] = pd.to_datetime(df_ref_pd["INICIOLINGOTAMENTO"])
df_ref_pd["FINALLINGOTAMENTO"]  = pd.to_datetime(df_ref_pd["FINALLINGOTAMENTO"])


# COMMAND ----------

row = df_ref_pd[df_ref_pd["CORRIDA"] == 130595].iloc[0]

plt.figure(figsize=(14,5))
plt.plot(df_lc_pd["timestamp"], df_lc_pd["peso"], label="Peso Torre")

plt.axvline(row["INICIOLINGOTAMENTO"], color="green", linestyle="--", label="Início Oficial")
plt.axvline(row["FINALLINGOTAMENTO"], color="red", linestyle="--", label="Fim Oficial")

plt.legend()
plt.grid(True)
plt.title("LC – Peso Real + Início/Fim Oficial")
plt.show()



# COMMAND ----------

START_THRESHOLD = 5    # subida brusca
END_THRESHOLD   = 5    # queda brusca
LOW_WEIGHT      = 5    # quase zero
STABLE_SECONDS = 60     # quanto tempo precisa ficar baixo


# COMMAND ----------

# ============================================================
# PARÂMETROS (AJUSTÁVEIS)
# ============================================================

START_THRESHOLD = 5      # subida brusca (kg em 1 segundo)
LOW_WEIGHT = 5           # peso considerado "zero"
STABLE_SECONDS = 60      # segundos em peso baixo para considerar fim

initial_corrida = 130595  # <<< VOCÊ DEFINE AQUI

# ============================================================
# CÁLCULOS AUXILIARES
# ============================================================

df_lc_pd = df_lc_pd.copy()

df_lc_pd["delta"] = df_lc_pd["peso"].diff()
df_lc_pd["is_low"] = df_lc_pd["peso"] <= LOW_WEIGHT

df_lc_pd["low_sustained"] = (
    df_lc_pd["is_low"]
    .rolling(window=STABLE_SECONDS, min_periods=STABLE_SECONDS)
    .mean()
    == 1.0
)

df_lc_pd["start_detected"] = df_lc_pd["delta"] > START_THRESHOLD

# ============================================================
# DETECÇÃO SEQUENCIAL DE CORRIDAS
# ============================================================

events = []
corrida_atual = initial_corrida

i = 0
n = len(df_lc_pd)

while i < n:
    # ---------------- INÍCIO ----------------
    if df_lc_pd.iloc[i]["start_detected"]:
        start_time = df_lc_pd.iloc[i]["timestamp"]

        # ---------------- FIM ----------------
        j = i + 1
        end_time = None

        while j < n:
            if df_lc_pd.iloc[j]["low_sustained"]:
                end_time = df_lc_pd.iloc[j]["timestamp"]
                break
            j += 1

        if end_time is not None:
            events.append({
                "corrida": corrida_atual,
                "start_time": start_time,
                "end_time": end_time
            })

            corrida_atual += 1     # <<< INCREMENTO DA CORRIDA
            i = j                  # pula para depois do fim
        else:
            break
    else:
        i += 1

events_df = pd.DataFrame(events)

print("\nCORRIDAS DETECTADAS:")
display(events_df)


# COMMAND ----------

plt.figure(figsize=(14,5))
plt.plot(df_lc_pd["timestamp"], df_lc_pd["peso"], label="Peso Torre")

for _, r in events_df.iterrows():
    # Início
    plt.axvline(r["start_time"], color="blue", linestyle=":", alpha=0.8)
    plt.text(
        r["start_time"],
        df_lc_pd["peso"].max() * 0.95,
        f"Início {r['corrida']}",
        rotation=90,
        fontsize=9,
        color="blue"
    )

    # Fim
    plt.axvline(r["end_time"], color="orange", linestyle=":", alpha=0.8)
    plt.text(
        r["end_time"],
        df_lc_pd["peso"].max() * 0.95,
        f"Fim {r['corrida']}",
        rotation=90,
        fontsize=9,
        color="orange"
    )

plt.legend()
plt.grid(True)
plt.title("LC – Corridas Detectadas (Sequencial)")
plt.show()


# COMMAND ----------

# ============================================================
# BLOCO A — DADOS OFICIAIS (OPERADOR)
# ============================================================

import pandas as pd

df_official = (
    spark.read.table(
        "industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_lc_corridalingotamento"
    )
    .select("CORRIDA", "INICIOLINGOTAMENTO", "FINALLINGOTAMENTO")
    .toPandas()
)

df_official["INICIOLINGOTAMENTO"] = pd.to_datetime(df_official["INICIOLINGOTAMENTO"])
df_official["FINALLINGOTAMENTO"] = pd.to_datetime(df_official["FINALLINGOTAMENTO"])

# Recorte temporal do gráfico
PLOT_START = pd.Timestamp("2025-12-09 12:00:00")
PLOT_END   = pd.Timestamp("2025-12-09 17:00:00")

df_official_window = df_official[
    (df_official["INICIOLINGOTAMENTO"] >= PLOT_START) &
    (df_official["FINALLINGOTAMENTO"] <= PLOT_END)
].sort_values("INICIOLINGOTAMENTO")

display(df_official_window)


# COMMAND ----------

# -----------------------------
# PARÂMETROS DA LÓGICA
# -----------------------------
THR_UP = 120                 # cruzamento "pra entrar" (>=)  (ajuste: 100~150)
THR_DOWN = 80                # histerese "pra considerar que estava fora" (<) (ajuste: 50~100)

SMOOTH_WINDOW_S = 15         # suavização (segundos)
SLOPE_WINDOW_S = 30          # janela pra calcular inclinação (segundos)
MIN_SLOPE = 10               # quanto tem que subir na janela (kg em SLOPE_WINDOW_S) (ajuste: 5~30)

LOOKBACK_LOW_S = 5 * 60      # precisa ter ficado "baixo" em algum momento nos últimos X segundos
LOW_WEIGHT = 5               # quase zero (fim)
STABLE_SECONDS = 60          # fim = low sustentado
initial_corrida = 130594





# COMMAND ----------

# ============================================================
# BLOCO B — LÓGICA DE DETECÇÃO LC
# ============================================================

from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd

# --- Ler LC (UTC → UTC-3)
df_lc_raw = spark.read.table("industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_lc")

df_lc = (
    df_lc_raw
    .withColumn("timestamp_sp", F.col("timestamp") - F.expr("INTERVAL 3 HOURS"))
    .withColumn(
        "peso_raw",
        F.regexp_replace("ACI@LC_TORRE_PESOREAL", ",", ".").cast("double")
    )
)

# Forward fill (PIMS)
w = Window.orderBy("timestamp_sp").rowsBetween(Window.unboundedPreceding, 0)
df_lc = df_lc.withColumn("peso", F.last("peso_raw", ignorenulls=True).over(w))

# Recorte
df_lc_pd = (
    df_lc
    .filter((F.col("timestamp_sp") >= PLOT_START) & (F.col("timestamp_sp") <= PLOT_END))
    .select("timestamp_sp", "peso")
    .orderBy("timestamp_sp")
    .toPandas()
)

df_lc_pd["timestamp"] = pd.to_datetime(df_lc_pd["timestamp_sp"])
df_lc_pd = df_lc_pd.drop(columns="timestamp_sp")




# COMMAND ----------

# -----------------------------
# PRÉ-COND: df_lc_pd precisa ter ["timestamp","peso"] e estar ordenado
# -----------------------------
df = df_lc_pd.copy().sort_values("timestamp").reset_index(drop=True)

# Se o timestamp não for regular, a suavização por "segundos" fica ruim.
# Assumindo 1 linha ~ 1 segundo (o seu caso). Se faltar segundos, ainda funciona.
w = int(SMOOTH_WINDOW_S)
k = int(SLOPE_WINDOW_S)
lb = int(LOOKBACK_LOW_S)

# -----------------------------
# 1) Suavizar peso (robusto a ruído)
#    mediana é ótima para "dentinhos"
# -----------------------------
df["peso_smooth"] = (
    df["peso"]
    .rolling(window=w, min_periods=max(3, w//3))
    .median()
)

# -----------------------------
# 2) Inclinação em janela (não delta de 1 segundo)
# -----------------------------
df["slope"] = df["peso_smooth"] - df["peso_smooth"].shift(k)

# -----------------------------
# 3) Flag de "baixo recente" (para ignorar corrida já em andamento)
#    Queremos: nos últimos LOOKBACK_LOW_S, existiu algum ponto < THR_DOWN
# -----------------------------
df["was_low_recent"] = (
    (df["peso_smooth"] < THR_DOWN)
    .rolling(window=lb, min_periods=max(10, lb//5))
    .max()
    .fillna(0)
    .astype(bool)
)

# -----------------------------
# 4) Start candidate:
#    - peso suavizado cruza THR_UP para cima
#    - slope >= MIN_SLOPE (subiu "de verdade" nos últimos k segundos)
#    - havia estado baixo recente (não pegar descida já em andamento)
# -----------------------------
df["above_up"] = df["peso_smooth"] >= THR_UP
df["cross_up"] = df["above_up"] & (~df["above_up"].shift(1, fill_value=False))

df["start_detected_flag"] = (
    df["cross_up"] &
    (df["slope"] >= MIN_SLOPE) &
    (df["was_low_recent"])
)

# -----------------------------
# 5) Fim: low sustentado (usando peso_smooth)
# -----------------------------
df["is_low"] = df["peso_smooth"] <= LOW_WEIGHT
df["low_sustained"] = (
    df["is_low"]
    .rolling(window=STABLE_SECONDS, min_periods=STABLE_SECONDS)
    .mean()
    == 1.0
)

# -----------------------------
# 6) Extrair eventos (múltiplas corridas)
# -----------------------------
events = []
corrida = initial_corrida
i = 0
n = len(df)

while i < n:
    if bool(df.iloc[i]["start_detected_flag"]):
        start_time = df.iloc[i]["timestamp"]

        # procurar fim
        j = i + 1
        end_time = None
        while j < n:
            if bool(df.iloc[j]["low_sustained"]):
                end_time = df.iloc[j]["timestamp"]
                break
            j += 1

        if end_time is not None:
            events.append({
                "corrida": corrida,
                "start_detected": start_time,
                "end_detected": end_time
            })
            corrida += 1
            i = j
        else:
            break
    else:
        i += 1

df_detected = pd.DataFrame(events)

# -----------------------------
# 7) DEBUG ÚTIL (pra ajustar parâmetros rápido)
# -----------------------------
print("N starts detectados:", int(df["start_detected_flag"].sum()))
print("N eventos fechados :", len(df_detected))

# Mostra candidatos a início (se existirem)
cand = df[df["start_detected_flag"]][["timestamp","peso","peso_smooth","slope","was_low_recent"]].head(20)
if len(cand) > 0:
    display(cand)

# Resultado final
if df_detected.empty:
    print("⚠️ Nenhuma corrida detectada — ajuste THR_UP/THR_DOWN, MIN_SLOPE, SLOPE_WINDOW_S, LOOKBACK_LOW_S.")
else:
    display(df_detected)

# COMMAND ----------

# ============================================================
# BLOCO C — PLOT
# ============================================================

import matplotlib.pyplot as plt
import pandas as pd

plt.figure(figsize=(14,5))
plt.plot(df_lc_pd["timestamp"], df_lc_pd["peso"], label="Peso Torre")

# -----------------------------
# ALTURAS DOS TEXTOS (EM KG)
# -----------------------------
y_max = df_lc_pd["peso"].max()

Y_INICIO_DET = y_max * 0.20
Y_INICIO_OP  = y_max * 0.70
Y_FIM_DET    = y_max * 0.50
Y_FIM_OP     = y_max * 0.90

# -----------------------------
# DESLOCAMENTO HORIZONTAL
# -----------------------------
TEXT_OFFSET = pd.Timedelta(minutes=0)  # ajuste se quiser (1–3 min)

# -----------------------------
# OFICIAL (OPERADOR)
# -----------------------------
for _, r in df_official_window.iterrows():
    c = int(r["CORRIDA"])

    # Início OP → texto à DIREITA
    plt.axvline(r["INICIOLINGOTAMENTO"], color="green", linestyle="--")
    plt.text(
        r["INICIOLINGOTAMENTO"] + TEXT_OFFSET,
        Y_INICIO_OP,
        f"Início (OP) {c}",
        rotation=90,
        color="green",
        va="top",
        ha="left"
    )

    # Fim OP → texto à ESQUERDA
    plt.axvline(r["FINALLINGOTAMENTO"], color="red", linestyle="--")
    plt.text(
        r["FINALLINGOTAMENTO"] - TEXT_OFFSET,
        Y_FIM_OP,
        f"Fim (OP) {c}",
        rotation=90,
        color="red",
        va="top",
        ha="right"
    )

# -----------------------------
# DETECTADO (NOSSA LÓGICA)
# -----------------------------
for _, r in df_detected.iterrows():
    c = int(r["corrida"])

    # Início Detectado → texto à DIREITA
    plt.axvline(r["start_detected"], color="blue", linestyle=":")
    plt.text(
        r["start_detected"] + TEXT_OFFSET,
        Y_INICIO_DET,
        f"Início {c}",
        rotation=90,
        color="blue",
        va="top",
        ha="left"
    )

    # Fim Detectado → texto à ESQUERDA
    plt.axvline(r["end_detected"], color="orange", linestyle=":")
    plt.text(
        r["end_detected"] - TEXT_OFFSET,
        Y_FIM_DET,
        f"Fim {c}",
        rotation=90,
        color="orange",
        va="top",
        ha="right"
    )

# -----------------------------
# FINALIZAÇÃO
# -----------------------------
plt.legend([
    "Peso Torre",
    "Início Oficial", "Fim Oficial",
    "Início Detectado", "Fim Detectado"
])

plt.title("LC — Oficial x Detectado")
plt.xlabel("Data / Hora")
plt.ylabel("Peso Torre")
plt.grid(True)
plt.show()



# COMMAND ----------

# MAGIC %md
# MAGIC ## 📘 Lógica de Detecção de Início e Fim de Corridas no LC
# MAGIC
# MAGIC ### 1. Contexto do Processo
# MAGIC
# MAGIC No processo de **Lingotamento Contínuo (LC)**, a torre de pesagem recebe panelas de aço antes e durante o vazamento.  
# MAGIC O sinal **ACI@LC_TORRE_PESOREAL** é fornecido pelo sistema **PIMS** com alta frequência (segundos).
# MAGIC
# MAGIC Características do sinal:
# MAGIC
# MAGIC - O valor só é enviado quando muda
# MAGIC - Quando o valor não muda, o PIMS envia `NULL`
# MAGIC - O timestamp do PIMS está em **UTC**
# MAGIC - Os dados operacionais de referência (início e fim de lingotamento) estão em **UTC-3**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2. Tratamento Inicial dos Dados
# MAGIC
# MAGIC Antes de qualquer lógica de detecção, os dados passam por correções obrigatórias.
# MAGIC
# MAGIC #### 2.1 Correção de Fuso Horário
# MAGIC
# MAGIC O timestamp do PIMS é convertido de **UTC para UTC-3**, garantindo alinhamento temporal com os eventos do operador.
# MAGIC
# MAGIC #### 2.2 Preenchimento de Valores Nulos (Forward Fill)
# MAGIC
# MAGIC Valores `NULL` no PIMS **não representam ausência de peso**.  
# MAGIC Eles indicam que o valor atual é **igual ao último valor válido**.
# MAGIC
# MAGIC Portanto:
# MAGIC
# MAGIC - Não se deve remover linhas com `NULL`
# MAGIC - Aplica-se **forward fill** para manter a continuidade da série temporal
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3. Comportamento Esperado do Sinal de Peso
# MAGIC
# MAGIC #### 3.1 Início de uma Corrida
# MAGIC
# MAGIC O início real de uma corrida ocorre quando:
# MAGIC
# MAGIC - O peso começa a **subir**
# MAGIC - A subida é **contínua**, não um degrau isolado
# MAGIC - O peso ultrapassa um **valor mínimo configurável**
# MAGIC - A subida representa uma **rampa real**, e não tara
# MAGIC
# MAGIC Visualmente, o início aparece como uma **rampa ascendente clara**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 3.2 O que NÃO é Início de Corrida
# MAGIC
# MAGIC A lógica deve ignorar:
# MAGIC
# MAGIC - Pequenos degraus de peso (tara)
# MAGIC - Subidas curtas que estabilizam
# MAGIC - Trechos em que o peso já está alto
# MAGIC - Situações em que o peso está **descendo**
# MAGIC
# MAGIC Para ser considerado início, o peso **precisa estar em regime de subida**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 3.3 Fim de uma Corrida
# MAGIC
# MAGIC O fim de uma corrida ocorre quando:
# MAGIC
# MAGIC - O peso entra em regime de **queda**
# MAGIC - O valor atinge um patamar próximo de zero
# MAGIC - O peso permanece baixo por um período mínimo
# MAGIC
# MAGIC A descida pode ser **gradual**.  
# MAGIC O fim **não é instantâneo**, mas confirmado após estabilização em peso baixo.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 4. Estratégia de Detecção
# MAGIC
# MAGIC A estratégia geral consiste em:
# MAGIC
# MAGIC 1. Detectar rampas de subida
# MAGIC 2. Confirmar o início apenas quando a rampa ultrapassa um peso mínimo
# MAGIC 3. Ignorar rampas incompletas (tara)
# MAGIC 4. Após o início, monitorar a queda do peso
# MAGIC 5. Confirmar o fim quando o peso permanece baixo por tempo suficiente
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 5. Parâmetros da Lógica
# MAGIC
# MAGIC Parâmetros configuráveis da detecção:
# MAGIC
# MAGIC - **MIN_START_WEIGHT**  
# MAGIC   Peso mínimo para confirmar o início de uma corrida
# MAGIC
# MAGIC - **RAMP_WINDOW_SECONDS**  
# MAGIC   Janela mínima de subida contínua
# MAGIC
# MAGIC - **RAMP_DELTA_MIN**  
# MAGIC   Variação mínima de peso para caracterizar subida
# MAGIC
# MAGIC - **LOW_WEIGHT**  
# MAGIC   Peso considerado próximo de zero
# MAGIC
# MAGIC - **STABLE_LOW_SECONDS**  
# MAGIC   Tempo mínimo sustentado em peso baixo
# MAGIC
# MAGIC - **INITIAL_CORRIDA**  
# MAGIC   Número da primeira corrida no período analisado
# MAGIC
# MAGIC Esses parâmetros devem ser ajustáveis para calibração da lógica.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 6. Numeração das Corridas
# MAGIC
# MAGIC O algoritmo **não identifica automaticamente** o número da corrida.
# MAGIC
# MAGIC É necessário fornecer uma **corrida inicial**.  
# MAGIC A cada novo evento detectado, o número da corrida é incrementado sequencialmente.
# MAGIC
# MAGIC Exemplo:
# MAGIC
# MAGIC - Corrida inicial: 130595  
# MAGIC - Próxima corrida detectada: 130596  
# MAGIC - Seguinte: 130597
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 7. Resultado Esperado
# MAGIC
# MAGIC A lógica gera uma estrutura contendo:
# MAGIC
# MAGIC - Número da corrida
# MAGIC - Timestamp de início detectado
# MAGIC - Timestamp de fim detectado
# MAGIC
# MAGIC Esses eventos podem ser:
# MAGIC
# MAGIC - Comparados com dados do operador
# MAGIC - Visualizados em gráficos
# MAGIC - Utilizados futuramente em processamento em tempo real (streaming)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧠 Pseudocódigo — Detecção de Corridas no LC 
# MAGIC
# MAGIC ### Entrada (Input)
# MAGIC
# MAGIC - Série temporal contendo:
# MAGIC   - `timestamp`
# MAGIC   - `peso`
# MAGIC - Número da corrida inicial (`initial_corrida`)
# MAGIC - Parâmetros de detecção (thresholds)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Preparação dos Dados
# MAGIC
# MAGIC - Converter o `timestamp` de **UTC para UTC-3**
# MAGIC - Aplicar **forward fill** nos valores de peso
# MAGIC   - Valores nulos passam a assumir o último valor válido
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Inicialização
# MAGIC
# MAGIC - Criar uma lista vazia de eventos detectados
# MAGIC - Definir `corrida_atual = initial_corrida`
# MAGIC - Definir `índice = 0`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Loop Principal de Detecção
# MAGIC
# MAGIC - Enquanto o índice for menor que o tamanho da série temporal:
# MAGIC
# MAGIC   - Verificar se o peso está em **regime de subida**
# MAGIC     - Subida não precisa ser perfeitamente monotônica
# MAGIC     - Pequenas oscilações são aceitáveis
# MAGIC
# MAGIC   - Se uma **rampa de subida** for identificada:
# MAGIC     - Verificar se a rampa se mantém por uma janela mínima de tempo
# MAGIC
# MAGIC     - Se a rampa persistir:
# MAGIC       - Verificar se o peso ultrapassa o valor mínimo (`MIN_START_WEIGHT`)
# MAGIC       - Se ultrapassar:
# MAGIC         - Confirmar **início da corrida**
# MAGIC         - Registrar o timestamp de início
# MAGIC
# MAGIC         - Iniciar a busca pelo **fim da corrida**:
# MAGIC
# MAGIC           - Avançar no tempo
# MAGIC           - Verificar se o peso cai abaixo de `LOW_WEIGHT`
# MAGIC           - Se cair:
# MAGIC             - Iniciar contagem de tempo em peso baixo
# MAGIC             - Se o peso permanecer abaixo do limite por `STABLE_LOW_SECONDS`:
# MAGIC               - Confirmar **fim da corrida**
# MAGIC               - Registrar o timestamp de fim
# MAGIC               - Salvar o evento:
# MAGIC                 - corrida
# MAGIC                 - início detectado
# MAGIC                 - fim detectado
# MAGIC               - Incrementar `corrida_atual`
# MAGIC               - Avançar o índice para depois do fim
# MAGIC               - Retornar ao loop principal
# MAGIC
# MAGIC   - Caso contrário:
# MAGIC     - Avançar o índice normalmente
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Saída (Output)
# MAGIC
# MAGIC - Estrutura contendo, para cada corrida detectada:
# MAGIC   - Número da corrida
# MAGIC   - Timestamp de início detectado
# MAGIC   - Timestamp de fim detectado
# MAGIC

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def detect_lc_events(
    spark_df,
    timestamp_col: str,
    peso_col: str,
    start_ts: str,
    end_ts: str,
    initial_corrida: int,

    # -----------------------------
    # PARÂMETROS DA LÓGICA
    # -----------------------------
    THR_UP: float = 120,
    THR_DOWN: float = 80,

    SMOOTH_WINDOW_S: int = 15,
    SLOPE_WINDOW_S: int = 30,

    MIN_POS_SLOPE: float = 10,      # subida mínima p/ start clássico
    MIN_ALT_POS_SLOPE: float = 30,  # subida forte p/ start alternativo

    MIN_NEG_SLOPE: float = -10,     # descida mínima p/ confirmar corrida

    MIN_DROP_CONFIRM: float = 40,   # queda acumulada mínima
    MAX_TIME_TO_DESCEND_S: int = 4 * 60,

    LOOKBACK_LOW_S: int = 5 * 60,

    LOW_WEIGHT: float = 50,
    STABLE_SECONDS: int = 60,
    END_SLOPE_EPS: float = -2,
):
    """
    Detecta eventos de Lingotamento Contínuo (LC) a partir do peso da torre.

    Retorna DataFrame Pandas com:
      - corrida
      - start_detected
      - end_detected
    """

    # =========================================================
    # 1. PREPARAÇÃO DOS DADOS (UTC → UTC-3 + forward fill)
    # =========================================================
    df_lc = (
        spark_df
        .withColumn("timestamp_sp", F.col(timestamp_col) - F.expr("INTERVAL 3 HOURS"))
        .withColumn(
            "peso_raw",
            F.regexp_replace(F.col(peso_col), ",", ".").cast("double")
        )
    )

    w_ffill = Window.orderBy("timestamp_sp").rowsBetween(Window.unboundedPreceding, 0)
    df_lc = df_lc.withColumn("peso", F.last("peso_raw", ignorenulls=True).over(w_ffill))

    df_pd = (
        df_lc
        .filter((F.col("timestamp_sp") >= start_ts) & (F.col("timestamp_sp") <= end_ts))
        .select("timestamp_sp", "peso")
        .orderBy("timestamp_sp")
        .toPandas()
    )

    if df_pd.empty:
        return pd.DataFrame(columns=["corrida", "start_detected", "end_detected"])

    df_pd["timestamp"] = pd.to_datetime(df_pd["timestamp_sp"])
    df = (
        df_pd
        .drop(columns="timestamp_sp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # =========================================================
    # 2. SUAVIZAÇÃO E INCLINAÇÃO
    # =========================================================
    w = int(SMOOTH_WINDOW_S)
    k = int(SLOPE_WINDOW_S)
    lb = int(LOOKBACK_LOW_S)

    df["peso_smooth"] = (
        df["peso"]
        .rolling(window=w, min_periods=max(3, w // 3))
        .median()
    )

    df["slope"] = df["peso_smooth"] - df["peso_smooth"].shift(k)

    # =========================================================
    # 3. BAIXO RECENTE
    # =========================================================
    df["was_low_recent"] = (
        (df["peso_smooth"] < THR_DOWN)
        .rolling(window=lb, min_periods=max(10, lb // 5))
        .max()
        .fillna(0)
        .astype(bool)
    )

    # =========================================================
    # 4. START CANDIDATES
    # =========================================================
    df["above_up"] = df["peso_smooth"] >= THR_UP
    df["cross_up"] = df["above_up"] & (~df["above_up"].shift(1, fill_value=False))

    df["start_classic"] = (
        df["cross_up"] &
        (df["slope"] >= MIN_POS_SLOPE) &
        (df["was_low_recent"])
    )

    df["start_alt"] = (
        (df["slope"] >= MIN_ALT_POS_SLOPE) &
        (df["peso_smooth"] >= THR_UP)
    )

    df["start_candidate"] = df["start_classic"] | df["start_alt"]

    # =========================================================
    # 5. END CONDITION
    # =========================================================
    df["is_low"] = df["peso_smooth"] <= LOW_WEIGHT
    df["end_not_descending"] = df["slope"] >= END_SLOPE_EPS
    df["end_candidate"] = df["is_low"] & df["end_not_descending"]

    df["low_sustained"] = (
        df["end_candidate"]
        .rolling(window=STABLE_SECONDS, min_periods=STABLE_SECONDS)
        .mean()
        == 1.0
    )

    # =========================================================
    # 6. MÁQUINA DE ESTADOS
    # =========================================================
    events = []
    corrida = initial_corrida

    state = "IDLE"
    candidate_start_time = None
    candidate_start_weight = None
    start_time = None

    end_candidate_time = None  # guarda o começo da janela atual de "fim"

    for _, row in df.iterrows():

        # -------------------------
        # IDLE → POSSIBLE_START
        # -------------------------
        if state == "IDLE":
            if bool(row["start_candidate"]):
                state = "POSSIBLE_START"
                candidate_start_time = row["timestamp"]
                candidate_start_weight = row["peso_smooth"]
            continue

        # -------------------------
        # POSSIBLE_START
        # -------------------------
        if state == "POSSIBLE_START":
            elapsed = (row["timestamp"] - candidate_start_time).total_seconds()
            drop = row["peso_smooth"] - candidate_start_weight

            # Confirma corrida
            if (row["slope"] <= MIN_NEG_SLOPE) or (drop <= -MIN_DROP_CONFIRM):
                state = "RUNNING"
                start_time = candidate_start_time
                end_candidate_time = None
                continue

            # Timeout → falso positivo
            if elapsed >= MAX_TIME_TO_DESCEND_S:
                state = "IDLE"
                candidate_start_time = None
                candidate_start_weight = None
                continue

        # -------------------------
        # RUNNING → END
        # -------------------------
        if state == "RUNNING":

            # 1) Se parou de ser candidato a fim, zera (descartando "tentativa antiga")
            if (not bool(row["end_candidate"])) and (end_candidate_time is not None):
                end_candidate_time = None

            # 2) Se virou candidato a fim agora, marca início da janela
            if bool(row["end_candidate"]) and (end_candidate_time is None):
                end_candidate_time = row["timestamp"]

            # 3) Se confirmou sustentado, fecha com o começo da janela atual
            if bool(row["low_sustained"]) and (end_candidate_time is not None):
                events.append({
                    "corrida": corrida,
                    "start_detected": start_time,
                    "end_detected": end_candidate_time,  # início real da janela válida
                })
                corrida += 1
                state = "IDLE"
                candidate_start_time = None
                candidate_start_weight = None
                start_time = None
                end_candidate_time = None

    return pd.DataFrame(events)


# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pyspark.sql import functions as F


def debug_lc_period(
    spark_df_lc,
    spark_df_official,
    timestamp_col: str,
    peso_col: str,
    start_ts: str,
    end_ts: str,
    initial_corrida: int,
    detect_params: dict,
):
    """
    Executa a detecção do LC em um período
    e plota:
      - Peso da torre
      - Início/Fim oficial
      - Início/Fim detectado
    """

    # ---------------------------------------------------------
    # 1. DETECTAR EVENTOS
    # ---------------------------------------------------------
    df_detected = detect_lc_events(
        spark_df=spark_df_lc,
        timestamp_col=timestamp_col,
        peso_col=peso_col,
        start_ts=start_ts,
        end_ts=end_ts,
        initial_corrida=initial_corrida,
        **detect_params,
    )

    # ---------------------------------------------------------
    # 2. PREPARAR SÉRIE PARA PLOT
    # ---------------------------------------------------------
    df_plot = (
        spark_df_lc
        .withColumn(
            "timestamp_sp",
            F.col(timestamp_col) - F.expr("INTERVAL 3 HOURS")
        )
        .withColumn(
            "peso",
            F.regexp_replace(F.col(peso_col), ",", ".").cast("double")
        )
        .filter((F.col("timestamp_sp") >= start_ts) & (F.col("timestamp_sp") <= end_ts))
        .orderBy("timestamp_sp")
        .toPandas()
    )

    # 🔥 GARANTIA DE DATETIME (FIX DO ERRO)
    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp_sp"])
    df_plot = df_plot.drop(columns=["timestamp_sp"])

    # ---------------------------------------------------------
    # 3. EVENTOS OFICIAIS (OPERADOR)
    # ---------------------------------------------------------
    df_official = (
        spark_df_official
        .filter(
            (F.col("INICIOLINGOTAMENTO") >= start_ts) &
            (F.col("FINALLINGOTAMENTO") <= end_ts)
        )
        .toPandas()
    )

    # Garantir datetime também
    if not df_official.empty:
        df_official["INICIOLINGOTAMENTO"] = pd.to_datetime(df_official["INICIOLINGOTAMENTO"])
        df_official["FINALLINGOTAMENTO"] = pd.to_datetime(df_official["FINALLINGOTAMENTO"])

    # ---------------------------------------------------------
    # 4. PLOT (COM CONVERSOR DE DATA EXPLÍCITO)
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 5))
    ax = plt.gca()

    ax.plot(df_plot["timestamp"], df_plot["peso"], label="Peso Torre")

    # 🔥 FIX DEFINITIVO PARA DATETIME NO MATPLOTLIB
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    y_max = df_plot["peso"].max()

    # ---------- OFICIAL ----------
    for _, r in df_official.iterrows():
        c = int(r["CORRIDA"])

        ax.axvline(r["INICIOLINGOTAMENTO"], color="green", linestyle="--")
        ax.text(
            r["INICIOLINGOTAMENTO"],
            y_max * 0.95,
            f"Início OP {c}",
            rotation=90,
            color="green",
            ha="right",
        )

        ax.axvline(r["FINALLINGOTAMENTO"], color="red", linestyle="--")
        ax.text(
            r["FINALLINGOTAMENTO"],
            y_max * 0.90,
            f"Fim OP {c}",
            rotation=90,
            color="red",
            ha="right",
        )

    # ---------- DETECTADO ----------
    for _, r in df_detected.iterrows():
        c = int(r["corrida"])

        ax.axvline(r["start_detected"], color="blue", linestyle=":")
        ax.text(
            r["start_detected"],
            y_max * 0.85,
            f"Início {c}",
            rotation=90,
            color="blue",
            ha="left",
        )

        ax.axvline(r["end_detected"], color="orange", linestyle=":")
        ax.text(
            r["end_detected"],
            y_max * 0.80,
            f"Fim {c}",
            rotation=90,
            color="orange",
            ha="right",
        )

    ax.set_title("LC — Oficial x Detectado")
    ax.set_xlabel("Horário")
    ax.set_ylabel("Peso")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return df_detected


# COMMAND ----------

df_lc_official = spark.read.table(
    "industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_lc_corridalingotamento"
).select(
    "CORRIDA",
    "INICIOLINGOTAMENTO",
    "FINALLINGOTAMENTO"
)


# COMMAND ----------

detect_params = dict(
    # =============================
    # START — níveis absolutos
    # =============================
    THR_UP=120,          # cruzamento clássico de entrada
    THR_DOWN=80,         # histerese: considerar "baixo recente"

    # =============================
    # JANELAS TEMPORAIS
    # =============================
    SMOOTH_WINDOW_S=15,  # suavização do peso (mediana)
    SLOPE_WINDOW_S=30,   # janela para cálculo de inclinação

    # =============================
    # START — dinâmica
    # =============================
    MIN_POS_SLOPE=10,        # subida mínima p/ start clássico
    MIN_ALT_POS_SLOPE=30,    # subida forte p/ start alternativo (corrida já alta)

    # =============================
    # CONFIRMAÇÃO DE CORRIDA
    # =============================
    MIN_NEG_SLOPE=-10,       # começou a cair (confirma corrida)
    MIN_DROP_CONFIRM=40,     # queda acumulada mínima
    MAX_TIME_TO_DESCEND_S=6 * 60,  # timeout p/ descartar falso start

    # =============================
    # HISTERESE TEMPORAL
    # =============================
    LOOKBACK_LOW_S=5 * 60,   # quanto tempo olhar para trás p/ ver se estava baixo

    # =============================
    # END — final da corrida
    # =============================
    LOW_WEIGHT=300,           # não precisa zerar
    STABLE_SECONDS=60,       # fim precisa sustentar
    END_SLOPE_EPS=-1.5,        # não pode mais estar descendo
)




df_events = debug_lc_period(
    spark_df_lc=df_lc,
    spark_df_official=df_lc_official,
    timestamp_col="timestamp",
    peso_col="ACI@LC_TORRE_PESOREAL",
    start_ts="2025-12-03 19:01:00",
    end_ts="2025-12-03 23:20:00",
    initial_corrida=130494,
    detect_params=detect_params,
)


# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd


def get_lc_events_comparison(
    spark_df_lc,
    spark_df_official,
    start_ts: str,
    end_ts: str,
    initial_corrida: int,
    detect_params: dict,
):
    """
    Retorna um DataFrame Pandas com:
      - corrida
      - start_official
      - end_official
      - start_detected
      - end_detected
      - diff_start_min
      - diff_end_min
    """

    # =====================================================
    # 1) Detectado (NOSSA lógica)
    # =====================================================
    df_detected = detect_lc_events(
        spark_df=spark_df_lc,
        timestamp_col="timestamp",
        peso_col="ACI@LC_TORRE_PESOREAL",
        start_ts=start_ts,
        end_ts=end_ts,
        initial_corrida=initial_corrida,
        **detect_params,
    )

    # =====================================================
    # 2) Oficial (OPERADOR)
    # =====================================================
    df_official = (
        spark_df_official
        .filter(
            (F.col("INICIOLINGOTAMENTO") >= start_ts) &
            (F.col("FINALLINGOTAMENTO") <= end_ts)
        )
        .select(
            F.col("CORRIDA").alias("corrida"),
            F.col("INICIOLINGOTAMENTO").alias("start_official"),
            F.col("FINALLINGOTAMENTO").alias("end_official"),
        )
        .toPandas()
    )

    # =====================================================
    # 3) Merge (mantendo sua lógica original)
    # =====================================================
    df_result = (
        df_detected
        .merge(df_official, on="corrida", how="outer")
        .sort_values("corrida")
        .reset_index(drop=True)
    )

    # =====================================================
    # 4) Normalizar timestamps (MESMO FUSO, só formato)
    # =====================================================
    for col in ["start_detected", "end_detected"]:
        if col in df_result.columns:
            df_result[col] = (
                pd.to_datetime(df_result[col])
                .dt.tz_localize(None)
            )

    for col in ["start_official", "end_official"]:
        if col in df_result.columns:
            df_result[col] = pd.to_datetime(df_result[col])

    # =====================================================
    # 5) Diferenças em minutos
    # =====================================================
    df_result["diff_start_min"] = (
        (df_result["start_detected"] - df_result["start_official"])
        .dt.total_seconds() / 60
    )

    df_result["diff_end_min"] = (
        (df_result["end_detected"] - df_result["end_official"])
        .dt.total_seconds() / 60
    )

    return df_result


# COMMAND ----------

df_events = get_lc_events_comparison(
    spark_df_lc=df_lc_raw,
    spark_df_official=df_lc_official,
    start_ts="2025-12-02 00:01:00",
    end_ts="2025-12-21 18:00:00",
    initial_corrida=130449,
    detect_params=detect_params,
)

display(df_events)


# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# Clipar as diferenças para o histograma
start_clip = df_events["diff_start_min"].clip(-5, 10)
end_clip = df_events["diff_end_min"].clip(-5, 10)

plt.figure(figsize=(10, 5))
sns.histplot(start_clip, color="blue", label="Start", alpha=0.5, bins=30)
sns.histplot(end_clip, color="orange", label="End", alpha=0.5, bins=30)
plt.legend()
plt.title("Histograma das diferenças (min) — Start vs End ")
plt.xlabel("Diferença (min)")
plt.ylabel("Contagem")
plt.grid(True)
plt.tight_layout()
plt.show()

df_gt10 = df_events[(df_events["diff_start_min"] > 10) | (df_events["diff_end_min"] > 10)]
display(df_gt10)

n_total = len(df_events)
n_gt5 = ((df_events["diff_start_min"] > 5) | (df_events["diff_end_min"] > 5)).sum()
n_gt10 = ((df_events["diff_start_min"] > 10) | (df_events["diff_end_min"] > 10)).sum()
n_gt20 = ((df_events["diff_start_min"] > 20) | (df_events["diff_end_min"] > 20)).sum()
n_gt50 = ((df_events["diff_start_min"] > 50) | (df_events["diff_end_min"] > 50)).sum()

pct_gt5 = 100 * n_gt5 / n_total if n_total > 0 else 0
pct_gt10 = 100 * n_gt10 / n_total if n_total > 0 else 0
pct_gt20 = 100 * n_gt20 / n_total if n_total > 0 else 0
pct_gt50 = 100 * n_gt50 / n_total if n_total > 0 else 0

print(f"Total de corridas: {n_total}")
print(f"N > 5 min: {n_gt5} ({pct_gt5:.1f}%)")
print(f"N > 10 min: {n_gt10} ({pct_gt10:.1f}%)")
print(f"N > 20 min: {n_gt20} ({pct_gt20:.1f}%)")
print(f"N > 50 min: {n_gt50} ({pct_gt50:.1f}%)")