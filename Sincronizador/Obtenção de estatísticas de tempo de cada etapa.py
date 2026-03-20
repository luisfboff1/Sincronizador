# Databricks notebook source
# MAGIC %md
# MAGIC catalog.schema
# MAGIC
# MAGIC [_industrial_trusted_prd.gsb_cha_iba_aciaria_](https://dbc-de54e264-692b.cloud.databricks.com/explore/data/industrial_trusted_prd/gsb_cha_iba_aciaria?o=2630899921881782&activeListType=TABLE)
# MAGIC > [industrial_trusted_prd.gsb_cha_iba_aciaria.tb_aciaria_lc](https://dbc-de54e264-692b.cloud.databricks.com/explore/data/industrial_trusted_prd/gsb_cha_iba_aciaria/tb_aciaria_lc?o=2630899921881782)
# MAGIC
# MAGIC [_industrial_trusted_prd.gsb_cha_mes_aciaria_](https://dbc-de54e264-692b.cloud.databricks.com/explore/data/industrial_trusted_prd/gsb_cha_mes_aciaria?o=2630899921881782&activeListType=TABLE)
# MAGIC > 
# MAGIC
# MAGIC [_industrial_trusted_prd.gsb_cha_mes_bra_](https://dbc-de54e264-692b.cloud.databricks.com/explore/data/industrial_trusted_prd/gsb_cha_mes_bra?o=2630899921881782&activeListType=TABLE)
# MAGIC
# MAGIC [_industrial_trusted_prd.gsb_cha_opc_aciaria_](https://dbc-de54e264-692b.cloud.databricks.com/explore/data/industrial_trusted_prd/gsb_cha_opc_aciaria?o=2630899921881782&activeListType=TABLE)
# MAGIC > [industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_fp](https://dbc-de54e264-692b.cloud.databricks.com/explore/data/industrial_trusted_prd/gsb_cha_opc_aciaria/tb_aciaria_fp?o=2630899921881782)
# MAGIC > [industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_vd](https://dbc-de54e264-692b.cloud.databricks.com/explore/data/industrial_trusted_prd/gsb_cha_opc_aciaria/tb_aciaria_vd?o=2630899921881782)
# MAGIC > [industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_lc](https://dbc-de54e264-692b.cloud.databricks.com/explore/data/industrial_trusted_prd/gsb_cha_opc_aciaria/tb_aciaria_lc?o=2630899921881782)
# MAGIC  

# COMMAND ----------

# MAGIC %md
# MAGIC # Update libraries, modules and packages

# COMMAND ----------

!pip install skimpy
!pip install summarytools --upgrade

# COMMAND ----------

# # Reset kernel
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Imports

# COMMAND ----------

# Standard library imports
import sys
from typing import Optional

# Related third party imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, FloatType, StringType, TimestampType, DoubleType
from pyspark.sql.window import Window
import seaborn as sns
# import skimpy  # Light weight tool for creating summary statistics from dataframes
import summarytools  # DataFrame Summary Toolst

# COMMAND ----------

# MAGIC %md
# MAGIC # Versions

# COMMAND ----------

print(f'Python: {sys.version}')
# Print Spark version and cluster details
cluster_details = f"""
Spark Version: {spark.version}
Cluster Name: {spark.conf.get("spark.databricks.clusterUsageTags.clusterName")}
Runtime: {spark.conf.get("spark.databricks.clusterUsageTags.effectiveSparkVersion")}
Driver: {spark.conf.get("spark.databricks.clusterUsageTags.driverNodeType")}
"""
print(f"{cluster_details}")
# spark.conf.getAll

print(
    f"{'Package':<20} {'Version':<20}\n"
    f"{'matplotlib':<20} {matplotlib.__version__}\n"
    f"{'numpy':<20} {np.__version__}\n"
    f"{'pandas':<20} {pd.__version__}\n"
    f"{'pyspark':<20} {pyspark.__version__}\n"
    f"{'seaborn':<20} {sns.__version__}\n"
    # f"{'skimpy':<20} {skimpy.__version__}\n"
    f"{'summarytools':<20} {summarytools.__version__}\n"
)

# COMMAND ----------

!pip freeze

# COMMAND ----------

# MAGIC %md
# MAGIC # Constant

# COMMAND ----------

# MIN_HEAT = 114654  # 2022-12-31
# MIN_HEAT = 114655  # 2023-01-01
# MIN_HEAT = 119773  # 2023-12-24
MIN_HEAT = 119774  # 2024-01-02
# MIN_HEAT = 125732  # 2024-12-17
# MIN_HEAT = 125733  # 2025-01-04

# Percentiles to be used in pandas DataFrame describe method
PERCENTILES = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]

DISPLAY_DF_SUMMARY_GERAL = True
# DISPLAY_DF_SUMMARY = True
DISPLAY_SKIMPY_DF_SUMMARY = False

# COMMAND ----------

# MAGIC %md
# MAGIC # Functions

# COMMAND ----------


def format_time_column(
    df: pyspark.sql.DataFrame, column_name: str, result_column_name: Optional[str] = None
    ) -> pyspark.sql.DataFrame:
    """
    This function takes a dataframe and a column name, and transforms the time data
    in the given column into the HH:MM format.
    If the value is in HHMM format, it will insert ":" between hours and minutes.
    
    df (pyspark.sql.DataFrame): The input dataframe
    column_name (str): The name of the column containing the time data
    result_column_name (str): The name of the result column containing the time data
    return: A pyspark.sql.DataFrame with the transformed column
    """
    if result_column_name is None:
        result_column_name = column_name

    return df.withColumn(
        f"{result_column_name}",
        F.when(
            # Checks if the value is in HHMM format (4 digits)
            F.col(column_name).rlike(r"^\d{4}$"),
            F.concat(
                F.col(column_name).substr(1, 2),
                F.lit(":"),
                F.col(column_name).substr(3, 2)
            )
        ).otherwise(
            # If already in HH:MM format, leave it unchanged
            F.col(column_name)
        )
    )

def print_grouped_df_columns_names(
    df: pd.DataFrame | pyspark.sql.dataframe.DataFrame, n_first_chars_to_group: int = 2
) -> None:
    last_first_letters = ""
    columns = sorted(df.columns)
    for column in columns:
        if last_first_letters == column[0:n_first_chars_to_group]:
            print(column, end=", ")
        else:
            print("\n" + column, end=", ")
        last_first_letters = column[0:n_first_chars_to_group]
    print(f"\n{len(columns)} columns")


def inspect_df(
    df: pd.DataFrame, 
    id_col: str = 'corrida', 
    date_col: Optional[str] = 'datasaidafp',
    date_format: str = '%Y-%m-%d'
) -> None:
    """
    Performs sanity checks on the DataFrame (duplicates) and prints a summary 
    of the ID range and Date range.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.
        id_col (str): The name of the identifier column (e.g., run/corrida ID). 
                      Defaults to 'corrida'.
        date_col (Optional[str]): The name of the date column. If provided, 
                                  min/max dates will be printed. Defaults to 'datasaidafp'.
        date_format (str): The output format for dates. Defaults to '%Y-%m-%d'.

    Raises:
        ValueError: If the DataFrame contains duplicate column names (sanity check).
        KeyError: If id_col is not found in the DataFrame.
    """
    
    # --- 1. Sanity Checks ---
    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"🚨 CRITICAL: Duplicate column names detected: {dups}. Please deduplicate before proceeding.")

    # Warn about index duplicates
    if df.index.duplicated().any():
        print(f"\n⚠️  WARNING: Found {df.index.duplicated().sum()} duplicate index values.")

    if id_col not in df.columns:
        raise KeyError(f"Column '{id_col}' not found in DataFrame.")

    # --- 2. Data Preparation ---
    # Calculate ID stats
    min_id = df[id_col].min()
    max_id = df[id_col].max()
    
    # Calculate Date stats (handle both string and datetime objects)
    min_date_str = ""
    max_date_str = ""
    
    if date_col and date_col in df.columns:
        # We convert to datetime temporarily to ensure .min()/.max() works chronologically, 
        # not alphabetically (which happens with strings).
        # 'dayfirst=True' assumes DD/MM/YYYY format if parsing strings.
        temp_dates = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        
        if temp_dates.notna().any():
            min_date_str = f"({temp_dates.min().strftime(date_format)})"
            max_date_str = f"({temp_dates.max().strftime(date_format)})"
        else:
            min_date_str = "(Date parse error)"
            max_date_str = "(Date parse error)"

    # --- 3. Output ---
    print("-" * 60)
    print("DATASET INSPECTION SUMMARY")
    print("-" * 60)
    print("In the current dataset:")
    print(f"\t• Oldest {id_col}: {min_id} {min_date_str}")
    print(f"\t• Newest {id_col}: {max_id} {max_date_str}")
    print(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
    print("-" * 60)


def plot_time_histogram(
    df: pd.DataFrame,
    duration_col: str,
    grades: Optional[list[str]] = None,
    title: str = "Histograma tempo entre chegada e saída FP (min)",
    binwidth: int = 1,
    figsize: tuple[float, float] = (12, 6)
    ) -> None:
    plt.figure(figsize=figsize)
    if grades is None:
        sns.histplot(
            df[duration_col].dropna(), stat='count', binwidth=binwidth, alpha=0.7, kde=True, kde_kws={'cut': 0}
        )
        plt.title(f"{title} - Geral")
        plt.xlabel("Tempo (min)")
        plt.ylabel("Frequência")
        plt.show()
    else:
        for grade in grades:
            subset = df[df["qualidade"] == grade]
            sns.histplot(
                subset[duration_col].dropna(), stat='count', binwidth=binwidth, alpha=0.5, kde=True, kde_kws={'cut': 0}, label=grade
            )
        plt.title(f"{title} - por qualidade")
        plt.legend()
        plt.xlabel("Tempo (min)")
        plt.ylabel("Frequência")
        plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Read Tables

# COMMAND ----------

# MAGIC %md
# MAGIC ### tb_admaci_aci_retorno
# MAGIC Area served by the system: Steel mill of the Charqueadas plant. Data handled by the system: Charqueadas still mill production data
# MAGIC
# MAGIC Purpose: Obtain heats that have made two or more visits, to discard them

# COMMAND ----------

tb_admaci_aci_retorno = (
    spark.read.table(
        f"industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_aci_retorno"
    )
    .withColumn("corrida", F.col("corrida").cast("int"))
    .filter(F.col("corrida") >= MIN_HEAT)
)

returned_heats = list(
    tb_admaci_aci_retorno.toPandas().apply(pd.to_numeric, errors="ignore")["corrida"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### tb_admaci_aci_corrida

# COMMAND ----------

tb_admaci_aci_corrida = (
    spark.table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_aci_corrida")
    .withColumn("corrida", F.col("corrida").cast("int"))
    .withColumn("qualidade", F.col("qual_uni").cast("string"))
    .select(
        "corrida",
        "qualidade"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### tb_admaci_aci_corrida_vst

# COMMAND ----------

tb_admaci_aci_corrida_vst = (
    spark.read.table(
        f"industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_aci_corrida_vst"
    )
    .withColumn("corrida", F.col("corrida").cast("int"))
    .filter(F.col("corrida") >= MIN_HEAT)
)


tb_admaci_aci_corrida_vst = (
    tb_admaci_aci_corrida_vst.withColumn(
        "num_id", F.rank().over(Window.orderBy(F.desc("visit_no")))
    )
    .dropDuplicates(subset=["corrida"])
    # .filter(F.col("visit_no")==1)
)

print_grouped_df_columns_names(tb_admaci_aci_corrida_vst)

tb_admaci_aci_corrida_vst = (
    tb_admaci_aci_corrida_vst
    .select(
        "corrida",
        "visit_no",
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## FEA

# COMMAND ----------

# industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fea_corridas
# TIPOCARGA
# CARGA
# CESTOES
# PESOPOTEINI
# PESOPOTEFIM
# PESOPOTELIMPEZA
# RECEITA
# SECAGEMNAPROXIMA
# SITUACAOPANELA
# TEMPOESPVAZAR
# TEMPOFLOTACAO
# TEMPOMORTO
# ACOLIQUIDOSFUND
# industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fea_interrupcoes
# industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fea_par_codinterrupcao

tb_admaci_fea_corridas = (
    spark.table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fea_corridas")
    .withColumn("corrida", F.col("corrida").cast("int"))
    .filter(F.col("corrida") >= MIN_HEAT)
    .filter(~F.col("corrida").isin(returned_heats))
)

tb_fea = (
    tb_admaci_fea_corridas
    .join(tb_admaci_aci_corrida_vst, on="corrida", how="left",)
    .filter(F.col("visit_no")==1)
)

df_fea = tb_fea.toPandas().apply(
    lambda column: pd.to_numeric(column, errors='ignore')
    if not np.issubdtype(column.dtype, np.datetime64) else column
)
# display(df_fea)
print_grouped_df_columns_names(df_fea)
inspect_df(df_fea, id_col="corrida", date_col="data")

# COMMAND ----------

if DISPLAY_DF_SUMMARY_GERAL:
    if DISPLAY_SKIMPY_DF_SUMMARY:
        skimpy.skim(df_fea)
    display(
        summarytools.dfSummary(df_fea.sort_index(axis=1), is_collapsible=True)
    )
print(f"Memory usage: {df_fea.memory_usage(deep=True).sum()} bytes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## FP

# COMMAND ----------

tb_admaci_fp_corrida = (
    spark.table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fp_corrida")
    .withColumn(
        "data_hora_saidafp",
        F.to_timestamp(
            F.concat(
                F.date_format(F.col("datasaidafp"), "yyyy-MM-dd"), F.lit(" "), F.col("horasaidafp")
            ),
            "yyyy-MM-dd HH:mm"
        )
    )
    .withColumn(
        "data_hora_saidalc",
        F.to_timestamp(
            F.concat(
                F.date_format(F.col("datasaidalingotamento"), "yyyy-MM-dd"), F.lit(" "), F.col("horasaidalingotamento")
            ),
            "yyyy-MM-dd HH:mm"
        )
    )
    .withColumn(
        "data_hora_prevista_saidalc",
        F.to_timestamp(
            F.concat(
                F.date_format(F.col("dataprevistasaidalingotamento"), "yyyy-MM-dd"), F.lit(" "), F.col("horaprevistasaidalingotamento")
            ),
            "yyyy-MM-dd HH:mm"
        )
    )
    .withColumn("horavazamento_original", F.col("horavazamento"))
    .withColumn("horachegadafp_original", F.col("horachegadafp"))
    .withColumn("tttfp", F.col("tttfp").cast("int"))
    .withColumn("tttvd", F.col("tttvd").cast("int"))
    .withColumn("corrida", F.col("corrida").cast("int"))
    .filter(F.col("corrida") >= MIN_HEAT)
    .filter(~F.col("corrida").isin(returned_heats))
    .filter(F.col("adicionaltttfp") == 0)
    .filter(F.col("adicionaltttvd") == 0)
)


# industrial_iatemperaturascha_refined_prd = (
#     spark.table("industrial_iatemperaturascha_refined_prd.iatemperaturascha")
# )


tb_admaci_fp_corrida = format_time_column(
    df=tb_admaci_fp_corrida, column_name="horachegadafp", result_column_name="horachegadafp"
)
tb_admaci_fp_corrida = (
    tb_admaci_fp_corrida
    .withColumn(
        "data_hora_chegadafp",
        F.to_timestamp(
            F.concat(
                F.date_format(F.col("datachegadafp"), "yyyy-MM-dd"), F.lit(" "), F.col("horachegadafp")
            ),
            "yyyy-MM-dd HH:mm"
        )
    )
)

tb_admaci_fp_corrida = format_time_column(
    df=tb_admaci_fp_corrida, column_name="horavazamento", result_column_name="horavazamento"
)

# Unir data e hora, calcular tempo entre chegada e saída em minutos
tb_admaci_fp_corrida = (
    tb_admaci_fp_corrida
    # .withColumn(
    #     "chegada_fp", 
    #     F.expr("F.to_timestamp(DATACHEGADAFP || ' ' || HORACHEGADAFP, 'yyyy-MM-dd HH:mm:ss')")
    # )
    # .withColumn(
    #     "saida_fp", 
    #     F.expr("F.to_timestamp(DATASAIDAFP || ' ' || HORASAIDAFP, 'yyyy-MM-dd HH:mm:ss')")
    # )
    .withColumn(
        "tempo_fp_min", 
        (F.col("data_hora_saidafp").cast("long") - F.col("data_hora_chegadafp").cast("long")) / 60
        # (F.col("data_hora_saidafp") - F.col("data_hora_chegadafp"))
    )
)


tb_admaci_fp_corridavd = (
    spark.table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fp_corridavd")
    .withColumn("corrida", F.col("corrida").cast("int"))
    .select(
        "corrida",
        "tempovacuototal"
    )
)

# data_preparation.tb_operator_input"

tb_fp = (
    tb_admaci_fp_corrida
    .join(tb_admaci_aci_corrida, on="corrida", how="left",)
    .join(tb_admaci_fp_corridavd, on="corrida", how="left",)
    .join(tb_admaci_aci_corrida_vst, on="corrida", how="left",)
    .filter(F.col("visit_no")==1)
)


# Converter para pandas
df_fp = tb_fp.toPandas().apply(
    lambda column: pd.to_numeric(column, errors='ignore')
    if not np.issubdtype(column.dtype, np.datetime64) else column
)
print_grouped_df_columns_names(df_fp)
inspect_df(df_fp, id_col="corrida", date_col="DATASAIDAFP")

# COMMAND ----------

if DISPLAY_DF_SUMMARY_GERAL:
    if DISPLAY_SKIMPY_DF_SUMMARY:
        skimpy.skim(df_fp)
    display(
        summarytools.dfSummary(df_fp.sort_index(axis=1), is_collapsible=True)
    )
print(f"Memory usage: {df_fp.memory_usage(deep=True).sum()} bytes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## VD

# COMMAND ----------

tb_admaci_fp_corridavd = (
    spark.table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fp_corridavd")
    .withColumn("corrida", F.col("corrida").cast("int"))
    .filter(F.col("corrida") >= MIN_HEAT)
    .filter(~F.col("corrida").isin(returned_heats))
)

tb_vd = (
    tb_admaci_fp_corridavd
    .join(tb_admaci_aci_corrida_vst, on="corrida", how="left",)
    .filter(F.col("visit_no")==1)
)

df_vd = tb_vd.toPandas().apply(
    lambda column: pd.to_numeric(column, errors='ignore')
    if not np.issubdtype(column.dtype, np.datetime64) else column
)

print_grouped_df_columns_names(df_vd)
inspect_df(df_vd, id_col="corrida")

# COMMAND ----------

if DISPLAY_DF_SUMMARY_GERAL:
    if DISPLAY_SKIMPY_DF_SUMMARY:
        skimpy.skim(df_vd)
    display(
        summarytools.dfSummary(df_vd.sort_index(axis=1), is_collapsible=True)
    )
print(f"Memory usage: {df_vd.memory_usage(deep=True).sum()} bytes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## LC

# COMMAND ----------

tb_admaci_lc_corridalingotamento = (
    spark.table("industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_lc_corridalingotamento")
    .withColumn('iniciolingotamento', F.col('iniciolingotamento').cast(TimestampType()))
    .withColumn('finallingotamento', F.col('finallingotamento').cast(TimestampType()))

    .withColumn("corrida", F.col("corrida").cast("int"))
    .filter(F.col("corrida") >= MIN_HEAT)
    .filter(~F.col("corrida").isin(returned_heats))
)

tb_lc = (
    tb_admaci_lc_corridalingotamento
    .join(tb_admaci_aci_corrida_vst, on="corrida", how="left",)
    .join(
        (
            tb_admaci_fp_corrida
            .select(
                'corrida',
                'data_hora_prevista_saidalc',
                'data_hora_saidalc',
            )
        ),
        on="corrida",
        how="left",
    )
    .filter(F.col("visit_no")==1)
)

df_lc = tb_lc.toPandas().apply(
    lambda column: pd.to_numeric(column, errors='ignore')
    if not np.issubdtype(column.dtype, np.datetime64) else column
)
# iniciolingotamento
# finallingotamento

# datasaidalingotamento
# horasaidalingotamento

# dataprevistasaidalingotamento
# horaprevistasaidalingotamento

print_grouped_df_columns_names(df_lc)
inspect_df(df_lc, id_col="corrida")

# COMMAND ----------

if DISPLAY_DF_SUMMARY_GERAL:
    if DISPLAY_SKIMPY_DF_SUMMARY:
        skimpy.skim(df_lc)
    display(
        summarytools.dfSummary(df_lc.sort_index(axis=1), is_collapsible=True)
    )
print(f"Memory usage: {df_lc.memory_usage(deep=True).sum()} bytes")

# COMMAND ----------

# MAGIC %md
# MAGIC # Start

# COMMAND ----------

# MAGIC %md
# MAGIC ## FP

# COMMAND ----------

# MAGIC %md
# MAGIC ### tempo_fp_min

# COMMAND ----------

df_fp["tempo_fp_min"] = (
    df_fp["data_hora_saidafp"] - df_fp["data_hora_chegadafp"]
).dt.total_seconds() / 60
df_fp["tempo_fp_min"].value_counts(dropna=False).head()

# COMMAND ----------

plot_time_histogram(
    df_fp,
    title = "Histograma tempo entre saída e chagada FP (min)",
    duration_col="tempo_fp_min",
    binwidth=5,
    figsize=(12, 3)
)
plot_time_histogram(
    df_fp[(df_fp["tempo_fp_min"] < 140) & (df_fp["tempo_fp_min"] > 10)],
    title = "Histograma tempo entre saída e chagada FP (min)",
    duration_col="tempo_fp_min"
)
df_fp[["corrida", "tempo_fp_min"]].describe(percentiles=PERCENTILES).T

# COMMAND ----------

print(df_fp[["qualidade"]].value_counts().head(15))

# COMMAND ----------

# Visão por qualidade
grades = [
    "1515F",
    "4120HC",
    "5115HB",
    "51B20HZF",
    "15V38HD",
    "10B38HA",
    "1141HE",
    "50095A",
]
plot_time_histogram(
    df_fp[(df_fp["tempo_fp_min"] < 140) & (df_fp["tempo_fp_min"] > 10)],
    title = "Histograma tempo entre saída e chagada FP (min)",
    duration_col="tempo_fp_min",
    grades=grades
)
for grade in grades:
    subset = df_fp[df_fp["qualidade"] == grade]
    plot_time_histogram(
        subset,
        title = "Histograma tempo entre saída e chagada FP (min)",
        duration_col="tempo_fp_min",
        grades=grades
    )
    subset[["corrida", "tempo_fp_min"]].describe(percentiles=PERCENTILES).T
# # Visão por tempo de vácuo (se houver)
# if "TEMPOVACUOREAL" in df_fp.columns:
#     for vacuo in sorted(df_fp["TEMPOVACUOREAL"].dropna().unique()):
#         subset = df_fp[df_fp["TEMPOVACUOREAL"] == vacuo]
#         plt.figure(figsize=(8,5))
#         plt.hist(subset["tempo_fp_min"].dropna(), bins=20, alpha=0.7)
#         plt.title(f"Histograma tempo FP - Tempo de vácuo: {vacuo}")
#         plt.xlabel("Tempo (min)")
#         plt.ylabel("Frequência")
#         plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### tttfp

# COMMAND ----------

plot_time_histogram(df_fp, title = "Histograma TTTFP (min)", duration_col="tttfp", binwidth=5, figsize=(12, 3))
plot_time_histogram(
    df_fp[(df_fp["tttfp"] < 140) & (df_fp["tttfp"] > 10)],
    title = "Histograma TTTFP (min)",
    duration_col="tttfp"
)
df_fp[["corrida", "tttfp"]].describe(percentiles=PERCENTILES).T

# COMMAND ----------

# MAGIC %md
# MAGIC ## FP - LC

# COMMAND ----------

df_fp_lc = df_fp.merge(df_lc, on="corrida", how="left")

# COMMAND ----------

# MAGIC %md
# MAGIC ### tempo_fp_lc_min

# COMMAND ----------

# Quero compararar o tttvd com iniciolingotamento  - data_hora_saidafp
df_fp_lc["tempo_fp_lc_min"] = (
    pd.to_datetime(df_fp_lc["iniciolingotamento"]) - pd.to_datetime(df_fp_lc["data_hora_saidafp"])
).dt.total_seconds() / 60
df_fp_lc["tempo_fp_lc_min"].value_counts(dropna=False).head()


# COMMAND ----------

plot_time_histogram(
    df_fp_lc,
    title="Histograma tempo entre saída do FP e Início LC (min)",
    duration_col="tempo_fp_lc_min",
    binwidth=5,
    figsize=(12, 3)
)
plot_time_histogram(
    df_fp_lc[(df_fp_lc["tempo_fp_lc_min"] < 140) & (df_fp_lc["tempo_fp_lc_min"] > 10)],
    title="Histograma tempo entre saída do FP e Início LC (min)",
    duration_col="tempo_fp_lc_min"
)
df_fp_lc[["corrida", "tempo_fp_lc_min"]].describe(percentiles=PERCENTILES).T

# COMMAND ----------

# MAGIC %md
# MAGIC ### tttvd

# COMMAND ----------

plot_time_histogram(
    df_fp_lc,
    title = "Histograma TTTVD (min)",
    duration_col="tttvd",
    binwidth=5,
    figsize=(12, 3)
)
plot_time_histogram(
    df_fp_lc[(df_fp_lc["tttvd"] < 140) & (df_fp_lc["tttvd"] > 10)],
    title = "Histograma TTTVD (min)",
    duration_col="tttvd"
)
df_fp_lc[["corrida", "tttvd"]].describe(percentiles=PERCENTILES).T

# COMMAND ----------

df_fp_lc["tttvd - tempovacuototal"] = df_fp_lc["tttvd"] - df_fp_lc["tempovacuototal"]
plot_time_histogram(
    df_fp_lc,
    title = "Histograma TTTVD (min)",
    duration_col="tttvd - tempovacuototal",
    binwidth=5,
    figsize=(12, 3)
)
plot_time_histogram(
    df_fp_lc[(df_fp_lc["tttvd - tempovacuototal"] < 140) & (df_fp_lc["tttvd - tempovacuototal"] > 10)],
    title = "Histograma TTTVD - tempovacuototal (min)",
    duration_col="tttvd - tempovacuototal"
)
df_fp_lc[["corrida", "tttvd - tempovacuototal"]].describe(percentiles=PERCENTILES).T

# COMMAND ----------

# MAGIC %md
# MAGIC ## LC

# COMMAND ----------

df_lc["tempo_lc_min"] = (
    df_lc["finallingotamento"] - df_lc["iniciolingotamento"]
).dt.total_seconds() / 60
df_lc["tempo_lc_min"].value_counts(dropna=False).head()

# COMMAND ----------

plot_time_histogram(
    df_lc,
    title="Histograma tempo entre Final e Início LC (min)",
    duration_col="tempo_lc_min",
    binwidth=5,
    figsize=(12, 3)
)
plot_time_histogram(
    df_lc[(df_lc["tempo_lc_min"] < 140) & (df_lc["tempo_lc_min"] > 10)],
    title="Histograma tempo entre Final e Início LC (min)",
    duration_col="tempo_lc_min"
)
df_lc[["corrida", "tempo_lc_min"]].describe(percentiles=PERCENTILES).T

# COMMAND ----------

df_lc["delta_previsao_hora_saida_lc"] = (
    df_lc["data_hora_prevista_saidalc"] - df_lc["data_hora_saidalc"]
).dt.total_seconds() / 60
df_lc["delta_previsao_hora_saida_lc"].value_counts(dropna=False).head()

# COMMAND ----------

plot_time_histogram(
    df_lc[(df_lc["delta_previsao_hora_saida_lc"] < 1000) & (df_lc["delta_previsao_hora_saida_lc"] > -1000)],
    title="Histograma tempo entre Saída Prevista e Saída do LC (min)",
    duration_col="delta_previsao_hora_saida_lc",
    binwidth=5,
    figsize=(12, 3)
)
plot_time_histogram(
    df_lc[(df_lc["delta_previsao_hora_saida_lc"] < 140) & (df_lc["delta_previsao_hora_saida_lc"] > -140)],
    title="Histograma tempo entre Saída Prevista e Saída do LC (min)",
    duration_col="delta_previsao_hora_saida_lc"
)
df_lc[["corrida", "delta_previsao_hora_saida_lc"]].describe(percentiles=PERCENTILES).T