# Databricks notebook source
# MAGIC %md
# MAGIC # Análise de Lingotamento Contínuo (LC)
# MAGIC ## Comparação entre Tempos do Operador vs Detecção Automática por Sensores PIMS
# MAGIC
# MAGIC **Objetivo:** Detectar automaticamente eventos de início e fim de corridas de LC usando dados de peso dos braços (PESOBRA1/PESOBRA2) e comparar com os registros manuais do operador.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📦 1. Imports e Configuração Inicial

# COMMAND ----------

# ============================================================
# CÉLULA 1 — IMPORTS
# ============================================================

import pandas as pd
import numpy as np

from typing import List, Dict, Any, Optional

import pyspark
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import matplotlib.pyplot as plt

# COMMAND ----------

# DBTITLE 1,Funções Auxiliares
# ============================================================
# CÉLULA 2 — FUNÇÕES AUXILIARES (PIMS preprocessing + helpers)
# ============================================================

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

def fill_nulls_with_last_valid(
    df: pyspark.sql.DataFrame, timestamp_column: str, columns: List[str]
) -> pyspark.sql.DataFrame:
    """
    Forward-fill (PIMS): se valor veio null, significa "igual ao último".
    """
    window_spec = Window.orderBy(timestamp_column).rowsBetween(Window.unboundedPreceding, 0)
    select_exprs = []

    target_cols_set = set(columns)
    for col_name in df.columns:
        if col_name in target_cols_set:
            select_exprs.append(
                F.last(F.col(col_name), ignorenulls=True).over(window_spec).alias(col_name)
            )
        else:
            select_exprs.append(F.col(col_name))

    return df.select(*select_exprs)


def fill_first_row_with_first_valid(
    df: pyspark.sql.DataFrame, columns: List[str]
) -> pyspark.sql.DataFrame:
    """
    Backfill apenas na primeira linha: garante que a série comece com valor válido.
    """
    select_exprs = []
    target_cols_set = set(columns)

    w_first = Window.orderBy(F.monotonically_increasing_id())

    for col_name in df.columns:
        if col_name in target_cols_set:
            first_valid_row = df.filter(F.col(col_name).isNotNull()).select(col_name).first()
            first_valid_value = first_valid_row[0] if first_valid_row else None

            select_exprs.append(
                F.when(
                    F.row_number().over(w_first) == 1,
                    F.lit(first_valid_value)
                ).otherwise(F.col(col_name)).alias(col_name)
            )
        else:
            select_exprs.append(F.col(col_name))

    return df.select(*select_exprs)


def repair_data_glitches(
    df: pyspark.sql.DataFrame,
    column: str,
    active_threshold: float = 1.0,
    lookahead_steps: int = 2,
    timestamp_col: str = "timestamp"
) -> pyspark.sql.DataFrame:
    """
    Repara glitches: queda rápida pra ~0 e volta rápido => considera ruído e faz forward-fill.
    """
    w_lag = Window.orderBy(timestamp_col)
    w_lookahead = Window.orderBy(timestamp_col).rowsBetween(1, lookahead_steps)
    w_ffill = Window.orderBy(timestamp_col).rowsBetween(Window.unboundedPreceding, Window.currentRow)

    is_glitch_cond = (
        (F.col(column) < 1) &
        (F.lag(column, 1, 0).over(w_lag) > active_threshold) &
        (F.max(column).over(w_lookahead) > active_threshold)
    )

    col_temp = f"{column}_temp_clean"
    df_masked = df.select(
        "*",
        F.when(is_glitch_cond, F.lit(None)).otherwise(F.col(column)).alias(col_temp)
    )

    df_fixed = df_masked.withColumn(
        column,
        F.last(col_temp, ignorenulls=True).over(w_ffill)
    ).drop(col_temp)

    return df_fixed


def preprocess_pims_data(
    spark_df: pyspark.sql.DataFrame,
    first_timestamp: str,
    last_timestamp: str,
    timestamp_column: str,
    columns_to_fill: List[str],
) -> pyspark.sql.DataFrame:
    """
    1) filtra janela
    2) forward-fill nulls (PIMS)
    3) backfill primeira linha
    4) repair glitches
    """
    filtered_tb = spark_df.filter(
        (F.col(timestamp_column) >= F.lit(first_timestamp)) &
        (F.col(timestamp_column) <= F.lit(last_timestamp))
    )

    filled = fill_nulls_with_last_valid(
        filtered_tb.orderBy(timestamp_column, ascending=True),
        timestamp_column=timestamp_column,
        columns=columns_to_fill,
    )

    filled = fill_first_row_with_first_valid(filled, columns=columns_to_fill)

    for c in columns_to_fill:
        filled = repair_data_glitches(
            filled,
            column=c,
            active_threshold=1,
            lookahead_steps=2,
            timestamp_col=timestamp_column
        )

    return filled


def to_pandas_ts(df_pd: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Converte colunas datetime (string ou timestamp) para pandas datetime.
    """
    out = df_pd.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    return out

# COMMAND ----------

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






# COMMAND ----------

# DBTITLE 1,START/END
START_TS = "2025-11-20 10:00:00"
END_TS   = "2025-11-21 15:59:59"

# COMMAND ----------

df_final = df_final.filter(
    F.col("inicio_lc").between(START_TS, END_TS) 
)

display(df_final.orderBy(F.desc("corrida")))

# COMMAND ----------

df_fea_pims_raw = spark.table("industrial_iatemperaturascha_refined_dev.computed_features.hist_tags_novembro_2025")

df_fea_pims_raw.display()

# COMMAND ----------

# DBTITLE 1,Ler, tratar e juntar tb_aciaria_lc (OPC) com PIMS
# ============================================================
# LEITURA E TRATAMENTO DA tb_aciaria_lc (OPC - segundo a segundo)
# ============================================================

# 1. Ler tabela OPC
df_lc_opc = spark.read.table(
    "industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_lc"
)
print(f"tb_aciaria_lc: {df_lc_opc.count():,} linhas")
print(f"Colunas: {df_lc_opc.columns}")

# 2. Converter UTC -> UTC-3
df_lc_opc = df_lc_opc.withColumn(
    "timestamp", F.col("timestamp") - F.expr("INTERVAL 3 HOURS")
)

# 3. Identificar colunas de valores e converter string -> double
opc_value_cols = [c for c in df_lc_opc.columns if c.startswith("ACI@")]
print(f"\nColunas OPC: {opc_value_cols}")

for c in opc_value_cols:
    df_lc_opc = df_lc_opc.withColumn(c, F.col(c).cast("double"))

# 4. Truncar para minuto e agregar com média
df_lc_opc = df_lc_opc.withColumn(
    "minute_ts", F.date_trunc("minute", "timestamp")
)

agg_exprs = [F.avg(F.col(c)).alias(c) for c in opc_value_cols]
df_lc_opc_min = df_lc_opc.groupBy("minute_ts").agg(*agg_exprs).orderBy("minute_ts")

print(f"\nApós resample para minuto: {df_lc_opc_min.count():,} linhas")
print("\nAmostra (OPC - minuto, UTC-3):")
df_lc_opc_min.show(5, truncate=False)

# 5. Preparar df_fea_pims_raw: criar coluna minute_ts a partir de Tempo
df_pims_with_min = df_fea_pims_raw.withColumn(
    "minute_ts",
    F.date_trunc(
        "minute",
        F.to_timestamp(F.col("Tempo"), "dd-MMM-yy HH:mm:ss.S")
    )
)

print("\nAmostra PIMS (minute_ts):")
df_pims_with_min.select("Tempo", "minute_ts").show(5, truncate=False)

# 6. Renomear colunas OPC com sufixo _opc para evitar conflito
opc_renamed = df_lc_opc_min
for c in opc_value_cols:
    opc_renamed = opc_renamed.withColumnRenamed(c, f"{c}_opc")

# 7. JOIN: PIMS (minuto) LEFT JOIN OPC (minuto)
df_fea_pims_raw = df_pims_with_min.join(
    opc_renamed,
    on="minute_ts",
    how="left"
).drop("minute_ts")

print(f"\n\u2705 Join conclu\u00eddo! df_fea_pims_raw agora tem {len(df_fea_pims_raw.columns)} colunas")
print(f"Novas colunas OPC adicionadas:")
for c in opc_value_cols:
    print(f"  - {c}_opc")

df_fea_pims_raw.select(
    "Tempo",
    *[f"`{c}_opc`" for c in opc_value_cols]
).show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📊 2. Carregamento e Preparação dos Dados PIMS LC

# COMMAND ----------

# df_fea_pims_raw = spark.table("industrial_iatemperaturascha_refined_dev.computed_features.hist_tags_complete")

# df_fea_pims_raw.display()

# COMMAND ----------

# DBTITLE 1,Ler tabela PIMS LC
# ============================================================
# LEITURA DA TABELA PIMS LC (UTC → UTC-3)
# ============================================================

# Converter coluna Tempo (string "01-DEC-25 00:00:00.0") para timestamp
# Formato: DD-MON-YY HH:mm:ss.S
df_lc_pims = df_fea_pims_raw.withColumn(
    "timestamp",
    F.to_timestamp(F.col("Tempo"), "dd-MMM-yy HH:mm:ss.S")
)

# # Converter de UTC para UTC-3 (subtrair 3 horas)
# df_lc_pims = df_lc_pims.withColumn(
#     "timestamp",
#     F.col("timestamp") - F.expr("INTERVAL 3 HOURS")
# )

# Remover a coluna Tempo original
df_lc_pims = df_lc_pims.drop("Tempo")

print(f"Total de colunas: {len(df_lc_pims.columns)}")
print(f"Total de linhas: {df_lc_pims.count()}")

# Verificar conversão
print("\nPrimeiras linhas com timestamp convertido (UTC-3):")
display(df_lc_pims.select("timestamp").limit(10))

# Identificar colunas que começam com ACI@LC
lc_columns_all = [col for col in df_lc_pims.columns if col.startswith("ACI@LC")]
print(f"\nColunas ACI@LC encontradas (total): {len(lc_columns_all)}")

# Filtrar colunas que têm pelo menos algum valor não-nulo
lc_columns = []
for col in lc_columns_all:
    non_null_count = df_lc_pims.filter(F.col(col).isNotNull()).count()
    if non_null_count > 0:
        lc_columns.append(col)
        print(f"  ✓ {col} - {non_null_count} valores não-nulos")
    else:
        print(f"  ✗ {col} - TODOS NULOS (removida)")

print(f"\nColunas ACI@LC válidas (com dados): {len(lc_columns)}")

# COMMAND ----------

# DBTITLE 1,Análise Estatística LC
# ============================================================
# ANÁLISE ESTATÍSTICA DAS COLUNAS LC
# ============================================================

import pandas as pd
from pyspark.sql import functions as F

print("="*80)
print("ANÁLISE ESTATÍSTICA DAS VARIÁVEIS LC")
print("="*80)

# Lista para armazenar estatísticas de cada coluna
stats_list = []

for col in lc_columns:
    print(f"\nProcessando: {col}...")
    
    # Converter para double se necessário (algumas colunas podem ser string)
    df_col = df_lc_pims.withColumn(f"{col}_numeric", F.col(col).cast("double"))
    
    # Calcular estatísticas usando approxQuantile para percentis
    stats = df_col.select(
        F.count(f"{col}_numeric").alias("count"),
        F.sum(F.when(F.col(f"{col}_numeric").isNull(), 1).otherwise(0)).alias("null_count"),
        F.min(f"{col}_numeric").alias("min"),
        F.max(f"{col}_numeric").alias("max"),
        F.mean(f"{col}_numeric").alias("mean"),
        F.stddev(f"{col}_numeric").alias("std"),
    ).collect()[0]
    
    # Calcular percentis (5%, 25%, 50%, 75%, 95%)
    percentiles = df_col.stat.approxQuantile(
        f"{col}_numeric", 
        [0.05, 0.25, 0.50, 0.75, 0.95], 
        0.01  # precisão
    )
    
    # Montar dicionário com todas as estatísticas
    col_stats = {
        "Coluna": col,
        "Count": stats["count"],
        "Nulls": stats["null_count"],
        "Min": stats["min"],
        "P5%": percentiles[0] if len(percentiles) > 0 else None,
        "Q1 (25%)": percentiles[1] if len(percentiles) > 1 else None,
        "Mediana (50%)": percentiles[2] if len(percentiles) > 2 else None,
        "Q3 (75%)": percentiles[3] if len(percentiles) > 3 else None,
        "P95%": percentiles[4] if len(percentiles) > 4 else None,
        "Max": stats["max"],
        "Média": stats["mean"],
        "Desvio Padrão": stats["std"],
    }
    
    stats_list.append(col_stats)

# Criar DataFrame Pandas com todas as estatísticas
df_stats = pd.DataFrame(stats_list)

# Formatar para melhor visualização
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.float_format', lambda x: f'{x:.2f}' if pd.notna(x) else 'NaN')

print("\n" + "="*80)
print("RESUMO ESTATÍSTICO COMPLETO")
print("="*80)
display(df_stats)

# Salvar em variável para uso posterior
print(f"\n✓ Estatísticas calculadas para {len(lc_columns)} colunas LC")
print(f"✓ DataFrame salvo em: df_stats")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📊 3. Análise Estatística das Variáveis LC

# COMMAND ----------

# DBTITLE 1,Estatísticas por Tipo
# ============================================================
# ANÁLISE ESTATÍSTICA AGRUPADA POR TIPO DE VARIÁVEL
# ============================================================

import re

# Extrair tipo de variável do nome da coluna
def extract_variable_type(col_name):
    """
    Extrai o tipo de variável do nome da coluna.
    Ex: ACI@LC_TORRE_PESOREAL -> TORRE
        ACI@LC_GERAL_NUMCORR -> GERAL
    """
    match = re.search(r'ACI@LC_([A-Z]+)_', col_name)
    if match:
        return match.group(1)
    return "OUTROS"

# Adicionar coluna de tipo
df_stats['Tipo'] = df_stats['Coluna'].apply(extract_variable_type)

# Agrupar por tipo e mostrar resumo
print("\n" + "="*80)
print("RESUMO POR TIPO DE VARIÁVEL")
print("="*80)

for tipo in df_stats['Tipo'].unique():
    df_tipo = df_stats[df_stats['Tipo'] == tipo]
    print(f"\n{'='*80}")
    print(f"TIPO: {tipo} ({len(df_tipo)} variáveis)")
    print(f"{'='*80}")
    display(df_tipo.drop(columns=['Tipo']))

# Estatísticas gerais
print("\n" + "="*80)
print("ESTATÍSTICAS GERAIS")
print("="*80)
print(f"Total de variáveis LC: {len(lc_columns)}")
print(f"\nVariáveis por tipo:")
for tipo, count in df_stats['Tipo'].value_counts().items():
    print(f"  - {tipo}: {count} variáveis")

print(f"\nVariáveis com dados válidos (sem nulls): {len(df_stats[df_stats['Nulls'] == 0])}")
print(f"Variáveis com alguns nulls: {len(df_stats[df_stats['Nulls'] > 0])}")

# Identificar variáveis constantes (std = 0 ou muito baixo)
df_constant = df_stats[df_stats['Desvio Padrão'] < 0.01]
if len(df_constant) > 0:
    print(f"\n⚠️ Variáveis praticamente constantes (std < 0.01): {len(df_constant)}")
    for col in df_constant['Coluna'].values:
        print(f"  - {col}")

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Preprocessar dados PIMS LC
# ============================================================
# PRÉ-PROCESSAMENTO PIMS LC
# ============================================================

# Aplicar preprocess_pims_data nas colunas LC
df_lc_processed = preprocess_pims_data(
    spark_df=df_lc_pims,
    first_timestamp=START_TS,
    last_timestamp=END_TS,
    timestamp_column="timestamp",
    columns_to_fill=lc_columns,
)

print(f"Dados processados: {df_lc_processed.count()} linhas")
print(f"Período: {START_TS} até {END_TS}")

# Converter para Pandas para plotagem
df_lc_plot = df_lc_processed.select(["timestamp"] + lc_columns).toPandas()
df_lc_plot["timestamp"] = pd.to_datetime(df_lc_plot["timestamp"])

print(f"\nDataFrame Pandas: {len(df_lc_plot)} linhas, {len(df_lc_plot.columns)} colunas")

# COMMAND ----------

# DBTITLE 1,Preparar dados do operador (LC)
# ============================================================
# PREPARAR DADOS DO OPERADOR (INICIO/FIM LC)
# ============================================================

# Filtrar corridas do período e coletar dados de inicio/fim LC
df_operator_lc = (
    df_final
    .filter(
        (F.col("inicio_lc").isNotNull()) &
        (F.col("final_lc").isNotNull()) &
        (F.col("inicio_lc").between(START_TS, END_TS))
    )
    .select("corrida", "inicio_lc", "final_lc")
    .orderBy("corrida")
    .toPandas()
)

df_operator_lc["inicio_lc"] = pd.to_datetime(df_operator_lc["inicio_lc"])
df_operator_lc["final_lc"] = pd.to_datetime(df_operator_lc["final_lc"])

print(f"Corridas LC no período: {len(df_operator_lc)}")
display(df_operator_lc)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📈 4. Visualização dos Dados LC com Marcadores de Corrida

# COMMAND ----------

# DBTITLE 1,Plotar variáveis FEA com marcadores de corrida
# ============================================================
# PLOTAR CADA VARIÁVEL LC COM MARCADORES DE CORRIDA
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Criar um plot para cada coluna LC
for col in lc_columns:
    plt.figure(figsize=(16, 6))
    
    # Plot da série temporal
    plt.plot(df_lc_plot["timestamp"], df_lc_plot[col], label=col, linewidth=1.5, alpha=0.8)
    
    # Adicionar linhas verticais para inicio e fim de cada corrida
    if not df_operator_lc.empty:
        y_max = df_lc_plot[col].max()
        y_min = df_lc_plot[col].min()
        
        for idx, row in df_operator_lc.iterrows():
            if pd.isna(row["inicio_lc"]) or pd.isna(row["final_lc"]):
                continue
            if not (np.isfinite(y_max) and np.isfinite(y_min)):
                continue
            corrida = int(row["corrida"])
            
            # Linha verde para início
            plt.axvline(row["inicio_lc"], color="green", linestyle="--", linewidth=2, alpha=0.7)
            plt.text(
                row["inicio_lc"],
                y_max * 0.95,
                f"Início {corrida}",
                rotation=90,
                color="green",
                ha="left",
                fontsize=9
            )
            
            # Linha vermelha para fim
            plt.axvline(row["final_lc"], color="red", linestyle="--", linewidth=2, alpha=0.7)
            plt.text(
                row["final_lc"],
                y_max * 0.90,
                f"Fim {corrida}",
                rotation=90,
                color="red",
                ha="right",
                fontsize=9
            )
    
    # Formatação do gráfico
    plt.title(f"LC - {col}\n(Período: {START_TS} até {END_TS})", fontsize=14, fontweight="bold")
    plt.xlabel("Tempo", fontsize=12)
    plt.ylabel("Valor", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    
    # Formatar eixo X com locator automático
    ax = plt.gca()
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{col} - Estatísticas:")
    print(f"  Min: {df_lc_plot[col].min():.2f}")
    print(f"  Max: {df_lc_plot[col].max():.2f}")
    print(f"  Média: {df_lc_plot[col].mean():.2f}")
    print(f"  Desvio padrão: {df_lc_plot[col].std():.2f}")
    print("=" * 80)

# COMMAND ----------

# ============================================================
# PLOTAR ACI@LC_GERAL_NUMCORR AO LONGO DO TEMPO
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

col = "ACI@LC_GERAL_NUMCORR"

# Converter coluna para numérico se necessário
df_lc_plot[col] = pd.to_numeric(df_lc_plot[col], errors='coerce')

plt.figure(figsize=(18, 7))

# Plot da série temporal com step (melhor para variáveis discretas como número de corrida)
plt.plot(df_lc_plot["timestamp"], df_lc_plot[col], label=col, linewidth=2, alpha=0.9, drawstyle='steps-post')

# Adicionar linhas verticais para inicio e fim de cada corrida
if not df_operator_lc.empty:
    y_max = df_lc_plot[col].max()
    y_min = df_lc_plot[col].min()
    y_range = y_max - y_min if y_max != y_min else 1
    
    for idx, row in df_operator_lc.iterrows():
        corrida = int(row["corrida"])
        
        # Verificar se os timestamps são finitos
        if pd.isna(row["inicio_lc"]) or pd.isna(row["final_lc"]):
            continue
        if not (np.isfinite(y_max) and np.isfinite(y_min)):
            continue
        
        # Linha verde para início
        plt.axvline(row["inicio_lc"], color="green", linestyle="--", linewidth=2, alpha=0.7)
        plt.text(
            row["inicio_lc"],
            y_max - (y_range * 0.05),
            f"Início\n{corrida}",
            rotation=0,
            color="green",
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="green", alpha=0.7)
        )
        
        # Linha vermelha para fim
        plt.axvline(row["final_lc"], color="red", linestyle="--", linewidth=2, alpha=0.7)
        plt.text(
            row["final_lc"],
            y_max - (y_range * 0.15),
            f"Fim\n{corrida}",
            rotation=0,
            color="red",
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.7)
        )

# Formatação do gráfico
plt.title(f"LC - Número da Corrida (PIMS)\n(Período: {START_TS} até {END_TS})", fontsize=14, fontweight="bold")
plt.xlabel("Tempo", fontsize=12)
plt.ylabel("Número da Corrida", fontsize=12)
plt.grid(True, alpha=0.3, linestyle=':')
plt.legend(loc="upper left", fontsize=10)

# Formatar eixo X com locator automático (evita gerar milhares de ticks)
ax = plt.gca()
locator = mdates.AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
plt.xticks(rotation=45, ha='right')

# Ajustar limites do eixo Y para melhor visualização
if not df_lc_plot[col].isna().all():
    y_min_val = df_lc_plot[col].min()
    y_max_val = df_lc_plot[col].max()
    y_padding = (y_max_val - y_min_val) * 0.1 if y_max_val != y_min_val else 1
    plt.ylim(y_min_val - y_padding, y_max_val + y_padding)

plt.tight_layout()
plt.show()

print(f"\n{col} - Estatísticas:")
print(f"  Min: {df_lc_plot[col].min():.0f}")
print(f"  Max: {df_lc_plot[col].max():.0f}")
print(f"  Média: {df_lc_plot[col].mean():.2f}")
print(f"  Desvio padrão: {df_lc_plot[col].std():.2f}")
print(f"  Valores únicos: {df_lc_plot[col].nunique()}")
print("=" * 80)

# COMMAND ----------

# ============================================================
# DETECÇÃO DE EVENTOS LC — PESOBRA1 / PESOBRA2 (VERSÃO ROBUSTA)
# ============================================================

import pandas as pd
import numpy as np


def detect_lc_events_pesobra(
    df: pd.DataFrame,
    initial_corrida: int = 130473,
    initial_braco: int = 2,  # corrida inicial começa em qual braço

    # -----------------------------
    # SUAVIZAÇÃO / JANELAS
    # -----------------------------
    SMOOTH_WINDOW_S: int = 15,
    SLOPE_WINDOW_S: int = 20,

    # -----------------------------
    # DETECÇÃO DE INÍCIO
    # -----------------------------
    STABLE_WINDOW: int = 30,
    STABLE_VAR: float = 2.0,

    MIN_NEG_SLOPE: float = -3.0,
    DROP_WINDOW: int = 60,
    MIN_REAL_DROP: float = 8.0,

    MAX_TIME_TO_CONFIRM_START_S: int = 4 * 60,

    # -----------------------------
    # DETECÇÃO DE FIM (BICO)
    # -----------------------------
    BOTTOM_WEIGHT: float = 15.0,
    POS_SLOPE_END: float = 2.0,

    STABLE_SECONDS_END: int = 30,

    # -----------------------------
    # SANIDADE
    # -----------------------------
    min_duration_seconds: int = 300,
):
    """
    Retorna DataFrame com:
    corrida | inicio_lc_sys | final_lc_sys | braco | duracao_min
    """

    df = df.copy().reset_index(drop=True)

    # ======================================================
    # 1. PESOS NUMÉRICOS
    # ======================================================
    df['peso_bra1'] = pd.to_numeric(df['ACI@LC_TORRE_PESOBRA1'], errors='coerce')
    df['peso_bra2'] = pd.to_numeric(df['ACI@LC_TORRE_PESOBRA2'], errors='coerce')

    # ======================================================
    # 2. SUAVIZAÇÃO (MEDIANA)
    # ======================================================
    w = int(SMOOTH_WINDOW_S)
    df['peso_bra1_smooth'] = df['peso_bra1'].rolling(w, min_periods=max(3, w // 3)).median()
    df['peso_bra2_smooth'] = df['peso_bra2'].rolling(w, min_periods=max(3, w // 3)).median()

    # ======================================================
    # 3. SLOPES
    # ======================================================
    k = int(SLOPE_WINDOW_S)
    df['slope_bra1'] = df['peso_bra1_smooth'] - df['peso_bra1_smooth'].shift(k)
    df['slope_bra2'] = df['peso_bra2_smooth'] - df['peso_bra2_smooth'].shift(k)

    # ======================================================
    # 4. ESTABILIDADE (ANTES DO INÍCIO)
    # ======================================================
    df['std_bra1'] = (
        df['peso_bra1_smooth']
        .rolling(STABLE_WINDOW, min_periods=STABLE_WINDOW)
        .std()
    )

    df['std_bra2'] = (
        df['peso_bra2_smooth']
        .rolling(STABLE_WINDOW, min_periods=STABLE_WINDOW)
        .std()
    )

    df['was_stable_bra1'] = df['std_bra1'] <= STABLE_VAR
    df['was_stable_bra2'] = df['std_bra2'] <= STABLE_VAR

    # ======================================================
    # 5. QUEDA REAL (ANTI FALSO INÍCIO)
    # ======================================================
    df['drop_bra1'] = df['peso_bra1_smooth'] - df['peso_bra1_smooth'].shift(DROP_WINDOW)
    df['drop_bra2'] = df['peso_bra2_smooth'] - df['peso_bra2_smooth'].shift(DROP_WINDOW)

    # ======================================================
    # 6. CANDIDATOS DE INÍCIO
    # ======================================================
    df['start_candidate_bra1'] = (
        df['was_stable_bra1'].shift(5).fillna(False)
        & (df['slope_bra1'] <= MIN_NEG_SLOPE)
        & (df['drop_bra1'] <= -MIN_REAL_DROP)
    )

    df['start_candidate_bra2'] = (
        df['was_stable_bra2'].shift(5).fillna(False)
        & (df['slope_bra2'] <= MIN_NEG_SLOPE)
        & (df['drop_bra2'] <= -MIN_REAL_DROP)
    )

    # ======================================================
    # 7. FUNDO E SAÍDA DO FUNDO (FIM REAL)
    # ======================================================
    df['in_bottom_bra1'] = df['peso_bra1_smooth'] <= BOTTOM_WEIGHT
    df['in_bottom_bra2'] = df['peso_bra2_smooth'] <= BOTTOM_WEIGHT

    df['exit_bottom_bra1'] = df['in_bottom_bra1'].shift(1) & (~df['in_bottom_bra1'])
    df['exit_bottom_bra2'] = df['in_bottom_bra2'].shift(1) & (~df['in_bottom_bra2'])

    df['end_candidate_bra1'] = (
        df['exit_bottom_bra1']
        | (df['slope_bra1'] >= POS_SLOPE_END)
    )

    df['end_candidate_bra2'] = (
        df['exit_bottom_bra2']
        | (df['slope_bra2'] >= POS_SLOPE_END)
    )

    # Sustentação curta para evitar ruído
    df['end_sustained_bra1'] = (
        df['end_candidate_bra1']
        .rolling(STABLE_SECONDS_END, min_periods=STABLE_SECONDS_END)
        .mean() == 1.0
    )

    df['end_sustained_bra2'] = (
        df['end_candidate_bra2']
        .rolling(STABLE_SECONDS_END, min_periods=STABLE_SECONDS_END)
        .mean() == 1.0
    )

    # ======================================================
    # 8. MÁQUINA DE ESTADOS
    # ======================================================
    events = []
    corrida = int(initial_corrida)
    braco_atual = int(initial_braco)

    state = "IDLE"
    start_time = None
    start_candidate_time = None

    for i in range(len(df)):
        row = df.iloc[i]
        ts = row['timestamp']

        if braco_atual == 1:
            start_candidate = row['start_candidate_bra1']
            end_sustained = row['end_sustained_bra1']
        else:
            start_candidate = row['start_candidate_bra2']
            end_sustained = row['end_sustained_bra2']

        # -------------------------
        if state == "IDLE":
            if bool(start_candidate):
                state = "POSSIBLE_START"
                start_candidate_time = ts
            continue

        # -------------------------
        if state == "POSSIBLE_START":
            elapsed = (ts - start_candidate_time).total_seconds()

            if bool(start_candidate):
                state = "RUNNING"
                start_time = start_candidate_time
                continue

            if elapsed > MAX_TIME_TO_CONFIRM_START_S:
                state = "IDLE"
                start_candidate_time = None
                continue

        # -------------------------
        if state == "RUNNING":
            if bool(end_sustained):
                end_time = df.iloc[i - STABLE_SECONDS_END + 1]['timestamp']
                dur = (end_time - start_time).total_seconds()

                if dur >= min_duration_seconds:
                    events.append({
                        'corrida': corrida,
                        'inicio_lc_sys': start_time,
                        'final_lc_sys': end_time,
                        'braco': braco_atual,
                        'duracao_min': dur / 60
                    })

                    corrida += 1
                    braco_atual = 2 if braco_atual == 1 else 1

                state = "IDLE"
                start_candidate_time = None
                start_time = None

    # ======================================================
    # 9. RESULTADO
    # ======================================================
    df_events = pd.DataFrame(events)

    if not df_events.empty:
        df_events['inicio_lc_sys'] = pd.to_datetime(df_events['inicio_lc_sys'])
        df_events['final_lc_sys'] = pd.to_datetime(df_events['final_lc_sys'])

    return df_events


# COMMAND ----------

# MAGIC %md
# MAGIC ### 🤖 5. Algoritmos de Detecção Automática de Eventos LC
# MAGIC
# MAGIC **Objetivo:** Detectar automaticamente início e fim de corridas usando dados de peso dos braços (PESOBRA1/PESOBRA2)
# MAGIC
# MAGIC **Abordagens implementadas:**
# MAGIC * **Versão 1:** Detecção completa com FSM (início + fim)
# MAGIC * **Versão 2:** Detecção apenas de início com confirmação
# MAGIC * **Versão 3:** Candidatos com scoring e clustering
# MAGIC * **Versão 4:** Detecção sequencial (candidatos + fim simples)
# MAGIC * **Versão 5:** FSM completo com estados (IDLE → START_ARMED → RUNNING)

# COMMAND ----------

# DBTITLE 1,Lógica de Detecção LC (PESOBRA)
# ============================================================
# DETECÇÃO DE INÍCIO LC — QUEDA REAL CONFIRMADA
# ============================================================

import pandas as pd
import numpy as np


def detect_lc_start_events_pesobra(
    df: pd.DataFrame,
    initial_corrida: int = 130473,
    initial_braco: int = 2,

    # -------------------------------
    # SUAVIZAÇÃO
    # -------------------------------
    SMOOTH_WINDOW_S: int = 5,

    # -------------------------------
    # ÚLTIMO SALTO (carga final)
    # -------------------------------
    LOAD_JUMP_WINDOW_S: int = 5,
    LOAD_JUMP_TON: float = 6.0,
    REFRACTORY_AFTER_LOAD_S: int = 20,

    # -------------------------------
    # INÍCIO (queda real)
    # -------------------------------
    START_MIN_WEIGHT: float = 95.0,
    START_DROP_WINDOW_S: int = 5,
    START_DROP_TON: float = 4.0,

    # Confirmação
    CONFIRM_WINDOW_S: int = 15,
    CONFIRM_EXTRA_DROP_TON: float = 2.0,
):
    """
    Retorna:
    corrida | inicio_lc_sys | braco
    """

    df = df.copy().reset_index(drop=True)

    # ======================================================
    # 1) PESOS
    # ======================================================
    df["peso_bra1"] = pd.to_numeric(df["ACI@LC_TORRE_PESOBRA1"], errors="coerce")
    df["peso_bra2"] = pd.to_numeric(df["ACI@LC_TORRE_PESOBRA2"], errors="coerce")

    # ======================================================
    # 2) SUAVIZAÇÃO
    # ======================================================
    w = int(SMOOTH_WINDOW_S)
    df["p1"] = df["peso_bra1"].rolling(w, min_periods=1).median()
    df["p2"] = df["peso_bra2"].rolling(w, min_periods=1).median()

    # ======================================================
    # 3) FEATURES
    # ======================================================
    def build_features(p: pd.Series):
        jump = p - p.shift(LOAD_JUMP_WINDOW_S)
        is_load = jump >= LOAD_JUMP_TON

        idx = np.arange(len(p))
        last_load_idx = np.where(is_load.fillna(False).to_numpy(), idx, np.nan)
        last_load_idx = pd.Series(last_load_idx).ffill()

        drop = p - p.shift(START_DROP_WINDOW_S)
        is_big_drop = drop <= -START_DROP_TON

        return last_load_idx, is_big_drop

    df["last_load_idx1"], df["is_big_drop1"] = build_features(df["p1"])
    df["last_load_idx2"], df["is_big_drop2"] = build_features(df["p2"])

    # ======================================================
    # 4) FSM SIMPLES: IDLE → CONFIRM
    # ======================================================
    results = []
    corrida = int(initial_corrida)
    braco_atual = int(initial_braco)

    state = "IDLE"
    cand_time = None
    cand_weight = None

    for i in range(len(df)):
        ts = df.at[i, "timestamp"]

        if braco_atual == 1:
            p = df.at[i, "p1"]
            is_big_drop = bool(df.at[i, "is_big_drop1"])
            last_load_idx = df.at[i, "last_load_idx1"]
        else:
            p = df.at[i, "p2"]
            is_big_drop = bool(df.at[i, "is_big_drop2"])
            last_load_idx = df.at[i, "last_load_idx2"]

        # refractory
        ok_refractory = True
        if pd.notna(last_load_idx):
            j = int(last_load_idx)
            t_load = df.at[j, "timestamp"]
            ok_refractory = (ts - t_load).total_seconds() >= REFRACTORY_AFTER_LOAD_S

        # ---------------- IDLE ----------------
        if state == "IDLE":
            if (
                pd.notna(p)
                and p >= START_MIN_WEIGHT
                and ok_refractory
                and is_big_drop
            ):
                state = "CONFIRM"
                cand_time = ts
                cand_weight = float(p)
            continue

        # -------------- CONFIRM --------------
        if state == "CONFIRM":
            elapsed = (ts - cand_time).total_seconds()

            if elapsed >= CONFIRM_WINDOW_S:
                extra_drop = float(p) - cand_weight if pd.notna(p) else 0.0

                if extra_drop <= -CONFIRM_EXTRA_DROP_TON:
                    results.append({
                        "corrida": corrida,
                        "inicio_lc_sys": cand_time,
                        "braco": braco_atual,
                    })

                    corrida += 1
                    braco_atual = 2 if braco_atual == 1 else 1

                state = "IDLE"
                cand_time = None
                cand_weight = None

    return pd.DataFrame(results)


# COMMAND ----------

# df_lc_start_sys = detect_lc_start_events_pesobra(
#     df=df_lc_plot,

#     initial_corrida=130473,
#     initial_braco=2,

#     SMOOTH_WINDOW_S=5,

#     LOAD_JUMP_WINDOW_S=5,
#     LOAD_JUMP_TON=6.0,
#     REFRACTORY_AFTER_LOAD_S=20,

#     START_MIN_WEIGHT=95.0,
#     START_DROP_WINDOW_S=5,
#     START_DROP_TON=4.0,

#     CONFIRM_WINDOW_S=15,
#     CONFIRM_EXTRA_DROP_TON=2.0,
# )

# print(f"Inícios detectados: {len(df_lc_start_sys)}")
# display(df_lc_start_sys)


# COMMAND ----------

# DBTITLE 1,Comparação Operador vs Sistema
# # ============================================================
# # PLOTAR COMPARAÇÃO: OPERADOR vs SISTEMA
# # ============================================================

# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates

# # Plotar PESOBRA1 e PESOBRA2 com marcadores
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)

# # Converter para numérico
# df_lc_plot['peso_bra1'] = pd.to_numeric(df_lc_plot['ACI@LC_TORRE_PESOBRA1'], errors='coerce')
# df_lc_plot['peso_bra2'] = pd.to_numeric(df_lc_plot['ACI@LC_TORRE_PESOBRA2'], errors='coerce')

# # ============================================================
# # SUBPLOT 1: PESOBRA1
# # ============================================================
# ax1.plot(df_lc_plot['timestamp'], df_lc_plot['peso_bra1'], 
#          label='PESOBRA1', linewidth=1.5, alpha=0.8, color='blue')

# # Marcadores do OPERADOR (braço 1)
# if not df_operator_lc.empty:
#     for idx, row in df_operator_lc.iterrows():
#         corrida = int(row['corrida'])
        
#         # Linha verde tracejada para início (operador)
#         ax1.axvline(row['inicio_lc'], color='green', linestyle='--', linewidth=2, alpha=0.6, label='Início OP' if idx == 0 else '')
#         ax1.text(row['inicio_lc'], ax1.get_ylim()[1] * 0.95, f"OP\nIn {corrida}", 
#                 rotation=0, color='green', ha='left', va='top', fontsize=8,
#                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.7))
        
#         # Linha vermelha tracejada para fim (operador)
#         ax1.axvline(row['final_lc'], color='red', linestyle='--', linewidth=2, alpha=0.6, label='Fim OP' if idx == 0 else '')
#         ax1.text(row['final_lc'], ax1.get_ylim()[1] * 0.85, f"OP\nFim {corrida}", 
#                 rotation=0, color='red', ha='right', va='top', fontsize=8,
#                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.7))

# # Marcadores do SISTEMA (braço 1)
# if not df_lc_sys.empty:
#     df_sys_bra1 = df_lc_sys[df_lc_sys['braco'] == 1]
#     for idx, row in df_sys_bra1.iterrows():
#         corrida = int(row['corrida'])
        
#         # Linha verde sólida para início (sistema)
#         ax1.axvline(row['inicio_lc_sys'], color='darkgreen', linestyle='-', linewidth=2, alpha=0.8, label='Início SYS' if idx == df_sys_bra1.index[0] else '')
#         ax1.text(row['inicio_lc_sys'], ax1.get_ylim()[1] * 0.75, f"SYS\nIn {corrida}", 
#                 rotation=0, color='darkgreen', ha='left', va='top', fontsize=8,
#                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', edgecolor='darkgreen', alpha=0.9))
        
#         # Linha vermelha sólida para fim (sistema)
#         ax1.axvline(row['final_lc_sys'], color='darkred', linestyle='-', linewidth=2, alpha=0.8, label='Fim SYS' if idx == df_sys_bra1.index[0] else '')
#         ax1.text(row['final_lc_sys'], ax1.get_ylim()[1] * 0.65, f"SYS\nFim {corrida}", 
#                 rotation=0, color='darkred', ha='right', va='top', fontsize=8,
#                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', edgecolor='darkred', alpha=0.9))

# ax1.set_title('LC - PESOBRA1 (Braço 1) - Operador vs Sistema', fontsize=14, fontweight='bold')
# ax1.set_ylabel('Peso (kg)', fontsize=12)
# ax1.grid(True, alpha=0.3, linestyle=':')
# ax1.legend(loc='upper left', fontsize=10)

# # ============================================================
# # SUBPLOT 2: PESOBRA2
# # ============================================================
# ax2.plot(df_lc_plot['timestamp'], df_lc_plot['peso_bra2'], 
#          label='PESOBRA2', linewidth=1.5, alpha=0.8, color='orange')

# # Marcadores do OPERADOR (braço 2)
# if not df_operator_lc.empty:
#     for idx, row in df_operator_lc.iterrows():
#         corrida = int(row['corrida'])
        
#         ax2.axvline(row['inicio_lc'], color='green', linestyle='--', linewidth=2, alpha=0.6, label='Início OP' if idx == 0 else '')
#         ax2.text(row['inicio_lc'], ax2.get_ylim()[1] * 0.95, f"OP\nIn {corrida}", 
#                 rotation=0, color='green', ha='left', va='top', fontsize=8,
#                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.7))
        
#         ax2.axvline(row['final_lc'], color='red', linestyle='--', linewidth=2, alpha=0.6, label='Fim OP' if idx == 0 else '')
#         ax2.text(row['final_lc'], ax2.get_ylim()[1] * 0.85, f"OP\nFim {corrida}", 
#                 rotation=0, color='red', ha='right', va='top', fontsize=8,
#                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.7))

# # Marcadores do SISTEMA (braço 2)
# if not df_lc_sys.empty:
#     df_sys_bra2 = df_lc_sys[df_lc_sys['braco'] == 2]
#     for idx, row in df_sys_bra2.iterrows():
#         corrida = int(row['corrida'])
        
#         ax2.axvline(row['inicio_lc_sys'], color='darkgreen', linestyle='-', linewidth=2, alpha=0.8, label='Início SYS' if idx == df_sys_bra2.index[0] else '')
#         ax2.text(row['inicio_lc_sys'], ax2.get_ylim()[1] * 0.75, f"SYS\nIn {corrida}", 
#                 rotation=0, color='darkgreen', ha='left', va='top', fontsize=8,
#                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', edgecolor='darkgreen', alpha=0.9))
        
#         ax2.axvline(row['final_lc_sys'], color='darkred', linestyle='-', linewidth=2, alpha=0.8, label='Fim SYS' if idx == df_sys_bra2.index[0] else '')
#         ax2.text(row['final_lc_sys'], ax2.get_ylim()[1] * 0.65, f"SYS\nFim {corrida}", 
#                 rotation=0, color='darkred', ha='right', va='top', fontsize=8,
#                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', edgecolor='darkred', alpha=0.9))

# ax2.set_title('LC - PESOBRA2 (Braço 2) - Operador vs Sistema', fontsize=14, fontweight='bold')
# ax2.set_xlabel('Tempo', fontsize=12)
# ax2.set_ylabel('Peso (kg)', fontsize=12)
# ax2.grid(True, alpha=0.3, linestyle=':')
# ax2.legend(loc='upper left', fontsize=10)

# # Formatar eixo X
# ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
# plt.xticks(rotation=45, ha='right')

# plt.suptitle(f'Comparação: Operador (tracejado) vs Sistema (sólido)\nPeríodo: {START_TS} até {END_TS}', 
#              fontsize=16, fontweight='bold', y=0.995)

# plt.tight_layout()
# plt.show()

# print("\n" + "="*80)
# print("LEGENDA:")
# print("="*80)
# print("  Linhas TRACEJADAS (--) = Dados do OPERADOR (df_final)")
# print("  Linhas SÓLIDAS (─) = Detecção do SISTEMA (lógica PESOBRA)")
# print("  Verde = INÍCIO da corrida")
# print("  Vermelho = FIM da corrida")
# print("="*80)

# COMMAND ----------

import pandas as pd
import numpy as np


def compute_start_candidates_pesobra(
    df: pd.DataFrame,

    # ---- Smoothing ----
    SMOOTH_WINDOW_S: int = 5,

    # ---- Regra base: "já carregou alto" ----
    HIGH_WEIGHT: float = 100.0,
    HIGH_LOOKBACK_S: int = 60,

    # ---- Regra R1: queda brusca ----
    DROP_WINDOW_S: int = 5,
    DROP_TON: float = 6.0,

    # ---- Regra R2: sequência de quedas (joelho) ----
    DP_SEQ_S: int = 8,
    DP_EPS: float = 0.0,

    # ---- Filtro: ignorar acomodação pós-salto ----
    USE_REFRACTORY: bool = True,
    LOAD_JUMP_WINDOW_S: int = 5,
    LOAD_JUMP_TON: float = 6.0,
    REFRACTORY_AFTER_LOAD_S: int = 40,

    # ---- PEAK GATE (depois do pico) ----
    USE_PEAK_GATE: bool = True,
    PEAK_MIN_WEIGHT: float = 112.0,
    PEAK_LOOKBACK_S: int = 180,
    PEAK_AFTER_MIN_S: int = 0,
    PEAK_AFTER_MAX_S: int = 120,

    # ---- Faixa do candidato ----
    MIN_CAND_WEIGHT: float = 80.0,
    MAX_CAND_WEIGHT: float = 115.0,

    # ---- Score (para escolher o melhor dentro do cluster) ----
    SCORE_FUTURE_WINDOW_S: int = 60,     # olha 60s à frente
    SCORE_DROP_WEIGHT: float = 2.0,      # peso da queda futura
    SCORE_R1_BONUS: float = 3.0,         # bônus se r1 (queda brusca)
    SCORE_R2_BONUS: float = 2.0,         # bônus se r2 (fallseq)

    # ---- pós-processamento ----
    CLUSTER_GAP_S: int = 180,  # agrupa candidatos que estão "na mesma corrida"
):
    """
    Retorna:
      df_debug: df com features/flags
      df_cands_all: todos candidatos (antes do cluster)
      df_cands_best: 1 candidato por cluster (melhor score)
    """
    df = df.copy().reset_index(drop=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # pesos numéricos
    df["peso_bra1"] = pd.to_numeric(df["ACI@LC_TORRE_PESOBRA1"], errors="coerce")
    df["peso_bra2"] = pd.to_numeric(df["ACI@LC_TORRE_PESOBRA2"], errors="coerce")

    def s2n(s: int) -> int:
        return max(1, int(s))

    # smoothing
    w = s2n(SMOOTH_WINDOW_S)
    df["p1"] = df["peso_bra1"].rolling(w, min_periods=1).median()
    df["p2"] = df["peso_bra2"].rolling(w, min_periods=1).median()

    # esteve alto recentemente
    lb = s2n(HIGH_LOOKBACK_S)
    df["was_high_bra1"] = (df["p1"].rolling(lb, min_periods=1).max() >= HIGH_WEIGHT).fillna(False).astype(bool)
    df["was_high_bra2"] = (df["p2"].rolling(lb, min_periods=1).max() >= HIGH_WEIGHT).fillna(False).astype(bool)

    # R1: queda brusca
    dw = s2n(DROP_WINDOW_S)
    df["drop1"] = df["p1"] - df["p1"].shift(dw)
    df["drop2"] = df["p2"] - df["p2"].shift(dw)
    df["r1_bra1"] = (df["was_high_bra1"] & (df["drop1"] <= -DROP_TON).fillna(False)).fillna(False).astype(bool)
    df["r1_bra2"] = (df["was_high_bra2"] & (df["drop2"] <= -DROP_TON).fillna(False)).fillna(False).astype(bool)

    # R2: sequência de quedas
    df["dp1"] = df["p1"].diff()
    df["dp2"] = df["p2"].diff()
    seq = s2n(DP_SEQ_S)
    fall1 = ((df["dp1"] < -DP_EPS).fillna(False).rolling(seq, min_periods=seq).sum() == seq)
    fall2 = ((df["dp2"] < -DP_EPS).fillna(False).rolling(seq, min_periods=seq).sum() == seq)
    df["fallseq1"] = fall1.fillna(False).astype(bool)
    df["fallseq2"] = fall2.fillna(False).astype(bool)
    df["r2_bra1"] = (df["was_high_bra1"] & df["fallseq1"]).fillna(False).astype(bool)
    df["r2_bra2"] = (df["was_high_bra2"] & df["fallseq2"]).fillna(False).astype(bool)

    # REFRACTORY
    if USE_REFRACTORY:
        jw = s2n(LOAD_JUMP_WINDOW_S)
        df["jump1"] = df["p1"] - df["p1"].shift(jw)
        df["jump2"] = df["p2"] - df["p2"].shift(jw)

        df["is_load1"] = (df["jump1"] >= LOAD_JUMP_TON).fillna(False).astype(bool)
        df["is_load2"] = (df["jump2"] >= LOAD_JUMP_TON).fillna(False).astype(bool)

        idx = np.arange(len(df))
        last1 = np.where(df["is_load1"].to_numpy(), idx, np.nan)
        last2 = np.where(df["is_load2"].to_numpy(), idx, np.nan)
        df["last_load_idx1"] = pd.Series(last1).ffill()
        df["last_load_idx2"] = pd.Series(last2).ffill()

        ref = s2n(REFRACTORY_AFTER_LOAD_S)
        df["ok_ref1"] = (df["last_load_idx1"].isna() | ((idx - df["last_load_idx1"]) >= ref)).fillna(False).astype(bool)
        df["ok_ref2"] = (df["last_load_idx2"].isna() | ((idx - df["last_load_idx2"]) >= ref)).fillna(False).astype(bool)

        df["r1_bra1"] = (df["r1_bra1"] & df["ok_ref1"]).fillna(False).astype(bool)
        df["r1_bra2"] = (df["r1_bra2"] & df["ok_ref2"]).fillna(False).astype(bool)
        df["r2_bra1"] = (df["r2_bra1"] & df["ok_ref1"]).fillna(False).astype(bool)
        df["r2_bra2"] = (df["r2_bra2"] & df["ok_ref2"]).fillna(False).astype(bool)
    else:
        df["ok_ref1"] = True
        df["ok_ref2"] = True

    # junta regras
    df["start_any_bra1"] = (df["r1_bra1"] | df["r2_bra1"]).fillna(False).astype(bool)
    df["start_any_bra2"] = (df["r1_bra2"] | df["r2_bra2"]).fillna(False).astype(bool)

    # faixa do candidato
    df["cand_range1"] = ((df["p1"] >= MIN_CAND_WEIGHT) & (df["p1"] <= MAX_CAND_WEIGHT)).fillna(False).astype(bool)
    df["cand_range2"] = ((df["p2"] >= MIN_CAND_WEIGHT) & (df["p2"] <= MAX_CAND_WEIGHT)).fillna(False).astype(bool)
    df["start_any_bra1"] = (df["start_any_bra1"] & df["cand_range1"]).fillna(False).astype(bool)
    df["start_any_bra2"] = (df["start_any_bra2"] & df["cand_range2"]).fillna(False).astype(bool)

    # PEAK gate
    if USE_PEAK_GATE:
        idx = np.arange(len(df))
        df["is_peak1"] = (df["p1"] >= PEAK_MIN_WEIGHT).fillna(False).astype(bool)
        df["is_peak2"] = (df["p2"] >= PEAK_MIN_WEIGHT).fillna(False).astype(bool)

        lastp1 = np.where(df["is_peak1"].to_numpy(), idx, np.nan)
        lastp2 = np.where(df["is_peak2"].to_numpy(), idx, np.nan)
        df["last_peak_idx1"] = pd.Series(lastp1).ffill()
        df["last_peak_idx2"] = pd.Series(lastp2).ffill()

        look = s2n(PEAK_LOOKBACK_S)
        df["has_peak_recent1"] = (df["is_peak1"].rolling(look, min_periods=1).max() > 0).fillna(False).astype(bool)
        df["has_peak_recent2"] = (df["is_peak2"].rolling(look, min_periods=1).max() > 0).fillna(False).astype(bool)

        amin = s2n(PEAK_AFTER_MIN_S)
        amax = s2n(PEAK_AFTER_MAX_S)

        df["after_peak1"] = (
            df["last_peak_idx1"].notna() &
            ((idx - df["last_peak_idx1"]) >= amin) &
            ((idx - df["last_peak_idx1"]) <= amax)
        ).fillna(False).astype(bool)

        df["after_peak2"] = (
            df["last_peak_idx2"].notna() &
            ((idx - df["last_peak_idx2"]) >= amin) &
            ((idx - df["last_peak_idx2"]) <= amax)
        ).fillna(False).astype(bool)

        df["gate1"] = (df["has_peak_recent1"] & df["after_peak1"]).fillna(False).astype(bool)
        df["gate2"] = (df["has_peak_recent2"] & df["after_peak2"]).fillna(False).astype(bool)

        df["start_any_bra1"] = (df["start_any_bra1"] & df["gate1"]).fillna(False).astype(bool)
        df["start_any_bra2"] = (df["start_any_bra2"] & df["gate2"]).fillna(False).astype(bool)
    else:
        df["gate1"] = True
        df["gate2"] = True

    # EDGE
    prev1 = df["start_any_bra1"].shift(1).fillna(False).astype(bool)
    prev2 = df["start_any_bra2"].shift(1).fillna(False).astype(bool)
    df["edge_bra1"] = (df["start_any_bra1"] & (~prev1)).fillna(False).astype(bool)
    df["edge_bra2"] = (df["start_any_bra2"] & (~prev2)).fillna(False).astype(bool)

    # ======================================================
    # SCORE: queda futura (quanto mais cai, melhor)
    # ======================================================
    fw = s2n(SCORE_FUTURE_WINDOW_S)
    fmin1 = df["p1"].shift(-1)[::-1].rolling(fw, min_periods=1).min()[::-1]
    fmin2 = df["p2"].shift(-1)[::-1].rolling(fw, min_periods=1).min()[::-1]
    df["future_drop1"] = (df["p1"] - fmin1)
    df["future_drop2"] = (df["p2"] - fmin2)

    df["score1"] = (
        SCORE_DROP_WEIGHT * df["future_drop1"].fillna(0)
        + SCORE_R1_BONUS * df["r1_bra1"].astype(int)
        + SCORE_R2_BONUS * df["r2_bra1"].astype(int)
    )
    df["score2"] = (
        SCORE_DROP_WEIGHT * df["future_drop2"].fillna(0)
        + SCORE_R1_BONUS * df["r1_bra2"].astype(int)
        + SCORE_R2_BONUS * df["r2_bra2"].astype(int)
    )

    # ======================================================
    # MONTA TODOS candidatos (antes do cluster)
    # ======================================================
    rows = []
    for i in df.index[df["edge_bra1"]]:
        rows.append({
            "i": int(i),
            "timestamp": df.at[i, "timestamp"],
            "braco": 1,
            "p": float(df.at[i, "p1"]) if pd.notna(df.at[i, "p1"]) else np.nan,
            "drop": float(df.at[i, "drop1"]) if pd.notna(df.at[i, "drop1"]) else np.nan,
            "future_drop": float(df.at[i, "future_drop1"]) if pd.notna(df.at[i, "future_drop1"]) else np.nan,
            "r1": bool(df.at[i, "r1_bra1"]),
            "r2": bool(df.at[i, "r2_bra1"]),
            "score": float(df.at[i, "score1"]) if pd.notna(df.at[i, "score1"]) else 0.0,
        })
    for i in df.index[df["edge_bra2"]]:
        rows.append({
            "i": int(i),
            "timestamp": df.at[i, "timestamp"],
            "braco": 2,
            "p": float(df.at[i, "p2"]) if pd.notna(df.at[i, "p2"]) else np.nan,
            "drop": float(df.at[i, "drop2"]) if pd.notna(df.at[i, "drop2"]) else np.nan,
            "future_drop": float(df.at[i, "future_drop2"]) if pd.notna(df.at[i, "future_drop2"]) else np.nan,
            "r1": bool(df.at[i, "r1_bra2"]),
            "r2": bool(df.at[i, "r2_bra2"]),
            "score": float(df.at[i, "score2"]) if pd.notna(df.at[i, "score2"]) else 0.0,
        })

    df_cands_all = pd.DataFrame(rows).sort_values(["braco", "timestamp"]).reset_index(drop=True)

    # ======================================================
    # CLUSTER por braço: candidatos com delta <= CLUSTER_GAP_S
    # Escolhe o MAIOR score dentro do cluster (em vez do primeiro)
    # ======================================================
    if df_cands_all.empty:
        return df, df_cands_all, df_cands_all

    gap = pd.Timedelta(seconds=int(CLUSTER_GAP_S))
    best_rows = []

    for braco in [1, 2]:
        sub = df_cands_all[df_cands_all["braco"] == braco].sort_values("timestamp").reset_index(drop=True)
        if sub.empty:
            continue

        cluster_id = 0
        clusters = [cluster_id]
        for k in range(1, len(sub)):
            if (sub.loc[k, "timestamp"] - sub.loc[k - 1, "timestamp"]) > gap:
                cluster_id += 1
            clusters.append(cluster_id)
        sub["cluster"] = clusters

        # pega o melhor por cluster (maior score; empate -> mais tarde)
        sub_best = (
            sub.sort_values(["cluster", "score", "timestamp"])
               .groupby("cluster", as_index=False)
               .tail(1)
        )

        best_rows.append(sub_best)

    df_start_cands = (
        pd.concat(best_rows, ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df, df_cands_all, df_start_cands

# COMMAND ----------

# MAGIC %md
# MAGIC #### 🎯 5.1. Detecção de Candidatos de Início (Scoring + Clustering)
# MAGIC
# MAGIC **Características:**
# MAGIC * Duas regras de detecção: R1 (queda brusca) e R2 (sequência de quedas)
# MAGIC * Período refratário após saltos de carga
# MAGIC * Peak gate (deve ocorrer após pico de peso)
# MAGIC * Sistema de scoring baseado em queda futura
# MAGIC * Clustering para agrupar candidatos próximos

# COMMAND ----------

df_debug, df_cands_all, df_start_cands = compute_start_candidates_pesobra(
    df=df_lc_plot,

    # ---- smoothing ----
    SMOOTH_WINDOW_S=5,

    # ---- já carregou ----
    HIGH_WEIGHT=100.0,
    HIGH_LOOKBACK_S=60,

    # ---- queda brusca ----
    DROP_WINDOW_S=5,
    DROP_TON=2.0,

    # ---- joelho ----
    DP_SEQ_S=8,
    DP_EPS=0.2,

    # ---- refractory ----
    USE_REFRACTORY=True,
    LOAD_JUMP_WINDOW_S=5,
    LOAD_JUMP_TON=6.0,
    REFRACTORY_AFTER_LOAD_S=40,

    # ---- peak gate ----
    USE_PEAK_GATE=False,
    PEAK_MIN_WEIGHT=108.0,
    PEAK_LOOKBACK_S=60,
    PEAK_AFTER_MIN_S=0,
    PEAK_AFTER_MAX_S=60,

    # ---- faixa válida de início ----
    MIN_CAND_WEIGHT=80.0,
    MAX_CAND_WEIGHT=115.0,

    # ---- score ----
    SCORE_FUTURE_WINDOW_S=10,
    SCORE_DROP_WEIGHT=2.0,
    SCORE_R1_BONUS=3.0,
    SCORE_R2_BONUS=2.0,

    # ---- cluster ----
    CLUSTER_GAP_S=180,
)

print("DEBUG – todos candidatos:", len(df_cands_all))
display(df_cands_all)

print("CANDIDATOS FINAIS (1 por corrida):", len(df_start_cands))
display(df_start_cands)


# COMMAND ----------

# ============================================================
# 3) PLOT: OPERADOR vs SISTEMA + CANDIDATOS
# ============================================================
import matplotlib.dates as mdates

# Plotar PESOBRA1 e PESOBRA2 com marcadores
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)

# Converter para numérico (garantia)
df_lc_plot['peso_bra1'] = pd.to_numeric(df_lc_plot['ACI@LC_TORRE_PESOBRA1'], errors='coerce')
df_lc_plot['peso_bra2'] = pd.to_numeric(df_lc_plot['ACI@LC_TORRE_PESOBRA2'], errors='coerce')

# ============================================================
# SUBPLOT 1: PESOBRA1
# ============================================================
ax1.plot(df_lc_plot['timestamp'], df_lc_plot['peso_bra1'],
         label='PESOBRA1', linewidth=1.5, alpha=0.8, color='blue')

# Marcadores do OPERADOR (ambos braços no df_operator_lc, você usa igual)
if not df_operator_lc.empty:
    for idx, row in df_operator_lc.iterrows():
        corrida = int(row['corrida'])

        ax1.axvline(row['inicio_lc'], color='green', linestyle='--', linewidth=2, alpha=0.6,
                    label='Início OP' if idx == df_operator_lc.index[0] else '')
        ax1.text(row['inicio_lc'], ax1.get_ylim()[1] * 0.95, f"OP\nIn {corrida}",
                 rotation=0, color='green', ha='left', va='top', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.7))

        ax1.axvline(row['final_lc'], color='red', linestyle='--', linewidth=2, alpha=0.6,
                    label='Fim OP' if idx == df_operator_lc.index[0] else '')
        ax1.text(row['final_lc'], ax1.get_ylim()[1] * 0.85, f"OP\nFim {corrida}",
                 rotation=0, color='red', ha='right', va='top', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.7))

# Marcadores do SISTEMA (se existir df_lc_sys)
if "df_lc_sys" in globals() and isinstance(df_lc_sys, pd.DataFrame) and (not df_lc_sys.empty):
    df_sys_bra1 = df_lc_sys[df_lc_sys['braco'] == 1]
    for j, row in enumerate(df_sys_bra1.itertuples(index=False)):
        ax1.axvline(row.inicio_lc_sys, color='darkgreen', linestyle='-', linewidth=2, alpha=0.8,
                    label='Início SYS' if j == 0 else '')
        ax1.axvline(row.final_lc_sys, color='darkred', linestyle='-', linewidth=2, alpha=0.8,
                    label='Fim SYS' if j == 0 else '')

# Candidatos de início (SYS) — Braço 1
c1 = df_start_cands[df_start_cands["braco"] == 1]["timestamp"] if not df_start_cands.empty else []
for j, t in enumerate(c1):
    ax1.axvline(t, color="purple", linestyle=":", linewidth=2, alpha=0.7,
                label="Start cand SYS" if j == 0 else "")
    ax1.text(t, ax1.get_ylim()[1]*0.55, "cand", color="purple",
             ha="left", va="top", fontsize=7,
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="purple", alpha=0.6))

ax1.set_title('LC - PESOBRA1 (Braço 1) - Operador vs Sistema + Candidatos', fontsize=14, fontweight='bold')
ax1.set_ylabel('Peso (kg)', fontsize=12)
ax1.grid(True, alpha=0.3, linestyle=':')
ax1.legend(loc='upper left', fontsize=10)

# ============================================================
# SUBPLOT 2: PESOBRA2
# ============================================================
ax2.plot(df_lc_plot['timestamp'], df_lc_plot['peso_bra2'],
         label='PESOBRA2', linewidth=1.5, alpha=0.8, color='orange')

# Marcadores do OPERADOR
if not df_operator_lc.empty:
    for idx, row in df_operator_lc.iterrows():
        corrida = int(row['corrida'])

        ax2.axvline(row['inicio_lc'], color='green', linestyle='--', linewidth=2, alpha=0.6,
                    label='Início OP' if idx == df_operator_lc.index[0] else '')
        ax2.text(row['inicio_lc'], ax2.get_ylim()[1] * 0.95, f"OP\nIn {corrida}",
                 rotation=0, color='green', ha='left', va='top', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.7))

        ax2.axvline(row['final_lc'], color='red', linestyle='--', linewidth=2, alpha=0.6,
                    label='Fim OP' if idx == df_operator_lc.index[0] else '')
        ax2.text(row['final_lc'], ax2.get_ylim()[1] * 0.85, f"OP\nFim {corrida}",
                 rotation=0, color='red', ha='right', va='top', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.7))

# Marcadores do SISTEMA (se existir df_lc_sys)
if "df_lc_sys" in globals() and isinstance(df_lc_sys, pd.DataFrame) and (not df_lc_sys.empty):
    df_sys_bra2 = df_lc_sys[df_lc_sys['braco'] == 2]
    for j, row in enumerate(df_sys_bra2.itertuples(index=False)):
        ax2.axvline(row.inicio_lc_sys, color='darkgreen', linestyle='-', linewidth=2, alpha=0.8,
                    label='Início SYS' if j == 0 else '')
        ax2.axvline(row.final_lc_sys, color='darkred', linestyle='-', linewidth=2, alpha=0.8,
                    label='Fim SYS' if j == 0 else '')

# Candidatos de início (SYS) — Braço 2
c2 = df_start_cands[df_start_cands["braco"] == 2]["timestamp"] if not df_start_cands.empty else []
for j, t in enumerate(c2):
    ax2.axvline(t, color="purple", linestyle=":", linewidth=2, alpha=0.7,
                label="Start cand SYS" if j == 0 else "")
    ax2.text(t, ax2.get_ylim()[1]*0.55, "cand", color="purple",
             ha="left", va="top", fontsize=7,
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="purple", alpha=0.6))

ax2.set_title('LC - PESOBRA2 (Braço 2) - Operador vs Sistema + Candidatos', fontsize=14, fontweight='bold')
ax2.set_xlabel('Tempo', fontsize=12)
ax2.set_ylabel('Peso (kg)', fontsize=12)
ax2.grid(True, alpha=0.3, linestyle=':')
ax2.legend(loc='upper left', fontsize=10)

# Formatar eixo X com locator automático
locator = mdates.AutoDateLocator()
ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
plt.xticks(rotation=45, ha='right')

plt.suptitle(
    f'Comparação: Operador (tracejado) vs Sistema (sólido) + Candidatos (roxo pontilhado)\nPeríodo: {START_TS} até {END_TS}',
    fontsize=16, fontweight='bold', y=0.995
)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("LEGENDA:")
print("="*80)
print("  TRACEJADAS (--) = Operador")
print("  SÓLIDAS (─) = Sistema (df_lc_sys), se existir")
print("  ROXO (:) = Candidatos de INÍCIO do Sistema (DEBUG)")
print("="*80)

# COMMAND ----------

import pandas as pd
import numpy as np


def detect_lc_events_sequential_pesobra(
    df: pd.DataFrame,

    # -------- INÍCIO (SUA FUNÇÃO, INTACTA) --------
    start_params: dict,

    # -------- fluxo --------
    initial_corrida: int,
    initial_braco: int,  # 1 ou 2

    # -------- FIM (SIMPLES E PARAMÉTRICO) --------
    END_ARM_WEIGHT: float = 60.0,   # t
    END_RISE_TON: float = 2.0,      # t
    END_RISE_WINDOW_S: int = 20,    # s

    DEBUG: bool = True,
):
    """
    Retorna:
      df_sys_lc: corrida | inicio_lc_sys | final_lc_sys | braco
      df_debug_trace
    """

    # ======================================================
    # 1) CANDIDATOS DE INÍCIO (SEM MEXER)
    # ======================================================
    _, _, df_start_cands = compute_start_candidates_pesobra(
        df=df,
        **start_params
    )

    if df_start_cands.empty:
        return pd.DataFrame(), pd.DataFrame()

    # ======================================================
    # 2) PREPARA SÉRIES
    # ======================================================
    df = df.reset_index(drop=True)

    df["p1"] = pd.to_numeric(df["ACI@LC_TORRE_PESOBRA1"], errors="coerce")
    df["p2"] = pd.to_numeric(df["ACI@LC_TORRE_PESOBRA2"], errors="coerce")

    # ======================================================
    # 3) LOOP SEQUENCIAL
    # ======================================================
    corrida = int(initial_corrida)
    braco = int(initial_braco)

    results = []
    debug_rows = []

    n = len(df)
    cand_idx = 0

    while cand_idx < len(df_start_cands):

        cand = df_start_cands.iloc[cand_idx]

        # respeita alternância de braço
        if int(cand["braco"]) != braco:
            cand_idx += 1
            continue

        i_start = int(cand["i"])
        ts_start = cand["timestamp"]

        debug_rows.append({
            "event": "RUN_START",
            "corrida": corrida,
            "braco": braco,
            "i": i_start,
            "timestamp": ts_start,
        })

        crossed_60 = False
        i_cross_60 = None
        p_at_60 = None

        end_i = None

        i = i_start + 1

        while i < n:

            ts = df.at[i, "timestamp"]
            p = df.at[i, "p1"] if braco == 1 else df.at[i, "p2"]

            if pd.isna(p):
                i += 1
                continue

            # -----------------------------
            # CRUZOU 60t
            # -----------------------------
            if (not crossed_60) and p <= END_ARM_WEIGHT:
                crossed_60 = True
                i_cross_60 = i
                p_at_60 = p

                if DEBUG:
                    debug_rows.append({
                        "event": "CROSSED_60",
                        "corrida": corrida,
                        "braco": braco,
                        "i": i,
                        "timestamp": ts,
                        "p": p,
                    })

                i += 1
                continue

            # -----------------------------
            # APÓS CRUZAR 60 → PROCURA PICO
            # -----------------------------
            if crossed_60:

                dt = (ts - df.at[i_cross_60, "timestamp"]).total_seconds()

                # só olha dentro da janela
                if dt <= END_RISE_WINDOW_S:

                    if (p - p_at_60) >= END_RISE_TON:
                        end_i = i

                        debug_rows.append({
                            "event": "RUN_CLOSE",
                            "corrida": corrida,
                            "braco": braco,
                            "i": i,
                            "timestamp": ts,
                            "p": p,
                            "delta_p": p - p_at_60,
                            "dt_s": dt,
                            "reason": "rise_after_60",
                        })
                        break

                else:
                    # passou a janela → atualiza referência
                    i_cross_60 = i
                    p_at_60 = p

            i += 1

        # -----------------------------
        # FECHOU CORRIDA
        # -----------------------------
        if end_i is not None:
            results.append({
                "corrida": corrida,
                "inicio_lc_sys": ts_start,
                "final_lc_sys": df.at[end_i, "timestamp"],
                "braco": braco,
                "close_reason": "rise_after_60",
            })

            corrida += 1
            braco = 2 if braco == 1 else 1
            cand_idx += 1
        else:
            # não achou fim → aborta fluxo
            break

    return pd.DataFrame(results), pd.DataFrame(debug_rows)


# COMMAND ----------

# MAGIC %md
# MAGIC #### 🔄 5.2. Detecção Sequencial (Candidatos + Fim Simples)
# MAGIC
# MAGIC **Estratégia:**
# MAGIC * Usa candidatos de início da seção anterior
# MAGIC * Detecta fim quando peso cruza 60t e depois sobe 2t em 20s
# MAGIC * Alterna entre braços automaticamente

# COMMAND ----------

start_params = dict(
    SMOOTH_WINDOW_S=5,
    HIGH_WEIGHT=100.0,
    HIGH_LOOKBACK_S=60,
    DROP_WINDOW_S=8,
    DROP_TON=2.0,
    DP_SEQ_S=5,
    DP_EPS=0.05,
    USE_REFRACTORY=True,
    LOAD_JUMP_WINDOW_S=5,
    LOAD_JUMP_TON=6.0,
    REFRACTORY_AFTER_LOAD_S=80,
    USE_PEAK_GATE=False,
    PEAK_MIN_WEIGHT=108.0,
    PEAK_LOOKBACK_S=60,
    PEAK_AFTER_MIN_S=0,
    PEAK_AFTER_MAX_S=60,
    MIN_CAND_WEIGHT=80.0,
    MAX_CAND_WEIGHT=115.0,
    SCORE_FUTURE_WINDOW_S=10,
    SCORE_DROP_WEIGHT=2.0,
    SCORE_R1_BONUS=3.0,
    SCORE_R2_BONUS=2.0,
    CLUSTER_GAP_S=180,
)

df_sys_lc, df_debug = detect_lc_events_sequential_pesobra(
    df=df_lc_plot,

    start_params=start_params,

    initial_corrida=130460,
    initial_braco=1,

    END_ARM_WEIGHT=60.0,
    END_RISE_TON=2.0,
    END_RISE_WINDOW_S=20,

    DEBUG=True
)




display(df_sys_lc)
display(df_debug)
df_lc_sys = df_sys_lc



# COMMAND ----------

display(df_start_cands)


# COMMAND ----------

# ============================================================
# 3) PLOT: OPERADOR vs SISTEMA + CANDIDATOS (com filtro de período)
# ============================================================

from datetime import datetime
import matplotlib.dates as mdates

# Selecione o período desejado (edite aqui)
start_plot = pd.to_datetime("2025-11-29 14:00:00")
end_plot = pd.to_datetime("2025-11-29 20:00:00")

# --- Normalizar timestamps (remover timezone se existir) ---
def tz_strip(s):
    """Remove timezone de uma Series datetime."""
    if hasattr(s, 'dt'):
        if s.dt.tz is not None:
            return s.dt.tz_localize(None)
    return s

df_lc_plot["timestamp"] = tz_strip(df_lc_plot["timestamp"])

# Filtrar df_lc_plot
mask = (df_lc_plot["timestamp"] >= start_plot) & (df_lc_plot["timestamp"] <= end_plot)
df_lc_plot_filt = df_lc_plot[mask].copy()
print(f"DEBUG: df_lc_plot total={len(df_lc_plot)}, filtrado={len(df_lc_plot_filt)}, ts_dtype={df_lc_plot['timestamp'].dtype}")

# Filtrar df_operator_lc
if not df_operator_lc.empty:
    df_operator_lc["inicio_lc"] = tz_strip(df_operator_lc["inicio_lc"])
    df_operator_lc["final_lc"] = tz_strip(df_operator_lc["final_lc"])
    mask_op = (df_operator_lc["inicio_lc"] <= end_plot) & (df_operator_lc["final_lc"] >= start_plot)
    df_operator_lc_filt = df_operator_lc[mask_op].copy()
else:
    df_operator_lc_filt = df_operator_lc
print(f"DEBUG: operador filtrado={len(df_operator_lc_filt)}")

# Filtrar df_lc_sys se existir
if "df_lc_sys" in globals() and isinstance(df_lc_sys, pd.DataFrame) and (not df_lc_sys.empty):
    df_lc_sys_cp = df_lc_sys.copy()
    df_lc_sys_cp["inicio_lc_sys"] = tz_strip(df_lc_sys_cp["inicio_lc_sys"])
    df_lc_sys_cp["final_lc_sys"] = tz_strip(df_lc_sys_cp["final_lc_sys"])
    mask_sys = (df_lc_sys_cp["inicio_lc_sys"] <= end_plot) & (df_lc_sys_cp["final_lc_sys"] >= start_plot)
    df_lc_sys_filt = df_lc_sys_cp[mask_sys].copy()
else:
    df_lc_sys_filt = pd.DataFrame()
print(f"DEBUG: sistema filtrado={len(df_lc_sys_filt)}")

# Filtrar df_start_cands
if "df_start_cands" in globals() and not df_start_cands.empty:
    df_sc = df_start_cands.copy()
    df_sc["timestamp"] = tz_strip(df_sc["timestamp"])
    mask_cand = (df_sc["timestamp"] >= start_plot) & (df_sc["timestamp"] <= end_plot)
    df_start_cands_filt = df_sc[mask_cand].copy()
else:
    df_start_cands_filt = pd.DataFrame()
print(f"DEBUG: candidatos filtrados={len(df_start_cands_filt)}")

if df_lc_plot_filt.empty:
    print("\n\u26a0\ufe0f AVISO: Nenhum dado no per\u00edodo selecionado!")
    print(f"  Timestamps no df_lc_plot: {df_lc_plot['timestamp'].min()} a {df_lc_plot['timestamp'].max()}")
    print(f"  Filtro: {start_plot} a {end_plot}")
else:
    # Plotar PESOBRA1 e PESOBRA2 com marcadores
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)

    # Converter para numérico (garantia)
    df_lc_plot_filt['peso_bra1'] = pd.to_numeric(df_lc_plot_filt['ACI@LC_TORRE_PESOBRA1'], errors='coerce')
    df_lc_plot_filt['peso_bra2'] = pd.to_numeric(df_lc_plot_filt['ACI@LC_TORRE_PESOBRA2'], errors='coerce')

    # ============================================================
    # SUBPLOT 1: PESOBRA1
    # ============================================================
    ax1.plot(df_lc_plot_filt['timestamp'], df_lc_plot_filt['peso_bra1'],
             label='PESOBRA1', linewidth=1.5, alpha=0.8, color='blue')

    # Marcadores do OPERADOR
    if not df_operator_lc_filt.empty:
        for idx, row in df_operator_lc_filt.iterrows():
            corrida = int(row['corrida'])
            ax1.axvline(row['inicio_lc'], color='green', linestyle='--', linewidth=2, alpha=0.6,
                        label='Início OP' if idx == df_operator_lc_filt.index[0] else '')
            ax1.text(row['inicio_lc'], ax1.get_ylim()[1] * 0.95, f"OP\nIn {corrida}",
                     rotation=0, color='green', ha='left', va='top', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.7))
            ax1.axvline(row['final_lc'], color='red', linestyle='--', linewidth=2, alpha=0.6,
                        label='Fim OP' if idx == df_operator_lc_filt.index[0] else '')
            ax1.text(row['final_lc'], ax1.get_ylim()[1] * 0.85, f"OP\nFim {corrida}",
                     rotation=0, color='red', ha='right', va='top', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.7))

    # Marcadores do SISTEMA
    if not df_lc_sys_filt.empty:
        df_sys_bra1 = df_lc_sys_filt[df_lc_sys_filt['braco'] == 1]
        for j, row in enumerate(df_sys_bra1.itertuples(index=False)):
            ax1.axvline(row.inicio_lc_sys, color='darkgreen', linestyle='-', linewidth=2, alpha=0.8,
                        label='Início SYS' if j == 0 else '')
            ax1.axvline(row.final_lc_sys, color='darkred', linestyle='-', linewidth=2, alpha=0.8,
                        label='Fim SYS' if j == 0 else '')

    # Candidatos de início — Braço 1
    if not df_start_cands_filt.empty and "braco" in df_start_cands_filt.columns:
        c1 = df_start_cands_filt[df_start_cands_filt["braco"] == 1]["timestamp"]
        for j, t in enumerate(c1):
            ax1.axvline(t, color="purple", linestyle=":", linewidth=2, alpha=0.7,
                        label="Start cand SYS" if j == 0 else "")
            ax1.text(t, ax1.get_ylim()[1]*0.55, "cand", color="purple",
                     ha="left", va="top", fontsize=7,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="purple", alpha=0.6))

    ax1.set_title('LC - PESOBRA1 (Braço 1) - Operador vs Sistema + Candidatos', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Peso (kg)', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(loc='upper left', fontsize=10)

    # ============================================================
    # SUBPLOT 2: PESOBRA2
    # ============================================================
    ax2.plot(df_lc_plot_filt['timestamp'], df_lc_plot_filt['peso_bra2'],
             label='PESOBRA2', linewidth=1.5, alpha=0.8, color='orange')

    # Marcadores do OPERADOR
    if not df_operator_lc_filt.empty:
        for idx, row in df_operator_lc_filt.iterrows():
            corrida = int(row['corrida'])
            ax2.axvline(row['inicio_lc'], color='green', linestyle='--', linewidth=2, alpha=0.6,
                        label='Início OP' if idx == df_operator_lc_filt.index[0] else '')
            ax2.axvline(row['final_lc'], color='red', linestyle='--', linewidth=2, alpha=0.6,
                        label='Fim OP' if idx == df_operator_lc_filt.index[0] else '')

    # Marcadores do SISTEMA
    if not df_lc_sys_filt.empty:
        df_sys_bra2 = df_lc_sys_filt[df_lc_sys_filt['braco'] == 2]
        for j, row in enumerate(df_sys_bra2.itertuples(index=False)):
            ax2.axvline(row.inicio_lc_sys, color='darkgreen', linestyle='-', linewidth=2, alpha=0.8,
                        label='Início SYS' if j == 0 else '')
            ax2.axvline(row.final_lc_sys, color='darkred', linestyle='-', linewidth=2, alpha=0.8,
                        label='Fim SYS' if j == 0 else '')

    # Candidatos de início — Braço 2
    if not df_start_cands_filt.empty and "braco" in df_start_cands_filt.columns:
        c2 = df_start_cands_filt[df_start_cands_filt["braco"] == 2]["timestamp"]
        for j, t in enumerate(c2):
            ax2.axvline(t, color="purple", linestyle=":", linewidth=2, alpha=0.7,
                        label="Start cand SYS" if j == 0 else "")
            ax2.text(t, ax2.get_ylim()[1]*0.55, "cand", color="purple",
                     ha="left", va="top", fontsize=7,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="purple", alpha=0.6))

    ax2.set_title('LC - PESOBRA2 (Braço 2) - Operador vs Sistema + Candidatos', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Tempo', fontsize=12)
    ax2.set_ylabel('Peso (kg)', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.legend(loc='upper left', fontsize=10)

    # Formatar eixo X com locator automático
    locator = mdates.AutoDateLocator()
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plt.xticks(rotation=45, ha='right')

    plt.suptitle(
        f'Comparação: Operador (tracejado) vs Sistema (sólido) + Candidatos (roxo pontilhado)\nPeríodo: {start_plot} até {end_plot}',
        fontsize=16, fontweight='bold', y=0.995
    )

    plt.tight_layout()
    plt.show()

    print("\n" + "="*80)
    print("LEGENDA:")
    print("="*80)
    print("  TRACEJADAS (--) = Operador")
    print("  SÓLIDAS (─) = Sistema (df_lc_sys), se existir")
    print("  ROXO (:) = Candidatos de INÍCIO do Sistema (DEBUG)")
    print("="*80)

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,COMPARAÇÃO OPERADOR
import pandas as pd

def comparacao_operador(df_sys_lc, df_final):
    # ============================================================
    # 1) CONVERTER DF DO OPERADOR (SPARK → PANDAS)
    # ============================================================
    df_op = (
        df_final
        .select("corrida", "inicio_lc", "final_lc")
        .toPandas()
    )

    if df_op.empty:
        print("AVISO: df_final (operador) está vazio no período selecionado!")
        return pd.DataFrame()

    # ============================================================
    # 2) DF DO SISTEMA JÁ É PANDAS
    # ============================================================
    df_sys = df_sys_lc.copy()

    if df_sys.empty:
        print("AVISO: df_sys_lc (sistema) está vazio!")
        return pd.DataFrame()

    # ============================================================
    # 3) GARANTIR DATETIME
    # ============================================================
    df_sys["inicio_lc_sys"] = pd.to_datetime(df_sys["inicio_lc_sys"])
    df_sys["final_lc_sys"] = pd.to_datetime(df_sys["final_lc_sys"])
    df_op["inicio_lc"] = pd.to_datetime(df_op["inicio_lc"])
    df_op["final_lc"] = pd.to_datetime(df_op["final_lc"])

    # ============================================================
    # 4) MATCH POR PROXIMIDADE TEMPORAL (em vez de corrida)
    #    Para cada corrida do operador, encontra o evento do
    #    sistema com inicio_lc_sys mais próximo.
    # ============================================================
    matches = []
    sys_used = set()

    for _, op_row in df_op.sort_values("inicio_lc").iterrows():
        best_idx = None
        best_delta = pd.Timedelta.max

        for sys_idx, sys_row in df_sys.iterrows():
            if sys_idx in sys_used:
                continue
            delta = abs(sys_row["inicio_lc_sys"] - op_row["inicio_lc"])
            if delta < best_delta:
                best_delta = delta
                best_idx = sys_idx

        # Só aceitar match se a diferença for menor que 15 min
        if best_idx is not None and best_delta <= pd.Timedelta(minutes=15):
            sys_used.add(best_idx)
            sys_row = df_sys.loc[best_idx]
            matches.append({
                "corrida": op_row["corrida"],
                "inicio_lc": op_row["inicio_lc"],
                "inicio_lc_sys": sys_row["inicio_lc_sys"],
                "delta_inicio_s": (sys_row["inicio_lc_sys"] - op_row["inicio_lc"]).total_seconds(),
                "final_lc": op_row["final_lc"],
                "final_lc_sys": sys_row["final_lc_sys"],
                "delta_fim_s": (sys_row["final_lc_sys"] - op_row["final_lc"]).total_seconds(),
                "braco_sys": sys_row.get("braco", None),
            })

    if not matches:
        print("AVISO: Nenhum match temporal encontrado entre operador e sistema (tolerância: 15 min).")
        print(f"  Corridas operador: {sorted(df_op['corrida'].tolist())}")
        print(f"  Período sistema:   {df_sys['inicio_lc_sys'].min()} a {df_sys['inicio_lc_sys'].max()}")
        print(f"  Período operador:  {df_op['inicio_lc'].min()} a {df_op['inicio_lc'].max()}")
        return pd.DataFrame()

    df_cmp_final = pd.DataFrame(matches)
    display(df_cmp_final)

    # Estatísticas resumidas
    print(f"\nMatches encontrados: {len(df_cmp_final)} de {len(df_op)} corridas do operador")
    print(f"Delta início - Média: {df_cmp_final['delta_inicio_s'].mean():.1f}s | "
          f"Mediana: {df_cmp_final['delta_inicio_s'].median():.1f}s | "
          f"Std: {df_cmp_final['delta_inicio_s'].std():.1f}s")
    print(f"Delta fim    - Média: {df_cmp_final['delta_fim_s'].mean():.1f}s | "
          f"Mediana: {df_cmp_final['delta_fim_s'].median():.1f}s | "
          f"Std: {df_cmp_final['delta_fim_s'].std():.1f}s")

    return df_cmp_final

# Exemplo de uso:
df_cmp_final = comparacao_operador(df_sys_lc, df_final)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📊 6. Comparação: Operador vs Sistema
# MAGIC
# MAGIC **Análise de Precisão:**
# MAGIC * Calcula diferenças de tempo (deltas) entre detecção automática e registros do operador
# MAGIC * Compara início e fim de cada corrida
# MAGIC * Identifica desvios e acurácia do algoritmo

# COMMAND ----------

import pandas as pd
import numpy as np


def detect_lc_events_fsm_pesobra(
    df: pd.DataFrame,

    # =========================
    # CONFIGURAÇÃO – INÍCIO
    # =========================
    SMOOTH_WINDOW_S: int = 5,

    HIGH_WEIGHT: float = 100.0,
    HIGH_LOOKBACK_S: int = 60,

    DROP_WINDOW_S: int = 5,
    DROP_TON: float = 6.0,

    DP_SEQ_S: int = 8,
    DP_EPS: float = 0.0,

    USE_REFRACTORY: bool = True,
    LOAD_JUMP_WINDOW_S: int = 5,
    LOAD_JUMP_TON: float = 6.0,
    REFRACTORY_AFTER_LOAD_S: int = 40,

    USE_PEAK_GATE: bool = True,
    PEAK_MIN_WEIGHT: float = 112.0,
    PEAK_LOOKBACK_S: int = 180,
    PEAK_AFTER_MIN_S: int = 0,
    PEAK_AFTER_MAX_S: int = 120,

    MIN_CAND_WEIGHT: float = 80.0,
    MAX_CAND_WEIGHT: float = 115.0,

    SCORE_FUTURE_WINDOW_S: int = 60,
    SCORE_DROP_WEIGHT: float = 2.0,
    SCORE_R1_BONUS: float = 3.0,
    SCORE_R2_BONUS: float = 2.0,

    CLUSTER_GAP_S: int = 180,

    # =========================
    # CONFIGURAÇÃO – FIM
    # =========================
    END_ARM_WEIGHT: float = 60.0,
    END_RISE_TON: float = 2.0,
    END_RISE_WINDOW_S: int = 20,

    # =========================
    # FLUXO
    # =========================
    initial_corrida: int = 0,
    initial_braco: int = 1,

    DEBUG: bool = True,
):
    """
    FSM sequencial (streaming-ready).

    Retorna:
      df_sys_lc   : corrida | inicio_lc_sys | final_lc_sys | braco
      df_debug    : trace completo de estados e decisões
    """

    # ======================================================
    # 0) PREPARAÇÃO
    # ======================================================
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    df["p1_raw"] = pd.to_numeric(df["ACI@LC_TORRE_PESOBRA1"], errors="coerce")
    df["p2_raw"] = pd.to_numeric(df["ACI@LC_TORRE_PESOBRA2"], errors="coerce")

    w = max(1, int(SMOOTH_WINDOW_S))
    df["p1"] = df["p1_raw"].rolling(w, min_periods=1).median()
    df["p2"] = df["p2_raw"].rolling(w, min_periods=1).median()

    df["dp1"] = df["p1"].diff()
    df["dp2"] = df["p2"].diff()

    lb = max(1, int(HIGH_LOOKBACK_S))
    df["was_high_1"] = df["p1"].rolling(lb, min_periods=1).max() >= HIGH_WEIGHT
    df["was_high_2"] = df["p2"].rolling(lb, min_periods=1).max() >= HIGH_WEIGHT

    dw = max(1, int(DROP_WINDOW_S))
    df["drop1"] = df["p1"] - df["p1"].shift(dw)
    df["drop2"] = df["p2"] - df["p2"].shift(dw)

    df["r1_1"] = df["was_high_1"] & (df["drop1"] <= -DROP_TON)
    df["r1_2"] = df["was_high_2"] & (df["drop2"] <= -DROP_TON)

    seq = max(1, int(DP_SEQ_S))
    df["fallseq1"] = (df["dp1"] < -DP_EPS).rolling(seq, min_periods=seq).sum() == seq
    df["fallseq2"] = (df["dp2"] < -DP_EPS).rolling(seq, min_periods=seq).sum() == seq

    df["r2_1"] = df["was_high_1"] & df["fallseq1"]
    df["r2_2"] = df["was_high_2"] & df["fallseq2"]

    if USE_REFRACTORY:
        jw = max(1, int(LOAD_JUMP_WINDOW_S))
        df["jump1"] = df["p1"] - df["p1"].shift(jw)
        df["jump2"] = df["p2"] - df["p2"].shift(jw)

        df["is_load1"] = df["jump1"] >= LOAD_JUMP_TON
        df["is_load2"] = df["jump2"] >= LOAD_JUMP_TON

        idx = np.arange(len(df))
        last1 = np.where(df["is_load1"], idx, np.nan)
        last2 = np.where(df["is_load2"], idx, np.nan)

        df["last_load1"] = pd.Series(last1).ffill()
        df["last_load2"] = pd.Series(last2).ffill()

        ref = max(1, int(REFRACTORY_AFTER_LOAD_S))
        df["ok_ref1"] = df["last_load1"].isna() | ((idx - df["last_load1"]) >= ref)
        df["ok_ref2"] = df["last_load2"].isna() | ((idx - df["last_load2"]) >= ref)
    else:
        df["ok_ref1"] = True
        df["ok_ref2"] = True

    df["start_any_1"] = (df["r1_1"] | df["r2_1"]) & df["ok_ref1"]
    df["start_any_2"] = (df["r1_2"] | df["r2_2"]) & df["ok_ref2"]

    df["start_any_1"] &= (df["p1"] >= MIN_CAND_WEIGHT) & (df["p1"] <= MAX_CAND_WEIGHT)
    df["start_any_2"] &= (df["p2"] >= MIN_CAND_WEIGHT) & (df["p2"] <= MAX_CAND_WEIGHT)

    # ======================================================
    # FSM
    # ======================================================
    state = "IDLE"
    corrida = int(initial_corrida)
    braco = int(initial_braco)

    cluster = []
    last_cand_ts = None

    crossed_60 = False
    p_at_60 = None
    t_at_60 = None

    results = []
    debug = []

    n = len(df)

    for i in range(n):
        row = df.iloc[i]
        ts = row["timestamp"]

        p = row["p1"] if braco == 1 else row["p2"]
        dp = row["dp1"] if braco == 1 else row["dp2"]

        # ==================================================
        # IDLE → START_ARMED
        # ==================================================
        if state == "IDLE":
            is_start = row["start_any_1"] if braco == 1 else row["start_any_2"]

            if is_start:
                cluster = [{
                    "i": i,
                    "timestamp": ts,
                    "p": p,
                }]
                last_cand_ts = ts
                state = "START_ARMED"

                debug.append({
                    "event": "START_CAND",
                    "state": state,
                    "i": i,
                    "timestamp": ts,
                    "braco": braco,
                    "p": p,
                })

            continue

        # ==================================================
        # START_ARMED
        # ==================================================
        if state == "START_ARMED":
            is_start = row["start_any_1"] if braco == 1 else row["start_any_2"]

            if is_start:
                cluster.append({
                    "i": i,
                    "timestamp": ts,
                    "p": p,
                })
                last_cand_ts = ts

            if (ts - last_cand_ts).total_seconds() > CLUSTER_GAP_S:
                best = cluster[-1]

                start_i = best["i"]
                start_ts = best["timestamp"]

                debug.append({
                    "event": "RUNNING_START",
                    "corrida": corrida,
                    "braco": braco,
                    "i": start_i,
                    "timestamp": start_ts,
                    "confirmed_at": ts,
                })

                state = "RUNNING"
                crossed_60 = False
                p_at_60 = None
                t_at_60 = None

            continue

        # ==================================================
        # RUNNING
        # ==================================================
        if state == "RUNNING":
            if not crossed_60 and pd.notna(p) and p <= END_ARM_WEIGHT:
                crossed_60 = True
                p_at_60 = p
                t_at_60 = ts

                debug.append({
                    "event": "CROSSED_60",
                    "corrida": corrida,
                    "braco": braco,
                    "i": i,
                    "timestamp": ts,
                    "p": p,
                })
                continue

            if crossed_60:
                dt = (ts - t_at_60).total_seconds()
                dp_rise = p - p_at_60

                if dp_rise >= END_RISE_TON and dt <= END_RISE_WINDOW_S:
                    results.append({
                        "corrida": corrida,
                        "inicio_lc_sys": start_ts,
                        "final_lc_sys": ts,
                        "braco": braco,
                    })

                    debug.append({
                        "event": "RUN_CLOSE",
                        "corrida": corrida,
                        "braco": braco,
                        "i": i,
                        "timestamp": ts,
                        "p": p,
                        "delta_p": dp_rise,
                        "dt_s": dt,
                    })

                    corrida += 1
                    braco = 2 if braco == 1 else 1
                    state = "IDLE"
                    cluster = []
                    crossed_60 = False

            continue

    return pd.DataFrame(results), pd.DataFrame(debug)


# COMMAND ----------

# MAGIC %md
# MAGIC #### ⚙️ 5.3. Máquina de Estados Finitos (FSM) - Versão Completa
# MAGIC
# MAGIC **Arquitetura:**
# MAGIC * **Estados:** IDLE → START_ARMED → RUNNING → IDLE
# MAGIC * **IDLE:** Aguardando candidato de início
# MAGIC * **START_ARMED:** Coletando candidatos em cluster, aguarda gap de 180s para confirmar
# MAGIC * **RUNNING:** Corrida em andamento, aguarda cruzar 60t e subir 2t para detectar fim
# MAGIC
# MAGIC **Vantagens:**
# MAGIC * Pronto para streaming (processa linha por linha)
# MAGIC * Unifica detecção de início e fim em uma única função
# MAGIC * Alterna automaticamente entre braços

# COMMAND ----------

import pandas as pd
import numpy as np


def detect_lc_events_fsm_pesobra(
    df: pd.DataFrame,

    # =========================
    # CONFIGURAÇÃO – INÍCIO
    # =========================
    SMOOTH_WINDOW_S: int = 5,

    HIGH_WEIGHT: float = 100.0,
    HIGH_LOOKBACK_S: int = 60,

    DROP_WINDOW_S: int = 5,
    DROP_TON: float = 6.0,

    DP_SEQ_S: int = 8,
    DP_EPS: float = 0.0,

    USE_REFRACTORY: bool = True,
    LOAD_JUMP_WINDOW_S: int = 5,
    LOAD_JUMP_TON: float = 6.0,
    REFRACTORY_AFTER_LOAD_S: int = 40,

    USE_PEAK_GATE: bool = True,
    PEAK_MIN_WEIGHT: float = 112.0,
    PEAK_LOOKBACK_S: int = 180,
    PEAK_AFTER_MIN_S: int = 0,
    PEAK_AFTER_MAX_S: int = 120,

    MIN_CAND_WEIGHT: float = 80.0,
    MAX_CAND_WEIGHT: float = 115.0,

    SCORE_FUTURE_WINDOW_S: int = 60,
    SCORE_DROP_WEIGHT: float = 2.0,
    SCORE_R1_BONUS: float = 3.0,
    SCORE_R2_BONUS: float = 2.0,

    CLUSTER_GAP_S: int = 180,

    # =========================
    # CONFIGURAÇÃO – FIM
    # =========================
    END_ARM_WEIGHT: float = 60.0,
    END_RISE_TON: float = 2.0,
    END_RISE_WINDOW_S: int = 20,

    # =========================
    # FLUXO
    # =========================
    initial_corrida: int = 0,
    initial_braco: int = 1,

    DEBUG: bool = True,
):
    """
    FSM sequencial (streaming-ready), tudo dentro de uma função.

    Retorna:
      df_sys_lc   : corrida | inicio_lc_sys | final_lc_sys | braco | close_reason
      df_debug    : trace completo
      df_features : df com features calculadas (pra inspecionar flags)
    """

    # ======================================================
    # 0) PREPARAÇÃO / FEATURES (iguais à lógica antiga)
    # ======================================================
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # garantir datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    # pesos numéricos
    df["peso_bra1"] = pd.to_numeric(df["ACI@LC_TORRE_PESOBRA1"], errors="coerce")
    df["peso_bra2"] = pd.to_numeric(df["ACI@LC_TORRE_PESOBRA2"], errors="coerce")

    def s2n(s: int) -> int:
        return max(1, int(s))

    # smoothing (mediana)
    w = s2n(SMOOTH_WINDOW_S)
    df["p1"] = df["peso_bra1"].rolling(w, min_periods=1).median()
    df["p2"] = df["peso_bra2"].rolling(w, min_periods=1).median()

    # esteve alto recentemente
    lb = s2n(HIGH_LOOKBACK_S)
    df["was_high_bra1"] = (df["p1"].rolling(lb, min_periods=1).max() >= HIGH_WEIGHT).fillna(False).astype(bool)
    df["was_high_bra2"] = (df["p2"].rolling(lb, min_periods=1).max() >= HIGH_WEIGHT).fillna(False).astype(bool)

    # R1: queda brusca
    dw = s2n(DROP_WINDOW_S)
    df["drop1"] = df["p1"] - df["p1"].shift(dw)
    df["drop2"] = df["p2"] - df["p2"].shift(dw)
    df["r1_bra1"] = (df["was_high_bra1"] & (df["drop1"] <= -DROP_TON).fillna(False)).fillna(False).astype(bool)
    df["r1_bra2"] = (df["was_high_bra2"] & (df["drop2"] <= -DROP_TON).fillna(False)).fillna(False).astype(bool)

    # R2: sequência de quedas
    df["dp1"] = df["p1"].diff()
    df["dp2"] = df["p2"].diff()
    seq = s2n(DP_SEQ_S)
    fall1 = ((df["dp1"] < -DP_EPS).fillna(False).rolling(seq, min_periods=seq).sum() == seq)
    fall2 = ((df["dp2"] < -DP_EPS).fillna(False).rolling(seq, min_periods=seq).sum() == seq)
    df["fallseq1"] = fall1.fillna(False).astype(bool)
    df["fallseq2"] = fall2.fillna(False).astype(bool)
    df["r2_bra1"] = (df["was_high_bra1"] & df["fallseq1"]).fillna(False).astype(bool)
    df["r2_bra2"] = (df["was_high_bra2"] & df["fallseq2"]).fillna(False).astype(bool)

    # refractory
    if USE_REFRACTORY:
        jw = s2n(LOAD_JUMP_WINDOW_S)
        df["jump1"] = df["p1"] - df["p1"].shift(jw)
        df["jump2"] = df["p2"] - df["p2"].shift(jw)

        df["is_load1"] = (df["jump1"] >= LOAD_JUMP_TON).fillna(False).astype(bool)
        df["is_load2"] = (df["jump2"] >= LOAD_JUMP_TON).fillna(False).astype(bool)

        idx = np.arange(len(df))
        last1 = np.where(df["is_load1"].to_numpy(), idx, np.nan)
        last2 = np.where(df["is_load2"].to_numpy(), idx, np.nan)
        df["last_load_idx1"] = pd.Series(last1).ffill()
        df["last_load_idx2"] = pd.Series(last2).ffill()

        ref = s2n(REFRACTORY_AFTER_LOAD_S)
        df["ok_ref1"] = (df["last_load_idx1"].isna() | ((idx - df["last_load_idx1"]) >= ref)).fillna(False).astype(bool)
        df["ok_ref2"] = (df["last_load_idx2"].isna() | ((idx - df["last_load_idx2"]) >= ref)).fillna(False).astype(bool)

        df["r1_bra1"] = (df["r1_bra1"] & df["ok_ref1"]).fillna(False).astype(bool)
        df["r1_bra2"] = (df["r1_bra2"] & df["ok_ref2"]).fillna(False).astype(bool)
        df["r2_bra1"] = (df["r2_bra1"] & df["ok_ref1"]).fillna(False).astype(bool)
        df["r2_bra2"] = (df["r2_bra2"] & df["ok_ref2"]).fillna(False).astype(bool)
    else:
        df["ok_ref1"] = True
        df["ok_ref2"] = True

    # start any
    df["start_any_bra1"] = (df["r1_bra1"] | df["r2_bra1"]).fillna(False).astype(bool)
    df["start_any_bra2"] = (df["r1_bra2"] | df["r2_bra2"]).fillna(False).astype(bool)

    # faixa do candidato
    df["cand_range1"] = ((df["p1"] >= MIN_CAND_WEIGHT) & (df["p1"] <= MAX_CAND_WEIGHT)).fillna(False).astype(bool)
    df["cand_range2"] = ((df["p2"] >= MIN_CAND_WEIGHT) & (df["p2"] <= MAX_CAND_WEIGHT)).fillna(False).astype(bool)
    df["start_any_bra1"] = (df["start_any_bra1"] & df["cand_range1"]).fillna(False).astype(bool)
    df["start_any_bra2"] = (df["start_any_bra2"] & df["cand_range2"]).fillna(False).astype(bool)

    # peak gate
    if USE_PEAK_GATE:
        idx = np.arange(len(df))
        df["is_peak1"] = (df["p1"] >= PEAK_MIN_WEIGHT).fillna(False).astype(bool)
        df["is_peak2"] = (df["p2"] >= PEAK_MIN_WEIGHT).fillna(False).astype(bool)

        lastp1 = np.where(df["is_peak1"].to_numpy(), idx, np.nan)
        lastp2 = np.where(df["is_peak2"].to_numpy(), idx, np.nan)
        df["last_peak_idx1"] = pd.Series(lastp1).ffill()
        df["last_peak_idx2"] = pd.Series(lastp2).ffill()

        look = s2n(PEAK_LOOKBACK_S)
        df["has_peak_recent1"] = (df["is_peak1"].rolling(look, min_periods=1).max() > 0).fillna(False).astype(bool)
        df["has_peak_recent2"] = (df["is_peak2"].rolling(look, min_periods=1).max() > 0).fillna(False).astype(bool)

        amin = s2n(PEAK_AFTER_MIN_S)
        amax = s2n(PEAK_AFTER_MAX_S)

        df["after_peak1"] = (
            df["last_peak_idx1"].notna() &
            ((idx - df["last_peak_idx1"]) >= amin) &
            ((idx - df["last_peak_idx1"]) <= amax)
        ).fillna(False).astype(bool)

        df["after_peak2"] = (
            df["last_peak_idx2"].notna() &
            ((idx - df["last_peak_idx2"]) >= amin) &
            ((idx - df["last_peak_idx2"]) <= amax)
        ).fillna(False).astype(bool)

        df["gate1"] = (df["has_peak_recent1"] & df["after_peak1"]).fillna(False).astype(bool)
        df["gate2"] = (df["has_peak_recent2"] & df["after_peak2"]).fillna(False).astype(bool)

        df["start_any_bra1"] = (df["start_any_bra1"] & df["gate1"]).fillna(False).astype(bool)
        df["start_any_bra2"] = (df["start_any_bra2"] & df["gate2"]).fillna(False).astype(bool)
    else:
        df["gate1"] = True
        df["gate2"] = True

    # EDGE (flanco)
    prev1 = df["start_any_bra1"].shift(1).fillna(False).astype(bool)
    prev2 = df["start_any_bra2"].shift(1).fillna(False).astype(bool)
    df["edge_bra1"] = (df["start_any_bra1"] & (~prev1)).fillna(False).astype(bool)
    df["edge_bra2"] = (df["start_any_bra2"] & (~prev2)).fillna(False).astype(bool)

    # SCORE (queda futura)
    fw = s2n(SCORE_FUTURE_WINDOW_S)
    fmin1 = df["p1"].shift(-1)[::-1].rolling(fw, min_periods=1).min()[::-1]
    fmin2 = df["p2"].shift(-1)[::-1].rolling(fw, min_periods=1).min()[::-1]
    df["future_drop1"] = (df["p1"] - fmin1)
    df["future_drop2"] = (df["p2"] - fmin2)

    df["score1"] = (
        SCORE_DROP_WEIGHT * df["future_drop1"].fillna(0)
        + SCORE_R1_BONUS * df["r1_bra1"].astype(int)
        + SCORE_R2_BONUS * df["r2_bra1"].astype(int)
    )
    df["score2"] = (
        SCORE_DROP_WEIGHT * df["future_drop2"].fillna(0)
        + SCORE_R1_BONUS * df["r1_bra2"].astype(int)
        + SCORE_R2_BONUS * df["r2_bra2"].astype(int)
    )

    # ======================================================
    # 1) FSM
    # ======================================================
    state = "IDLE"
    corrida = int(initial_corrida)
    braco = int(initial_braco)

    # cluster: lista de candidatos (i, ts, score, p, r1, r2)
    cluster = []
    last_cand_ts = None

    # running context
    running_start_ts = None
    running_start_i = None

    crossed_60 = False
    p_ref = None
    t_ref = None

    results = []
    debug_rows = []
    events = []   # tabela append-only de estados da corrida

    

    def dbg(event: str, **kwargs):
        if not DEBUG:
            return
        row = {"event": event, "state": state}
        row.update(kwargs)
        debug_rows.append(row)

    n = len(df)

    for i in range(n):
        ts = df.at[i, "timestamp"]

        # valores do braço atual (para fim)
        p_run = df.at[i, "p1"] if braco == 1 else df.at[i, "p2"]

        # flags de EDGE por braço (para início)
        edge1 = bool(df.at[i, "edge_bra1"])
        edge2 = bool(df.at[i, "edge_bra2"])

        # ==================================================
        # IDLE
        # ==================================================
        if state == "IDLE":
            is_edge = edge1 if braco == 1 else edge2

            if is_edge:
                score = float(df.at[i, "score1"]) if braco == 1 else float(df.at[i, "score2"])
                p_here = float(df.at[i, "p1"]) if braco == 1 else float(df.at[i, "p2"])
                r1 = bool(df.at[i, "r1_bra1"]) if braco == 1 else bool(df.at[i, "r1_bra2"])
                r2 = bool(df.at[i, "r2_bra1"]) if braco == 1 else bool(df.at[i, "r2_bra2"])

                cluster = [{
                    "i": int(i),
                    "timestamp": ts,
                    "score": score,
                    "p": p_here,
                    "r1": r1,
                    "r2": r2,
                }]
                last_cand_ts = ts
                state = "START_ARMED"

                dbg("CLUSTER_OPEN", i=i, timestamp=ts, braco=braco, p=p_here, score=score)

            continue

        # ==================================================
        # START_ARMED (acumula edges e fecha cluster por gap)
        # ==================================================
        if state == "START_ARMED":
            is_edge = edge1 if braco == 1 else edge2

            if is_edge:
                score = float(df.at[i, "score1"]) if braco == 1 else float(df.at[i, "score2"])
                p_here = float(df.at[i, "p1"]) if braco == 1 else float(df.at[i, "p2"])
                r1 = bool(df.at[i, "r1_bra1"]) if braco == 1 else bool(df.at[i, "r1_bra2"])
                r2 = bool(df.at[i, "r2_bra1"]) if braco == 1 else bool(df.at[i, "r2_bra2"])

                cluster.append({
                    "i": int(i),
                    "timestamp": ts,
                    "score": score,
                    "p": p_here,
                    "r1": r1,
                    "r2": r2,
                })
                last_cand_ts = ts

                dbg("EDGE_CAND", i=i, timestamp=ts, braco=braco, p=p_here, score=score)

            # fecha cluster ao atingir gap
            if last_cand_ts is not None and (ts - last_cand_ts).total_seconds() > CLUSTER_GAP_S:
                # melhor score; empate -> timestamp mais tarde
                best = sorted(cluster, key=lambda x: (x["score"], x["timestamp"]))[-1]

                running_start_i = int(best["i"])
                running_start_ts = best["timestamp"]

                dbg("RUNNING_START", corrida=corrida, braco=braco, i=running_start_i, timestamp=running_start_ts, confirmed_at=ts)
                events.append({
                    "corrida": corrida,
                    "status": "RUNNING",
                    "inicio_lc_sys": running_start_ts,
                    "final_lc_sys": None,
                    "braco": braco,
                    "event_ts": ts,   # quando o sistema confirmou o início
                })


                # muda para RUNNING
                state = "RUNNING"
                crossed_60 = False
                p_ref = None
                t_ref = None

            continue

        # ==================================================
        # RUNNING (fim simples: cruzou 60, subiu 2t em 20s)
        # ==================================================
        if state == "RUNNING":
            if pd.isna(p_run):
                continue

            # cruzou 60
            if (not crossed_60) and (p_run <= END_ARM_WEIGHT):
                crossed_60 = True
                p_ref = float(p_run)
                t_ref = ts
                dbg(
                    "CROSSED_60",
                    corrida=corrida,
                    braco=braco,
                    i=i,
                    timestamp=ts,
                    p=p_ref,
                    armed_at=ts,
                )
                continue

            if crossed_60:
                old_p_ref = p_ref
                old_t_ref = t_ref

                dt = (ts - t_ref).total_seconds()
                dp = float(p_run) - float(p_ref)

                dbg(
                    "END_EVAL",
                    corrida=corrida,
                    braco=braco,
                    i=i,
                    timestamp=ts,
                    p=float(p_run),
                    p_ref=p_ref,
                    delta_p=dp,
                    dt_s=dt,
                    within_window=(dt <= END_RISE_WINDOW_S),
                    enough_rise=(dp >= END_RISE_TON),
                )


                # condição do pico dentro da janela
                if (dt <= END_RISE_WINDOW_S) and (dp >= END_RISE_TON):
                    end_ts = ts

                    results.append({
                        "corrida": corrida,
                        "inicio_lc_sys": running_start_ts,
                        "final_lc_sys": end_ts,
                        "braco": braco,
                        "close_reason": "rise_after_60",
                    })

                    events.append({
                        "corrida": corrida,
                        "status": "CLOSED",
                        "inicio_lc_sys": running_start_ts,
                        "final_lc_sys": end_ts,
                        "braco": braco,
                        "event_ts": end_ts,
                    })


                    dbg("RUN_CLOSE", corrida=corrida, braco=braco, i=i, timestamp=end_ts, p=float(p_run), delta_p=dp, dt_s=dt)

                    # avança corrida
                    corrida += 1
                    braco = 2 if braco == 1 else 1
                    state = "IDLE"
                    cluster = []
                    last_cand_ts = None

                    running_start_ts = None
                    running_start_i = None

                    crossed_60 = False
                    p_ref = None
                    t_ref = None
                    continue

                # passou janela: atualiza referência (sliding reference)
                if dt > END_RISE_WINDOW_S:
                    p_ref = float(p_run)
                    t_ref = ts
                    dbg(
                        "END_REF_UPDATE",
                        corrida=corrida,
                        braco=braco,
                        i=i,
                        timestamp=ts,
                        p=float(p_run),
                        p_ref_before=old_p_ref,
                        p_ref_after=p_ref,
                        dt_s=(ts - t_ref).total_seconds(),
                    )


            continue

    # ======================================================
    # 2) RETORNOS — sempre com colunas fixas
    # ======================================================
    df_sys_lc_final = pd.DataFrame(
        results,
        columns=["corrida", "inicio_lc_sys", "final_lc_sys", "braco", "close_reason"]
    )

    df_sys_lc_events = pd.DataFrame(
        events,
        columns=["corrida", "status", "inicio_lc_sys", "final_lc_sys", "braco", "event_ts"]
    )

    df_debug = pd.DataFrame(debug_rows)

    return df_sys_lc_final, df_sys_lc_events, df_debug, df




# COMMAND ----------

df_sys_lc1, df_sys_lc_events, df_debug, df_features = detect_lc_events_fsm_pesobra(
    df=df_lc_plot,

    SMOOTH_WINDOW_S=5,
    HIGH_WEIGHT=100.0,
    HIGH_LOOKBACK_S=60,

    DROP_WINDOW_S=8,
    DROP_TON=2.0,          # <-- use seu valor calibrado

    DP_SEQ_S=5,
    DP_EPS=0.05,            # <-- use seu valor calibrado

    USE_REFRACTORY=True,
    LOAD_JUMP_WINDOW_S=5,
    LOAD_JUMP_TON=6.0,
    REFRACTORY_AFTER_LOAD_S=80,

    USE_PEAK_GATE=False,   # <-- comece assim pra validar, depois liga
    PEAK_MIN_WEIGHT=108.0,
    PEAK_LOOKBACK_S=60,
    PEAK_AFTER_MIN_S=0,
    PEAK_AFTER_MAX_S=60,

    MIN_CAND_WEIGHT=80.0,
    MAX_CAND_WEIGHT=115.0,

    SCORE_FUTURE_WINDOW_S=10,
    SCORE_DROP_WEIGHT=2.0,
    SCORE_R1_BONUS=3.0,
    SCORE_R2_BONUS=2.0,

    CLUSTER_GAP_S=180,

    END_ARM_WEIGHT=60.0,
    END_RISE_TON=2.0,
    END_RISE_WINDOW_S=25,

    initial_corrida=130268,
    initial_braco=1,

    DEBUG=True,
)



# COMMAND ----------

print("df_sys_lc1:", df_sys_lc1.shape)
print("df_debug:", df_debug.shape)

if not df_sys_lc1.empty:
    display(df_sys_lc1)
else:
    print("df_sys_lc1 vazio (nenhuma corrida fechada)")

if not df_debug.empty:
    display(df_debug)
else:
    print("df_debug vazio")

if not df_sys_lc_events.empty:
    display(df_sys_lc_events)
else:
    print("df_sys_lc_events vazio")


# COMMAND ----------

df_cmp_final = comparacao_operador(df_sys_lc1, df_final)


# COMMAND ----------

# DBTITLE 1,Plot Comparação Operador vs Sistema (df_cmp_final)
# ============================================================
# PLOT: OPERADOR vs SISTEMA — a partir de df_cmp_final
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# --- helper: remover timezone ---
def tz_strip(s):
    if hasattr(s, 'dt') and s.dt.tz is not None:
        return s.dt.tz_localize(None)
    return s

# Copiar df_lc_plot e garantir timestamp limpo
df_plot = df_lc_plot.copy()
df_plot["timestamp"] = tz_strip(df_plot["timestamp"])
df_plot["peso_bra1"] = pd.to_numeric(df_plot["ACI@LC_TORRE_PESOBRA1"], errors="coerce")
df_plot["peso_bra2"] = pd.to_numeric(df_plot["ACI@LC_TORRE_PESOBRA2"], errors="coerce")

# Copiar df_cmp_final e limpar tz
cmp = df_cmp_final.copy()
for c in ["inicio_lc", "inicio_lc_sys", "final_lc", "final_lc_sys"]:
    if c in cmp.columns:
        cmp[c] = pd.to_datetime(cmp[c], errors="coerce")
        cmp[c] = tz_strip(cmp[c])

# ============================================================
# PLOT
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)

# --- SUBPLOT 1: PESOBRA1 ---
ax1.plot(df_plot["timestamp"], df_plot["peso_bra1"],
         label="PESOBRA1", linewidth=1.5, alpha=0.8, color="blue")

# --- SUBPLOT 2: PESOBRA2 ---
ax2.plot(df_plot["timestamp"], df_plot["peso_bra2"],
         label="PESOBRA2", linewidth=1.5, alpha=0.8, color="orange")

# --- Marcadores a partir de df_cmp_final ---
first_op = True
first_sys = True

for _, row in cmp.iterrows():
    corrida = int(row["corrida"])
    braco = int(row["braco_sys"])
    ax_bra = ax1 if braco == 1 else ax2

    # --- OPERADOR (tracejado) ---
    if pd.notna(row.get("inicio_lc")):
        ax_bra.axvline(row["inicio_lc"], color="green", linestyle="--", linewidth=2, alpha=0.6,
                       label="Início OP" if first_op else "")
        ax_bra.text(row["inicio_lc"], ax_bra.get_ylim()[1] * 0.95,
                    f"OP In\n{corrida}", rotation=0, color="green",
                    ha="left", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="green", alpha=0.7))
    if pd.notna(row.get("final_lc")):
        ax_bra.axvline(row["final_lc"], color="red", linestyle="--", linewidth=2, alpha=0.6,
                       label="Fim OP" if first_op else "")
        ax_bra.text(row["final_lc"], ax_bra.get_ylim()[1] * 0.85,
                    f"OP Fim\n{corrida}", rotation=0, color="red",
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="red", alpha=0.7))

    # --- SISTEMA (sólido) ---
    if pd.notna(row.get("inicio_lc_sys")):
        ax_bra.axvline(row["inicio_lc_sys"], color="darkgreen", linestyle="-", linewidth=2, alpha=0.8,
                       label="Início SYS" if first_sys else "")
        delta_i = row.get("delta_inicio_s", "")
        ax_bra.text(row["inicio_lc_sys"], ax_bra.get_ylim()[1] * 0.75,
                    f"SYS In\n{corrida}\n({delta_i:.0f}s)" if isinstance(delta_i, (int, float)) else f"SYS In\n{corrida}",
                    rotation=0, color="darkgreen",
                    ha="left", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="darkgreen", alpha=0.7))
    if pd.notna(row.get("final_lc_sys")):
        ax_bra.axvline(row["final_lc_sys"], color="darkred", linestyle="-", linewidth=2, alpha=0.8,
                       label="Fim SYS" if first_sys else "")
        delta_f = row.get("delta_fim_s", "")
        ax_bra.text(row["final_lc_sys"], ax_bra.get_ylim()[1] * 0.65,
                    f"SYS Fim\n{corrida}\n({delta_f:.0f}s)" if isinstance(delta_f, (int, float)) else f"SYS Fim\n{corrida}",
                    rotation=0, color="darkred",
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="darkred", alpha=0.7))

    first_op = False
    first_sys = False

# --- Formatação ---
ax1.set_title("LC - PESOBRA1 (Braço 1) — Operador vs Sistema", fontsize=14, fontweight="bold")
ax1.set_ylabel("Peso (t)", fontsize=12)
ax1.grid(True, alpha=0.3, linestyle=":")
ax1.legend(loc="upper left", fontsize=10)

ax2.set_title("LC - PESOBRA2 (Braço 2) — Operador vs Sistema", fontsize=14, fontweight="bold")
ax2.set_xlabel("Tempo", fontsize=12)
ax2.set_ylabel("Peso (t)", fontsize=12)
ax2.grid(True, alpha=0.3, linestyle=":")
ax2.legend(loc="upper left", fontsize=10)

locator = mdates.AutoDateLocator()
ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
plt.xticks(rotation=45, ha="right")

plt.suptitle(
    f"Comparação: Operador (tracejado) vs Sistema (sólido) — df_cmp_final\nPeríodo: {START_TS} até {END_TS}",
    fontsize=16, fontweight="bold", y=0.995
)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("LEGENDA:")
print("="*80)
print("  TRACEJADAS (--) verde/vermelha = Operador (início/fim)")
print("  SÓLIDAS (─) darkgreen/darkred  = Sistema (início/fim)")
print("  Deltas em segundos entre parênteses (negativo = sys antecipou)")
print("="*80)

# COMMAND ----------

# DBTITLE 1,Teste: import dos .py (lc_pesobra_logic + comparacao_operador)
# ============================================================
# TESTE: importar e rodar a partir dos módulos .py
# ============================================================
import sys
import importlib

# Adicionar pasta do Sincronizador ao path
module_dir = "/Workspace/Users/execs1@ex.gerdau.com/Sincronizador"
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

# Forçar reload (caso já tenha sido importado antes)
import lc_pesobra_logic
importlib.reload(lc_pesobra_logic)
from lc_pesobra_logic import detect_lc_events_fsm_pesobra as detect_fsm_py

import comparacao_operador as _cmp_mod
importlib.reload(_cmp_mod)
from comparacao_operador import comparacao_operador as comparacao_py

print("✅ Imports OK")
print(f"   lc_pesobra_logic:    {lc_pesobra_logic.__file__}")
print(f"   comparacao_operador: {_cmp_mod.__file__}")

# ============================================================
# Rodar FSM importada (parâmetros calibrados)
# ============================================================
df_sys_py, df_events_py, df_debug_py, df_feat_py = detect_fsm_py(
    df=df_lc_plot,

    # --- Início (calibrados) ---
    SMOOTH_WINDOW_S=5,
    HIGH_WEIGHT=100.0,
    HIGH_LOOKBACK_S=60,
    DROP_WINDOW_S=8,
    DROP_TON=2.0,
    DP_SEQ_S=5,
    DP_EPS=0.05,
    USE_REFRACTORY=True,
    LOAD_JUMP_WINDOW_S=5,
    LOAD_JUMP_TON=6.0,
    REFRACTORY_AFTER_LOAD_S=80,
    USE_PEAK_GATE=False,
    PEAK_MIN_WEIGHT=108.0,
    PEAK_LOOKBACK_S=60,
    PEAK_AFTER_MIN_S=0,
    PEAK_AFTER_MAX_S=60,
    MIN_CAND_WEIGHT=80.0,
    MAX_CAND_WEIGHT=115.0,
    SCORE_FUTURE_WINDOW_S=10,
    SCORE_DROP_WEIGHT=2.0,
    SCORE_R1_BONUS=3.0,
    SCORE_R2_BONUS=2.0,
    CLUSTER_GAP_S=180,

    # --- Fim (calibrados) ---
    END_ARM_WEIGHT=60.0,
    END_RISE_TON=2.0,
    END_RISE_WINDOW_S=25,

    # --- Fluxo ---
    initial_corrida=130268,
    initial_braco=1,
    DEBUG=True,
)

print(f"\n📊 FSM (.py): {len(df_sys_py)} corridas detectadas")
display(df_sys_py)

print(f"\n📊 Events (.py): {len(df_events_py)} eventos")
display(df_events_py)

# ============================================================
# Rodar comparação importada
# ============================================================
df_cmp_py = comparacao_py(df_sys_py, df_final)
display(df_cmp_py)

# ============================================================
# Validar: resultado do .py == resultado do notebook?
# ============================================================
print("\n" + "="*80)
print("VALIDAÇÃO: .py vs notebook")
print("="*80)

if 'df_sys_lc1' in dir():
    match_count = len(df_sys_py) == len(df_sys_lc1)
    print(f"  Corridas notebook: {len(df_sys_lc1)}")
    print(f"  Corridas .py:      {len(df_sys_py)}")
    print(f"  Match quantidade:  {'✅' if match_count else '❌'}")

    if match_count and not df_sys_py.empty:
        # Comparar timestamps
        for col in ['inicio_lc_sys', 'final_lc_sys']:
            nb_vals = df_sys_lc1[col].astype(str).tolist()
            py_vals = df_sys_py[col].astype(str).tolist()
            ok = nb_vals == py_vals
            print(f"  {col} match: {'✅' if ok else '❌'}")
            if not ok:
                print(f"    notebook: {nb_vals}")
                print(f"    .py:      {py_vals}")
else:
    print("  (df_sys_lc1 não encontrado - rode as células 41-43 primeiro)")