# Databricks notebook source
# DBTITLE 0,TAGS INTRA LC
# MAGIC %md
# MAGIC # TAGS Intra LC — Análise de Tempos do Lingotamento Contínuo
# MAGIC
# MAGIC **Objetivo**: Detectar eventos de chegada na torre e ciclos de peso real via FSM,  
# MAGIC emparelhar em corridas, filtrar paradas/outliers e analisar distribuição dos tempos operacionais.
# MAGIC
# MAGIC | Fonte | Tabela | Tags |
# MAGIC |-------|--------|------|
# MAGIC | OPC | `tb_aciaria_lc` | PESOREAL, DISTRIB_PESO, NUMCORR, VEL veios |
# MAGIC | CSV | `csv_concat` | PESOBRA1, PESOBRA2 |
# MAGIC
# MAGIC **Detecções** (em `src/detection/`):
# MAGIC - `detect_lc_chegada_torre`: FSM queda de peso nos braços (PESOBRA1/2)
# MAGIC - `detect_lc_peso_real`: FSM início/fim por subida + estabilização (PESOREAL)
# MAGIC
# MAGIC **Análise** (em `src/analysis/`):
# MAGIC - `emparelhar_corridas`: associa chegada → início real → fim real
# MAGIC - `filtrar_corridas_padrao`: timeout 90min + IQR 1.5×
# MAGIC - `print_stats_tempos` / `plot_distribuicao_tempos`: estatísticas e visualização

# COMMAND ----------

# DBTITLE 0,Carga de dados — OPC e CSV
# ================================================================
# CONFIGURAÇÃO E CARGA DE DADOS
# ================================================================
import sys, importlib
sys.path.insert(0, "/Workspace/Users/execs1@ex.gerdau.com/Sincronizador")

import pyspark.sql.functions as F
import pandas as pd
import numpy as np

# Recarrega módulos (dev)
for mod_name in list(sys.modules):
    if mod_name.startswith("src."):
        del sys.modules[mod_name]

from src.config import TABLE_OPC_LC, TABLE_CSV, UTC_OFFSET_HOURS
from src.data.read import read_opc_table, read_csv_pesobra

# --- Período de análise ---
MES_FILTRO = "02.2026"   # Formato mm.yyyy do CSV
DATA_INI   = "2026-02-01" # Formato yyyy-mm-dd do OPC
DATA_FIM   = "2026-03-01"

# 1) OPC (PESOREAL + tags LC) — já converte UTC→3
print(f"📅 Carregando OPC: {DATA_INI} → {DATA_FIM}")
df_opc = (
    read_opc_table(spark, "lc", convert_utc=True, cast_values=True)
    .filter(F.col("timestamp") >= DATA_INI)
    .filter(F.col("timestamp") < DATA_FIM)
    .orderBy("timestamp")
)
pdf_opc = df_opc.toPandas()
if pdf_opc["timestamp"].dt.tz is not None:
    pdf_opc["timestamp"] = pdf_opc["timestamp"].dt.tz_localize(None)
print(f"   ✅ {len(pdf_opc):,} registros | {pdf_opc['timestamp'].dt.date.nunique()} dias")

# 2) CSV (PESOBRA1/2)
print(f"📅 Carregando CSV PESOBRA: {MES_FILTRO}")
pdf_csv = read_csv_pesobra(spark, mes_filtro=MES_FILTRO)
print(f"   ✅ {len(pdf_csv):,} registros | {pdf_csv['timestamp'].dt.date.nunique()} dias")
print(f"   Período: {pdf_csv['timestamp'].min()} → {pdf_csv['timestamp'].max()}")

# COMMAND ----------

# DBTITLE 0,Detecção e emparelhamento
# ================================================================
# DETECÇÃO FSM + EMPARELHAMENTO
# ================================================================
from src.detection import detect_lc_chegada_torre, detect_lc_peso_real
from src.analysis import emparelhar_corridas

# 1) Chegada na torre (PESOBRA1/2)
df_bra_sys, _, _ = detect_lc_chegada_torre(df=pdf_csv, DEBUG=False)
print(f"🏗️  Chegadas: {len(df_bra_sys)} "
      f"(B1: {(df_bra_sys['braco']==1).sum()}, B2: {(df_bra_sys['braco']==2).sum()})")

# 2) Peso real (PESOREAL)
df_real_sys, _, _ = detect_lc_peso_real(
    df=pdf_opc[["timestamp", "ACI@LC_TORRE_PESOREAL"]].copy(), DEBUG=False
)
print(f"⚖️  Ciclos peso real: {len(df_real_sys)}")

# 3) Emparelhamento
df_corridas = emparelhar_corridas(df_bra_sys, df_real_sys)

# COMMAND ----------

# DBTITLE 0,Filtragem e análise de tempos padrão
# ================================================================
# FILTRAGEM + ESTATÍSTICAS + PLOTS
# ================================================================
from src.analysis import (
    filtrar_corridas_padrao,
    print_stats_tempos,
    plot_distribuicao_tempos,
)

TIMEOUT_MIN = 90  # acima = parada / troca distribuidor

df_std, df_paradas, resumo = filtrar_corridas_padrao(
    df_corridas, timeout_min=TIMEOUT_MIN
)

print_stats_tempos(df_std)
plot_distribuicao_tempos(df_std, df_all=df_corridas)

# COMMAND ----------

# DBTITLE 1,Exemplo 7h — sinais + eventos detectados
# ================================================================
# PLOT EXEMPLO: JANELA DE OPERACAO COM EVENTOS DETECTADOS
# Altere DIA / H_INI / H_FIM nos widgets acima
# ================================================================
from src.plots import plot_janela_lc

plot_janela_lc(
    pdf_csv=pdf_csv,
    pdf_opc=pdf_opc,
    df_bra_sys=df_bra_sys,
    df_real_sys=df_real_sys,
    dia=dbutils.widgets.get("DIA"),
    h_ini=dbutils.widgets.get("H_INI"),
    h_fim=dbutils.widgets.get("H_FIM"),
)