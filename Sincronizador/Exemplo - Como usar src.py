# Databricks notebook source
# DBTITLE 0,Como usar os módulos src/
# MAGIC %md
# MAGIC # Como usar os módulos `src/`
# MAGIC
# MAGIC Este notebook mostra como importar e usar as funções centralizadas.
# MAGIC
# MAGIC **Regra:** Rode a **Célula 1 (Setup)** primeiro, depois qualquer célula abaixo.

# COMMAND ----------

# DBTITLE 0,1. Setup (rodar primeiro!)
# ============================================================
# CÉLULA 1 — SETUP (rodar sempre primeiro)
# ============================================================
import sys

# Adiciona o Sincronizador ao path para que 'src' seja encontrado
SINC_PATH = "/Workspace/Users/execs1@ex.gerdau.com/Sincronizador"
if SINC_PATH not in sys.path:
    sys.path.insert(0, SINC_PATH)

print("✅ Path configurado. Agora pode importar de src.*")

# COMMAND ----------

# DBTITLE 0,2. Ler tabela unificada (df_final)
# ============================================================
# EXEMPLO: Construir df_final com UMA linha
# ============================================================
from src.data.process import build_df_final

df_final = build_df_final(spark)

print(f"df_final: {df_final.count():,} corridas")
display(df_final.orderBy("corrida", ascending=False).limit(10))

# COMMAND ----------

# DBTITLE 0,3. Ler tabela OPC (sensores)
# ============================================================
# EXEMPLO: Ler OPC do LC (já com UTC-3 e colunas double)
# ============================================================
from src.data.read import read_opc_table

df_opc_lc = read_opc_table(spark, stage="lc")   # 'lc', 'vd', ou 'fp'

print(f"Colunas: {df_opc_lc.columns}")
display(df_opc_lc.limit(5))

# COMMAND ----------

# DBTITLE 0,4. Ler PIMS e preparar para plot
# ============================================================
# EXEMPLO: Preparar dados LC para plotagem
# ============================================================
from src.data.read import read_pims_hist
from src.data.process import prepare_lc_plot_data

START_TS = "2025-11-20 10:00:00"
END_TS   = "2025-11-20 15:59:59"

df_pims = read_pims_hist(spark)
df_lc_plot, lc_columns = prepare_lc_plot_data(df_pims, START_TS, END_TS)

print(f"Pronto para plotar: {len(df_lc_plot)} linhas, {len(lc_columns)} colunas LC")
print(f"Colunas: {lc_columns}")

# COMMAND ----------

# DBTITLE 0,5. Rodar detecção LC + comparar com operador
# ============================================================
# EXEMPLO: Detecção LC completa
# ============================================================
from src.detection.lc_logic import detect_lc_events_fsm_pesobra
from src.comparison.operador import comparacao_operador
from pyspark.sql import functions as F
import pandas as pd

# 1) Rodar FSM
df_sys, df_events, df_debug, df_feat = detect_lc_events_fsm_pesobra(
    df=df_lc_plot,
    initial_corrida=130267,
    initial_braco=1,
)
print(f"Corridas detectadas: {len(df_sys)}")
display(df_sys)

# 2) Comparar com operador
df_op = df_final.filter(F.col("inicio_lc").between(START_TS, END_TS))
df_cmp = comparacao_operador(df_sys, df_op)
display(df_cmp)

# COMMAND ----------

# DBTITLE 0,6. Acessar config diretamente
# ============================================================
# EXEMPLO: Ver todas as tabelas do config
# ============================================================
from src import config

print("TABELAS MES:")
print(f"  FEA: {config.TABLE_FEA_CORRIDAS}")
print(f"  FP:  {config.TABLE_FP_CORRIDA}")
print(f"  VD:  {config.TABLE_VD_CORRIDA_TEMPOS}")
print(f"  LC:  {config.TABLE_LC_CORRIDALINGOTAMENTO}")

print(f"\nTABELAS OPC:")
print(f"  LC: {config.TABLE_OPC_LC}")
print(f"  VD: {config.TABLE_OPC_VD}")
print(f"  FP: {config.TABLE_OPC_FP}")

print(f"\nDELAYS: FEA→FP={config.DELAY_FEA_FP_MIN}min | FP→VD={config.DELAY_FP_VD_MIN}min | VD→LC={config.DELAY_VD_LC_MIN}min")