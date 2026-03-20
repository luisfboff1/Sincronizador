# Databricks notebook source
# DBTITLE 1,Imports e Configuração
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

# Configuração
TIMEZONE = "America/Sao_Paulo"
TABLE_CSV = "industrial_composicaoquimicacha_refined_dev.csv_valid.csv_concat"
DIA_VALIDACAO = "20.02.2026"  # Dia para validação

# ================================================================
# MAPEAMENTO DE COLUNAS POR EQUIPAMENTO
# ================================================================

# ---------- FEA ----------
FEA_COLS = [
    # Energia e arcos
    "ACI@FEA_ELET_ENERGIA",
    "ACI@FEA_GERAL_INCLINOM",
    "ACI@FEA_TEMPO_VAZAMENTO",
    "ARCOS_CORRENTE_ELETRODO1",
    "ARCOS_CORRENTE_ELETRODO2",
    "ARCOS_CORRENTE_ELETRODO3",
    "ARCOS_POSICAO_ELETRODO1",
    "ARCOS_POSICAO_ELETRODO2",
    "ARCOS_POSICAO_ELETRODO3",
    # Peso
    "PESO_CARRO_PANELA",
    "PESO_VAZADO",
    # Digitais (botões, válvulas, abóbada)
    "BTN_VAZAMENTO_ABRE_VALV",
    "BTN_VAZAMENTO_FECHA_VALV",
    "BTN_PREPARAR_VAZAR",
    "LMP_PREPANDO_VAZAR",
    "VALV_SOBE_ABOBADA",
    "VALV_DESCE_ABOBADA",
    "ABOBADA_ALTA",
    "ABOBADA_BAIXA",
]

# ---------- LC ----------
# Colunas do CSV
LC_COLS_CSV = [
    "ACI@LC_TORRE_PESOBRA1",
    "ACI@LC_TORRE_PESOBRA2",
    "CMD_ABRE_VALV_GAVETA",
    "CMD_FECHA_VALV_GAVETA",
]

# Colunas do OPC (tb_aciaria_lc) - serão adicionadas via merge
LC_COLS_OPC = [
    "ACI@LC_TORRE_PESOREAL",     # Peso total da torre
    "ACI@LC_DISTRIB_PESO",       # Peso do distribuidor
    "ACI@LC_DISTRIB_TEMP",       # Temperatura do distribuidor
    "ACI@LC_GERAL_NUMCORR",      # Número da corrida (importante!)
    "ACI@LC_EXTEND_VEL2V1",      # Velocidade veio 1
    "ACI@LC_EXTEND_VEL2V2",      # Velocidade veio 2
    "ACI@LC_EXTEND_VEL2V3",      # Velocidade veio 3
]

# Lista combinada para plots
LC_COLS = LC_COLS_CSV + LC_COLS_OPC

print("✓ Imports e configuração carregados")
print(f"  FEA_COLS: {len(FEA_COLS)} tags")
print(f"  LC_COLS:  {len(LC_COLS)} tags ({len(LC_COLS_CSV)} CSV + {len(LC_COLS_OPC)} OPC)")

# COMMAND ----------

# DBTITLE 1,Leitura e parsing dos dados CSV para o dia 20/02
# Lê a tabela e cria uma ordem estável das linhas
df_raw = (
    spark.read.table(TABLE_CSV)
    .withColumn("_row_id", F.monotonically_increasing_id())
)

# Descobre o nome da primeira coluna atual do dataframe
first_col = df_raw.columns[0]

# Encontra a linha onde a primeira coluna == 'nomePIMS'
header_row_id = (
    df_raw
    .filter(F.col(first_col) == "nomePIMS")
    .select("_row_id")
    .orderBy("_row_id")
    .first()["_row_id"]
)

# Pega a linha de header completa
header_row = (
    df_raw
    .filter(F.col("_row_id") == header_row_id)
    .drop("_row_id")
    .first()
)

# Monta a lista de novos nomes de colunas a partir dessa linha
new_columns = []
for value in header_row:
    col_name = str(value).strip() if value is not None else ""
    if col_name == "nomePIMS":
        col_name = "timestamp"
    new_columns.append(col_name)

# Mantém somente as linhas após o header e a linha de tipo (analog/dig)
df_data = (
    df_raw
    .filter(F.col("_row_id") > header_row_id + 1)
    .drop("_row_id")
)

# Renomeia as colunas usando os headers capturados
tb_csvs = df_data.toDF(*new_columns)

# Filtra somente o dia de validação (20/02) - faz antes de converter pra pandas
tb_dia = tb_csvs.filter(F.col("timestamp").startswith(DIA_VALIDACAO))

# Converte para pandas
pdf = tb_dia.toPandas()

# Parse timestamp
pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], format="%d.%m.%Y %H:%M:%S.%f", errors="coerce")

# Converte todas as colunas de tags para numérico
for col in pdf.columns:
    if col != "timestamp":
        pdf[col] = pdf[col].str.replace(",", ".", regex=False)
        pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

pdf = pdf.sort_values("timestamp").reset_index(drop=True)

print(f"Dados do dia {DIA_VALIDACAO}:")
print(f"  Linhas: {len(pdf):,}")
print(f"  Período: {pdf['timestamp'].min()} → {pdf['timestamp'].max()}")
print(f"  Colunas: {len(pdf.columns)}")
print(f"\nColunas disponíveis:")
for i, c in enumerate(pdf.columns, 1):
    print(f"  {i:2d}. {c}")

# COMMAND ----------

# DBTITLE 1,Leitura e parsing dos dados CSV para o dia 20/02
# Lê a tabela e cria uma ordem estável das linhas
df_raw = (
    spark.read.table(TABLE_CSV)
    .withColumn("_row_id", F.monotonically_increasing_id())
)

# Descobre o nome da primeira coluna atual do dataframe
first_col = df_raw.columns[0]

# Encontra a linha onde a primeira coluna == 'nomePIMS'
header_row_id = (
    df_raw
    .filter(F.col(first_col) == "nomePIMS")
    .select("_row_id")
    .orderBy("_row_id")
    .first()["_row_id"]
)

# Pega a linha de header completa
header_row = (
    df_raw
    .filter(F.col("_row_id") == header_row_id)
    .drop("_row_id")
    .first()
)

# Monta a lista de novos nomes de colunas a partir dessa linha
new_columns = []
for value in header_row:
    col_name = str(value).strip() if value is not None else ""
    if col_name == "nomePIMS":
        col_name = "timestamp"
    new_columns.append(col_name)

# Mantém somente as linhas após o header e a linha de tipo (analog/dig)
df_data = (
    df_raw
    .filter(F.col("_row_id") > header_row_id + 1)
    .drop("_row_id")
)

# Renomeia as colunas usando os headers capturados
tb_csvs = df_data.toDF(*new_columns)

# Filtra somente o dia de validação (20/02) - faz antes de converter pra pandas
tb_dia = tb_csvs.filter(F.col("timestamp").startswith(DIA_VALIDACAO))

# Converte para pandas
pdf = tb_dia.toPandas()

# Parse timestamp
pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], format="%d.%m.%Y %H:%M:%S.%f", errors="coerce")

# Converte todas as colunas de tags para numérico
for col in pdf.columns:
    if col != "timestamp":
        pdf[col] = pdf[col].str.replace(",", ".", regex=False)
        pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

pdf = pdf.sort_values("timestamp").reset_index(drop=True)

print(f"Dados do dia {DIA_VALIDACAO}:")
print(f"  Linhas: {len(pdf):,}")
print(f"  Período: {pdf['timestamp'].min()} → {pdf['timestamp'].max()}")
print(f"  Colunas: {len(pdf.columns)}")
print(f"\nColunas disponíveis:")
for i, c in enumerate(pdf.columns, 1):
    print(f"  {i:2d}. {c}")

# COMMAND ----------

# DBTITLE 1,Ler tabela OPC LC (PIMS) e comparar com CSV
# ================================================================
# LÊ TABELA OPC LC E COMPARA COM DADOS DO CSV
# ================================================================
import sys
sys.path.insert(0, "/Workspace/Users/execs1@ex.gerdau.com/Sincronizador")
from src.config import TABLE_OPC_LC

print(f"Tabela OPC LC: {TABLE_OPC_LC}")
print("=" * 80)

# Lê a tabela OPC LC para o dia de validação
df_opc_lc = (
    spark.read.table(TABLE_OPC_LC)
    .filter(F.col("timestamp") >= "2026-02-20")
    .filter(F.col("timestamp") < "2026-02-21")
    .orderBy("timestamp")
)

# Converte para pandas
pdf_opc = df_opc_lc.toPandas()

# Converte colunas string para numérico (menos timestamp e surrogate_key)
for col in pdf_opc.columns:
    if col not in ["timestamp", "surrogate_key"]:
        pdf_opc[col] = pd.to_numeric(pdf_opc[col], errors="coerce")

print(f"\n📊 DADOS OPC LC:")
print(f"  Linhas: {len(pdf_opc):,}")
print(f"  Período: {pdf_opc['timestamp'].min()} → {pdf_opc['timestamp'].max()}")
print(f"  Colunas: {len(pdf_opc.columns)}")

# Colunas da OPC (sem metadata)
opc_cols = [c for c in pdf_opc.columns if c not in ["timestamp", "surrogate_key"]]
csv_cols = [c for c in pdf.columns if c != "timestamp"]

# Comparação de colunas
cols_only_opc = [c for c in opc_cols if c not in csv_cols]
cols_only_csv = [c for c in csv_cols if c not in opc_cols]
cols_both = [c for c in opc_cols if c in csv_cols]

print(f"\n🔍 COMPARAÇÃO DE COLUNAS:")
print(f"  Somente OPC ({len(cols_only_opc)}): {cols_only_opc}")
print(f"  Somente CSV ({len(cols_only_csv)}): {', '.join(cols_only_csv[:10])}..." if len(cols_only_csv) > 10 else f"  Somente CSV ({len(cols_only_csv)}): {cols_only_csv}")
print(f"  Em ambos ({len(cols_both)}): {cols_both}")

# ================================================================
# ANÁLISE DE GRANULARIDADE
# ================================================================
print(f"\n⏱️  GRANULARIDADE (diferença entre timestamps consecutivos):")

# CSV
pdf_sorted = pdf.sort_values("timestamp").reset_index(drop=True)
csv_diff = pdf_sorted["timestamp"].diff().dt.total_seconds().dropna()

print(f"\n  CSV:")
print(f"    Média: {csv_diff.mean():.3f}s")
print(f"    Mediana: {csv_diff.median():.3f}s")
print(f"    Min: {csv_diff.min():.3f}s | Max: {csv_diff.max():.3f}s")
print(f"    Std: {csv_diff.std():.3f}s")
print(f"    Valores únicos (top 5): {csv_diff.value_counts().head(5).to_dict()}")

# OPC
pdf_opc_sorted = pdf_opc.sort_values("timestamp").reset_index(drop=True)
opc_diff = pdf_opc_sorted["timestamp"].diff().dt.total_seconds().dropna()

print(f"\n  OPC:")
print(f"    Média: {opc_diff.mean():.3f}s")
print(f"    Mediana: {opc_diff.median():.3f}s")
print(f"    Min: {opc_diff.min():.3f}s | Max: {opc_diff.max():.3f}s")
print(f"    Std: {opc_diff.std():.3f}s")
print(f"    Valores únicos (top 5): {opc_diff.value_counts().head(5).to_dict()}")

# ================================================================
# QUALIDADE DOS DADOS
# ================================================================
print(f"\n📝 QUALIDADE DOS DADOS:")

print(f"\n  CSV - Valores nulos por coluna LC:")
for col in LC_COLS:
    if col in pdf.columns:
        n_null = pdf[col].isnull().sum()
        pct = n_null / len(pdf) * 100
        print(f"    {col}: {n_null:,} ({pct:.1f}%)")

print(f"\n  OPC - Valores nulos por coluna:")
for col in opc_cols:
    n_null = pdf_opc[col].isnull().sum()
    pct = n_null / len(pdf_opc) * 100
    print(f"    {col}: {n_null:,} ({pct:.1f}%)")

# Timestamps duplicados
csv_dups = pdf["timestamp"].duplicated().sum()
opc_dups = pdf_opc["timestamp"].duplicated().sum()
print(f"\n  Timestamps duplicados:")
print(f"    CSV: {csv_dups:,}")
print(f"    OPC: {opc_dups:,}")

# ================================================================
# VERIFICAR ALINHAMENTO DE TIMESTAMPS
# ================================================================
print(f"\n⏰ ALINHAMENTO DE TIMESTAMPS:")
print(f"  CSV primeiro: {pdf['timestamp'].min()}")
print(f"  OPC primeiro: {pdf_opc['timestamp'].min()}")
print(f"  CSV último:   {pdf['timestamp'].max()}")
print(f"  OPC último:   {pdf_opc['timestamp'].max()}")

# Verificar se o OPC timestamp é UTC ou local
# Se o CSV está em hora local (sem timezone) e OPC em UTC,
# haveria uma diferença de 3h (America/Sao_Paulo)
print(f"\n  ⚠️  Nota: CSV não tem timezone, OPC pode estar em UTC.")
print(f"      Se os dados não alinharem, pode ser necessário ajustar 3h.")

# COMMAND ----------

# DBTITLE 1,Merge tags OPC com CSV
# ================================================================
# MERGE: ADICIONA TAGS DO OPC COM AJUSTE UTC → UTC-3
# ================================================================

# Tags OPC que queremos adicionar (não estão no CSV)
tags_to_add = cols_only_opc
print(f"Tags a adicionar do OPC: {tags_to_add}")

# Prepara DataFrame OPC para merge
pdf_opc_adj = pdf_opc.copy()

# Remove surrogate_key
if 'surrogate_key' in pdf_opc_adj.columns:
    pdf_opc_adj = pdf_opc_adj.drop(columns=['surrogate_key'])

# ⏰ AJUSTA OPC DE UTC PARA LOCAL (UTC-3 = America/Sao_Paulo)
print(f"\n⏰ Ajustando timestamps OPC de UTC para UTC-3:")
print(f"  Antes:  {pdf_opc_adj['timestamp'].min()} → {pdf_opc_adj['timestamp'].max()}")
pdf_opc_adj["timestamp"] = pdf_opc_adj["timestamp"] - pd.Timedelta(hours=3)
print(f"  Depois: {pdf_opc_adj['timestamp'].min()} → {pdf_opc_adj['timestamp'].max()}")

# Remove timezone do OPC se existir (para merge com CSV que não tem tz)
if pdf_opc_adj["timestamp"].dt.tz is not None:
    pdf_opc_adj["timestamp"] = pdf_opc_adj["timestamp"].dt.tz_localize(None)

# Arredonda timestamps para segundo mais próximo para facilitar merge
pdf_csv_round = pdf.copy()
pdf_csv_round["ts_round"] = pdf_csv_round["timestamp"].dt.round("1s")

pdf_opc_round = pdf_opc_adj.copy()
pdf_opc_round["ts_round"] = pdf_opc_round["timestamp"].dt.round("1s")

# Merge por timestamp arredondado
pdf_merged = pd.merge(
    pdf_csv_round,
    pdf_opc_round.drop(columns=["timestamp"]),
    on="ts_round",
    how="left",
    suffixes=('', '_opc')
).drop(columns=["ts_round"])

print(f"\n✅ MERGE CONCLUÍDO:")
print(f"  Linhas CSV original: {len(pdf):,}")
print(f"  Linhas após merge:   {len(pdf_merged):,}")
print(f"  Colunas adicionadas: {tags_to_add}")

# Verifica quantos registros do OPC foram encontrados
print(f"\n📊 COBERTURA DAS TAGS OPC (após UTC-3):")
for col in tags_to_add:
    if col in pdf_merged.columns:
        n_found = pdf_merged[col].notna().sum()
        pct = n_found / len(pdf_merged) * 100
        print(f"    {col}: {n_found:,} valores encontrados ({pct:.1f}%)")

# Atualiza pdf para incluir as novas colunas
pdf = pdf_merged.copy()

# Adiciona as novas colunas à lista LC_COLS se ainda não estiverem
for col in tags_to_add:
    if col not in LC_COLS:
        LC_COLS.append(col)

print(f"\n📋 LC_COLS atualizado: {LC_COLS}")

# COMMAND ----------

# DBTITLE 1,Tempos observados manualmente - FEA e LC (PREENCHER)
# ================================================================
# TEMPOS OBSERVADOS MANUALMENTE - PREENCHER COM SEUS DADOS
# Formato: "dd.MM.yyyy HH:mm:ss" ou None se não tem
# ================================================================

def parse_ts(s):
    """Converte string de timestamp para datetime."""
    if s is None:
        return None
    for fmt in ["%d.%m.%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"]:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None

# ---------- FEA ----------
tempos_fea = {
    "chegada_fea":         parse_ts(None),
    "inicio_arco_fea":     parse_ts(None),
    "fim_arco_fea":        parse_ts(None),
    "inicio_vazamento":    parse_ts(None),
    "fim_vazamento":       parse_ts(None),
}

# ---------- LC - MÚLTIPLAS CORRIDAS ----------
# Lista de corridas com seus eventos observados
corridas_lc = [
    {
        "corrida": 131513,
        "braco": 2,
        "eventos": {
            "chegada_torre":       parse_ts("20.02.2026 15:00:00"),
            "abertura_vazamento":  parse_ts("20.02.2026 15:03:00"),
            "inicio_lingotamento": parse_ts("20.02.2026 15:12:00"),
            "fim_lingotamento":    parse_ts("20.02.2026 16:07:00"),  # quando 131514 começa
        }
    },
    {
        "corrida": 131514,
        "braco": 1,
        "eventos": {
            "chegada_torre":       parse_ts("20.02.2026 15:48:00"),
            "abertura_vazamento":  parse_ts("20.02.2026 15:54:00"),
            "inicio_lingotamento": parse_ts("20.02.2026 16:07:00"),
            "fim_lingotamento":    parse_ts(None),  # não temos ainda
        }
    },
]

# Cores por corrida (alterna entre paletas)
CORRIDA_COLORS = [
    {"chegada": "green", "vazamento": "purple", "lingotamento": "red"},
    {"chegada": "lime", "vazamento": "blue", "lingotamento": "orange"},
]

# Estilos por tipo de evento
EVENT_LINE_STYLES = {
    "chegada_torre":       {"ls": "--", "lw": 1.5},
    "abertura_vazamento":  {"ls": "-.", "lw": 1.5},
    "inicio_lingotamento": {"ls": "-",  "lw": 2.0},
    "fim_lingotamento":    {"ls": ":",  "lw": 2.0},
}

def add_event_lines_lc(ax, corridas_list):
    """Adiciona linhas verticais para múltiplas corridas LC."""
    for i, corrida_info in enumerate(corridas_list):
        colors = CORRIDA_COLORS[i % len(CORRIDA_COLORS)]
        corrida_num = corrida_info["corrida"]
        braco = corrida_info["braco"]
        
        for evento_nome, ts in corrida_info["eventos"].items():
            if ts is None:
                continue
            
            # Determina cor baseado no tipo de evento
            if "chegada" in evento_nome:
                color = colors["chegada"]
            elif "vazamento" in evento_nome:
                color = colors["vazamento"]
            else:
                color = colors["lingotamento"]
            
            style = EVENT_LINE_STYLES.get(evento_nome, {"ls": "--", "lw": 1})
            label = f"{corrida_num} B{braco}: {evento_nome.replace('_', ' ')}"
            ax.axvline(ts, color=color, linestyle=style["ls"],
                       linewidth=style["lw"], alpha=0.85, label=label)

# Função legada para FEA (mantém compatibilidade)
EVENT_STYLES = {
    "chegada_fea":     {"color": "green", "ls": "--", "lw": 1.5},
    "inicio_arco_fea": {"color": "blue",  "ls": "-",  "lw": 2.0},
    "fim_arco_fea":    {"color": "blue",  "ls": ":",  "lw": 2.0},
    "inicio_vazamento":{"color": "red",   "ls": "-",  "lw": 2.0},
    "fim_vazamento":   {"color": "red",   "ls": ":",  "lw": 2.0},
}

def add_event_lines(ax, tempos_dict):
    """Adiciona linhas verticais dos eventos FEA."""
    for nome, ts in tempos_dict.items():
        if ts is not None:
            style = EVENT_STYLES.get(nome, {"color": "gray", "ls": "--", "lw": 1})
            ax.axvline(ts, color=style["color"], linestyle=style["ls"],
                       linewidth=style["lw"], alpha=0.8, label=nome)

# Resumo
n_fea = sum(1 for v in tempos_fea.values() if v is not None)
n_lc_eventos = sum(sum(1 for v in c["eventos"].values() if v is not None) for c in corridas_lc)
print(f"Tempos FEA preenchidos: {n_fea}/{len(tempos_fea)}")
print(f"Corridas LC: {len(corridas_lc)}")
for c in corridas_lc:
    n_ev = sum(1 for v in c["eventos"].values() if v is not None)
    print(f"  • Corrida {c['corrida']} (Braço {c['braco']}): {n_ev} eventos")
print("\n✓ Tempos LC carregados!" if n_lc_eventos > 0 else "\n⚠️  Preencha os tempos!")

# COMMAND ----------

# DBTITLE 1,Detecção automática LC (FSM PesoBra)
# ================================================================
# DETECÇÃO AUTOMÁTICA DE CORRIDAS LC VIA FSM (PesoBra)
# Roda na janela hora_ini – hora_fim dos widgets.
# Converte PESOBRA de escala bruta (0–1200) para toneladas (/ 10).
# ================================================================

import sys
sys.path.insert(0, "/Workspace/Users/execs1@ex.gerdau.com/Sincronizador")
from src.detection.lc_logic import detect_lc_events_fsm_pesobra

# Filtra pelo mesmo intervalo dos plots
HORA_INI_FSM = dbutils.widgets.get("hora_ini")
HORA_FIM_FSM = dbutils.widgets.get("hora_fim")

cols_fsm = ["timestamp", "ACI@LC_TORRE_PESOBRA1", "ACI@LC_TORRE_PESOBRA2"]
df_fsm_input = pdf[cols_fsm].copy()
df_fsm_input = df_fsm_input[
    (df_fsm_input["timestamp"].dt.strftime("%H:%M") >= HORA_INI_FSM) &
    (df_fsm_input["timestamp"].dt.strftime("%H:%M") < HORA_FIM_FSM)
].reset_index(drop=True)

# Converte escala bruta → toneladas (1200 = 120t)
df_fsm_input["ACI@LC_TORRE_PESOBRA1"] = df_fsm_input["ACI@LC_TORRE_PESOBRA1"] / 10.0
df_fsm_input["ACI@LC_TORRE_PESOBRA2"] = df_fsm_input["ACI@LC_TORRE_PESOBRA2"] / 10.0

print(f"🔧 FSM PesoBra | Janela {HORA_INI_FSM}–{HORA_FIM_FSM} | {len(df_fsm_input):,} registros")
print(f"   PESOBRA1: {df_fsm_input['ACI@LC_TORRE_PESOBRA1'].min():.1f}–{df_fsm_input['ACI@LC_TORRE_PESOBRA1'].max():.1f} t")
print(f"   PESOBRA2: {df_fsm_input['ACI@LC_TORRE_PESOBRA2'].min():.1f}–{df_fsm_input['ACI@LC_TORRE_PESOBRA2'].max():.1f} t")

# Roda FSM com thresholds originais (calibrados em toneladas)
df_sys_lc, df_events_lc, df_debug_lc, df_feat_lc = detect_lc_events_fsm_pesobra(
    df=df_fsm_input,
    initial_corrida=131513,
    initial_braco=2,
    DEBUG=True,
)

print(f"\n✅ CORRIDAS DETECTADAS: {len(df_sys_lc)}")
if len(df_sys_lc) > 0:
    print(df_sys_lc.to_string(index=False))
else:
    print("  Nenhuma corrida detectada.")

print(f"\n📊 EVENTOS FSM: {len(df_events_lc)}")
if len(df_events_lc) > 0:
    print(df_events_lc.to_string(index=False))

# ================================================================
# CONVERTE PARA FORMATO DE LINHAS VERTICAIS NOS GRÁFICOS
# ================================================================

SYS_COLORS = [
    {"inicio": "cyan", "fim": "magenta"},
    {"inicio": "deepskyblue", "fim": "hotpink"},
]

SYS_LINE_STYLES = {
    "inicio_lc_sys":  {"ls": "-",  "lw": 2.5},
    "final_lc_sys":   {"ls": ":",  "lw": 2.5},
}

corridas_lc_sys = []
for _, row in df_sys_lc.iterrows():
    corridas_lc_sys.append({
        "corrida": int(row["corrida"]),
        "braco": int(row["braco"]),
        "eventos": {
            "inicio_lc_sys": row["inicio_lc_sys"],
            "final_lc_sys":  row["final_lc_sys"],
        }
    })

print(f"\n🎯 Corridas para plot: {len(corridas_lc_sys)}")
for c in corridas_lc_sys:
    print(f"  Corrida {c['corrida']} B{c['braco']}: "
          f"início={c['eventos']['inicio_lc_sys']}, "
          f"fim={c['eventos']['final_lc_sys']}")


def add_event_lines_sys(ax, corridas_sys_list):
    """Adiciona linhas verticais dos eventos DETECTADOS (FSM) nos gráficos."""
    for i, c in enumerate(corridas_sys_list):
        colors = SYS_COLORS[i % len(SYS_COLORS)]
        corrida_num = c["corrida"]
        braco = c["braco"]
        for evento_nome, ts in c["eventos"].items():
            if ts is None or pd.isna(ts):
                continue
            color = colors["inicio"] if "inicio" in evento_nome else colors["fim"]
            style = SYS_LINE_STYLES.get(evento_nome, {"ls": "--", "lw": 2})
            label = f"SYS {corrida_num} B{braco}: {evento_nome.replace('_', ' ')}"
            ax.axvline(ts, color=color, linestyle=style["ls"],
                       linewidth=style["lw"], alpha=0.9, label=label)
            ylim = ax.get_ylim()
            y_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.10
            ax.text(ts, y_pos, f"SYS {evento_nome.split('_')[0]}",
                    color=color, rotation=90, va='bottom', ha='right',
                    fontsize=7, fontweight='bold', alpha=0.9,
                    backgroundcolor='white')

print(f"\n✅ Pronto! Detecção restrita a {HORA_INI_FSM}–{HORA_FIM_FSM}")

# COMMAND ----------

# DBTITLE 1,Detecção simples LC - Chegada torre + Peso Real
# ================================================================
# DETECÇÃO SIMPLES DE EVENTOS LC
# - Chegada na torre: PESOBRA1/2 sobe de ~0 para >50t
# - Início peso real: PESOREAL sobe de ~0 para >20t
# - Fim peso real: PESOREAL desce e ESTABILIZA (independente do patamar)
#
# Escala: bruto / 10 = toneladas
# ================================================================

SMOOTH_W = 5  # mediana móvel (segundos)

def detect_rise(ts, signal_raw, low_t=10.0, high_t=50.0, smooth=SMOOTH_W, min_gap_s=120):
    """
    Detecta subidas: sinal estava abaixo de low_t e cruza acima de high_t.
    Começa com was_low=False — só dispara após observar o sinal baixo.
    """
    s = (signal_raw / 10.0).rolling(smooth, min_periods=1).median()
    events = []
    was_low = False

    for i in range(len(s)):
        v = s.iloc[i]
        if pd.isna(v):
            continue
        if v < low_t:
            was_low = True
        elif v > high_t and was_low:
            t = ts.iloc[i]
            if not events or (t - events[-1]).total_seconds() > min_gap_s:
                events.append(t)
            was_low = False
    return events


def detect_fall(ts, signal_raw, high_t=20.0, descent_t=5.0, smooth=SMOOTH_W,
                stable_window_s=60, max_range_t=0.3, min_gap_s=120):
    """
    Detecta fim do aço: sinal estava acima de high_t, caiu abaixo de descent_t,
    e ESTABILIZOU (variação < max_range_t nos últimos stable_window_s segundos).

    Não depende de um patamar fixo — funciona se estabilizar a 0.9t, 2.5t, etc.

    Parâmetros:
        high_t:           peso (t) que confirma que houve vaz. (fase alta)
        descent_t:        peso (t) abaixo do qual estamos na "zona de fim"
        stable_window_s:  janela (s) para checar estabilidade
        max_range_t:      variação máx (t) na janela para considerar estável
    """
    s = (signal_raw / 10.0).rolling(smooth, min_periods=1).median()

    # Variação (max - min) nos últimos stable_window_s amostras
    rolling_max = s.rolling(stable_window_s, min_periods=stable_window_s).max()
    rolling_min = s.rolling(stable_window_s, min_periods=stable_window_s).min()
    r_range = rolling_max - rolling_min

    events = []
    was_high = False

    for i in range(len(s)):
        v = s.iloc[i]
        if pd.isna(v):
            continue
        if v > high_t:
            was_high = True
        elif was_high and v < descent_t:
            rr = r_range.iloc[i]
            if pd.notna(rr) and rr < max_range_t:
                # Estabilizou! Início da janela estável ≈ stable_window_s atrás
                idx_start = max(0, i - stable_window_s)
                t_stable = ts.iloc[idx_start]
                if not events or (t_stable - events[-1]).total_seconds() > min_gap_s:
                    events.append(t_stable)
                was_high = False
    return events


# ================================================================
# APLICA NA JANELA DO PLOT
# ================================================================
HORA_INI_DET = dbutils.widgets.get("hora_ini")
HORA_FIM_DET = dbutils.widgets.get("hora_fim")

df_det = pdf[
    (pdf["timestamp"].dt.strftime("%H:%M") >= HORA_INI_DET) &
    (pdf["timestamp"].dt.strftime("%H:%M") < HORA_FIM_DET)
].copy().sort_values("timestamp").reset_index(drop=True)

ts = df_det["timestamp"]

print(f"🔍 Detecção simples LC | {HORA_INI_DET}–{HORA_FIM_DET} | {len(df_det):,} registros")
print("=" * 65)

# --- Chegada na torre ---
chegada_bra1 = detect_rise(ts, df_det["ACI@LC_TORRE_PESOBRA1"], low_t=10, high_t=50)
chegada_bra2 = detect_rise(ts, df_det["ACI@LC_TORRE_PESOBRA2"], low_t=10, high_t=50)

print(f"\n🏗️  CHEGADA NA TORRE:")
for t in chegada_bra1:
    print(f"  Braço 1: {t}")
for t in chegada_bra2:
    print(f"  Braço 2: {t}")
if not chegada_bra1 and not chegada_bra2:
    print("  Nenhuma chegada detectada")

# --- Início peso real (torre girou, panela na posição) ---
inicio_pesoreal = detect_rise(ts, df_det["ACI@LC_TORRE_PESOREAL"], low_t=5, high_t=20)

print(f"\n⚖️  INÍCIO PESO REAL (panela na posição):")
for t in inicio_pesoreal:
    print(f"  {t}")
if not inicio_pesoreal:
    print("  Nenhum início detectado")

# --- Fim peso real (aço acabou - detecção por estabilização) ---
# descent_t=5t: zona de fim (sinal já caiu bastante)
# stable_window_s=60s, max_range_t=0.3t: variou menos de 0.3t em 60s = estabilizou
fim_pesoreal = detect_fall(ts, df_det["ACI@LC_TORRE_PESOREAL"],
                           high_t=20, descent_t=5.0,
                           stable_window_s=60, max_range_t=0.3)

print(f"\n🏁 FIM PESO REAL (estabilizou):")
for t in fim_pesoreal:
    print(f"  {t}")
if not fim_pesoreal:
    print("  Nenhum fim detectado")

# ================================================================
# EMPARELHA EVENTOS EM CICLOS (sem repetir eventos)
# ================================================================
DET_EVENTS = []

all_chegadas = ([(t, 1) for t in chegada_bra1] +
                [(t, 2) for t in chegada_bra2])
all_chegadas.sort(key=lambda x: x[0])

inicio_pool = list(inicio_pesoreal)
fim_pool = list(fim_pesoreal)

for t_cheg, braco in all_chegadas:
    ev = {
        "braco": braco,
        "chegada_torre": t_cheg,
        "inicio_pesoreal": None,
        "fim_pesoreal": None,
    }
    for j, t_ini in enumerate(inicio_pool):
        if t_ini > t_cheg:
            ev["inicio_pesoreal"] = t_ini
            inicio_pool.pop(j)
            break
    if ev["inicio_pesoreal"]:
        for j, t_fim in enumerate(fim_pool):
            if t_fim > ev["inicio_pesoreal"]:
                ev["fim_pesoreal"] = t_fim
                fim_pool.pop(j)
                break
    DET_EVENTS.append(ev)

print(f"\n{'='*65}")
print(f"📊 RESUMO - EVENTOS DETECTADOS:")
print(f"{'='*65}")
for i, ev in enumerate(DET_EVENTS):
    print(f"\n  Ciclo {i+1} (Braço {ev['braco']}):")
    print(f"    Chegada torre:     {ev['chegada_torre']}")
    print(f"    Início peso real:  {ev['inicio_pesoreal']}")
    print(f"    Fim peso real:     {ev['fim_pesoreal']}")
    if ev['inicio_pesoreal'] and ev['chegada_torre']:
        delta = (ev['inicio_pesoreal'] - ev['chegada_torre']).total_seconds()
        print(f"    ⏱️  Chegada → Peso real: {delta:.0f}s ({delta/60:.1f}min)")
    if ev['fim_pesoreal'] and ev['inicio_pesoreal']:
        delta = (ev['fim_pesoreal'] - ev['inicio_pesoreal']).total_seconds()
        print(f"    ⏱️  Duração vazamento:   {delta:.0f}s ({delta/60:.1f}min)")

if len(DET_EVENTS) >= 2:
    print(f"\n{'='*65}")
    print(f"⏱️  TIMING ENTRE CORRIDAS:")
    for i in range(1, len(DET_EVENTS)):
        prev = DET_EVENTS[i-1]
        curr = DET_EVENTS[i]
        if prev['fim_pesoreal'] and curr['chegada_torre']:
            delta = (curr['chegada_torre'] - prev['fim_pesoreal']).total_seconds()
            status = "✅ JÁ ESPERANDO" if delta < 0 else "⚠️ ATRASOU"
            print(f"  Fim corrida {i} → Chegada corrida {i+1}: {delta:.0f}s ({delta/60:.1f}min) {status}")
        if prev['fim_pesoreal'] and curr['inicio_pesoreal']:
            delta = (curr['inicio_pesoreal'] - prev['fim_pesoreal']).total_seconds()
            print(f"  Fim corrida {i} → Início pesoreal {i+1}: {delta:.0f}s ({delta/60:.1f}min)")

# ================================================================
# HELPER PARA PLOTAR
# ================================================================
DET_COLORS = {"chegada_torre": "green", "inicio_pesoreal": "blue", "fim_pesoreal": "red"}
DET_STYLES = {"chegada_torre": "--", "inicio_pesoreal": "-", "fim_pesoreal": ":"}

def add_event_lines_det(ax, det_events):
    """Adiciona linhas verticais dos eventos detectados."""
    for ev in det_events:
        braco = ev["braco"]
        for key in ["chegada_torre", "inicio_pesoreal", "fim_pesoreal"]:
            t = ev[key]
            if t is None:
                continue
            color = DET_COLORS[key]
            label = f"DET B{braco}: {key.replace('_', ' ')}"
            ax.axvline(t, color=color, linestyle=DET_STYLES[key],
                       linewidth=2.0, alpha=0.85, label=label)
            ylim = ax.get_ylim()
            y_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.85
            ax.text(t, y_pos, key.replace('_', ' '),
                    color=color, rotation=90, va='top', ha='right',
                    fontsize=7, fontweight='bold', alpha=0.85,
                    backgroundcolor='white')

print(f"\n✅ Pronto! {len(DET_EVENTS)} ciclos detectados.")

# COMMAND ----------

# DBTITLE 1,Plot DET - Peso Braços e Peso Real com eventos detectados
# ================================================================
# PLOT FOCADO NA DETECÇÃO SIMPLES (DET)
# 3 gráficos: PESOBRA1, PESOBRA2, PESOREAL (em toneladas)
# Linhas verticais limpas: nome sem fundo, sem bold
# ================================================================

HORA_INI_P = dbutils.widgets.get("hora_ini")
HORA_FIM_P = dbutils.widgets.get("hora_fim")

df_plot = pdf[
    (pdf["timestamp"].dt.strftime("%H:%M") >= HORA_INI_P) &
    (pdf["timestamp"].dt.strftime("%H:%M") < HORA_FIM_P)
].copy().sort_values("timestamp")

print(f"Plot DET | {HORA_INI_P}–{HORA_FIM_P} | {len(df_plot):,} registros | {len(DET_EVENTS)} ciclos")

# Cores e estilos por tipo de evento
EV_STYLE = {
    "chegada_torre":   {"color": "#2ca02c", "ls": "--", "lw": 1.5},  # verde
    "inicio_pesoreal": {"color": "#1f77b4", "ls": "-",  "lw": 1.5},  # azul
    "fim_pesoreal":    {"color": "#d62728", "ls": ":",  "lw": 1.8},  # vermelho
}

EV_LABEL = {
    "chegada_torre":   "chegada torre",
    "inicio_pesoreal": "início peso real",
    "fim_pesoreal":    "fim peso real",
}

def add_det_lines(ax, det_events, y_frac=0.92):
    """Linhas verticais limpas: nome sem fundo, sem bold."""
    for ev in det_events:
        braco = ev["braco"]
        for key in ["chegada_torre", "inicio_pesoreal", "fim_pesoreal"]:
            t = ev[key]
            if t is None:
                continue
            st = EV_STYLE[key]
            label = f"B{braco} {EV_LABEL[key]}"
            ax.axvline(t, color=st["color"], ls=st["ls"], lw=st["lw"], alpha=0.8)
            ylim = ax.get_ylim()
            y_pos = ylim[0] + (ylim[1] - ylim[0]) * y_frac
            ax.text(t, y_pos, label, color=st["color"], rotation=90,
                    va='top', ha='right', fontsize=7, alpha=0.85)

# Tags para plotar (em tons)
tags = [
    ("ACI@LC_TORRE_PESOBRA1", "Peso Braço 1 (t)",  "#ff7f0e"),
    ("ACI@LC_TORRE_PESOBRA2", "Peso Braço 2 (t)",  "#ff7f0e"),
    ("ACI@LC_TORRE_PESOREAL", "Peso Real Torre (t)", "#9467bd"),
]

fig, axes = plt.subplots(len(tags), 1, figsize=(14, len(tags) * 4), sharex=True)

for idx, (col, title, color) in enumerate(tags):
    ax = axes[idx]
    vals = df_plot[col].dropna() / 10.0
    ts_vals = df_plot.loc[vals.index, "timestamp"]
    ax.plot(ts_vals, vals, linewidth=0.6, color=color, alpha=0.9)
    add_det_lines(ax, DET_EVENTS)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("toneladas")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, alpha=0.25)
    # Margem superior
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.08)

# Legenda única no topo
from matplotlib.lines import Line2D
leg_handles = [
    Line2D([0],[0], color=EV_STYLE[k]["color"], ls=EV_STYLE[k]["ls"],
           lw=EV_STYLE[k]["lw"], label=EV_LABEL[k])
    for k in ["chegada_torre", "inicio_pesoreal", "fim_pesoreal"]
]
fig.legend(handles=leg_handles, loc="upper right", fontsize=9, framealpha=0.9)
fig.suptitle(f"Detecção LC — Pesos em Toneladas ({DIA_VALIDACAO} {HORA_INI_P}–{HORA_FIM_P})",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# COMMAND ----------

# DBTITLE 1,Estatísticas e histogramas dos tempos detectados
# ================================================================
# ANÁLISE ESTATÍSTICA DOS TEMPOS DETECTADOS
# Histogramas + tabela resumo
# ================================================================

# Calcula métricas por ciclo
metrics = []
for i, ev in enumerate(DET_EVENTS):
    m = {"ciclo": i + 1, "braco": ev["braco"]}

    # Chegada → Início peso real (tempo de giro da torre)
    if ev["chegada_torre"] and ev["inicio_pesoreal"]:
        m["chegada_to_pesoreal_s"] = (ev["inicio_pesoreal"] - ev["chegada_torre"]).total_seconds()
    else:
        m["chegada_to_pesoreal_s"] = np.nan

    # Duração do vazamento (início → fim peso real)
    if ev["inicio_pesoreal"] and ev["fim_pesoreal"]:
        m["duracao_vazamento_s"] = (ev["fim_pesoreal"] - ev["inicio_pesoreal"]).total_seconds()
    else:
        m["duracao_vazamento_s"] = np.nan

    # Ciclo total (chegada → fim)
    if ev["chegada_torre"] and ev["fim_pesoreal"]:
        m["ciclo_total_s"] = (ev["fim_pesoreal"] - ev["chegada_torre"]).total_seconds()
    else:
        m["ciclo_total_s"] = np.nan

    metrics.append(m)

# Timing entre corridas
for i in range(1, len(DET_EVENTS)):
    prev = DET_EVENTS[i - 1]
    curr = DET_EVENTS[i]
    if prev["fim_pesoreal"] and curr["chegada_torre"]:
        metrics[i]["fim_to_chegada_s"] = (curr["chegada_torre"] - prev["fim_pesoreal"]).total_seconds()
    else:
        metrics[i]["fim_to_chegada_s"] = np.nan
    if prev["fim_pesoreal"] and curr["inicio_pesoreal"]:
        metrics[i]["fim_to_inicio_s"] = (curr["inicio_pesoreal"] - prev["fim_pesoreal"]).total_seconds()
    else:
        metrics[i]["fim_to_inicio_s"] = np.nan

df_metrics = pd.DataFrame(metrics)

# Converte segundos para minutos nas colunas de tempo
time_cols = [c for c in df_metrics.columns if c.endswith("_s")]
for c in time_cols:
    df_metrics[c.replace("_s", "_min")] = df_metrics[c] / 60.0

print("📊 TABELA DE MÉTRICAS POR CICLO:")
print("=" * 80)
cols_show = ["ciclo", "braco"]
cols_show += [c for c in df_metrics.columns if c.endswith("_min")]
print(df_metrics[cols_show].to_string(index=False, float_format="{:.1f}".format))

# Estatísticas descritivas
print(f"\n\n📈 ESTATÍSTICAS DESCRITIVAS (minutos):")
print("=" * 80)
min_cols = [c for c in df_metrics.columns if c.endswith("_min")]
stats = df_metrics[min_cols].describe().T
stats.index = [c.replace("_min", "") for c in stats.index]
print(stats.to_string(float_format="{:.1f}".format))

# ================================================================
# HISTOGRAMAS
# ================================================================
hist_data = {
    "Chegada \u2192 Peso Real\n(giro torre, min)": df_metrics["chegada_to_pesoreal_min"].dropna(),
    "Dura\u00e7\u00e3o Vazamento\n(min)": df_metrics["duracao_vazamento_min"].dropna(),
    "Ciclo Total\n(chegada \u2192 fim, min)": df_metrics["ciclo_total_min"].dropna(),
    "Fim corr. anterior \u2192\nChegada pr\u00f3xima (min)": df_metrics.get("fim_to_chegada_min", pd.Series(dtype=float)).dropna(),
    "Fim corr. anterior \u2192\nIn\u00edcio peso real (min)": df_metrics.get("fim_to_inicio_min", pd.Series(dtype=float)).dropna(),
}

# Filtra histogramas com dados
hist_data = {k: v for k, v in hist_data.items() if len(v) > 0}

if hist_data:
    n_plots = len(hist_data)
    fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    colors = ["#2ca02c", "#1f77b4", "#9467bd", "#d62728", "#ff7f0e"]

    for idx, (title, data) in enumerate(hist_data.items()):
        ax = axes[idx]
        n_bins = max(2, min(15, len(data)))
        ax.hist(data, bins=n_bins, color=colors[idx % len(colors)],
                edgecolor="white", alpha=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("minutos", fontsize=8)
        ax.set_ylabel("frequ\u00eancia", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2, axis="y")

        # Anota\u00e7\u00f5es de estat\u00edsticas
        if len(data) >= 1:
            mu = data.mean()
            ax.axvline(mu, color="black", ls="--", lw=1.2, alpha=0.7)
            ax.text(mu, ax.get_ylim()[1] * 0.9, f"m\u00e9dia\n{mu:.1f}",
                    ha="center", fontsize=7, color="black")
        # Valores negativos = j\u00e1 esperando (destaque)
        if (data < 0).any():
            ax.axvspan(data.min() - 1, 0, alpha=0.1, color="green")
            ax.text(data.min() / 2, ax.get_ylim()[1] * 0.5, "j\u00e1 esperando",
                    ha="center", fontsize=7, color="green", fontstyle="italic")

    fig.suptitle(f"Distribui\u00e7\u00e3o dos Tempos LC ({DIA_VALIDACAO})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()
else:
    print("\u26a0\ufe0f  Dados insuficientes para histogramas (precisa de mais corridas).")

# ================================================================
# ANÁLISE DE ATRASO / ANTECIPAÇÃO
# ================================================================
if "fim_to_chegada_min" in df_metrics.columns:
    ftc = df_metrics["fim_to_chegada_min"].dropna()
    if len(ftc) > 0:
        n_esperando = (ftc < 0).sum()
        n_atrasou = (ftc >= 0).sum()
        print(f"\n\n🚚 ATRASO vs ANTECIPA\u00c7\u00c3O (fim corrida anterior \u2192 chegada pr\u00f3xima):")
        print(f"  Total transi\u00e7\u00f5es: {len(ftc)}")
        print(f"  \u2705 J\u00e1 esperando (< 0min): {n_esperando} ({n_esperando/len(ftc)*100:.0f}%)")
        print(f"  \u26a0\ufe0f  Atrasou (>= 0min):     {n_atrasou} ({n_atrasou/len(ftc)*100:.0f}%)")
        if n_esperando > 0:
            antecipacao = ftc[ftc < 0]
            print(f"  M\u00e9dia antecipa\u00e7\u00e3o: {antecipacao.mean():.1f} min")
        if n_atrasou > 0:
            atraso = ftc[ftc >= 0]
            print(f"  M\u00e9dia atraso: {atraso.mean():.1f} min")

# COMMAND ----------

# DBTITLE 1,Plot FEA - Tags do dia 20/02 com linhas de eventos
# ================================================================
# PLOT FEA - Todas as tags FEA do dia com linhas verticais
# ================================================================

# Lê widgets de hora
HORA_INI = dbutils.widgets.get("hora_ini")
HORA_FIM = dbutils.widgets.get("hora_fim")

pdf_fea = pdf[(pdf["timestamp"].dt.strftime("%H:%M") >= HORA_INI) & (pdf["timestamp"].dt.strftime("%H:%M") < HORA_FIM)].copy()
print(f"Período: {HORA_INI} – {HORA_FIM}  |  {len(pdf_fea):,} registros")

# Filtra colunas FEA disponíveis no dataframe
fea_available = [c for c in FEA_COLS if c in pdf_fea.columns]

# Separa em análogos vs digitais (0/1)
fea_analog = []
fea_digital = []
for col in fea_available:
    vals = pdf_fea[col].dropna()
    if len(vals) > 0 and set(vals.unique()).issubset({0.0, 1.0, 0, 1}):
        fea_digital.append(col)
    else:
        fea_analog.append(col)

print(f"FEA análogas:  {len(fea_analog)} tags")
print(f"FEA digitais:  {len(fea_digital)} tags")

# --- Plot tags analógicas ---
if fea_analog:
    n_cols_plot = 3
    n_rows_plot = int(np.ceil(len(fea_analog) / n_cols_plot))
    fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(22, n_rows_plot * 3.5), sharex=True)
    axes = np.atleast_2d(axes).flatten()

    for idx, col in enumerate(fea_analog):
        ax = axes[idx]
        ax.plot(pdf_fea["timestamp"], pdf_fea[col], linewidth=0.5, color="steelblue")
        add_event_lines(ax, tempos_fea)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.grid(True, alpha=0.3)

    for idx in range(len(fea_analog), len(axes)):
        axes[idx].set_visible(False)

    handles, labels = [], []
    for nome, ts in tempos_fea.items():
        if ts is not None:
            style = EVENT_STYLES[nome]
            handles.append(plt.Line2D([0],[0], color=style["color"], ls=style["ls"], lw=style["lw"]))
            labels.append(nome)
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8, ncol=2)

    fig.suptitle(f"FEA - Tags Analógicas ({DIA_VALIDACAO} {HORA_INI}–{HORA_FIM})", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# --- Plot tags digitais ---
if fea_digital:
    n_cols_plot = 3
    n_rows_plot = int(np.ceil(len(fea_digital) / n_cols_plot))
    fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(22, n_rows_plot * 2.5), sharex=True)
    axes = np.atleast_2d(axes).flatten()

    for idx, col in enumerate(fea_digital):
        ax = axes[idx]
        ax.fill_between(pdf_fea["timestamp"], pdf_fea[col], step="post", alpha=0.4, color="coral")
        ax.step(pdf_fea["timestamp"], pdf_fea[col], where="post", linewidth=0.5, color="coral")
        add_event_lines(ax, tempos_fea)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_ylim(-0.1, 1.3)
        ax.set_yticks([0, 1])
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.grid(True, alpha=0.3)

    for idx in range(len(fea_digital), len(axes)):
        axes[idx].set_visible(False)

    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8, ncol=2)

    fig.suptitle(f"FEA - Tags Digitais ({DIA_VALIDACAO} {HORA_INI}–{HORA_FIM})", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# COMMAND ----------

# DBTITLE 1,Plot LC - Tags do dia 20/02 com linhas de eventos
# ================================================================
# PLOT LC - Tags LC com linhas de eventos:
#   OBS  = observações manuais (corridas_lc)
#   SYS  = FSM PesoBra (corridas_lc_sys)
#   DET  = detecção simples (chegada torre, peso real)
# ================================================================

HORA_INI_LC = dbutils.widgets.get("hora_ini")
HORA_FIM_LC = dbutils.widgets.get("hora_fim")

pdf_lc = pdf[(pdf["timestamp"].dt.strftime("%H:%M") >= HORA_INI_LC) & (pdf["timestamp"].dt.strftime("%H:%M") < HORA_FIM_LC)].copy()
print(f"Período: {HORA_INI_LC} – {HORA_FIM_LC}  |  {len(pdf_lc):,} registros")
print(f"Corridas manuais: {', '.join(str(c['corrida']) for c in corridas_lc)}")
print(f"Corridas FSM:     {', '.join(str(c['corrida']) for c in corridas_lc_sys)}")
print(f"Eventos DET:      {len(DET_EVENTS)} ciclos")

# Filtra colunas LC disponíveis
lc_available = [c for c in LC_COLS if c in pdf_lc.columns]
lc_analog, lc_digital = [], []
for col in lc_available:
    vals = pdf_lc[col].dropna()
    if len(vals) > 0 and set(vals.unique()).issubset({0.0, 1.0, 0, 1}):
        lc_digital.append(col)
    else:
        lc_analog.append(col)

print(f"LC análogas: {len(lc_analog)}  |  LC digitais: {len(lc_digital)}")

def add_event_lines_lc_with_text(ax, corridas_list):
    """Linhas verticais - observações manuais."""
    for i, corrida_info in enumerate(corridas_list):
        colors = CORRIDA_COLORS[i % len(CORRIDA_COLORS)]
        corrida_num = corrida_info["corrida"]
        braco = corrida_info["braco"]
        for evento_nome, ts in corrida_info["eventos"].items():
            if ts is None:
                continue
            if "chegada" in evento_nome:
                color = colors["chegada"]
            elif "vazamento" in evento_nome:
                color = colors["vazamento"]
            else:
                color = colors["lingotamento"]
            style = EVENT_LINE_STYLES.get(evento_nome, {"ls": "--", "lw": 1})
            label = f"OBS {corrida_num} B{braco}: {evento_nome.replace('_', ' ')}"
            ax.axvline(ts, color=color, linestyle=style["ls"],
                       linewidth=style["lw"], alpha=0.85, label=label)
            ylim = ax.get_ylim()
            y_pos = ylim[1] - (ylim[1] - ylim[0]) * 0.05
            ax.text(ts, y_pos, evento_nome.replace('_', ' '), color=color, rotation=90,
                    va='top', ha='left', fontsize=7, fontweight='bold', alpha=0.85, backgroundcolor='white')

# --- Plot tags analógicas ---
if lc_analog:
    fig, axes = plt.subplots(len(lc_analog), 1, figsize=(14, len(lc_analog) * 4.5), sharex=True)
    if len(lc_analog) == 1:
        axes = [axes]
    for idx, col in enumerate(lc_analog):
        ax = axes[idx]
        ax.plot(pdf_lc["timestamp"], pdf_lc[col], linewidth=0.5, color="darkorange")
        add_event_lines_lc_with_text(ax, corridas_lc)   # OBS manual
        add_event_lines_sys(ax, corridas_lc_sys)         # SYS FSM
        add_event_lines_det(ax, DET_EVENTS)              # DET simples
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.grid(True, alpha=0.3)
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.10)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper right", fontsize=6, framealpha=0.9, ncol=2)
    fig.suptitle(f"LC - Análogas ({DIA_VALIDACAO} {HORA_INI_LC}–{HORA_FIM_LC}) | OBS=manual  SYS=FSM  DET=simples",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# --- Plot tags digitais ---
if lc_digital:
    fig, axes = plt.subplots(len(lc_digital), 1, figsize=(14, len(lc_digital) * 3.0), sharex=True)
    if len(lc_digital) == 1:
        axes = [axes]
    for idx, col in enumerate(lc_digital):
        ax = axes[idx]
        ax.fill_between(pdf_lc["timestamp"], pdf_lc[col], step="post", alpha=0.4, color="teal")
        ax.step(pdf_lc["timestamp"], pdf_lc[col], where="post", linewidth=0.5, color="teal")
        add_event_lines_lc_with_text(ax, corridas_lc)
        add_event_lines_sys(ax, corridas_lc_sys)
        add_event_lines_det(ax, DET_EVENTS)
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_ylim(-0.1, 1.4)
        ax.set_yticks([0, 1])
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper right", fontsize=6, framealpha=0.9, ncol=2)
    fig.suptitle(f"LC - Digitais ({DIA_VALIDACAO} {HORA_INI_LC}–{HORA_FIM_LC}) | OBS=manual  SYS=FSM  DET=simples",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# DBTITLE 1,Validação 7 dias - Carga OPC e detecção via src
# ================================================================
# VALIDAÇÃO MÊS COMPLETO — Carrega OPC LC e roda detecções do src/
# ================================================================
import importlib, sys
sys.path.insert(0, "/Workspace/Users/execs1@ex.gerdau.com/Sincronizador")

# Recarrega módulos para pegar alterações recentes
import src.detection.lc_peso_bra as _bra_mod
import src.detection.lc_peso_real as _real_mod
importlib.reload(_bra_mod)
importlib.reload(_real_mod)
from src.detection.lc_peso_bra import detect_lc_chegada_torre
from src.detection.lc_peso_real import detect_lc_peso_real
from src.config import TABLE_OPC_LC

# --- Período: mês completo (limitado pelo CSV: 01/02 a 28/02) ---
DATA_INI = "2026-02-01"
DATA_FIM = "2026-03-01"

print(f"📅 Carregando OPC LC: {DATA_INI} → {DATA_FIM} (fevereiro completo)")
print(f"   Tabela: {TABLE_OPC_LC}")

df_opc_7d = (
    spark.read.table(TABLE_OPC_LC)
    .filter(F.col("timestamp") >= DATA_INI)
    .filter(F.col("timestamp") < DATA_FIM)
    .orderBy("timestamp")
)

pdf_opc_7d = df_opc_7d.toPandas()

# Ajuste UTC → UTC-3
pdf_opc_7d["timestamp"] = pd.to_datetime(pdf_opc_7d["timestamp"]) - pd.Timedelta(hours=3)
if pdf_opc_7d["timestamp"].dt.tz is not None:
    pdf_opc_7d["timestamp"] = pdf_opc_7d["timestamp"].dt.tz_localize(None)

# Converte tags para numérico
for col in pdf_opc_7d.columns:
    if col not in ["timestamp", "surrogate_key"]:
        pdf_opc_7d[col] = pd.to_numeric(pdf_opc_7d[col], errors="coerce")

print(f"\n✅ Dados OPC carregados: {len(pdf_opc_7d):,} registros")
print(f"   Período: {pdf_opc_7d['timestamp'].min()} → {pdf_opc_7d['timestamp'].max()}")
print(f"   Dias: {pdf_opc_7d['timestamp'].dt.date.nunique()}")

# CSV PESOBRA — todo o dataset disponível (01/02 a 28/02)
# NOTA: timestamps CSV em formato dd.mm.yyyy, comparação string só funciona
# dentro do mesmo mês, então filtramos por ".02.2026" no nome
print(f"\n📅 Carregando CSV para PESOBRA (fevereiro completo)...")

df_csv_7d = (
    spark.read.table(TABLE_CSV)
    .withColumn("_row_id", F.monotonically_increasing_id())
)
first_col = df_csv_7d.columns[0]
header_row_id = (
    df_csv_7d.filter(F.col(first_col) == "nomePIMS")
    .select("_row_id").orderBy("_row_id").first()["_row_id"]
)
header_row = df_csv_7d.filter(F.col("_row_id") == header_row_id).drop("_row_id").first()
new_columns = []
for value in header_row:
    col_name = str(value).strip() if value is not None else ""
    new_columns.append("timestamp" if col_name == "nomePIMS" else col_name)

df_data_7d = df_csv_7d.filter(F.col("_row_id") > header_row_id).drop("_row_id")
for i, new_name in enumerate(new_columns):
    df_data_7d = df_data_7d.withColumnRenamed(df_data_7d.columns[i], new_name)

# Filtra fevereiro completo pelo mês no timestamp (formato dd.mm.yyyy)
df_data_7d = df_data_7d.filter(F.col("timestamp").contains(".02.2026"))

pdf_csv_7d = df_data_7d.select(
    "timestamp", "ACI@LC_TORRE_PESOBRA1", "ACI@LC_TORRE_PESOBRA2"
).toPandas()

# Parse timestamps CSV
pdf_csv_7d["timestamp"] = pd.to_datetime(
    pdf_csv_7d["timestamp"], format="%d.%m.%Y %H:%M:%S.%f", errors="coerce"
)
pdf_csv_7d = pdf_csv_7d.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
for col in ["ACI@LC_TORRE_PESOBRA1", "ACI@LC_TORRE_PESOBRA2"]:
    pdf_csv_7d[col] = pd.to_numeric(pdf_csv_7d[col], errors="coerce")

print(f"\n✅ CSV PESOBRA: {len(pdf_csv_7d):,} registros")
print(f"   Período: {pdf_csv_7d['timestamp'].min()} → {pdf_csv_7d['timestamp'].max()}")
print(f"   Dias: {pdf_csv_7d['timestamp'].dt.date.nunique()}")

# ================================================================
# RODA DETECÇÕES
# ================================================================
print(f"\n{'='*70}")
print(f"🔧 RODANDO DETECÇÕES (fevereiro completo)...")
print(f"{'='*70}")

# 1) Chegada na torre (PESOBRA1/2 do CSV)
df_bra_sys, df_bra_ev, df_bra_dbg = detect_lc_chegada_torre(
    df=pdf_csv_7d, DEBUG=False
)
print(f"\n🏗️  Chegadas na torre: {len(df_bra_sys)} eventos")
if len(df_bra_sys) > 0:
    print(f"   Braço 1: {(df_bra_sys['braco'] == 1).sum()}")
    print(f"   Braço 2: {(df_bra_sys['braco'] == 2).sum()}")

# 2) Peso real (PESOREAL do OPC)
df_real_sys, df_real_ev, df_real_dbg = detect_lc_peso_real(
    df=pdf_opc_7d[["timestamp", "ACI@LC_TORRE_PESOREAL"]].copy(), DEBUG=False
)
print(f"\n⚖️  Peso real: {len(df_real_sys)} ciclos completos")
if len(df_real_sys) > 0:
    print(f"   Duração média: {df_real_sys['duracao_s'].mean()/60:.1f} min")

print(f"\n✅ Detecções concluídas — fevereiro completo!")
print(f"   Chegadas torre: {len(df_bra_sys)}")
print(f"   Ciclos peso real: {len(df_real_sys)}")

# COMMAND ----------

# DBTITLE 1,Plot validação 12h de um dia (7 dias)
# ================================================================
# PLOT VALIDAÇÃO — 12h de um dia específico
# Mostra PESOBRA1, PESOBRA2, PESOREAL com eventos detectados
# ================================================================

# Escolha o dia e a janela de 12h
DIA_PLOT = "2026-02-20"
H_INI_PLOT = "06:00"
H_FIM_PLOT = "18:00"

print(f"📊 Plot validação: {DIA_PLOT} {H_INI_PLOT}–{H_FIM_PLOT}")

# Filtra dados CSV (PESOBRA)
mask_csv = (
    (pdf_csv_7d["timestamp"].dt.date == pd.Timestamp(DIA_PLOT).date()) &
    (pdf_csv_7d["timestamp"].dt.strftime("%H:%M") >= H_INI_PLOT) &
    (pdf_csv_7d["timestamp"].dt.strftime("%H:%M") < H_FIM_PLOT)
)
df_p_bra = pdf_csv_7d.loc[mask_csv].copy()

# Filtra dados OPC (PESOREAL)
mask_opc = (
    (pdf_opc_7d["timestamp"].dt.date == pd.Timestamp(DIA_PLOT).date()) &
    (pdf_opc_7d["timestamp"].dt.strftime("%H:%M") >= H_INI_PLOT) &
    (pdf_opc_7d["timestamp"].dt.strftime("%H:%M") < H_FIM_PLOT)
)
df_p_real = pdf_opc_7d.loc[mask_opc].copy()

print(f"  PESOBRA: {len(df_p_bra):,} registros")
print(f"  PESOREAL: {len(df_p_real):,} registros")

# Filtra eventos detectados para esse dia e janela
dia_date = pd.Timestamp(DIA_PLOT).date()
t_ini = pd.Timestamp(f"{DIA_PLOT} {H_INI_PLOT}")
t_fim = pd.Timestamp(f"{DIA_PLOT} {H_FIM_PLOT}")

bra_day = df_bra_sys[
    (df_bra_sys["chegada_torre_ts"].dt.date == dia_date) &
    (df_bra_sys["chegada_torre_ts"] >= t_ini) &
    (df_bra_sys["chegada_torre_ts"] < t_fim)
]

real_day = df_real_sys[
    (df_real_sys["inicio_pesoreal_ts"].dt.date == dia_date) &
    (df_real_sys["inicio_pesoreal_ts"] >= t_ini) &
    (df_real_sys["inicio_pesoreal_ts"] < t_fim)
]

print(f"  Chegadas torre no período: {len(bra_day)}")
print(f"  Ciclos peso real no período: {len(real_day)}")

# Estilos
EV_STYLE_V = {
    "chegada": {"color": "#2ca02c", "ls": "--", "lw": 1.3},
    "inicio":  {"color": "#1f77b4", "ls": "-",  "lw": 1.3},
    "fim":     {"color": "#d62728", "ls": ":",  "lw": 1.5},
}

def add_validation_lines(ax, bra_events, real_events, y_frac=0.92):
    """Adiciona linhas de eventos detectados."""
    ylim = ax.get_ylim()
    y_pos = ylim[0] + (ylim[1] - ylim[0]) * y_frac
    for _, row in bra_events.iterrows():
        st = EV_STYLE_V["chegada"]
        ax.axvline(row["chegada_torre_ts"], color=st["color"], ls=st["ls"], lw=st["lw"], alpha=0.7)
        ax.text(row["chegada_torre_ts"], y_pos,
                f"B{int(row['braco'])} chegada",
                color=st["color"], rotation=90, va='top', ha='right', fontsize=6, alpha=0.8)
    for _, row in real_events.iterrows():
        # Início
        st = EV_STYLE_V["inicio"]
        ax.axvline(row["inicio_pesoreal_ts"], color=st["color"], ls=st["ls"], lw=st["lw"], alpha=0.7)
        ax.text(row["inicio_pesoreal_ts"], y_pos,
                "inicio real", color=st["color"], rotation=90, va='top', ha='right', fontsize=6, alpha=0.8)
        # Fim
        if pd.notna(row["fim_pesoreal_ts"]):
            st = EV_STYLE_V["fim"]
            ax.axvline(row["fim_pesoreal_ts"], color=st["color"], ls=st["ls"], lw=st["lw"], alpha=0.7)
            ax.text(row["fim_pesoreal_ts"], y_pos,
                    "fim real", color=st["color"], rotation=90, va='top', ha='right', fontsize=6, alpha=0.8)

# ================================================================
# PLOT
# ================================================================
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

# Peso Braço 1
ax = axes[0]
ax.plot(df_p_bra["timestamp"], df_p_bra["ACI@LC_TORRE_PESOBRA1"] / 10.0,
        lw=0.5, color="#ff7f0e", alpha=0.9)
add_validation_lines(ax, bra_day, real_day)
ax.set_title("Peso Braço 1 (t)", fontsize=11, fontweight="bold")
ax.set_ylabel("toneladas")
ax.grid(True, alpha=0.25)

# Peso Braço 2
ax = axes[1]
ax.plot(df_p_bra["timestamp"], df_p_bra["ACI@LC_TORRE_PESOBRA2"] / 10.0,
        lw=0.5, color="#ff7f0e", alpha=0.9)
add_validation_lines(ax, bra_day, real_day)
ax.set_title("Peso Braço 2 (t)", fontsize=11, fontweight="bold")
ax.set_ylabel("toneladas")
ax.grid(True, alpha=0.25)

# Peso Real
ax = axes[2]
ax.plot(df_p_real["timestamp"], df_p_real["ACI@LC_TORRE_PESOREAL"].apply(pd.to_numeric, errors='coerce') / 10.0,
        lw=0.5, color="#9467bd", alpha=0.9)
add_validation_lines(ax, bra_day, real_day)
ax.set_title("Peso Real Torre (t)", fontsize=11, fontweight="bold")
ax.set_ylabel("toneladas")
ax.grid(True, alpha=0.25)

for ax in axes:
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.08)

# Legenda
from matplotlib.lines import Line2D
leg = [
    Line2D([0],[0], color=EV_STYLE_V["chegada"]["color"], ls="--", lw=1.3, label="chegada torre"),
    Line2D([0],[0], color=EV_STYLE_V["inicio"]["color"],  ls="-",  lw=1.3, label="início peso real"),
    Line2D([0],[0], color=EV_STYLE_V["fim"]["color"],     ls=":",  lw=1.5, label="fim peso real"),
]
fig.legend(handles=leg, loc="upper right", fontsize=9, framealpha=0.9)
fig.suptitle(f"Validação 7 dias — {DIA_PLOT} {H_INI_PLOT}–{H_FIM_PLOT} (detecção via src/)",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Resumo do dia
print(f"\n📊 Resumo {DIA_PLOT} ({H_INI_PLOT}–{H_FIM_PLOT}):")
print(f"   Chegadas torre: {len(bra_day)} (B1: {(bra_day['braco']==1).sum()}, B2: {(bra_day['braco']==2).sum()})")
print(f"   Ciclos peso real: {len(real_day)}")
for _, r in real_day.iterrows():
    dur = f"{r['duracao_s']/60:.1f}min" if pd.notna(r['duracao_s']) else "em andamento"
    print(f"     Seq {int(r['corrida_seq'])}: {r['inicio_pesoreal_ts'].strftime('%H:%M')} → "
          f"{r['fim_pesoreal_ts'].strftime('%H:%M') if pd.notna(r['fim_pesoreal_ts']) else '???'} ({dur})")

# COMMAND ----------

# DBTITLE 1,Emparelhamento de corridas e análise estatística 7 dias
# ================================================================
# EMPARELHAMENTO: associa chegada_torre → inicio_real → fim_real
# em ciclos (corridas) e calcula todas as métricas de timing.
#
# Lógica: cada inicio_pesoreal se associa à chegada_torre
# mais recente ANTES dele (consumindo sem repetir).
# ================================================================

# Ordena eventos
bra_sorted = df_bra_sys.sort_values("chegada_torre_ts").reset_index(drop=True)
real_sorted = df_real_sys.sort_values("inicio_pesoreal_ts").reset_index(drop=True)

print(f"🔗 Emparelhando {len(bra_sorted)} chegadas com {len(real_sorted)} ciclos peso real...")

# Para cada ciclo de peso real, encontra a chegada mais próxima ANTES
corridas_7d = []
chegadas_used = set()

for _, real_row in real_sorted.iterrows():
    t_ini_real = real_row["inicio_pesoreal_ts"]
    t_fim_real = real_row["fim_pesoreal_ts"]
    duracao = real_row["duracao_s"]

    # Busca a chegada mais recente ANTES do inicio_pesoreal
    candidates = bra_sorted[
        (bra_sorted["chegada_torre_ts"] < t_ini_real) &
        (~bra_sorted.index.isin(chegadas_used))
    ]

    if len(candidates) > 0:
        best_idx = candidates["chegada_torre_ts"].idxmax()  # mais recente
        best = candidates.loc[best_idx]
        chegadas_used.add(best_idx)

        corridas_7d.append({
            "braco": int(best["braco"]),
            "chegada_torre_ts": best["chegada_torre_ts"],
            "inicio_pesoreal_ts": t_ini_real,
            "fim_pesoreal_ts": t_fim_real,
            "duracao_vazamento_s": duracao,
        })
    else:
        # Sem chegada associada
        corridas_7d.append({
            "braco": np.nan,
            "chegada_torre_ts": pd.NaT,
            "inicio_pesoreal_ts": t_ini_real,
            "fim_pesoreal_ts": t_fim_real,
            "duracao_vazamento_s": duracao,
        })

df_corridas = pd.DataFrame(corridas_7d)
print(f"✅ {len(df_corridas)} corridas emparelhadas")
print(f"   Com chegada associada: {df_corridas['chegada_torre_ts'].notna().sum()}")
print(f"   Sem chegada: {df_corridas['chegada_torre_ts'].isna().sum()}")

# ================================================================
# CALCULA MÉTRICAS DE TIMING (em minutos)
# ================================================================

# 1) Chegada torre → Início peso real (giro da torre)
df_corridas["chegada_to_inicio_min"] = (
    (df_corridas["inicio_pesoreal_ts"] - df_corridas["chegada_torre_ts"])
    .dt.total_seconds() / 60
)

# 2) Duração do vazamento (início → fim peso real)
df_corridas["duracao_vazamento_min"] = df_corridas["duracao_vazamento_s"] / 60

# 3) Ciclo total (chegada → fim)
df_corridas["ciclo_total_min"] = (
    (df_corridas["fim_pesoreal_ts"] - df_corridas["chegada_torre_ts"])
    .dt.total_seconds() / 60
)

# 4) Fim corrida anterior → Chegada próxima (intervalo entre corridas)
df_corridas["fim_to_chegada_min"] = np.nan
df_corridas["fim_to_inicio_min"] = np.nan
for i in range(1, len(df_corridas)):
    prev_fim = df_corridas.at[i - 1, "fim_pesoreal_ts"]
    curr_cheg = df_corridas.at[i, "chegada_torre_ts"]
    curr_ini = df_corridas.at[i, "inicio_pesoreal_ts"]
    if pd.notna(prev_fim) and pd.notna(curr_cheg):
        df_corridas.at[i, "fim_to_chegada_min"] = (curr_cheg - prev_fim).total_seconds() / 60
    if pd.notna(prev_fim) and pd.notna(curr_ini):
        df_corridas.at[i, "fim_to_inicio_min"] = (curr_ini - prev_fim).total_seconds() / 60

# ================================================================
# TABELA RESUMO
# ================================================================
metric_cols = [
    "chegada_to_inicio_min", "duracao_vazamento_min",
    "ciclo_total_min", "fim_to_chegada_min", "fim_to_inicio_min",
]
metric_labels = {
    "chegada_to_inicio_min": "Chegada → Início real",
    "duracao_vazamento_min": "Duração vazamento",
    "ciclo_total_min": "Ciclo total",
    "fim_to_chegada_min": "Fim ant. → Chegada próx.",
    "fim_to_inicio_min": "Fim ant. → Início próx.",
}

print(f"\n{'='*85}")
print(f"📊 ESTATÍSTICAS DE TIMING — {len(df_corridas)} corridas ({DATA_INI} a {DATA_FIM})")
print(f"{'='*85}")
print(f"{'Métrica':<28} | {'N':>4} | {'Média':>7} | {'Std':>6} | {'P05':>6} | {'P25':>6} | {'P50':>6} | {'P75':>6} | {'P95':>6} | {'Min':>6} | {'Max':>6}")
print("-" * 120)

for col in metric_cols:
    data = df_corridas[col].dropna()
    if len(data) == 0:
        continue
    label = metric_labels[col]
    print(
        f"{label:<28} | {len(data):>4} | {data.mean():>7.1f} | {data.std():>6.1f} | "
        f"{data.quantile(0.05):>6.1f} | {data.quantile(0.25):>6.1f} | {data.median():>6.1f} | "
        f"{data.quantile(0.75):>6.1f} | {data.quantile(0.95):>6.1f} | {data.min():>6.1f} | {data.max():>6.1f}"
    )

# Atraso vs antecipação
ftc = df_corridas["fim_to_chegada_min"].dropna()
if len(ftc) > 0:
    n_esp = (ftc < 0).sum()
    n_atr = (ftc >= 0).sum()
    print(f"\n🚚 Atraso vs Antecipação (fim ant. → chegada próx.):")
    print(f"   ✅ Já esperando: {n_esp} ({n_esp/len(ftc)*100:.0f}%)")
    print(f"   ⚠️  Atrasou:      {n_atr} ({n_atr/len(ftc)*100:.0f}%)")

# ================================================================
# HISTOGRAMAS COM P05/P95 E OUTLIERS
# ================================================================
hist_specs = [
    ("chegada_to_inicio_min", "Chegada torre \u2192 In\u00edcio peso real\n(giro torre, min)", "#2ca02c"),
    ("duracao_vazamento_min", "Dura\u00e7\u00e3o vazamento\n(in\u00edcio \u2192 fim peso real, min)", "#1f77b4"),
    ("ciclo_total_min", "Ciclo total\n(chegada \u2192 fim, min)", "#9467bd"),
    ("fim_to_chegada_min", "Fim ant. \u2192 Chegada pr\u00f3x.\n(min, negativo = esperando)", "#d62728"),
    ("fim_to_inicio_min", "Fim ant. \u2192 In\u00edcio pr\u00f3x.\n(intervalo entre corridas, min)", "#ff7f0e"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (col, title, color) in enumerate(hist_specs):
    ax = axes[idx]
    data = df_corridas[col].dropna()
    if len(data) == 0:
        ax.set_visible(False)
        continue

    p05, p95 = data.quantile(0.05), data.quantile(0.95)
    mu, med, sigma = data.mean(), data.median(), data.std()

    # Filtra outliers para range visual (mostra P01-P99)
    p01, p99 = data.quantile(0.01), data.quantile(0.99)
    data_vis = data[(data >= p01) & (data <= p99)]
    n_outliers = len(data) - len(data_vis)

    n_bins = min(30, max(5, len(data_vis) // 3))
    ax.hist(data_vis, bins=n_bins, color=color, edgecolor="white", alpha=0.75)

    # Linhas de referência
    ax.axvline(mu, color="black", ls="-", lw=1.5, alpha=0.8, label=f"m\u00e9dia={mu:.1f}")
    ax.axvline(med, color="black", ls="--", lw=1.2, alpha=0.6, label=f"mediana={med:.1f}")
    ax.axvline(p05, color="gray", ls=":", lw=1.2, alpha=0.7, label=f"P05={p05:.1f}")
    ax.axvline(p95, color="gray", ls=":", lw=1.2, alpha=0.7, label=f"P95={p95:.1f}")

    # Zona "j\u00e1 esperando" (valores negativos)
    if (data < 0).any():
        ax.axvspan(ax.get_xlim()[0], 0, alpha=0.08, color="green")

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("minutos", fontsize=8)
    ax.set_ylabel("frequ\u00eancia", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, axis="y")
    ax.legend(fontsize=7, loc="upper right")

    # Anota\u00e7\u00e3o de stats
    stats_text = f"n={len(data)}  \u03c3={sigma:.1f}"
    if n_outliers > 0:
        stats_text += f"  ({n_outliers} outliers)"
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
            fontsize=7, va='top', color='gray')

# Esconde subplot vazio
for idx in range(len(hist_specs), len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(f"Distribui\u00e7\u00e3o dos Tempos LC \u2014 {len(df_corridas)} corridas ({DATA_INI} a {DATA_FIM})",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()

# ================================================================
# BOXPLOTS COMPARATIVOS
# ================================================================
box_cols = [c for c in metric_cols if df_corridas[c].notna().sum() >= 3]
box_labels = [metric_labels[c] for c in box_cols]

if box_cols:
    fig, ax = plt.subplots(figsize=(12, 5))
    box_data = [df_corridas[c].dropna().values for c in box_cols]
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    showmeans=True, meanline=True,
                    meanprops={"color": "red", "ls": "-", "lw": 1.5},
                    medianprops={"color": "black", "lw": 1.5},
                    flierprops={"marker": "o", "markersize": 4, "alpha": 0.5})
    colors_box = ["#2ca02c", "#1f77b4", "#9467bd", "#d62728", "#ff7f0e"]
    for patch, c in zip(bp["boxes"], colors_box[:len(box_cols)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.4)
    ax.set_ylabel("minutos", fontsize=10)
    ax.set_title(f"Boxplot dos Tempos LC \u2014 {len(df_corridas)} corridas", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    ax.tick_params(axis='x', rotation=15, labelsize=9)
    plt.tight_layout()
    plt.show()

print(f"\n✅ Análise completa! {len(df_corridas)} corridas em {df_corridas['chegada_torre_ts'].dt.date.nunique()} dias.")

# COMMAND ----------

# DBTITLE 1,Investigação de outliers - plots dos casos suspeitos
# ================================================================
# INVESTIGAÇÃO DE OUTLIERS
# Plota PESOBRA1, PESOBRA2, PESOREAL para os casos suspeitos
# ================================================================
from matplotlib.lines import Line2D

def plot_investigation(title, t_start, t_end, bra_events, real_events,
                      pdf_csv=pdf_csv_7d, pdf_opc=pdf_opc_7d, margin_min=30):
    """Plota 3 sinais com margem ao redor do período suspeito."""
    t0 = t_start - pd.Timedelta(minutes=margin_min)
    t1 = t_end + pd.Timedelta(minutes=margin_min)

    # Filtra dados
    m_csv = (pdf_csv["timestamp"] >= t0) & (pdf_csv["timestamp"] <= t1)
    m_opc = (pdf_opc["timestamp"] >= t0) & (pdf_opc["timestamp"] <= t1)
    d_bra = pdf_csv.loc[m_csv]
    d_real = pdf_opc.loc[m_opc]

    # Filtra eventos no período
    bra_w = bra_events[(bra_events["chegada_torre_ts"] >= t0) & (bra_events["chegada_torre_ts"] <= t1)]
    real_w = real_events[
        (real_events["inicio_pesoreal_ts"] >= t0) & (real_events["inicio_pesoreal_ts"] <= t1)
    ]

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

    # PESOBRA1
    ax = axes[0]
    ax.plot(d_bra["timestamp"], d_bra["ACI@LC_TORRE_PESOBRA1"] / 10.0, lw=0.5, color="#ff7f0e")
    ax.set_title("PESOBRA1 (t)", fontsize=10, fontweight="bold")
    ax.set_ylabel("t")
    ax.grid(True, alpha=0.2)

    # PESOBRA2
    ax = axes[1]
    ax.plot(d_bra["timestamp"], d_bra["ACI@LC_TORRE_PESOBRA2"] / 10.0, lw=0.5, color="#ff7f0e")
    ax.set_title("PESOBRA2 (t)", fontsize=10, fontweight="bold")
    ax.set_ylabel("t")
    ax.grid(True, alpha=0.2)

    # PESOREAL
    ax = axes[2]
    real_vals = pd.to_numeric(d_real["ACI@LC_TORRE_PESOREAL"], errors="coerce") / 10.0
    ax.plot(d_real["timestamp"], real_vals, lw=0.5, color="#9467bd")
    ax.set_title("PESOREAL (t)", fontsize=10, fontweight="bold")
    ax.set_ylabel("t")
    ax.grid(True, alpha=0.2)

    # Adiciona linhas de eventos em todos os subplots
    ev_st = {
        "chegada": {"color": "#2ca02c", "ls": "--", "lw": 1.2},
        "inicio":  {"color": "#1f77b4", "ls": "-",  "lw": 1.2},
        "fim":     {"color": "#d62728", "ls": ":",  "lw": 1.5},
    }
    for ax in axes:
        ylim = ax.get_ylim()
        y_top = ylim[0] + (ylim[1] - ylim[0]) * 0.92
        for _, row in bra_w.iterrows():
            ax.axvline(row["chegada_torre_ts"], color=ev_st["chegada"]["color"],
                       ls=ev_st["chegada"]["ls"], lw=ev_st["chegada"]["lw"], alpha=0.7)
            ax.text(row["chegada_torre_ts"], y_top, f"B{int(row['braco'])} cheg",
                    color=ev_st["chegada"]["color"], rotation=90, va='top', ha='right', fontsize=6)
        for _, row in real_w.iterrows():
            ax.axvline(row["inicio_pesoreal_ts"], color=ev_st["inicio"]["color"],
                       ls=ev_st["inicio"]["ls"], lw=ev_st["inicio"]["lw"], alpha=0.7)
            ax.text(row["inicio_pesoreal_ts"], y_top, "ini real",
                    color=ev_st["inicio"]["color"], rotation=90, va='top', ha='right', fontsize=6)
            if pd.notna(row["fim_pesoreal_ts"]):
                ax.axvline(row["fim_pesoreal_ts"], color=ev_st["fim"]["color"],
                           ls=ev_st["fim"]["ls"], lw=ev_st["fim"]["lw"], alpha=0.7)
                ax.text(row["fim_pesoreal_ts"], y_top, "fim real",
                        color=ev_st["fim"]["color"], rotation=90, va='top', ha='right', fontsize=6)

        # Área destacada do período suspeito
        ax.axvspan(t_start, t_end, alpha=0.06, color="red")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))

    leg = [
        Line2D([0],[0], color=ev_st["chegada"]["color"], ls="--", lw=1.2, label="chegada torre"),
        Line2D([0],[0], color=ev_st["inicio"]["color"],  ls="-",  lw=1.2, label="início real"),
        Line2D([0],[0], color=ev_st["fim"]["color"],     ls=":",  lw=1.5, label="fim real"),
    ]
    fig.legend(handles=leg, loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


# ================================================================
# CASO 1: DURAÇÃO VAZAMENTO > 65 MIN
# ================================================================
print("\n" + "="*80)
print("🚨 INVESTIGAÇÃO: Duração vazamento > 65 min")
print("="*80)

outliers_vaz = df_corridas[df_corridas["duracao_vazamento_min"] > 65].sort_values(
    "duracao_vazamento_min", ascending=False
)

for i, (_, r) in enumerate(outliers_vaz.head(3).iterrows()):
    dur = r["duracao_vazamento_min"]
    t_ini = r["inicio_pesoreal_ts"]
    t_fim = r["fim_pesoreal_ts"]
    bra = int(r["braco"]) if pd.notna(r["braco"]) else "?"
    print(f"\n--- Caso {i+1}: B{bra} | {t_ini} → {t_fim} | {dur:.1f} min ---")
    plot_investigation(
        title=f"OUTLIER Vaz. #{i+1}: B{bra} duração={dur:.1f}min | {t_ini.strftime('%d/%m %H:%M')}–{t_fim.strftime('%d/%m %H:%M')}",
        t_start=t_ini, t_end=t_fim,
        bra_events=df_bra_sys, real_events=df_real_sys,
        margin_min=60,
    )

# ================================================================
# CASO 2: FIM ANT. → CHEGADA PRÓX > 20 MIN
# ================================================================
print("\n" + "="*80)
print("🚨 INVESTIGAÇÃO: Fim ant. → Chegada próx. > 20 min")
print("="*80)

outliers_gap = df_corridas[df_corridas["fim_to_chegada_min"] > 20].sort_values(
    "fim_to_chegada_min", ascending=False
)

for i, (idx, r) in enumerate(outliers_gap.head(3).iterrows()):
    gap = r["fim_to_chegada_min"]
    prev = df_corridas.iloc[idx - 1] if idx > 0 else None
    t_fim_prev = prev["fim_pesoreal_ts"] if prev is not None else r["inicio_pesoreal_ts"] - pd.Timedelta(minutes=gap)
    t_cheg = r["chegada_torre_ts"]
    bra = int(r["braco"]) if pd.notna(r["braco"]) else "?"
    print(f"\n--- Caso {i+1}: gap={gap:.1f}min | fim ant: {t_fim_prev} → chegada B{bra}: {t_cheg} ---")
    plot_investigation(
        title=f"OUTLIER Gap #{i+1}: {gap:.1f}min sem corrida | {t_fim_prev.strftime('%d/%m %H:%M')}–{t_cheg.strftime('%d/%m %H:%M')}",
        t_start=t_fim_prev, t_end=t_cheg,
        bra_events=df_bra_sys, real_events=df_real_sys,
        margin_min=30,
    )

print(f"\n✅ Investigação concluída! Analise os gráficos acima.")

# COMMAND ----------

# DBTITLE 1,Análise tempos padrão — filtra paradas (>90min) e outliers IQR
# ================================================================
# ANÁLISE TEMPOS PADRÃO
# Timeout 90 min: duração > 90 min = parada / troca distribuidor
# Outliers IQR: remove pontos fora de [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
# ================================================================

TIMEOUT_VAZAMENTO_MIN = 90  # máximo aceitável para operação normal

# ----- 1. Classifica paradas vs operação normal -----
df_corridas["tipo"] = np.where(
    df_corridas["duracao_vazamento_min"] > TIMEOUT_VAZAMENTO_MIN,
    "parada", "normal"
)

n_parada = (df_corridas["tipo"] == "parada").sum()
n_normal = (df_corridas["tipo"] == "normal").sum()
print(f"⚙️  Timeout vazamento: {TIMEOUT_VAZAMENTO_MIN} min")
print(f"   🛑 Paradas / troca distribuidor: {n_parada}")
print(f"   ✅ Operação normal: {n_normal}")
print()

# Lista as paradas
if n_parada > 0:
    print("Corridas classificadas como PARADA:")
    for _, r in df_corridas[df_corridas["tipo"] == "parada"].iterrows():
        bra = int(r["braco"]) if pd.notna(r["braco"]) else "?"
        print(f"  B{bra} | {r['inicio_pesoreal_ts']} → {r['fim_pesoreal_ts']} | "
              f"duração: {r['duracao_vazamento_min']:.1f} min")

# ----- 2. Filtra somente operação normal -----
df_normal = df_corridas[df_corridas["tipo"] == "normal"].copy().reset_index(drop=True)

# Recalcula fim_to_chegada e fim_to_inicio para corridas normais consecutivas
df_normal["fim_to_chegada_min"] = np.nan
df_normal["fim_to_inicio_min"] = np.nan
for i in range(1, len(df_normal)):
    prev_fim = df_normal.at[i - 1, "fim_pesoreal_ts"]
    curr_cheg = df_normal.at[i, "chegada_torre_ts"]
    curr_ini = df_normal.at[i, "inicio_pesoreal_ts"]
    if pd.notna(prev_fim) and pd.notna(curr_cheg):
        df_normal.at[i, "fim_to_chegada_min"] = (curr_cheg - prev_fim).total_seconds() / 60
    if pd.notna(prev_fim) and pd.notna(curr_ini):
        df_normal.at[i, "fim_to_inicio_min"] = (curr_ini - prev_fim).total_seconds() / 60

# ----- 3. Remove outliers IQR nas métricas de tempo -----
def iqr_mask(series):
    """Retorna mask True = dentro do intervalo IQR (não-outlier)."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return (series >= lo) & (series <= hi)

metric_cols = [
    "chegada_to_inicio_min", "duracao_vazamento_min",
    "ciclo_total_min", "fim_to_chegada_min", "fim_to_inicio_min",
]

# Aplica IQR em cada métrica e mostra quantos outliers removidos
print("\n📊 Remoção de outliers residuais (IQR 1.5×):")
outlier_mask = pd.Series(True, index=df_normal.index)  # começa mantendo todos
for col in metric_cols:
    data = df_normal[col].dropna()
    if len(data) < 5:
        continue
    m = iqr_mask(df_normal[col])
    m = m | df_normal[col].isna()  # mantém NaN (não penaliza)
    n_out = (~m).sum()
    if n_out > 0:
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        print(f"  {col}: {n_out} outliers removidos "
              f"(range aceito: [{q1-1.5*iqr:.1f}, {q3+1.5*iqr:.1f}])")
    outlier_mask &= m

df_std = df_normal[outlier_mask].copy().reset_index(drop=True)
print(f"\n✅ Corridas padrão finais: {len(df_std)} de {len(df_corridas)} originais")
print(f"   Removidos: {n_parada} paradas + {len(df_normal) - len(df_std)} outliers IQR")

# ================================================================
# TABELA RESUMO — TEMPOS PADRÃO
# ================================================================
metric_labels = {
    "chegada_to_inicio_min": "Chegada → Início real",
    "duracao_vazamento_min": "Duração vazamento",
    "ciclo_total_min": "Ciclo total",
    "fim_to_chegada_min": "Fim ant. → Chegada próx.",
    "fim_to_inicio_min": "Fim ant. → Início próx.",
}

print(f"\n{'='*90}")
print(f"📊 ESTATÍSTICAS TEMPOS PADRÃO — {len(df_std)} corridas (sem paradas/outliers)")
print(f"{'='*90}")
print(f"{'Métrica':<28} | {'N':>4} | {'Média':>7} | {'Std':>6} | "
      f"{'P05':>6} | {'P25':>6} | {'P50':>6} | {'P75':>6} | {'P95':>6} | {'Min':>6} | {'Max':>6}")
print("-" * 120)

for col in metric_cols:
    data = df_std[col].dropna()
    if len(data) == 0:
        continue
    label = metric_labels[col]
    print(
        f"{label:<28} | {len(data):>4} | {data.mean():>7.1f} | {data.std():>6.1f} | "
        f"{data.quantile(0.05):>6.1f} | {data.quantile(0.25):>6.1f} | {data.median():>6.1f} | "
        f"{data.quantile(0.75):>6.1f} | {data.quantile(0.95):>6.1f} | {data.min():>6.1f} | {data.max():>6.1f}"
    )

# Atraso vs antecipação (filtrado)
ftc = df_std["fim_to_chegada_min"].dropna()
if len(ftc) > 0:
    n_esp = (ftc < 0).sum()
    n_atr = (ftc >= 0).sum()
    print(f"\n🚚 Atraso vs Antecipação (corridas padrão):")
    print(f"   ✅ Já esperando: {n_esp} ({n_esp/len(ftc)*100:.0f}%)")
    print(f"   ⚠️  Atrasou:      {n_atr} ({n_atr/len(ftc)*100:.0f}%)")

# ================================================================
# HISTOGRAMAS — TEMPOS PADRÃO
# ================================================================
hist_specs = [
    ("chegada_to_inicio_min", "Chegada torre → Início peso real\n(giro torre, min)", "#2ca02c"),
    ("duracao_vazamento_min", "Duração vazamento\n(início → fim peso real, min)", "#1f77b4"),
    ("ciclo_total_min", "Ciclo total\n(chegada → fim, min)", "#9467bd"),
    ("fim_to_chegada_min", "Fim ant. → Chegada próx.\n(min, negativo = esperando)", "#d62728"),
    ("fim_to_inicio_min", "Fim ant. → Início próx.\n(intervalo entre corridas, min)", "#ff7f0e"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (col, title, color) in enumerate(hist_specs):
    ax = axes[idx]
    data = df_std[col].dropna()
    if len(data) == 0:
        ax.set_visible(False)
        continue

    p05, p95 = data.quantile(0.05), data.quantile(0.95)
    mu, med, sigma = data.mean(), data.median(), data.std()

    n_bins = min(30, max(5, len(data) // 3))
    ax.hist(data, bins=n_bins, color=color, edgecolor="white", alpha=0.75)

    ax.axvline(mu, color="black", ls="-", lw=1.5, alpha=0.8, label=f"média={mu:.1f}")
    ax.axvline(med, color="black", ls="--", lw=1.2, alpha=0.6, label=f"mediana={med:.1f}")
    ax.axvline(p05, color="gray", ls=":", lw=1.2, alpha=0.7, label=f"P05={p05:.1f}")
    ax.axvline(p95, color="gray", ls=":", lw=1.2, alpha=0.7, label=f"P95={p95:.1f}")

    if (data < 0).any():
        ax.axvspan(ax.get_xlim()[0], 0, alpha=0.08, color="green")

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("minutos", fontsize=8)
    ax.set_ylabel("frequência", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, axis="y")
    ax.legend(fontsize=7, loc="upper right")
    ax.text(0.02, 0.95, f"n={len(data)}  σ={sigma:.1f}",
            transform=ax.transAxes, fontsize=7, va='top', color='gray')

for idx in range(len(hist_specs), len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(f"Distribuição Tempos Padrão LC — {len(df_std)} corridas (sem paradas/outliers)",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()

# ================================================================
# BOXPLOTS — COMPARAÇÃO ANTES vs DEPOIS
# ================================================================
box_cols = [c for c in metric_cols if df_std[c].notna().sum() >= 3]
box_labels = [metric_labels[c].replace(" → ", "→\n").replace(" ant. ", " ant.\n") for c in box_cols]

if box_cols:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    colors_box = ["#2ca02c", "#1f77b4", "#9467bd", "#d62728", "#ff7f0e"]

    # ANTES (todos)
    ax = axes[0]
    box_data_all = [df_corridas[c].dropna().values for c in box_cols]
    bp1 = ax.boxplot(box_data_all, labels=box_labels, patch_artist=True,
                     showmeans=True, meanline=True,
                     meanprops={"color": "red", "ls": "-", "lw": 1.5},
                     medianprops={"color": "black", "lw": 1.5},
                     flierprops={"marker": "o", "markersize": 4, "alpha": 0.5})
    for patch, c in zip(bp1["boxes"], colors_box[:len(box_cols)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.4)
    ax.set_ylabel("minutos")
    ax.set_title(f"ANTES — Todas {len(df_corridas)} corridas", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    ax.tick_params(axis='x', rotation=15, labelsize=8)

    # DEPOIS (filtrado)
    ax = axes[1]
    box_data_std = [df_std[c].dropna().values for c in box_cols]
    bp2 = ax.boxplot(box_data_std, labels=box_labels, patch_artist=True,
                     showmeans=True, meanline=True,
                     meanprops={"color": "red", "ls": "-", "lw": 1.5},
                     medianprops={"color": "black", "lw": 1.5},
                     flierprops={"marker": "o", "markersize": 4, "alpha": 0.5})
    for patch, c in zip(bp2["boxes"], colors_box[:len(box_cols)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.4)
    ax.set_ylabel("minutos")
    ax.set_title(f"DEPOIS — {len(df_std)} corridas padrão", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    ax.tick_params(axis='x', rotation=15, labelsize=8)

    fig.suptitle("Comparação: Antes vs Depois da filtragem (paradas + IQR)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

print(f"\n✅ df_std pronto: {len(df_std)} corridas padrão para análise")