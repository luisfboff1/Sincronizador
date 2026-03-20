# Databricks notebook source
# DBTITLE 1,Imports e Configuração
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

# Configuracao
TIMEZONE = "America/Sao_Paulo"
TABLE_CSV = "industrial_composicaoquimicacha_refined_dev.csv_valid.csv_concat"
DIA_VALIDACAO = "04.02.2026"  # Dia para validacao

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
    # Digitais (botoes, valvulas, abobada)
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
LC_COLS_CSV = [
    "ACI@LC_TORRE_PESOBRA1",
    "ACI@LC_TORRE_PESOBRA2",
    "CMD_ABRE_VALV_GAVETA",
    "CMD_FECHA_VALV_GAVETA",
]

LC_COLS_OPC = [
    "ACI@LC_TORRE_PESOREAL",
    "ACI@LC_DISTRIB_PESO",
    "ACI@LC_DISTRIB_TEMP",
    "ACI@LC_GERAL_NUMCORR",
    "ACI@LC_EXTEND_VEL2V1",
    "ACI@LC_EXTEND_VEL2V2",
    "ACI@LC_EXTEND_VEL2V3",
]

LC_COLS = LC_COLS_CSV + LC_COLS_OPC

print("Imports e configuracao carregados")
print(f"  DIA: {DIA_VALIDACAO}")
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

# DBTITLE 1,Tempos observados manualmente - FEA e LC (PREENCHER)
# ================================================================
# TEMPOS OBSERVADOS + MES - FEA dia 04/02/2026
# ================================================================

def parse_ts(s):
    if s is None:
        return None
    for fmt in ["%d.%m.%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"]:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None

# ---------- Tempos observados manualmente ----------
tempos_obs = {
    "OBS inicio_fp":    parse_ts("04.02.2026 16:17:00"),
    "OBS limpa_porta":  parse_ts("04.02.2026 16:35:00"),
}

# ---------- Tempos MES (tb_admaci_fea_corridas) ----------
# Corrida 131261: HRVAZAMENTO=15:56, TTT=236min (!), Temp=1680, Peso=62.0t, Turma A
# Corrida 131262: HRVAZAMENTO=16:58, TTT=59min,  Temp=1717, Peso=62.3t, Turma B
# Corrida 131263: HRVAZAMENTO=17:51, TTT=53min,  Temp=1680, Peso=62.5t, Turma B
tempos_mes = {
    "MES vaz 131261 (15:56)": parse_ts("04.02.2026 15:56:00"),
    "MES vaz 131262 (16:58)": parse_ts("04.02.2026 16:58:00"),
    # "MES vaz 131263 (17:51)": parse_ts("04.02.2026 17:51:00"),
}

# ---------- Estilos ----------
EVENT_STYLES = {
    # Observados (tracejado)
    "OBS inicio_fp":         {"color": "green",  "ls": "--", "lw": 2.0},
    "OBS limpa_porta":       {"color": "orange", "ls": "--", "lw": 2.0},
    # MES (solido)
    "MES vaz 131261 (15:56)": {"color": "red",    "ls": "-",  "lw": 2.0},
    "MES vaz 131262 (16:58)": {"color": "blue",   "ls": "-",  "lw": 2.0},
    # "MES vaz 131263 (17:51)": {"color": "purple", "ls": "-",  "lw": 2.0},
}

# Combina todos os tempos
tempos_all = {**tempos_obs, **tempos_mes}

def add_event_lines(ax, tempos_dict):
    """Adiciona linhas verticais dos eventos."""
    for nome, ts in tempos_dict.items():
        if ts is not None:
            style = EVENT_STYLES.get(nome, {"color": "gray", "ls": "--", "lw": 1})
            ax.axvline(ts, color=style["color"], linestyle=style["ls"],
                       linewidth=style["lw"], alpha=0.8)

# Resumo
print("Tempos observados manualmente:")
for nome, ts in tempos_obs.items():
    if ts is not None:
        print(f"  {nome:30s} -> {ts.strftime('%H:%M:%S')}")

print("\nTempos MES (operador):")
for nome, ts in tempos_mes.items():
    if ts is not None:
        print(f"  {nome:30s} -> {ts.strftime('%H:%M:%S')}")

print(f"\nTotal eventos para plotar: {sum(1 for v in tempos_all.values() if v is not None)}")

# COMMAND ----------

# DBTITLE 1,Plot FEA - Tags do dia 20/02 com linhas de eventos
# ================================================================
# PLOT FEA - Todas as tags FEA do dia com linhas verticais
# ================================================================

# Le widgets de hora
HORA_INI = dbutils.widgets.get("hora_ini")
HORA_FIM = dbutils.widgets.get("hora_fim")

pdf_fea = pdf[(pdf["timestamp"].dt.strftime("%H:%M") >= HORA_INI) & (pdf["timestamp"].dt.strftime("%H:%M") < HORA_FIM)].copy()
print(f"Periodo: {HORA_INI} - {HORA_FIM}  |  {len(pdf_fea):,} registros")

# Filtra colunas FEA disponiveis no dataframe
fea_available = [c for c in FEA_COLS if c in pdf_fea.columns]

# Separa em analogos vs digitais (0/1)
fea_analog = []
fea_digital = []
for col in fea_available:
    vals = pdf_fea[col].dropna()
    if len(vals) > 0 and set(vals.unique()).issubset({0.0, 1.0, 0, 1}):
        fea_digital.append(col)
    else:
        fea_analog.append(col)

print(f"FEA analogas:  {len(fea_analog)} tags")
print(f"FEA digitais:  {len(fea_digital)} tags")

# Monta legenda de eventos
def _build_legend():
    handles, labels = [], []
    for nome, ts in tempos_all.items():
        if ts is not None:
            style = EVENT_STYLES.get(nome, {"color": "gray", "ls": "--", "lw": 1})
            handles.append(plt.Line2D([0],[0], color=style["color"], ls=style["ls"], lw=style["lw"]))
            labels.append(nome)
    return handles, labels

# --- Plot tags analogicas ---
if fea_analog:
    n_cols_plot = 3
    n_rows_plot = int(np.ceil(len(fea_analog) / n_cols_plot))
    fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(22, n_rows_plot * 3.5), sharex=True)
    axes = np.atleast_2d(axes).flatten()

    for idx, col in enumerate(fea_analog):
        ax = axes[idx]
        ax.plot(pdf_fea["timestamp"], pdf_fea[col], linewidth=0.5, color="steelblue")
        add_event_lines(ax, tempos_all)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.grid(True, alpha=0.3)

    for idx in range(len(fea_analog), len(axes)):
        axes[idx].set_visible(False)

    handles, labels = _build_legend()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8, ncol=2)

    fig.suptitle(f"FEA - Tags Analogicas ({DIA_VALIDACAO} {HORA_INI}-{HORA_FIM})", fontsize=14, fontweight="bold")
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
        add_event_lines(ax, tempos_all)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_ylim(-0.1, 1.3)
        ax.set_yticks([0, 1])
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.grid(True, alpha=0.3)

    for idx in range(len(fea_digital), len(axes)):
        axes[idx].set_visible(False)

    handles, labels = _build_legend()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8, ncol=2)

    fig.suptitle(f"FEA - Tags Digitais ({DIA_VALIDACAO} {HORA_INI}-{HORA_FIM})", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# COMMAND ----------

# DBTITLE 1,FSM Detector - FEA Energia (inicio, pausas, fim)
import sys, importlib

proj_path = "/Workspace/Users/execs1@ex.gerdau.com/Sincronizador"
if proj_path not in sys.path:
    sys.path.insert(0, proj_path)

# Forca reload para pegar versao mais recente do .py
import src.detection.fea_energia
importlib.reload(src.detection.fea_energia)
from src.detection.fea_energia import detect_fea_energia

print("detect_fea_energia importada de src/detection/fea_energia.py")
print(f"  Parametros default: RATE_RISE_T=3.0, FLAT_CONFIRM_S=30, MIN_PAUSE_S=60")

# COMMAND ----------

# DBTITLE 1,Executa detector e plota energia com eventos
# ================================================================
# EXECUTA DETECTOR FEA ENERGIA + PLOT VALIDACAO
# ================================================================

# Paleta EvcomX
EVCOMX_ORANGE = ["#FFF5F2","#FFE3D9","#FFC7B2","#FFA88C","#FF8C66","#FF7040","#FF541A","#FF4000"]
EVCOMX_GRAY   = ["#F2F5F5","#DBDEDE","#BABDBF","#969C9E","#73787D","#52595E","#2E363D","#172129"]

# Prepara dataframe
df_en = pdf[["timestamp", "ACI@FEA_ELET_ENERGIA"]].dropna().copy()
print(f"Registros com energia: {len(df_en):,}")
print(f"Periodo: {df_en['timestamp'].min()} -> {df_en['timestamp'].max()}")

# Roda detector com defaults do .py (v3)
df_sys, df_events, df_debug = detect_fea_energia(df_en)

print(f"\n=== RESULTADOS DIA {DIA_VALIDACAO} ===")
print(f"Ciclos detectados: {len(df_sys)}")

if len(df_sys) > 0:
    print(f"\n{'Seq':>4} | {'Inicio':>8} | {'Fim':>8} | {'Dur':>5} | {'E_pico':>6} | {'Pausas':>6} | {'Pausa_s':>7}")
    print("-" * 65)
    for _, row in df_sys.iterrows():
        ini_str = str(row['inicio_ts'].strftime('%H:%M')) if hasattr(row['inicio_ts'], 'strftime') else str(row['inicio_ts'])[-8:-3]
        fim_str = str(row['fim_ts'].strftime('%H:%M')) if hasattr(row['fim_ts'], 'strftime') else str(row['fim_ts'])[-8:-3]
        print(f"{row['ciclo_seq']:>4} | {ini_str:>8} | {fim_str:>8} | {row['duracao_s']/60:>5.1f} | {row['energia_pico']:>6.0f} | {row['n_pausas']:>6} | {row['pausas_total_s']:>7.0f}")

# Eventos na janela
HORA_INI = dbutils.widgets.get("hora_ini")
HORA_FIM = dbutils.widgets.get("hora_fim")

pdf_plot = pdf[(pdf["timestamp"].dt.strftime("%H:%M") >= HORA_INI) &
               (pdf["timestamp"].dt.strftime("%H:%M") < HORA_FIM)].copy()
t_ini = pdf_plot["timestamp"].min()
t_fim = pdf_plot["timestamp"].max()

if len(df_events) > 0:
    ev_window = df_events[
        (df_events["event_ts"] >= t_ini) & (df_events["event_ts"] <= t_fim)
    ].sort_values("event_ts")
    print(f"\nEventos na janela {HORA_INI}-{HORA_FIM}: {len(ev_window)}")
    for _, ev in ev_window.iterrows():
        print(f"  [{ev['event_type']:>10}] C{ev['ciclo_seq']:>2} | {ev['event_ts'].strftime('%H:%M:%S')}")

# ================================================================
# PLOT - Estilo EvcomX: sem grid, paleta laranja/cinza
# ================================================================
fig, ax = plt.subplots(figsize=(20, 5))

# Sinal de energia
ax.plot(pdf_plot["timestamp"], pdf_plot["ACI@FEA_ELET_ENERGIA"],
        linewidth=0.7, color=EVCOMX_GRAY[5], label="Energia FEA")

# Estilos FSM na paleta EvcomX
EVENT_STYLES_FSM = {
    "INICIO":    {"color": EVCOMX_ORANGE[7], "ls": "-",  "lw": 2.0, "tag": "INICIO"},
    "FIM":       {"color": EVCOMX_GRAY[7],   "ls": "-",  "lw": 2.0, "tag": "FIM"},
    "PAUSA_INI": {"color": EVCOMX_ORANGE[4], "ls": "--", "lw": 1.5, "tag": "PAUSA"},
    "PAUSA_FIM": {"color": EVCOMX_GRAY[4],   "ls": "--", "lw": 1.5, "tag": "RETOMADA"},
}

if len(df_events) > 0:
    for _, ev in ev_window.iterrows():
        etype = ev["event_type"]
        style = EVENT_STYLES_FSM.get(etype, {"color": EVCOMX_GRAY[3], "ls": "--", "lw": 1, "tag": etype})
        label_txt = f"{style['tag']} C{ev['ciclo_seq']} ({ev['event_ts'].strftime('%H:%M')})"
        ax.axvline(ev["event_ts"], color=style["color"], linestyle=style["ls"],
                   linewidth=style["lw"], alpha=0.85, label=label_txt)

    # Sombreia pausas
    pausas_ini = ev_window[ev_window["event_type"] == "PAUSA_INI"]
    pausas_fim = ev_window[ev_window["event_type"] == "PAUSA_FIM"]
    for _, pi in pausas_ini.iterrows():
        pf_match = pausas_fim[
            (pausas_fim["ciclo_seq"] == pi["ciclo_seq"]) &
            (pausas_fim["event_ts"] > pi["event_ts"])
        ]
        if len(pf_match) > 0:
            pf_ts = pf_match.iloc[0]["event_ts"]
            ax.axvspan(pi["event_ts"], pf_ts, alpha=0.10, color=EVCOMX_ORANGE[3])

ax.set_title(f"FEA Energia - Deteccao FSM  ({DIA_VALIDACAO}  {HORA_INI}-{HORA_FIM})",
             fontsize=13, fontweight="bold", color=EVCOMX_GRAY[7])
ax.set_ylabel("Energia (kWh)", fontsize=10, color=EVCOMX_GRAY[6])
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.tick_params(axis="x", rotation=45, labelsize=8, colors=EVCOMX_GRAY[5])
ax.tick_params(axis="y", labelsize=8, colors=EVCOMX_GRAY[5])
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(EVCOMX_GRAY[2])
ax.spines["bottom"].set_color(EVCOMX_GRAY[2])
ax.legend(loc="upper left", fontsize=7, ncol=2, frameon=False)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,FEA Energia - Deteccao FSM (fevereiro completo)
# ================================================================
# FEA ENERGIA - DETECCAO FSM PARA FEVEREIRO COMPLETO
# ================================================================

# Carrega CSV completo com coluna de energia FEA
MES_FEA = "02.2026"
print(f"Carregando CSV com ACI@FEA_ELET_ENERGIA para {MES_FEA}...")

df_raw_csv = (
    spark.read.table(TABLE_CSV)
    .withColumn("_row_id", F.monotonically_increasing_id())
)
first_col = df_raw_csv.columns[0]
hdr_id = (
    df_raw_csv.filter(F.col(first_col) == "nomePIMS")
    .select("_row_id").orderBy("_row_id").first()["_row_id"]
)
hdr = df_raw_csv.filter(F.col("_row_id") == hdr_id).drop("_row_id").first()
new_cols = ["timestamp" if str(v).strip() == "nomePIMS" else str(v).strip()
            for v in hdr]
df_csv_all = df_raw_csv.filter(F.col("_row_id") > hdr_id).drop("_row_id")
for i, n in enumerate(new_cols):
    df_csv_all = df_csv_all.withColumnRenamed(df_csv_all.columns[i], n)

# Filtra fevereiro e seleciona apenas timestamp + energia
df_csv_fea = (
    df_csv_all
    .filter(F.col("timestamp").contains(f".{MES_FEA.split('.')[0]}.{MES_FEA.split('.')[1]}"))
    .select("timestamp", "ACI@FEA_ELET_ENERGIA")
)
pdf_fea = df_csv_fea.toPandas()
pdf_fea["timestamp"] = pd.to_datetime(
    pdf_fea["timestamp"], format="%d.%m.%Y %H:%M:%S.%f", errors="coerce"
)
pdf_fea["ACI@FEA_ELET_ENERGIA"] = (
    pdf_fea["ACI@FEA_ELET_ENERGIA"].str.replace(",", ".", regex=False)
)
pdf_fea["ACI@FEA_ELET_ENERGIA"] = pd.to_numeric(
    pdf_fea["ACI@FEA_ELET_ENERGIA"], errors="coerce"
)
pdf_fea = pdf_fea.dropna().sort_values("timestamp").reset_index(drop=True)
print(f"   Registros: {len(pdf_fea):,}")
print(f"   Periodo:   {pdf_fea['timestamp'].min()} -> {pdf_fea['timestamp'].max()}")
print(f"   Dias:      {pdf_fea['timestamp'].dt.date.nunique()}")

# Roda detector FSM (defaults v3)
print(f"\nRodando detect_fea_energia...")
df_fea_sys, df_fea_events, _ = detect_fea_energia(pdf_fea)

print(f"\n{'='*70}")
print(f"RESULTADOS FEA - Fevereiro {MES_FEA}")
print(f"{'='*70}")
print(f"Ciclos detectados: {len(df_fea_sys)}")
print(f"Eventos totais:    {len(df_fea_events)}")
if len(df_fea_sys) > 0:
    print(f"Duracao media:     {df_fea_sys['duracao_s'].mean()/60:.1f} min")
    print(f"Pausas media:      {df_fea_sys['n_pausas'].mean():.1f} por ciclo")
    print(f"Energia delta:     {df_fea_sys['energia_delta'].mean():.0f} (media)")

# COMMAND ----------

# DBTITLE 1,FEA Energia - Filtragem, estatisticas e distribuicoes
# ================================================================
# FEA ENERGIA - FILTRAGEM + ESTATISTICAS + DISTRIBUICOES
# Usa src.viz para filtragem, stats, histogramas e boxplots
# ================================================================

from src.viz import (
    EVCOMX_ORANGE, EVCOMX_GRAY,
    filtrar_ciclos_iqr, print_stats,
    plot_histogramas, plot_boxplot_antes_depois,
)

# --- Prepara metricas em minutos ---
df_fea_m = df_fea_sys.copy()
df_fea_m["duracao_min"] = df_fea_m["duracao_s"] / 60
df_fea_m["pausas_total_min"] = df_fea_m["pausas_total_s"] / 60
df_fea_m["aquecimento_efetivo_min"] = (
    df_fea_m["duracao_min"] - df_fea_m["pausas_total_min"]
)
df_fea_m["pausa_media_min"] = np.where(
    df_fea_m["n_pausas"] > 0,
    df_fea_m["pausas_total_min"] / df_fea_m["n_pausas"],
    0
)

FEA_METRIC_COLS = ["duracao_min", "pausas_total_min", "aquecimento_efetivo_min"]
FEA_METRIC_LABELS = {
    "duracao_min": "Duracao ciclo FEA",
    "pausas_total_min": "Tempo total pausas",
    "aquecimento_efetivo_min": "Aquecimento efetivo",
}

# --- Filtragem 3 etapas ---
df_fea_std, df_fea_antes = filtrar_ciclos_iqr(
    df_fea_m,
    metric_cols=FEA_METRIC_COLS,
    metric_labels=FEA_METRIC_LABELS,
    timeout_col="duracao_min",
    timeout_max_min=90,
    timeout_label="Timeout duracao > 90 min",
)

# --- Estatisticas ---
print_stats(
    df_fea_std,
    metric_cols=FEA_METRIC_COLS,
    metric_labels=FEA_METRIC_LABELS,
    titulo=f"ESTATISTICAS FEA -- {len(df_fea_std)} ciclos padrao (Fev/2026)",
)

print(f"\nPausas por ciclo (padrao):")
print(f"  Media: {df_fea_std['n_pausas'].mean():.1f}")
print(f"  Min:   {df_fea_std['n_pausas'].min():.0f}")
print(f"  Max:   {df_fea_std['n_pausas'].max():.0f}")

# --- Histogramas ---
plot_histogramas(
    df_fea_std,
    hist_specs=[
        ("duracao_min",              "Duracao ciclo FEA (min)",      EVCOMX_ORANGE[7]),
        ("pausas_total_min",         "Tempo total pausas (min)",     EVCOMX_ORANGE[4]),
        ("aquecimento_efetivo_min",  "Aquecimento efetivo (min)",    EVCOMX_GRAY[5]),
    ],
    titulo=f"Distribuicao Tempos FEA -- {len(df_fea_std)} ciclos padrao (Fev/2026)",
)

# --- Boxplot antes vs depois ---
plot_boxplot_antes_depois(
    df_fea_antes,
    df_fea_std,
    metric_cols=FEA_METRIC_COLS,
    metric_labels=FEA_METRIC_LABELS,
    box_colors=[EVCOMX_ORANGE[7], EVCOMX_ORANGE[4], EVCOMX_GRAY[5]],
    titulo="FEA Energia: Antes vs Depois da filtragem",
)

# COMMAND ----------

# DBTITLE 1,FEA - Tendencia temporal do aquecimento (mes, semana, dia)
# ================================================================
# TENDENCIA TEMPORAL - DURACAO AQUECIMENTO POR CORRIDA
# 3 niveis de zoom: mes completo, 7 dias, 1 dia
# ================================================================

from src.viz import plot_tendencia_3zooms, EVCOMX_ORANGE, EVCOMX_GRAY

# Prepara dados
df_trend = df_fea_std[["ciclo_seq", "inicio_ts", "duracao_min",
                        "pausas_total_min", "aquecimento_efetivo_min"]].copy()
df_trend = df_trend.sort_values("inicio_ts").reset_index(drop=True)

METRICS_FEA = [
    ("duracao_min",              "Duracao total",       EVCOMX_ORANGE[5], EVCOMX_ORANGE[7]),
    ("aquecimento_efetivo_min",  "Aquecimento efetivo", EVCOMX_GRAY[4],   EVCOMX_GRAY[6]),
]

plot_tendencia_3zooms(
    df_trend,
    col_ts="inicio_ts",
    metrics=METRICS_FEA,
    titulo_base="FEA Energia",
)

# COMMAND ----------

# DBTITLE 1,FEA - Linha temporal corrida-a-corrida (mes, semana, dia)
# ================================================================
# LINHA TEMPORAL - CONECTANDO CORRIDAS SEQUENCIALMENTE
# Mesmo dado do plot anterior, mas com 2 metricas lado-a-lado
# ================================================================

from src.viz import plot_tendencia_3zooms, EVCOMX_ORANGE, EVCOMX_GRAY

# Inclui pausas como 3a metrica
METRICS_FEA_DETAIL = [
    ("duracao_min",              "Duracao total",       EVCOMX_ORANGE[5], EVCOMX_ORANGE[7]),
    ("aquecimento_efetivo_min",  "Aquecimento efetivo", EVCOMX_GRAY[4],   EVCOMX_GRAY[6]),
    ("pausas_total_min",         "Pausas",              EVCOMX_ORANGE[2], EVCOMX_ORANGE[4]),
]

plot_tendencia_3zooms(
    df_trend,
    col_ts="inicio_ts",
    metrics=METRICS_FEA_DETAIL,
    titulo_base="FEA Energia (detalhe)",
)

# COMMAND ----------

# DBTITLE 1,Detector FSM - Peso Carro Panela
import sys, importlib

proj_path = "/Workspace/Users/execs1@ex.gerdau.com/Sincronizador"
if proj_path not in sys.path:
    sys.path.insert(0, proj_path)

# Forca reload para pegar versao mais recente do .py
import src.detection.fea_peso_carro_panela
importlib.reload(src.detection.fea_peso_carro_panela)
from src.detection.fea_peso_carro_panela import detect_peso_carro_panela

print("detect_peso_carro_panela importada de src/detection/fea_peso_carro_panela.py")
print(f"  Parametros default: TH_CHEIO=48, MIN_CHEIO_S=60, PLATO_RATE_MAX=0.5")

# COMMAND ----------

# DBTITLE 1,Peso Carro Panela - Deteccao FSM (fevereiro completo)
# ================================================================
# PESO CARRO PANELA - DETECCAO FSM PARA FEVEREIRO COMPLETO
# ================================================================

# Reusa df_csv_all (ja em memoria, com todas as colunas)
MES_PCP = "02.2026"
print(f"Filtrando PESO_CARRO_PANELA para {MES_PCP}...")

df_csv_pcp = (
    df_csv_all
    .filter(F.col("timestamp").contains(f".{MES_PCP.split('.')[0]}.{MES_PCP.split('.')[1]}"))
    .select("timestamp", "PESO_CARRO_PANELA")
)
pdf_pcp = df_csv_pcp.toPandas()
pdf_pcp["timestamp"] = pd.to_datetime(
    pdf_pcp["timestamp"], format="%d.%m.%Y %H:%M:%S.%f", errors="coerce"
)
pdf_pcp["PESO_CARRO_PANELA"] = (
    pdf_pcp["PESO_CARRO_PANELA"].str.replace(",", ".", regex=False)
)
pdf_pcp["PESO_CARRO_PANELA"] = pd.to_numeric(
    pdf_pcp["PESO_CARRO_PANELA"], errors="coerce"
)
pdf_pcp = pdf_pcp.dropna().sort_values("timestamp").reset_index(drop=True)
print(f"   Registros: {len(pdf_pcp):,}")
print(f"   Periodo:   {pdf_pcp['timestamp'].min()} -> {pdf_pcp['timestamp'].max()}")
print(f"   Dias:      {pdf_pcp['timestamp'].dt.date.nunique()}")

# Roda detector FSM
print(f"\nRodando detect_peso_carro_panela...")
df_pcp_sys, df_pcp_events, df_pcp_debug = detect_peso_carro_panela(pdf_pcp)

print(f"\n{'='*90}")
print(f"RESULTADOS PESO CARRO PANELA - Fevereiro {MES_PCP}")
print(f"{'='*90}")
print(f"Ciclos detectados: {len(df_pcp_sys)}")
print(f"Eventos totais:    {len(df_pcp_events)}")

if len(df_pcp_sys) > 0:
    print(f"\n{'Seq':>4} | {'Chegou':>8} | {'Tara':>8} | {'Vaz':>8} | {'Plato':>8} | "
          f"{'Saiu':>8} | {'Reset':>8} | {'P.Max':>6} | {'Plato_s':>7} | "
          f"{'Espera':>7} | {'Vaz_s':>6} | {'Neg_s':>6}")
    print("-" * 115)
    for _, r in df_pcp_sys.iterrows():
        def _fmt(ts):
            return ts.strftime('%H:%M') if pd.notna(ts) and hasattr(ts, 'strftime') else '  --  '
        print(f"{r['ciclo_seq']:>4} | "
              f"{_fmt(r['ts_panela_chegou']):>8} | "
              f"{_fmt(r['ts_tara']):>8} | "
              f"{_fmt(r['ts_vazamento_inicio']):>8} | "
              f"{_fmt(r['ts_plato_inicio']):>8} | "
              f"{_fmt(r['ts_panela_saiu']):>8} | "
              f"{_fmt(r['ts_tara_reset']):>8} | "
              f"{r['peso_max']:>6.1f} | "
              f"{r['plato_dur_s']:>7.0f} | "
              f"{r.get('dur_panela_espera_s', 0) or 0:>7.0f} | "
              f"{r.get('dur_vazamento_s', 0) or 0:>6.0f} | "
              f"{r.get('dur_negativo_s', 0) or 0:>6.0f}")

    print(f"\n--- Resumo ---")
    print(f"Peso max medio:        {df_pcp_sys['peso_max'].mean():.1f} t")
    print(f"Plato medio:           {df_pcp_sys['plato_dur_s'].mean()/60:.1f} min")
    espera = df_pcp_sys['dur_panela_espera_s'].dropna()
    if len(espera) > 0:
        print(f"Espera panela media:   {espera.mean()/60:.1f} min")
    vaz = df_pcp_sys['dur_vazamento_s'].dropna()
    if len(vaz) > 0:
        print(f"Vazamento medio:       {vaz.mean()/60:.1f} min")

# COMMAND ----------

# DBTITLE 1,Peso Carro Panela - Plot validacao (dia com mais ciclos)
# ================================================================
# PESO CARRO PANELA - PLOT VALIDACAO (janela hora_ini/hora_fim)
# ================================================================

from src.viz import plot_deteccao, EVCOMX_ORANGE, EVCOMX_GRAY

hora_ini = dbutils.widgets.get("hora_ini")
hora_fim = dbutils.widgets.get("hora_fim")

# Dia com mais ciclos
if len(df_pcp_sys) > 0:
    df_pcp_sys["dia"] = df_pcp_sys["ts_panela_saiu"].dt.date
    top_dia = df_pcp_sys.groupby("dia").size().idxmax()
else:
    top_dia = pdf_pcp["timestamp"].dt.date.value_counts().idxmax()

EVENT_STYLES_PCP = {
    "PANELA_CHEGOU":  {"color": EVCOMX_ORANGE[5], "ls": "-",  "lw": 1.5, "tag": "CHEGOU"},
    "TARA":           {"color": EVCOMX_GRAY[4],   "ls": "--", "lw": 1.2, "tag": "TARA"},
    "VAZ_INICIO":     {"color": EVCOMX_ORANGE[7], "ls": "-",  "lw": 2.0, "tag": "VAZAMENTO"},
    "PLATO_INICIO":   {"color": "#E6550D",        "ls": ":",  "lw": 1.8, "tag": "PLATO"},
    "PANELA_SAIU":    {"color": EVCOMX_GRAY[7],   "ls": "-",  "lw": 2.0, "tag": "SAIU"},
    "TARA_RESET":     {"color": EVCOMX_GRAY[3],   "ls": "--", "lw": 1.0, "tag": "RESET"},
}

plot_deteccao(
    pdf=pdf_pcp,
    df_events=df_pcp_events,
    col_signal="PESO_CARRO_PANELA",
    event_styles=EVENT_STYLES_PCP,
    signal_label="Peso carro panela",
    dia=top_dia,
    hora_ini=hora_ini,
    hora_fim=hora_fim,
    ref_lines=[
        {"y": 48, "color": EVCOMX_ORANGE[2], "ls": ":", "lw": 0.8, "alpha": 0.5, "label": "Limiar CHEIO (48t)"},
        {"y": 25, "color": EVCOMX_GRAY[2],   "ls": ":", "lw": 0.8, "alpha": 0.5, "label": "Limiar PANELA (25t)"},
        {"y": 0,  "color": EVCOMX_GRAY[1],   "ls": "-", "lw": 0.5, "alpha": 0.4},
    ],
    shade_events=[("PLATO_INICIO", "PANELA_SAIU")],
    tick_interval_min=10,
)

# COMMAND ----------

# DBTITLE 1,Peso Carro Panela - Filtragem, estatisticas e distribuicoes
# ================================================================
# PESO CARRO PANELA - FILTRAGEM + ESTATISTICAS + DISTRIBUICOES
# Usa src.viz para filtragem, stats, histogramas e boxplots
# ================================================================

from src.viz import (
    EVCOMX_ORANGE, EVCOMX_GRAY,
    filtrar_ciclos_iqr, print_stats,
    plot_histogramas, plot_boxplot_antes_depois,
)

print("="*120)
print("METRICAS CALCULADAS - PESO CARRO PANELA")
print("="*120)
print("""
Ciclo tipico do sinal:
  IDLE(0) -> PANELA_CHEGOU(~37t) -> TARA(0) -> VAZAMENTO(subindo) -> PLATO(~60t) -> PANELA_SAIU(~-37t) -> TARA_RESET(0)

Metricas e seus intervalos:
  1. Espera panela     = PANELA_CHEGOU -> TARA           | Tempo que a panela fica no carro ate eles tararem a balanca
  2. Tara a vazamento  = TARA -> VAZ_INICIO              | Tempo entre a tara e o inicio do vazamento (comeca a botar aco)
  3. Duracao vazamento = VAZ_INICIO -> PLATO_INICIO       | Tempo do vazamento subindo ate estabilizar no plato
  4. Espera ponte      = PLATO_INICIO -> PANELA_SAIU      | Tempo que ficou parado la em cima esperando a ponte tirar a panela
  5. Negativo          = PANELA_SAIU -> TARA_RESET         | Tempo com peso negativo ate eles tararem a balanca de volta a zero
  6. Ciclo total       = PANELA_CHEGOU -> TARA_RESET       | Ciclo completo de uma panela (chegou ate reset)
""")

# --- Prepara metricas em minutos ---
df_pcp_m = df_pcp_sys.copy()
df_pcp_m["espera_panela_min"]    = df_pcp_m["dur_panela_espera_s"] / 60
df_pcp_m["vazamento_min"]        = df_pcp_m["dur_vazamento_s"] / 60
df_pcp_m["plato_espera_min"]     = df_pcp_m["plato_dur_s"] / 60
df_pcp_m["negativo_min"]         = df_pcp_m["dur_negativo_s"] / 60
df_pcp_m["cheio_total_min"]      = df_pcp_m["dur_cheio_total_s"] / 60

df_pcp_m["tara_a_vaz_s"] = df_pcp_m.apply(
    lambda r: (r["ts_vazamento_inicio"] - r["ts_tara"]).total_seconds()
    if pd.notna(r["ts_tara"]) and pd.notna(r["ts_vazamento_inicio"]) else None,
    axis=1
)
df_pcp_m["tara_a_vaz_min"] = df_pcp_m["tara_a_vaz_s"] / 60

df_pcp_m["ciclo_total_s"] = df_pcp_m.apply(
    lambda r: (r["ts_tara_reset"] - r["ts_panela_chegou"]).total_seconds()
    if pd.notna(r["ts_panela_chegou"]) and pd.notna(r["ts_tara_reset"]) else None,
    axis=1
)
df_pcp_m["ciclo_total_min"] = df_pcp_m["ciclo_total_s"] / 60

PCP_METRIC_COLS = [
    "espera_panela_min", "tara_a_vaz_min", "vazamento_min",
    "plato_espera_min", "negativo_min", "ciclo_total_min",
]
PCP_METRIC_LABELS = {
    "espera_panela_min":   "Espera panela (chegou->tara)",
    "tara_a_vaz_min":      "Tara a vazamento (tara->vaz)",
    "vazamento_min":       "Duracao vazamento (vaz->plato)",
    "plato_espera_min":    "Espera ponte (plato->saiu)",
    "negativo_min":        "Negativo (saiu->reset)",
    "ciclo_total_min":     "Ciclo total (chegou->reset)",
}
PCP_METRIC_INTERVALS = {
    "espera_panela_min":   "CHEGOU -> TARA",
    "tara_a_vaz_min":      "TARA -> VAZ_INICIO",
    "vazamento_min":       "VAZ_INICIO -> PLATO",
    "plato_espera_min":    "PLATO -> PANELA_SAIU",
    "negativo_min":        "PANELA_SAIU -> RESET",
    "ciclo_total_min":     "CHEGOU -> RESET",
}

# --- Filtragem 3 etapas ---
df_pcp_std, df_pcp_antes = filtrar_ciclos_iqr(
    df_pcp_m,
    metric_cols=PCP_METRIC_COLS,
    metric_labels=PCP_METRIC_LABELS,
    anomaly_col="peso_max",
    anomaly_op="<",
    anomaly_val=55.0,
    anomaly_label="Fantasmas (peso_max < 55t)",
    timeout_col="plato_espera_min",
    timeout_max_min=30,
    timeout_label="Paradas (plato > 30 min)",
)

# --- Estatisticas ---
print_stats(
    df_pcp_std,
    metric_cols=PCP_METRIC_COLS,
    metric_labels=PCP_METRIC_LABELS,
    metric_intervals=PCP_METRIC_INTERVALS,
    titulo=f"ESTATISTICAS PESO CARRO PANELA -- {len(df_pcp_std)} ciclos padrao (Fev/2026)",
)

print(f"\nPeso maximo por ciclo (padrao):")
data_pmax = df_pcp_std["peso_max"].dropna()
print(f"  Media: {data_pmax.mean():.1f} t  |  Min: {data_pmax.min():.1f}  |  "
      f"Max: {data_pmax.max():.1f}  |  Std: {data_pmax.std():.1f}")

# --- Histogramas ---
plot_histogramas(
    df_pcp_std,
    hist_specs=[
        ("espera_panela_min",   "Espera panela (min)\nCHEGOU -> TARA",              EVCOMX_ORANGE[7]),
        ("tara_a_vaz_min",      "Tara a vazamento (min)\nTARA -> VAZ_INICIO",       EVCOMX_ORANGE[5]),
        ("vazamento_min",       "Duracao vazamento (min)\nVAZ_INICIO -> PLATO",     EVCOMX_ORANGE[3]),
        ("plato_espera_min",    "Espera ponte (min)\nPLATO -> PANELA_SAIU",        EVCOMX_GRAY[5]),
        ("negativo_min",        "Negativo ate reset (min)\nPANELA_SAIU -> RESET",  EVCOMX_GRAY[3]),
        ("ciclo_total_min",     "Ciclo total (min)\nCHEGOU -> RESET",              EVCOMX_ORANGE[4]),
    ],
    titulo=f"Distribuicao Tempos Peso Carro Panela -- {len(df_pcp_std)} ciclos padrao (Fev/2026)",
)

# --- Boxplot antes vs depois ---
plot_boxplot_antes_depois(
    df_pcp_antes,
    df_pcp_std,
    metric_cols=PCP_METRIC_COLS,
    metric_labels=PCP_METRIC_LABELS,
    box_colors=[EVCOMX_ORANGE[7], EVCOMX_ORANGE[5], EVCOMX_ORANGE[3],
                EVCOMX_GRAY[5], EVCOMX_GRAY[3], EVCOMX_ORANGE[4]],
    titulo="Peso Carro Panela: Antes vs Depois da filtragem",
)

# COMMAND ----------

# DBTITLE 1,Peso Carro Panela - Tendencia temporal (mes, semana, dia)
# ================================================================
# PESO CARRO PANELA - TENDENCIA TEMPORAL POR CICLO
# 3 niveis de zoom: mes completo, 7 dias, 1 dia
# ================================================================

from src.viz import plot_tendencia_3zooms, EVCOMX_ORANGE, EVCOMX_GRAY

METRICS_PCP = [
    ("espera_panela_min",  "Espera panela",        EVCOMX_ORANGE[5], EVCOMX_ORANGE[7]),
    ("vazamento_min",      "Vazamento",            EVCOMX_ORANGE[3], EVCOMX_ORANGE[4]),
    ("plato_espera_min",   "Espera ponte (plato)",  EVCOMX_GRAY[4],  EVCOMX_GRAY[6]),
]

plot_tendencia_3zooms(
    df_pcp_std,
    col_ts="ts_panela_saiu",
    metrics=METRICS_PCP,
    titulo_base="Peso Carro Panela",
)

# COMMAND ----------

# DBTITLE 1,FSM Detector - FEA Inclinometro (import .py)
import sys, importlib

proj_path = "/Workspace/Users/execs1@ex.gerdau.com/Sincronizador"
if proj_path not in sys.path:
    sys.path.insert(0, proj_path)

# Forca reload para pegar versao mais recente do .py
import src.detection.fea_inclinom
importlib.reload(src.detection.fea_inclinom)
from src.detection.fea_inclinom import detect_fea_inclinom

print("detect_fea_inclinom importada de src/detection/fea_inclinom.py")
print(f"  Parametros default: TH_INCLINANDO=5.0, TH_PICO=8.0, MIN_INCL_S=30")

# COMMAND ----------

# DBTITLE 1,FEA Inclinometro - Deteccao FSM (fevereiro completo)
# ================================================================
# FEA INCLINOMETRO - DETECCAO FSM PARA FEVEREIRO COMPLETO
# ================================================================

# Reusa df_csv_all (ja em memoria)
MES_INC = "02.2026"
print(f"Filtrando ACI@FEA_GERAL_INCLINOM para {MES_INC}...")

df_csv_inc = (
    df_csv_all
    .filter(F.col("timestamp").contains(f".{MES_INC.split('.')[0]}.{MES_INC.split('.')[1]}"))
    .select("timestamp", "ACI@FEA_GERAL_INCLINOM")
)
pdf_inc = df_csv_inc.toPandas()
pdf_inc["timestamp"] = pd.to_datetime(
    pdf_inc["timestamp"], format="%d.%m.%Y %H:%M:%S.%f", errors="coerce"
)
pdf_inc["ACI@FEA_GERAL_INCLINOM"] = (
    pdf_inc["ACI@FEA_GERAL_INCLINOM"].str.replace(",", ".", regex=False)
)
pdf_inc["ACI@FEA_GERAL_INCLINOM"] = pd.to_numeric(
    pdf_inc["ACI@FEA_GERAL_INCLINOM"], errors="coerce"
)
pdf_inc = pdf_inc.dropna().sort_values("timestamp").reset_index(drop=True)
print(f"   Registros: {len(pdf_inc):,}")
print(f"   Periodo:   {pdf_inc['timestamp'].min()} -> {pdf_inc['timestamp'].max()}")
print(f"   Dias:      {pdf_inc['timestamp'].dt.date.nunique()}")

# Roda detector FSM
print(f"\nRodando detect_fea_inclinom...")
df_inc_sys, df_inc_events, df_inc_debug = detect_fea_inclinom(pdf_inc)

print(f"\n{'='*90}")
print(f"RESULTADOS FEA INCLINOMETRO - Fevereiro {MES_INC}")
print(f"{'='*90}")
print(f"Ciclos detectados: {len(df_inc_sys)}")
print(f"Eventos totais:    {len(df_inc_events)}")

if len(df_inc_sys) > 0:
    print(f"\n{'Seq':>4} | {'Disturbio':>9} | {'Inclin':>8} | {'PicoMax':>8} | {'Queda':>8} | "
          f"{'Negativo':>8} | {'Retorno':>8} | {'Amax':>5} | {'Incl_s':>6} | "
          f"{'Dist_s':>6} | {'Neg_s':>6}")
    print("-" * 110)
    for _, r in df_inc_sys.head(50).iterrows():
        def _fmt(ts):
            return ts.strftime('%H:%M') if pd.notna(ts) and hasattr(ts, 'strftime') else '  -- '
        print(f"{r['ciclo_seq']:>4} | "
              f"{_fmt(r['ts_disturbio_inicio']):>9} | "
              f"{_fmt(r['ts_inclinando']):>8} | "
              f"{_fmt(r['ts_pico_max']):>8} | "
              f"{_fmt(r['ts_queda']):>8} | "
              f"{_fmt(r['ts_negativo']):>8} | "
              f"{_fmt(r['ts_retorno']):>8} | "
              f"{r['angulo_max']:>5.1f} | "
              f"{r['dur_inclinado_s']:>6.0f} | "
              f"{r.get('dur_disturbio_s', 0) or 0:>6.0f} | "
              f"{r.get('dur_negativo_s', 0) or 0:>6.0f}")

    print(f"\n--- Resumo ---")
    print(f"Angulo max medio:      {df_inc_sys['angulo_max'].mean():.1f} graus")
    print(f"Inclinado medio:       {df_inc_sys['dur_inclinado_s'].mean()/60:.1f} min")
    dist = df_inc_sys['dur_disturbio_s'].dropna()
    if len(dist) > 0:
        print(f"Disturbio medio:       {dist.mean()/60:.1f} min")
    neg = df_inc_sys['dur_negativo_s'].dropna()
    if len(neg) > 0:
        print(f"Negativo medio:        {neg.mean()/60:.1f} min")

# COMMAND ----------

# DBTITLE 1,FEA Inclinometro - Plot validacao (hora_ini/hora_fim)
# ================================================================
# FEA INCLINOMETRO - PLOT VALIDACAO (janela hora_ini/hora_fim)
# ================================================================

from src.viz import plot_deteccao, EVCOMX_ORANGE, EVCOMX_GRAY

hora_ini = dbutils.widgets.get("hora_ini")
hora_fim = dbutils.widgets.get("hora_fim")
print(f"Janela: {hora_ini} -> {hora_fim}")

# Dia com mais ciclos
if len(df_inc_sys) > 0:
    df_inc_sys["dia"] = df_inc_sys["ts_queda"].dt.date
    top_dia_inc = df_inc_sys.groupby("dia").size().idxmax()
else:
    top_dia_inc = pdf_inc["timestamp"].dt.date.value_counts().idxmax()

dia_ciclos_inc = df_inc_sys[df_inc_sys["dia"] == top_dia_inc] if len(df_inc_sys) > 0 else pd.DataFrame()
print(f"Dia: {top_dia_inc}  ({len(dia_ciclos_inc)} ciclos)")

EVENT_STYLES_INC = {
    "DISTURBIO_INICIO": {"color": EVCOMX_ORANGE[4], "ls": "--", "lw": 1.2, "tag": "DISTURBIO"},
    "INCLINANDO":       {"color": EVCOMX_ORANGE[6], "ls": "-",  "lw": 1.8, "tag": "INCLINANDO"},
    "PICO_MAX":         {"color": EVCOMX_ORANGE[7], "ls": "-",  "lw": 2.0, "tag": "PICO MAX"},
    "QUEDA":            {"color": EVCOMX_GRAY[7],   "ls": "-",  "lw": 2.0, "tag": "QUEDA"},
    "NEGATIVO":         {"color": EVCOMX_GRAY[5],   "ls": "--", "lw": 1.5, "tag": "NEGATIVO"},
    "RETORNO":          {"color": EVCOMX_GRAY[3],   "ls": "--", "lw": 1.0, "tag": "RETORNO"},
}

plot_deteccao(
    pdf=pdf_inc,
    df_events=df_inc_events,
    col_signal="ACI@FEA_GERAL_INCLINOM",
    event_styles=EVENT_STYLES_INC,
    signal_label="Inclinometro",
    dia=top_dia_inc,
    hora_ini=hora_ini,
    hora_fim=hora_fim,
    ref_lines=[
        {"y": 5,   "color": EVCOMX_ORANGE[2], "ls": ":", "lw": 0.8, "alpha": 0.5, "label": "Limiar 5\u00b0"},
        {"y": 1.5, "color": EVCOMX_ORANGE[1], "ls": ":", "lw": 0.6, "alpha": 0.4, "label": "Limiar 1.5\u00b0"},
        {"y": 0,   "color": EVCOMX_GRAY[1],   "ls": "-", "lw": 0.5, "alpha": 0.4},
        {"y": -3,  "color": EVCOMX_GRAY[2],   "ls": ":", "lw": 0.8, "alpha": 0.5, "label": "Limiar NEG (-3\u00b0)"},
    ],
    shade_events=[("INCLINANDO", "QUEDA")],
)

# COMMAND ----------

# DBTITLE 1,FEA Inclinometro - Filtragem, estatisticas, distribuicoes e tendencia
# ================================================================
# FEA INCLINOMETRO - FILTRAGEM + ESTATISTICAS + DISTRIBUICOES
# Usa src.viz para filtragem, stats, histogramas, boxplots e tendencia
# ================================================================

from src.viz import (
    EVCOMX_ORANGE, EVCOMX_GRAY,
    filtrar_ciclos_iqr, print_stats,
    plot_histogramas, plot_boxplot_antes_depois,
    plot_tendencia_3zooms,
)

print("="*120)
print("METRICAS CALCULADAS - FEA INCLINOMETRO")
print("="*120)
print("""
Ciclo tipico do sinal:
  IDLE(~0) -> DISTURBIO(1.5-5) -> INCLINANDO(>5) -> PICO(~8-12) -> QUEDA -> NEGATIVO(~-5) -> RETORNO(~0)

Metricas e seus intervalos:
  1. Disturbio          = DISTURBIO_INICIO -> INCLINANDO  | Oscilacao antes de inclinar efetivamente
  2. Inclinado          = INCLINANDO -> QUEDA              | Tempo total acima de 5 graus (vazamento)
  3. Pico               = PICO_MAX -> QUEDA                | Tempo no plato do pico maximo
  4. Negativo           = NEGATIVO -> RETORNO              | Tempo em angulo negativo ate voltar
  5. Ciclo total        = DISTURBIO_INICIO -> RETORNO      | Ciclo completo
""")

# --- Prepara metricas em minutos ---
df_inc_m = df_inc_sys.copy()
df_inc_m["disturbio_min"]    = df_inc_m["dur_disturbio_s"] / 60
df_inc_m["inclinado_min"]    = df_inc_m["dur_inclinado_s"] / 60
df_inc_m["pico_min"]         = df_inc_m["dur_pico_s"] / 60
df_inc_m["negativo_min"]     = df_inc_m["dur_negativo_s"] / 60
df_inc_m["ciclo_total_min"]  = df_inc_m["dur_ciclo_total_s"] / 60

INC_METRIC_COLS = [
    "disturbio_min", "inclinado_min", "pico_min",
    "negativo_min", "ciclo_total_min",
]
INC_METRIC_LABELS = {
    "disturbio_min":   "Disturbio (dist->incl)",
    "inclinado_min":   "Inclinado (incl->queda)",
    "pico_min":        "Pico (pico_max->queda)",
    "negativo_min":    "Negativo (neg->retorno)",
    "ciclo_total_min": "Ciclo total (dist->retorno)",
}
INC_METRIC_INTERVALS = {
    "disturbio_min":   "DISTURBIO -> INCLINANDO",
    "inclinado_min":   "INCLINANDO -> QUEDA",
    "pico_min":        "PICO_MAX -> QUEDA",
    "negativo_min":    "NEGATIVO -> RETORNO",
    "ciclo_total_min": "DISTURBIO -> RETORNO",
}

# --- Filtragem 3 etapas ---
df_inc_std, df_inc_antes = filtrar_ciclos_iqr(
    df_inc_m,
    metric_cols=INC_METRIC_COLS,
    metric_labels=INC_METRIC_LABELS,
    anomaly_col="angulo_max",
    anomaly_op="<",
    anomaly_val=5.0,
    anomaly_label="Angulo max < 5",
    timeout_col="inclinado_min",
    timeout_max_min=10,
    timeout_label="Timeout inclinado > 10 min",
)

# --- Estatisticas ---
print_stats(
    df_inc_std,
    metric_cols=INC_METRIC_COLS,
    metric_labels=INC_METRIC_LABELS,
    metric_intervals=INC_METRIC_INTERVALS,
    titulo=f"ESTATISTICAS FEA INCLINOMETRO -- {len(df_inc_std)} ciclos padrao (Fev/2026)",
)

print(f"\nAngulo maximo por ciclo:")
data_amax = df_inc_std["angulo_max"].dropna()
print(f"  Media: {data_amax.mean():.1f} graus  |  Min: {data_amax.min():.1f}  |"
      f"  Max: {data_amax.max():.1f}  |  Std: {data_amax.std():.2f}")

# --- Histogramas ---
plot_histogramas(
    df_inc_std,
    hist_specs=[
        ("disturbio_min",   "Disturbio (min)\nDISTURBIO -> INCLINANDO",  EVCOMX_ORANGE[5]),
        ("inclinado_min",   "Inclinado (min)\nINCLINANDO -> QUEDA",     EVCOMX_ORANGE[7]),
        ("pico_min",        "Pico (min)\nPICO_MAX -> QUEDA",            EVCOMX_ORANGE[3]),
        ("negativo_min",    "Negativo (min)\nNEGATIVO -> RETORNO",      EVCOMX_GRAY[5]),
        ("ciclo_total_min", "Ciclo total (min)\nDISTURBIO -> RETORNO",  EVCOMX_ORANGE[4]),
    ],
    titulo=f"Distribuicao Tempos Inclinometro -- {len(df_inc_std)} ciclos padrao (Fev/2026)",
)

# --- Boxplot antes vs depois ---
plot_boxplot_antes_depois(
    df_inc_antes,
    df_inc_std,
    metric_cols=INC_METRIC_COLS,
    metric_labels=INC_METRIC_LABELS,
    box_colors=[EVCOMX_ORANGE[5], EVCOMX_ORANGE[7], EVCOMX_ORANGE[3],
                EVCOMX_GRAY[5], EVCOMX_ORANGE[4]],
    titulo="FEA Inclinometro: Antes vs Depois da filtragem",
)

# --- Tendencia temporal (mes, semana, dia) ---
METRICS_INC = [
    ("disturbio_min",  "Disturbio",  EVCOMX_ORANGE[3], EVCOMX_ORANGE[5]),
    ("inclinado_min",  "Inclinado",  EVCOMX_ORANGE[5], EVCOMX_ORANGE[7]),
    ("negativo_min",   "Negativo",   EVCOMX_GRAY[4],   EVCOMX_GRAY[6]),
]

plot_tendencia_3zooms(
    df_inc_std,
    col_ts="ts_queda",
    metrics=METRICS_INC,
    titulo_base="FEA Inclinometro",
)