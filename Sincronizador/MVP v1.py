# Databricks notebook source
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

import math
from datetime import timedelta
import pyspark.sql
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, DoubleType, StringType, TimestampType
)
import datetime
from datetime import datetime, timedelta
from typing import Optional, Tuple

# Related third party imports
import pyspark
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from pyspark.sql.window import Window


# COMMAND ----------

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

# DBTITLE 1,detect_FEA


def detect_fea_events_fsm_energia(
    df: pd.DataFrame,

    # colunas
    timestamp_col: str = "timestamp",
    tag_col: str = "ACI@FEA_ELET_ENERGIA",

    # =========================
    # FEATURES / FILTRO
    # =========================
    SMOOTH_WINDOW_S: int = 5,
    
    # =========================
    # START (INÍCIO)
    # =========================
    ZERO_MAX: float = 200.0,
    ZERO_HOLD_S: int = 10,

    START_TS_MODE: str = "prev_zero",  # <<< recomendado
    START_MIN: float = 250.0,

    # =========================
    # RUNNING / HIGH GATE
    # =========================
    HIGH_MIN: float = 5000.0,
    
    # =========================
    # END (FIM)
    # =========================
    END_MAX: float = 200.0,
    DROP_WINDOW_S: int = 5,
    DROP_DELTA: float = 2000.0,

    END_TS_MODE: str = "prev",

    # =========================
    # FLUXO
    # =========================
    REFRACTORY_AFTER_END_S: int = 0,
    MIN_RUN_S: int = 0,

    # =========================
    # INICIALIZAÇÃO
    # =========================
    initial_corrida: int = 0,

    DEBUG: bool = True,
    ASSUME_RUNNING_IF_STARTS_HIGH: bool = False,
):

    # --------------------------
    # 0) Preparação
    # --------------------------
    if df is None or len(df) == 0:
        return (
            pd.DataFrame(columns=["corrida", "inicio_fea_sys", "final_fea_sys", "close_reason"]),
            pd.DataFrame(columns=["corrida", "status", "inicio_fea_sys", "final_fea_sys", "event_ts"]),
            pd.DataFrame(columns=["event", "state"]),
            pd.DataFrame(columns=[timestamp_col, tag_col]),
        )

    df = df.copy()

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)

    df["energia_raw"] = pd.to_numeric(df[tag_col], errors="coerce")

    # --------------------------
    # 0.1) Converter segundos → amostras
    # --------------------------
    dts = df[timestamp_col].diff().dt.total_seconds()
    dt_med = dts[(dts > 0) & np.isfinite(dts)].median()
    if not np.isfinite(dt_med) or dt_med <= 0:
        dt_med = 1.0

    def s2n(seconds: int) -> int:
        return max(1, int(round(seconds / dt_med)))

    w_smooth = s2n(SMOOTH_WINDOW_S)
    w_zero_hold = s2n(ZERO_HOLD_S)
    w_drop = s2n(DROP_WINDOW_S)

    # --------------------------
    # 0.2) Features
    # --------------------------
    df["energia"] = df["energia_raw"].rolling(w_smooth, min_periods=1).median()
    df["de"] = df["energia"].diff()
    df["drop_w"] = df["energia"] - df["energia"].shift(w_drop)

    # ZERO lógico (FSM) → energia suavizada
    df["is_zero"] = df["energia"].fillna(np.inf) <= ZERO_MAX

    # ZERO físico (timestamp) → energia bruta
    df["is_zero_raw"] = df["energia_raw"].fillna(np.inf) <= ZERO_MAX

    df["is_high"] = df["energia"].fillna(-np.inf) >= HIGH_MIN

    # --------------------------
    # 1) FSM
    # --------------------------
    state = "IDLE"
    corrida = int(initial_corrida)

    running_start_ts = None
    running_high_seen = False

    last_zero_ts = None
    zero_run = 0
    refractory_until = None

    results = []
    events = []
    debug_rows = []

    def dbg(event: str, **kwargs):
        if DEBUG:
            row = {"event": event, "state": state}
            row.update(kwargs)
            debug_rows.append(row)

    prev_is_zero = False

    for i in range(len(df)):
        ts = df.at[i, timestamp_col]
        e = df.at[i, "energia"]
        drop_w = df.at[i, "drop_w"]

        if pd.isna(e):
            continue

        is_zero = bool(df.at[i, "is_zero"])
        is_high = bool(df.at[i, "is_high"])

        # --------------------------
        # ZERO RUN (FSM)
        # --------------------------
        zero_run_prev = zero_run
        zero_run = zero_run + 1 if is_zero else 0

        # ZERO físico (timestamp)
        if df.at[i, "is_zero_raw"]:
            last_zero_ts = ts

        ts_prev = df.at[i - 1, timestamp_col] if i > 0 else None

        # --------------------------
        # IDLE → RUNNING (START)
        # --------------------------
        if state == "IDLE":
            left_zero = (not is_zero) and prev_is_zero
            zero_stable = zero_run_prev >= w_zero_hold

            start_ok = left_zero and zero_stable and (e >= START_MIN)

            if left_zero:
                dbg(
                    "START_EVAL",
                    i=i,
                    timestamp=ts,
                    energia=float(e),
                    zero_run_prev=int(zero_run_prev),
                    w_zero_hold=int(w_zero_hold),
                    start_ok=bool(start_ok),
                )

            if start_ok:
                start_ts = last_zero_ts if START_TS_MODE == "prev_zero" else ts

                running_start_ts = start_ts
                running_high_seen = is_high
                state = "RUNNING"

                events.append({
                    "corrida": corrida,
                    "status": "RUNNING",
                    "inicio_fea_sys": running_start_ts,
                    "final_fea_sys": None,
                    "event_ts": ts,
                })

                dbg(
                    "RUNNING_START",
                    corrida=corrida,
                    timestamp=ts,
                    inicio_fea_sys=running_start_ts,
                )

            prev_is_zero = is_zero
            continue

        # --------------------------
        # RUNNING → IDLE (END)
        # --------------------------
        if state == "RUNNING":
            running_high_seen = running_high_seen or is_high

            drop_ok = pd.notna(drop_w) and (drop_w <= -DROP_DELTA)

            end_ok = (
                running_high_seen
                and is_zero
                and (e <= END_MAX)
                and drop_ok
            )

            if is_zero:
                dbg(
                    "END_EVAL",
                    corrida=corrida,
                    timestamp=ts,
                    energia=float(e),
                    drop_w=float(drop_w) if pd.notna(drop_w) else None,
                    end_ok=bool(end_ok),
                )

            if end_ok:
                end_ts = ts_prev if END_TS_MODE == "prev" and ts_prev is not None else ts

                results.append({
                    "corrida": corrida,
                    "inicio_fea_sys": running_start_ts,
                    "final_fea_sys": end_ts,
                    "close_reason": "reset_to_zero",
                })

                events.append({
                    "corrida": corrida,
                    "status": "CLOSED",
                    "inicio_fea_sys": running_start_ts,
                    "final_fea_sys": end_ts,
                    "event_ts": ts,
                })

                dbg(
                    "RUN_CLOSE",
                    corrida=corrida,
                    final_fea_sys=end_ts,
                )

                corrida += 1
                state = "IDLE"
                running_start_ts = None
                running_high_seen = False

            prev_is_zero = is_zero
            continue

    return (
        pd.DataFrame(results),
        pd.DataFrame(events),
        pd.DataFrame(debug_rows),
        df.copy(),
    )


# COMMAND ----------

# DBTITLE 1,Teste detect_VD_NEW com hist_tags_complete
# ============================================================
# TESTE DA FUNÇÃO detect_vd_events COM hist_tags_complete
# ============================================================

# Definir timezone (se necessário)
TIMEZONE = "America/Sao_Paulo"

# Ler a tabela hist_tags_complete
tb_hist_tags = spark.read.table("industrial_iatemperaturascha_refined_dev.computed_features.hist_tags_complete")

print(f"Total de registros na tabela: {tb_hist_tags.count():,}")
print(f"\nColunas disponíveis: {len(tb_hist_tags.columns)}")

# Verificar se a coluna Tempo já está em formato timestamp ou precisa conversão
print("\nSchema da coluna Tempo:")
tb_hist_tags.select("Tempo").printSchema()

# Pegar uma amostra para ver o formato
print("\nAmostra dos dados:")
tb_hist_tags.select(
    "Tempo",
    "ACI@VD_VASO1_VOLUMEAR",
    "ACI@VD_VASO1_VOLUMEN2",
    "ACI@VD_VASO1_VZFORTE",
    "ACI@VD_VASO1_VZFRACA",
    "ACI@VD_PRESSAO_DO_VACUO"
).limit(5).show(truncate=False)

# COMMAND ----------

# DBTITLE 1,Preparar dados e executar detect_vd_events
# ============================================================
# PREPARAR DADOS E EXECUTAR detect_vd_events
# ============================================================

# Definir período de teste (1 dia para começar)
start_ts = "2025-09-01 00:00:00"
end_ts = "2025-09-01 23:59:59"
initial_heat = 1

# Preparar DataFrame com as colunas necessárias
tb_vd_prepared = (
    tb_hist_tags
    # Converter Tempo para timestamp
    .withColumn("timestamp", F.to_timestamp(F.col("Tempo"), "dd-MMM-yy HH:mm:ss.S"))
    
    # Converter colunas de string para double (VASO1)
    .withColumn("ACI@VD_VASO1_VOLUMEAR", F.col("ACI@VD_VASO1_VOLUMEAR").cast("double"))
    .withColumn("ACI@VD_VASO1_VOLUMEN2", F.col("ACI@VD_VASO1_VOLUMEN2").cast("double"))
    .withColumn("ACI@VD_VASO1_VZFORTE", F.col("ACI@VD_VASO1_VZFORTE").cast("double"))
    .withColumn("ACI@VD_VASO1_VZFRACA", F.col("ACI@VD_VASO1_VZFRACA").cast("double"))
    
    # Converter colunas de string para double (VASO2)
    .withColumn("ACI@VD_VASO2_VOLUMEAR", F.col("ACI@VD_VASO2_VOLUMEAR").cast("double"))
    .withColumn("ACI@VD_VASO2_VOLUMEN2", F.col("ACI@VD_VASO2_VOLUMEN2").cast("double"))
    .withColumn("ACI@VD_VASO2_VZFORTE", F.col("ACI@VD_VASO2_VZFORTE").cast("double"))
    .withColumn("ACI@VD_VASO2_VZFRACA", F.col("ACI@VD_VASO2_VZFRACA").cast("double"))
    
    # Converter pressão
    .withColumn("ACI@VD_PRESSAO_DO_VACUO", F.col("ACI@VD_PRESSAO_DO_VACUO").cast("double"))
)

print("Dados preparados!")
print(f"\nSchema das colunas VD:")
tb_vd_prepared.select(
    "timestamp",
    "ACI@VD_VASO1_VOLUMEAR",
    "ACI@VD_VASO1_VOLUMEN2",
    "ACI@VD_VASO1_VZFORTE",
    "ACI@VD_VASO1_VZFRACA",
    "ACI@VD_PRESSAO_DO_VACUO"
).printSchema()

print("\nAmostra dos dados preparados:")
tb_vd_prepared.select(
    "timestamp",
    "ACI@VD_VASO1_VOLUMEAR",
    "ACI@VD_VASO1_VOLUMEN2",
    "ACI@VD_VASO1_VZFORTE",
    "ACI@VD_VASO1_VZFRACA",
    "ACI@VD_PRESSAO_DO_VACUO"
).limit(5).show(truncate=False)

# COMMAND ----------

# DBTITLE 1,Executar detect_vd_events
# ============================================================
# EXECUTAR detect_vd_events
# ============================================================

print("Executando detect_vd_events...")
print(f"Período: {start_ts} até {end_ts}")
print(f"Corrida inicial: {initial_heat}\n")

try:
    # Executar a função detect_vd_events
    df_vd_final, df_vd_events, df_vd_debug = detect_vd_events(
        spark_df_vd_pims=tb_vd_prepared,
        start_ts=start_ts,
        end_ts=end_ts,
        initial_heat_number=initial_heat,
        reset_threshold=0.99,
        flow_activity_threshold=0.2406,
        min_vd_duration_seconds=35,
        deep_vac_end_pressure_threshold=750.0,
        debug=False
    )
    
    print("\u2713 Função executada com sucesso!\n")
    
    # Mostrar resultados
    print("=" * 60)
    print("RESULTADOS - df_vd_final")
    print("=" * 60)
    print(f"Total de eventos VD detectados: {df_vd_final.count()}")
    if df_vd_final.count() > 0:
        print("\nPrimeiros 10 eventos:")
        df_vd_final.orderBy("start_vd_time").show(10, truncate=False)
    else:
        print("Nenhum evento VD detectado no período.")
    
    print("\n" + "=" * 60)
    print("RESULTADOS - df_vd_events")
    print("=" * 60)
    print(f"Total de eventos (todos os tipos): {df_vd_events.count()}")
    if df_vd_events.count() > 0:
        print("\nPrimeiros 10 eventos:")
        df_vd_events.orderBy("event_ts").show(10, truncate=False)
    else:
        print("Nenhum evento detectado no período.")
        
except Exception as e:
    print(f"\u2717 Erro ao executar detect_vd_events:")
    print(f"Tipo: {type(e).__name__}")
    print(f"Mensagem: {str(e)}")
    import traceback
    print("\nTraceback completo:")
    traceback.print_exc()

# COMMAND ----------

# DBTITLE 1,Resumo e análise dos resultados
# ============================================================
# RESUMO E ANÁLISE DOS RESULTADOS
# ============================================================

print("\u2713 FUNÇÃO detect_vd_events FUNCIONANDO CORRETAMENTE!\n")
print("=" * 70)
print("RESUMO DOS RESULTADOS")
print("=" * 70)

# Estatísticas gerais
print(f"\n1. EVENTOS VD DETECTADOS (df_vd_final):")
print(f"   - Total de corridas VD: {df_vd_final.count()}")
print(f"   - Período analisado: {start_ts} até {end_ts}")

# Estatísticas por tanque
print(f"\n2. DISTRIBUIÇÃO POR TANQUE:")
df_vd_final.groupBy("tank_id").count().orderBy("tank_id").show()

# Estatísticas de duração
print(f"\n3. ESTATÍSTICAS DE DURAÇÃO (segundos):")
df_vd_final.select(
    F.min("duration_seconds").alias("min_duration"),
    F.avg("duration_seconds").alias("avg_duration"),
    F.max("duration_seconds").alias("max_duration")
).show()

# Estatísticas de volume
print(f"\n4. ESTATÍSTICAS DE VOLUME MÁXIMO:")
df_vd_final.select(
    F.min("max_volume").alias("min_volume"),
    F.avg("max_volume").alias("avg_volume"),
    F.max("max_volume").alias("max_volume")
).show()

# Tipos de eventos
print(f"\n5. TIPOS DE EVENTOS (df_vd_events):")
df_vd_events.groupBy("status").count().orderBy("status").show()

print("\n" + "=" * 70)
print("PRÓXIMOS PASSOS")
print("=" * 70)
print("""
1. ✓ Função detect_vd_events está funcionando com a tabela hist_tags_complete
2. ✓ Todas as colunas necessárias estão disponíveis
3. ✓ Conversão de tipos (string -> double) funcionando
4. ✓ Detecção de eventos VD operacional

Agora você pode:
- Integrar esta função no pipeline principal
- Ajustar parâmetros se necessário
- Testar com períodos maiores
- Combinar com outras funções (FEA, FP, LC)
""")

# COMMAND ----------

# DBTITLE 1,detect_FP
import math
import heapq
from datetime import timedelta
import pyspark.sql
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, DoubleType, StringType, TimestampType
)

# -----------------------------
# Preprocess (streaming-safe)
# -----------------------------
def streaming_ffill_last_valid(
    df: pyspark.sql.DataFrame,
    timestamp_column: str,
    columns: list[str],
) -> pyspark.sql.DataFrame:
    w = Window.orderBy(F.col(timestamp_column)).rowsBetween(Window.unboundedPreceding, Window.currentRow)
    out = df
    for c in columns:
        out = out.withColumn(c, F.last(F.col(c), ignorenulls=True).over(w))
    return out


def preprocess_pims_data_streaming(
    spark_df: pyspark.sql.DataFrame,
    first_timestamp: str,
    last_timestamp: str,
    timestamp_column: str,
    columns_to_fill: list[str],
) -> pyspark.sql.DataFrame:
    filtered_tb = spark_df.filter(
        (F.col(timestamp_column) >= F.lit(first_timestamp)) &
        (F.col(timestamp_column) <= F.lit(last_timestamp))
    )

    casted = filtered_tb
    for c in columns_to_fill:
        casted = casted.withColumn(c, F.col(c).cast("double"))

    ordered = (
        casted
        .withColumn("_tie", F.monotonically_increasing_id())
        .orderBy(F.col(timestamp_column).asc(), F.col("_tie").asc())
        .drop("_tie")
    )

    return streaming_ffill_last_valid(
        df=ordered,
        timestamp_column=timestamp_column,
        columns=columns_to_fill,
    )


# -----------------------------
# Detector (ALL Spark + 1 pass FSM in mapPartitions, sem Pandas)
# -----------------------------
def detect_fp_events_gas_based_streaming(
    spark_df: pyspark.sql.DataFrame,
    initial_heat_number: int,
    initial_timestamp: str,
    last_timestamp: str,
    min_peak_gas: float = 100.0,
    min_peak_window_s: int = 300,
    energy_delta_eps: float = 1.5,
    reopen_merge_threshold: float = 100.0,   
):
    """
    SIMPLE + streaming-safe + melhorias pedidas:
    - START por carro (gas_total): cruza > threshold e sustenta Xs => RUNNING (fp_start = 1º ts acima)
    - END por carro: <= threshold sustenta Ys => entra em END_PENDING
        * espera REOPEN_MERGE_WINDOW_S (2 min) p/ ver se reabre
        * se reabrir dentro da janela -> cancela fechamento e continua MESMA corrida
        * se não reabrir -> fecha de fato (CLOSED)
    - CANCELLED por weak peak:
        * NÃO "consome" número de corrida (reutiliza)
        * cria threshold temporário = patamar do cancelamento (override) até iniciar a próxima corrida do carro
    - HEATING GLOBAL:
        * 1 FSM (energia única), atribuído à corrida ativa mais antiga
        * emite HEAT_ON e HEAT_OFF em df_fp_events + segmentos em df_heat
    """

    # -----------------------------
    # Schemas
    # -----------------------------
    FP_FINAL_SCHEMA = StructType([
        StructField("corrida", IntegerType(), True),
        StructField("fp_start", TimestampType(), True),
        StructField("fp_end", TimestampType(), True),
        StructField("duration_seconds", DoubleType(), True),
        StructField("carro_fp", IntegerType(), True),
        StructField("peak_gas", DoubleType(), True),
    ])

    FP_EVENTS_SCHEMA = StructType([
        StructField("corrida", IntegerType(), True),
        StructField("status", StringType(), True),  # RUNNING | CLOSED | CANCELLED | RUN_LEFT_OPEN | HEAT_ON | HEAT_OFF
        StructField("fp_start", TimestampType(), True),
        StructField("fp_end", TimestampType(), True),
        StructField("carro_fp", IntegerType(), True),
        StructField("confirmed_at", TimestampType(), True),
        StructField("event_ts", TimestampType(), True),
        StructField("reason", StringType(), True),
    ])

    HEAT_SCHEMA = StructType([
        StructField("corrida", IntegerType(), True),
        StructField("carro_fp", IntegerType(), True),
        StructField("heat_start", TimestampType(), True),
        StructField("heat_end", TimestampType(), True),
        StructField("duration_seconds", DoubleType(), True),
        StructField("energy_start", DoubleType(), True),
        StructField("energy_end", DoubleType(), True),
    ])

    spark = spark_df.sparkSession

    TIMESTAMP_COL = "timestamp"
    COL_AR_C1 = "ACI@FP_Volume_Argonio_Carro_1"
    COL_N2_C1 = "ACI@FP_Volume_Nitrogenio_Carro_1"
    COL_AR_C2 = "ACI@FP_Volume_Argonio_Carro_2"
    COL_N2_C2 = "ACI@FP_Volume_Nitrogenio_Carro_2"
    COL_ENERGY = "ACI@FP_Energia_Consumida"

    columns_to_fill = [COL_AR_C1, COL_N2_C1, COL_AR_C2, COL_N2_C2, COL_ENERGY]

    # -----------------------------
    # 1) preprocess + colapsa duplicados por timestamp (MAX)
    # -----------------------------
    base = preprocess_pims_data_streaming(
        spark_df=spark_df,
        first_timestamp=initial_timestamp,
        last_timestamp=last_timestamp,
        timestamp_column=TIMESTAMP_COL,
        columns_to_fill=columns_to_fill,
    )

    agg = (
        base
        .groupBy(F.col(TIMESTAMP_COL).alias(TIMESTAMP_COL))
        .agg(
            F.max(F.col(COL_AR_C1)).alias(COL_AR_C1),
            F.max(F.col(COL_N2_C1)).alias(COL_N2_C1),
            F.max(F.col(COL_AR_C2)).alias(COL_AR_C2),
            F.max(F.col(COL_N2_C2)).alias(COL_N2_C2),
            F.max(F.col(COL_ENERGY)).alias(COL_ENERGY),
        )
        .orderBy(F.col(TIMESTAMP_COL).asc())
        .withColumn("gas_carro1", F.coalesce(F.col(COL_AR_C1), F.lit(0.0)) + F.coalesce(F.col(COL_N2_C1), F.lit(0.0)))
        .withColumn("gas_carro2", F.coalesce(F.col(COL_AR_C2), F.lit(0.0)) + F.coalesce(F.col(COL_N2_C2), F.lit(0.0)))
        .withColumn("energia", F.coalesce(F.col(COL_ENERGY), F.lit(0.0)))
    )

    if agg.rdd.isEmpty():
        return (
            spark.createDataFrame([], schema=FP_FINAL_SCHEMA),
            spark.createDataFrame([], schema=FP_EVENTS_SCHEMA),
            spark.createDataFrame([], schema=HEAT_SCHEMA),
        )

    ordered_1p = agg.coalesce(1)

    def fsm_partition(rows_iter):
        # -----------------------------
        # Ajustes simples
        # -----------------------------
        START_THRESHOLD = 10.0
        END_THRESHOLD = 100.0

        START_CONFIRM_SUSTAIN_S = 5.0
        END_CONFIRM_SUSTAIN_S = 120.0

        # merge fechamento: se reabrir em até 2 min, não fecha
        REOPEN_MERGE_WINDOW_S = 120.0

        # segurar zeros espúrios do gás
        HOLD_ZERO_S = 10

        # heating
        HEAT_OFF_CONFIRM_S = 30.0

        # -----------------------------
        # Numeração com REUSO em cancelamento
        # -----------------------------
        next_corrida = int(initial_heat_number)
        reusable = []  # min-heap de números cancelados para reuso

        def alloc_corrida():
            nonlocal next_corrida
            if reusable:
                return int(heapq.heappop(reusable))
            c = int(next_corrida)
            next_corrida += 1
            return c

        def release_corrida(num):
            # coloca de volta para reuso (não consome)
            if num is None:
                return
            heapq.heappush(reusable, int(num))

        # -----------------------------
        # Estado por carro
        # -----------------------------
        def new_ctx():
            return {
                "state": "IDLE",  # IDLE | START_CANDIDATE | RUNNING | END_PENDING

                "_prev_gas": 0.0,
                "last_nonzero_val": 0.0,
                "last_nonzero_ts": None,

                # start candidate
                "cand_start_ts": None,
                "cand_sustain_s": 0.0,

                # running
                "run_corrida": None,
                "run_start_ts": None,
                "run_confirmed_at": None,
                "peak_total": None,

                # end pending (pra merge)
                "pending_end_ts0": None,        # fp_end planejado (início do sustain abaixo)
                "pending_confirmed_at": None,   # quando confirmou sustain
                "pending_deadline": None,       # ts limite = pending_confirmed_at + 2min

                # override threshold após cancelamento por weak peak
                "override_start_thr": None,     # patamar do cancelamento (vale só até próximo start)
            }

        ctx = {1: new_ctx(), 2: new_ctx()}

        out_events = []
        out_final = []
        out_heat = []

        # -----------------------------
        # HEATING GLOBAL (1 FSM)
        # -----------------------------
        heat_global = {
            "assigned_corrida": None,
            "assigned_carro": None,

            "prev_energy": None,

            "heat_on": False,
            "heat_start_ts": None,
            "heat_start_energy": None,

            "last_energy_inc_ts": None,
            "last_energy_inc_val": None,
        }

        def emit_event(corrida, status, fp_start, fp_end, carro_fp, confirmed_at, event_ts, reason):
            out_events.append(("E", (
                int(corrida) if corrida is not None else None,
                status,
                fp_start,
                fp_end,
                int(carro_fp) if carro_fp is not None else None,
                confirmed_at,
                event_ts,
                reason,
            )))

        def emit_event_fp(carro, status, event_ts, reason=None, fp_end=None, confirmed_at=None):
            c = ctx[carro]
            emit_event(
                corrida=c.get("run_corrida"),
                status=status,
                fp_start=c.get("run_start_ts"),
                fp_end=fp_end,
                carro_fp=carro,
                confirmed_at=confirmed_at,
                event_ts=event_ts,
                reason=reason,
            )

        def emit_final(carro, fp_end):
            c = ctx[carro]
            fp_start = c.get("run_start_ts")
            corrida = c.get("run_corrida")
            peak = c.get("peak_total")

            duration_s = (fp_end - fp_start).total_seconds() if (fp_end is not None and fp_start is not None) else None
            out_final.append(("F", (
                int(corrida) if corrida is not None else None,
                fp_start,
                fp_end,
                float(duration_s) if duration_s is not None else None,
                int(carro),
                float(peak) if peak is not None else None,
            )))

        def emit_heat_event(status, event_ts, reason=None):
            corrida = heat_global.get("assigned_corrida")
            carro = heat_global.get("assigned_carro")
            if corrida is None or carro not in (1, 2):
                return
            fp_start = ctx[carro].get("run_start_ts")
            emit_event(
                corrida=corrida,
                status=status,           # HEAT_ON / HEAT_OFF
                fp_start=fp_start,
                fp_end=None,
                carro_fp=carro,
                confirmed_at=None,
                event_ts=event_ts,
                reason=reason,
            )

        def close_heat_global_if_open(close_ts, close_energy):
            if heat_global["heat_on"] and heat_global["heat_start_ts"] is not None:
                dur = (close_ts - heat_global["heat_start_ts"]).total_seconds()
                out_heat.append(("H", (
                    int(heat_global["assigned_corrida"]) if heat_global["assigned_corrida"] is not None else None,
                    int(heat_global["assigned_carro"]) if heat_global["assigned_carro"] is not None else None,
                    heat_global["heat_start_ts"],
                    close_ts,
                    float(dur) if dur is not None else None,
                    float(heat_global["heat_start_energy"]) if heat_global["heat_start_energy"] is not None else None,
                    float(close_energy) if close_energy is not None else None,
                )))
                emit_heat_event(
                    status="HEAT_OFF",
                    event_ts=close_ts,
                    reason=f"no_inc>={HEAT_OFF_CONFIRM_S}s (end at last_inc)"
                )

            heat_global["heat_on"] = False
            heat_global["heat_start_ts"] = None
            heat_global["heat_start_energy"] = None

        def release_heat_assignment(close_ts=None, close_energy=None):
            if close_ts is not None:
                ce = close_energy if close_energy is not None else heat_global.get("last_energy_inc_val")
                if heat_global.get("last_energy_inc_ts") is not None and heat_global["heat_on"]:
                    close_ts = heat_global["last_energy_inc_ts"]
                    ce = heat_global.get("last_energy_inc_val")
                close_heat_global_if_open(close_ts, ce)

            heat_global["assigned_corrida"] = None
            heat_global["assigned_carro"] = None
            heat_global["last_energy_inc_ts"] = None
            heat_global["last_energy_inc_val"] = None

        def is_carro_active_for_heating(c):
            # considera RUNNING e END_PENDING como “corrida ainda ativa”
            return c["state"] in ("RUNNING", "END_PENDING") and c["run_corrida"] is not None and c["run_start_ts"] is not None

        def pick_active_corrida():
            """
            Regra:
            - se já tem assigned e ela ainda está ativa -> mantém
            - senão, escolhe a ativa mais antiga (menor run_start_ts)
            """
            ac = heat_global["assigned_carro"]
            ar = heat_global["assigned_corrida"]
            if ac in (1, 2) and ar is not None:
                c = ctx[ac]
                if is_carro_active_for_heating(c) and c["run_corrida"] == ar:
                    return (ar, ac)

            running = []
            for carro in (1, 2):
                c = ctx[carro]
                if is_carro_active_for_heating(c):
                    running.append((c["run_start_ts"], int(c["run_corrida"]), carro))

            if not running:
                return (None, None)

            running.sort(key=lambda x: (x[0], x[2]))
            _, corrida, carro = running[0]
            return (corrida, carro)

        def reset_to_idle(carro, ts, release_corrida_num=False, set_override_thr=None):
            c = ctx[carro]

            # se esse carro era dono do heating, libera (e fecha heat no ts)
            if heat_global["assigned_carro"] == carro and heat_global["assigned_corrida"] == c.get("run_corrida"):
                release_heat_assignment(close_ts=ts, close_energy=heat_global.get("prev_energy"))

            if release_corrida_num:
                release_corrida(c.get("run_corrida"))

            c["state"] = "IDLE"

            c["cand_start_ts"] = None
            c["cand_sustain_s"] = 0.0

            c["pending_end_ts0"] = None
            c["pending_confirmed_at"] = None
            c["pending_deadline"] = None

            # se pediu override (caso cancelamento por weak peak)
            if set_override_thr is not None:
                c["override_start_thr"] = float(set_override_thr)

            c["run_corrida"] = None
            c["run_start_ts"] = None
            c["run_confirmed_at"] = None
            c["peak_total"] = None

        prev_ts_global = None
        last_seen_ts = None

        for r in rows_iter:
            ts = r[TIMESTAMP_COL]
            last_seen_ts = ts

            # dt global
            if prev_ts_global is None or ts is None:
                dt_s = 0.0
            else:
                dt = (ts - prev_ts_global).total_seconds()
                dt_s = float(dt) if dt is not None and dt >= 0 and math.isfinite(dt) else 0.0
            prev_ts_global = ts

            gas_raw = {
                1: float(r["gas_carro1"] or 0.0),
                2: float(r["gas_carro2"] or 0.0),
            }
            energy_now = float(r["energia"] or 0.0)

            # -----------------------------
            # 1) FSM por carro
            # -----------------------------
            for carro in (1, 2):
                c = ctx[carro]

                # HOLD-ZERO (só GAS)
                gas = gas_raw[carro]
                if gas > 0:
                    c["last_nonzero_val"] = gas
                    c["last_nonzero_ts"] = ts
                else:
                    if HOLD_ZERO_S > 0 and c["last_nonzero_ts"] is not None and c["last_nonzero_val"] > 0:
                        dz = (ts - c["last_nonzero_ts"]).total_seconds()
                        if dz is not None and 0 <= dz <= float(HOLD_ZERO_S):
                            gas = float(c["last_nonzero_val"])

                prev_gas = float(c.get("_prev_gas", 0.0) or 0.0)
                c["_prev_gas"] = gas

                # start threshold (override só após cancel por weak peak)
                start_thr = float(c["override_start_thr"]) if c.get("override_start_thr") is not None else float(START_THRESHOLD)

                # -----------------------------------
                # END_PENDING: decide se fecha ou reabre
                # -----------------------------------
                if c["state"] == "END_PENDING":
                    # reabriu (merge): precisa ultrapassar um threshold MAIS ALTO (ex.: 100)
                    if gas > float(reopen_merge_threshold):
                        # cancela fechamento pendente e volta a RUNNING
                        c["state"] = "RUNNING"
                        c["pending_end_ts0"] = None
                        c["pending_confirmed_at"] = None
                        c["pending_deadline"] = None
                        # segue na mesma corrida, sem emitir nada
                    else:
                        # ainda não reabriu; se passou a janela -> fecha agora
                        if c["pending_deadline"] is not None and ts >= c["pending_deadline"]:
                            fp_end = c["pending_end_ts0"]
                            confirmed_at = c["pending_confirmed_at"]

                            emit_final(carro, fp_end=fp_end)
                            emit_event_fp(
                                carro=carro,
                                status="CLOSED",
                                event_ts=ts,
                                reason=f"end<= {END_THRESHOLD} sustained {END_CONFIRM_SUSTAIN_S}s + no_reopen<{REOPEN_MERGE_WINDOW_S}s (need>{reopen_merge_threshold})",
                                fp_end=fp_end,
                                confirmed_at=confirmed_at,
                            )

                            if heat_global["assigned_corrida"] == c["run_corrida"] and heat_global["assigned_carro"] == carro:
                                release_heat_assignment(close_ts=fp_end, close_energy=energy_now)

                            reset_to_idle(carro, ts, release_corrida_num=False)
                    continue


                # -----------------------------------
                # IDLE -> START_CANDIDATE
                # -----------------------------------
                if c["state"] == "IDLE":
                    crossed_up = (prev_gas <= start_thr) and (gas > start_thr)
                    if crossed_up:
                        c["state"] = "START_CANDIDATE"
                        c["cand_start_ts"] = ts
                        c["cand_sustain_s"] = 0.0
                        # IMPORTANT: override vale só até o próximo start; ao "tentar" start já podemos limpar
                        c["override_start_thr"] = None
                    continue

                # -----------------------------------
                # START_CANDIDATE
                # -----------------------------------
                if c["state"] == "START_CANDIDATE":
                    if gas > start_thr:
                        c["cand_sustain_s"] += dt_s
                        if c["cand_sustain_s"] >= float(START_CONFIRM_SUSTAIN_S):
                            corrida_num = alloc_corrida()

                            c["run_corrida"] = int(corrida_num)
                            c["run_start_ts"] = c["cand_start_ts"]  # 1º ts acima do thr
                            c["run_confirmed_at"] = ts
                            c["peak_total"] = float(gas)

                            c["cand_start_ts"] = None
                            c["cand_sustain_s"] = 0.0
                            c["state"] = "RUNNING"

                            emit_event_fp(
                                carro=carro,
                                status="RUNNING",
                                event_ts=ts,
                                reason=f"gas>={start_thr} sustained {START_CONFIRM_SUSTAIN_S}s",
                                fp_end=None,
                                confirmed_at=ts,
                            )
                    else:
                        # caiu antes de confirmar
                        c["state"] = "IDLE"
                        c["cand_start_ts"] = None
                        c["cand_sustain_s"] = 0.0
                    continue

                # -----------------------------------
                # RUNNING
                # -----------------------------------
                if c["state"] == "RUNNING":
                    # peak
                    if c["peak_total"] is None or gas >= float(c["peak_total"]):
                        c["peak_total"] = float(gas)

                    # CANCELLED por weak peak
                    if c["run_start_ts"] is not None and c["peak_total"] is not None:
                        run_elapsed = (ts - c["run_start_ts"]).total_seconds()
                        if run_elapsed is not None and run_elapsed >= float(min_peak_window_s):
                            if float(c["peak_total"]) < float(min_peak_gas):
                                emit_event_fp(
                                    carro=carro,
                                    status="CANCELLED",
                                    event_ts=ts,
                                    reason=f"weak_fp_peak<{min_peak_gas}_after_{min_peak_window_s}s",
                                    fp_end=None,
                                    confirmed_at=None,
                                )

                                # regra pedida:
                                # - corrida cancelada não consome número: devolve para reuso
                                # - define override_start_thr = patamar no cancelamento
                                #   (até o próximo START desse carro)
                                reset_to_idle(
                                    carro=carro,
                                    ts=ts,
                                    release_corrida_num=True,
                                    set_override_thr=gas,
                                )
                                continue

                    # END detect (abaixo por sustain)
                    if gas <= float(END_THRESHOLD):
                        # usamos "end_sustain" local com variáveis "pending_*" pra simplificar
                        if c["pending_end_ts0"] is None:
                            # abre sustain de fim
                            c["pending_end_ts0"] = ts
                            c["pending_confirmed_at"] = None  # ainda não confirmou
                            c["pending_deadline"] = None
                            c["_end_sustain_acc"] = 0.0  # atributo "inline"
                        c["_end_sustain_acc"] += dt_s

                        if c["_end_sustain_acc"] >= float(END_CONFIRM_SUSTAIN_S) and c["pending_confirmed_at"] is None:
                            # confirmou o sustain de fim, mas NÃO fecha ainda: entra em END_PENDING
                            c["pending_confirmed_at"] = ts
                            c["pending_deadline"] = ts + timedelta(seconds=int(REOPEN_MERGE_WINDOW_S))
                            c["state"] = "END_PENDING"
                            # mantém run ativo e heating ainda pode ficar atribuído aqui
                    else:
                        # voltou acima -> reseta fim
                        c["pending_end_ts0"] = None
                        c["pending_confirmed_at"] = None
                        c["pending_deadline"] = None
                        if "_end_sustain_acc" in c:
                            c["_end_sustain_acc"] = 0.0

            # -----------------------------
            # 2) HEATING GLOBAL (1 vez por timestamp)
            # -----------------------------
            prev_energy = heat_global["prev_energy"]
            if prev_energy is None:
                prev_energy = float(energy_now)
                heat_global["prev_energy"] = prev_energy

            dE = float(energy_now) - float(prev_energy)
            heat_global["prev_energy"] = float(energy_now)

            desired_corrida, desired_carro = pick_active_corrida()

            if desired_corrida is None:
                if heat_global["assigned_corrida"] is not None:
                    release_heat_assignment(close_ts=ts, close_energy=prev_energy)
            else:
                if heat_global["assigned_corrida"] != desired_corrida or heat_global["assigned_carro"] != desired_carro:
                    if heat_global["assigned_corrida"] is not None:
                        release_heat_assignment(close_ts=ts, close_energy=prev_energy)

                    heat_global["assigned_corrida"] = int(desired_corrida)
                    heat_global["assigned_carro"] = int(desired_carro)

                    heat_global["last_energy_inc_ts"] = None
                    heat_global["last_energy_inc_val"] = None
                    heat_global["heat_on"] = False
                    heat_global["heat_start_ts"] = None
                    heat_global["heat_start_energy"] = None

                if dE > float(energy_delta_eps):
                    heat_global["last_energy_inc_ts"] = ts
                    heat_global["last_energy_inc_val"] = float(energy_now)

                if heat_global["heat_on"]:
                    if heat_global["last_energy_inc_ts"] is not None:
                        no_inc_s = (ts - heat_global["last_energy_inc_ts"]).total_seconds()
                        if no_inc_s is not None and no_inc_s >= float(HEAT_OFF_CONFIRM_S):
                            heat_end = heat_global["last_energy_inc_ts"]
                            close_heat_global_if_open(heat_end, heat_global["last_energy_inc_val"])
                else:
                    if dE > float(energy_delta_eps):
                        heat_global["heat_on"] = True
                        heat_global["heat_start_ts"] = ts
                        heat_global["heat_start_energy"] = float(prev_energy)
                        emit_heat_event(
                            status="HEAT_ON",
                            event_ts=ts,
                            reason=f"dE={dE:.3f} > eps={float(energy_delta_eps):.3f}"
                        )

        # fim da janela: corridas ainda ativas -> RUN_LEFT_OPEN
        if last_seen_ts is not None:
            for carro in (1, 2):
                c = ctx[carro]
                if c["state"] in ("RUNNING", "END_PENDING") and c["run_corrida"] is not None:
                    emit_final(carro, fp_end=None)
                    emit_event_fp(
                        carro=carro,
                        status="RUN_LEFT_OPEN",
                        event_ts=last_seen_ts,
                        reason="window_ended_before_close",
                        fp_end=None,
                        confirmed_at=None,
                    )

            if heat_global["assigned_corrida"] is not None:
                end_ts = heat_global["last_energy_inc_ts"] or last_seen_ts
                end_energy = heat_global["last_energy_inc_val"] or heat_global["prev_energy"]
                close_heat_global_if_open(end_ts, end_energy)

        for e in out_events:
            yield e
        for f in out_final:
            yield f
        for h in out_heat:
            yield h

    tagged = ordered_1p.rdd.mapPartitions(fsm_partition)

    events_rdd = tagged.filter(lambda x: x[0] == "E").map(lambda x: x[1])
    final_rdd  = tagged.filter(lambda x: x[0] == "F").map(lambda x: x[1])
    heat_rdd   = tagged.filter(lambda x: x[0] == "H").map(lambda x: x[1])

    df_fp_events = spark.createDataFrame(events_rdd, schema=FP_EVENTS_SCHEMA)
    df_fp_final  = spark.createDataFrame(final_rdd,  schema=FP_FINAL_SCHEMA)
    df_heat      = spark.createDataFrame(heat_rdd,   schema=HEAT_SCHEMA)

    return df_fp_final, df_fp_events, df_heat


# COMMAND ----------

# DBTITLE 1,DETECT_VD
import pyspark
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def detect_vd_events_gas_based(
    spark_df_vd_pims: pyspark.sql.DataFrame,
    start_ts: str,
    end_ts: str,
    initial_heat_number: int,

    # ---- parâmetros VD
    reset_threshold: float = 1.0,
    minimal_duration_seconds: int = 30,

    # ---- debug
    debug: bool = False,
) -> pyspark.sql.DataFrame:
    """
    Detecta eventos de VD a partir de volumes acumulados por vaso,
    com pré-processamento PIMS embutido.

    MODELO:
      - volumes acumulados (Ar + N2)
      - reset físico = volume <= reset_threshold
      - VD ocorre ENTRE resets
      - início = 1ª amostra > threshold após reset
      - fim    = última amostra > threshold antes do próximo reset

    Retorna Spark DF:
      corrida | inicio_vd | final_vd | vd_tanque
    """

    TIMESTAMP_COL = "timestamp"

    # ======================================================
    # 1. PRÉ-PROCESSAMENTO PIMS (OBRIGATÓRIO PARA VD)
    # ======================================================
    df_pims = (
        spark_df_vd_pims
        .filter(
            (F.col(TIMESTAMP_COL) >= start_ts) &
            (F.col(TIMESTAMP_COL) <= end_ts)
        )
        .fillna(0, subset=[
            "ACI@VD_VASO1_VOLUMEAR",
            "ACI@VD_VASO1_VOLUMEN2",
            "ACI@VD_VASO2_VOLUMEAR",
            "ACI@VD_VASO2_VOLUMEN2",
        ])
        .select(
            TIMESTAMP_COL,
            (
                F.col("ACI@VD_VASO1_VOLUMEAR") +
                F.col("ACI@VD_VASO1_VOLUMEN2")
            ).alias("vol_v1"),
            (
                F.col("ACI@VD_VASO2_VOLUMEAR") +
                F.col("ACI@VD_VASO2_VOLUMEN2")
            ).alias("vol_v2"),
        )
        .orderBy(TIMESTAMP_COL)
    )

    # ======================================================
    # 2. UNIFICAR VASOS (stack)
    # ======================================================
    df_stack = (
        df_pims
        .select(
            F.explode(
                F.array(
                    F.struct(F.lit(1).alias("vd_tanque"), F.col("vol_v1").alias("volume")),
                    F.struct(F.lit(2).alias("vd_tanque"), F.col("vol_v2").alias("volume")),
                )
            ).alias("d"),
            TIMESTAMP_COL,
        )
        .select(
            TIMESTAMP_COL,
            F.col("d.vd_tanque").alias("vd_tanque"),
            F.col("d.volume").alias("volume"),
        )
    )

    # ======================================================
    # 3. DETECÇÃO DE RESETS
    # ======================================================
    w = Window.partitionBy("vd_tanque").orderBy(TIMESTAMP_COL)

    df_reset = (
        df_stack
        .withColumn("is_reset", F.col("volume") <= reset_threshold)
        .withColumn(
            "reset_id",
            F.sum(F.col("is_reset").cast("int")).over(w)
        )
    )

    # ======================================================
    # 4. MANTER SOMENTE VOLUME ATIVO
    # ======================================================
    df_active = df_reset.filter(F.col("volume") > reset_threshold)

    # ======================================================
    # 5. AGREGAÇÃO ENTRE RESETS
    # ======================================================
    agg_raw = (
        df_active
        .groupBy("vd_tanque", "reset_id")
        .agg(
            F.min(TIMESTAMP_COL).alias("inicio_vd"),
            F.max(TIMESTAMP_COL).alias("final_vd"),
        )
        .withColumn(
            "duration_seconds",
            F.col("final_vd").cast("long") - F.col("inicio_vd").cast("long")
        )
    )

    # ======================================================
    # 6. FILTRAR EVENTOS REAIS
    # ======================================================
    agg_valid = agg_raw.filter(
        F.col("duration_seconds") >= minimal_duration_seconds
    )

    # ======================================================
    # 7. GERAR CORRIDA (ORDENADO NO TEMPO)
    # ======================================================
    w_evt = Window.orderBy("inicio_vd")

    result = (
        agg_valid
        .withColumn(
            "event_idx",
            (F.row_number().over(w_evt) - 1).cast("int")
        )
        .withColumn(
            "corrida",
            (F.lit(initial_heat_number) + F.col("event_idx")).cast("int")
        )
        .select("corrida", "inicio_vd", "final_vd", "vd_tanque")
        .orderBy("corrida")
    )

    if debug:
        display(result)

    return result


# COMMAND ----------

# DBTITLE 1,DETECT_VD_NEW
def detect_vd_events(
    spark_df_vd_pims: pyspark.sql.DataFrame,
    start_ts,
    end_ts,
    initial_heat_number: int,
    reset_threshold: float = 0.99,
    flow_activity_threshold: float = 0.2406,
    min_event_confirm_seconds: Optional[int] = None,
    min_vd_arrival_confirm_seconds: Optional[int] = 5,
    min_vd_exit_confirm_seconds: Optional[int] = 5,
    min_deep_vac_start_confirm_seconds: Optional[int] = 30,
    min_deep_vac_end_confirm_seconds: Optional[int] = 0,
    deep_vac_end_pressure_threshold: float = 750.0,
    min_vd_duration_seconds: int = 35,
    startup_lookback_seconds: int = 180,
    startup_grace_seconds: int = 0,
    hold_inactive_seconds_flow: int = 3,
    hold_inactive_seconds_volume: int = 3,
    debug: bool = False,
    predicted_tank_input: Optional[int] = None,
) -> Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame, pyspark.sql.DataFrame]:
    """
    Detect VD (Vacuum Degassing) usage cycles and deep vacuum intervals using a
    streaming-friendly, 100% Spark approach (no lead / no future windows).

    The detector is designed for micro-batch streaming (e.g., foreachBatch) and
    batch runs. All detections are causal, using only past data via lag() and
    "unboundedPreceding -> currentRow" windows.

    Logical events emitted (append-only):
      1) VD_ARRIVAL
         - Definition: Rising edge of FLOW (VZFORTE + VZFRACA) for a given tank.
         - event_ts: The first timestamp where flow becomes active.
         - confirmed_at: The first timestamp where the VD session has remained active
                         for at least min_vd_arrival_confirm_seconds.

      2) DEEP_VAC_START
         - Definition: Rising edge of VOLUME (VOLUMEAR + VOLUMEN2) for a given tank.
         - event_ts: The first timestamp where volume becomes active.
         - confirmed_at: The first timestamp where the deep vacuum interval has remained
                         active for at least min_deep_vac_start_confirm_seconds.

      3) DEEP_VAC_END
         - Rising edge of PRESSURE: ACI@VD_PRESSAO_DO_VACUO >= deep_vac_end_pressure_threshold
           (previous samples below threshold)
         - event_ts: first timestamp of the crossing above threshold
         - confirmed_at: first timestamp where pressure stayed continuously above threshold
                         for at least min_deep_vac_end_confirm_seconds

      4) VD_EXIT
         - Definition: Falling edge of FLOW (VZFORTE + VZFRACA) for a given tank.
         - event_ts: The falling edge timestamp (first point where flow becomes inactive).
         - confirmed_at:
             * If min_vd_exit_confirm_seconds == 0, equals event_ts (immediate).
             * Otherwise, the first timestamp where flow has remained inactive for
               at least min_vd_exit_confirm_seconds.

    Anti-noise / robustness protections:
      - FLOW_TOTAL tag quantization:
          * values < flow_activity_threshold are forced to 0.0
        This avoids tiny float noise (e.g., 0.05) triggering false VD sessions.

      - Causal glitch repair (streaming-safe "hold"):
          * If a signal briefly drops below its activity threshold but returns within
            hold_inactive_seconds_*, keep the last active value for up to that duration
        This is causal and does not require future lookahead.

      - VD minimum duration filter:
          * VD sessions shorter than min_vd_duration_seconds are dropped BEFORE
            assigning corrida and emitting events.

      - Confirmation delays (sustain-based confirmation):
          * VD_ARRIVAL is emitted only after min_vd_arrival_confirm_seconds
          * VD_EXIT can be delayed by min_vd_exit_confirm_seconds
          * DEEP_VAC_START is emitted only after min_deep_vac_start_confirm_seconds
          * DEEP_VAC_END can be delayed by min_deep_vac_end_confirm_seconds

      - Optional global confirm seconds:
          * If min_event_confirm_seconds is provided, any per-event confirm parameter
            that is None will inherit its value.

      - Start boundary robustness:
          * startup_lookback_seconds: reads some history before start_ts to avoid
            artificial rising edges caused by missing lag context.
          * startup_grace_seconds: optionally suppress events confirmed too soon after
            start_ts (useful when the sensor chatters after boot).

    Args:
        spark_df_vd_pims (DataFrame): Input Spark DataFrame with timestamp, volume tags,
            and flow tags.
        start_ts, end_ts: Can be str (ISO-like), datetime, or pandas.Timestamp.
        initial_heat_number (int): Base corrida number used to assign sequential IDs.
        reset_threshold (float): Threshold for considering flow/volume active.
        min_event_confirm_seconds (Optional[int]): Global default confirm delay.
        min_vd_arrival_confirm_seconds (Optional[int]): Confirm delay for VD_ARRIVAL.
        min_vd_exit_confirm_seconds (Optional[int]): Confirm delay for VD_EXIT.
        min_deep_vac_start_confirm_seconds (Optional[int]): Confirm delay for DEEP_VAC_START.
        min_deep_vac_end_confirm_seconds (Optional[int]): Confirm delay for DEEP_VAC_END
            (pressure-above sustain).
        deep_vac_end_pressure_threshold (float): Pressure threshold for DEEP_VAC_END.
        min_vd_duration_seconds (int): Minimum VD session duration kept.
        startup_lookback_seconds (int): Lookback window before start_ts for context.
        startup_grace_seconds (int): Suppress events confirmed before start_ts + grace.
        hold_inactive_seconds_flow (int): Causal hold for flow dropouts (seconds).
        hold_inactive_seconds_volume (int): Causal hold for volume dropouts (seconds).
        debug (bool): If True, returns a third dataframe with trace/flags.
        predicted_tank_input (Optional[int]): Metadata passthrough to df_vd_final.

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame]:
            (df_vd_final, df_vd_events, df_vd_debug)

            df_vd_final:
              corrida, tank_id, start_vd_time, end_vd_time, duration_seconds,
              max_volume, predicted_tank_input

            df_vd_events:
              corrida, status, tank_id, start_vd_time, end_vd_time, duration_seconds,
              confirmed_at, event_ts

            df_vd_debug:
              Debug/trace flags; empty if debug=False.
    """
        # 3) DEEP_VAC_END
        #  - Definition: Falling edge of VOLUME (VOLUMEAR + VOLUMEN2) for a given tank.
        #  - event_ts: The falling edge timestamp (first point where volume becomes inactive).
        #  - confirmed_at:
        #      * If min_deep_vac_end_confirm_seconds == 0, equals event_ts (immediate).
        #      * Otherwise, the first timestamp where volume has remained inactive for
        #        at least min_deep_vac_end_confirm_seconds.


    # ----------------------------
    # Helpers: accept str/datetime/pandas.Timestamp for start_ts/end_ts
    # ----------------------------
    def _to_py_datetime(x) -> datetime:
        if isinstance(x, datetime):
            dt = x
        else:
            # pandas.Timestamp has .to_pydatetime()
            if hasattr(x, "to_pydatetime"):
                dt = x.to_pydatetime()
            else:
                dt = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
        # Spark timestamps are usually tz-naive in pipelines
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt

    def _shift_dt(dt: datetime, seconds: int) -> datetime:
        return dt + timedelta(seconds=int(seconds))

    start_dt = _to_py_datetime(start_ts)
    end_dt = _to_py_datetime(end_ts)

    # ----------------------------
    # Confirmation defaults wiring:
    # if min_event_confirm_seconds is provided, any None per-event value inherits it
    # ----------------------------
    if min_event_confirm_seconds is not None:
        if min_vd_arrival_confirm_seconds is None:
            min_vd_arrival_confirm_seconds = int(min_event_confirm_seconds)
        if min_vd_exit_confirm_seconds is None:
            min_vd_exit_confirm_seconds = int(min_event_confirm_seconds)
        if min_deep_vac_start_confirm_seconds is None:
            min_deep_vac_start_confirm_seconds = int(min_event_confirm_seconds)
        if min_deep_vac_end_confirm_seconds is None:
            min_deep_vac_end_confirm_seconds = int(min_event_confirm_seconds)

    min_vd_arrival_confirm_seconds = int(min_vd_arrival_confirm_seconds or 0)
    min_vd_exit_confirm_seconds = int(min_vd_exit_confirm_seconds or 0)
    min_deep_vac_start_confirm_seconds = int(min_deep_vac_start_confirm_seconds or 0)
    min_deep_vac_end_confirm_seconds = int(min_deep_vac_end_confirm_seconds or 0)
    min_vd_duration_seconds = int(min_vd_duration_seconds)

    startup_lookback_seconds = int(startup_lookback_seconds or 0)
    startup_grace_seconds = int(startup_grace_seconds or 0)
    hold_inactive_seconds_flow = int(hold_inactive_seconds_flow or 0)
    hold_inactive_seconds_volume = int(hold_inactive_seconds_volume or 0)

    thr = float(reset_threshold)
    pressure_thr = float(deep_vac_end_pressure_threshold)

    # ----------------------------
    # Schemas for empty returns
    # ----------------------------
    VD_FINAL_SCHEMA = StructType(
        [
            StructField("corrida", IntegerType(), True),
            StructField("tank_id", IntegerType(), True),
            StructField("start_vd_time", TimestampType(), True),
            StructField("end_vd_time", TimestampType(), True),
            StructField("duration_seconds", LongType(), True),
            StructField("max_volume", DoubleType(), True),
            StructField("predicted_tank_input", IntegerType(), True),
        ]
    )

    VD_EVENTS_SCHEMA = StructType(
        [
            StructField("corrida", IntegerType(), True),
            StructField("status", StringType(), True),
            StructField("tank_id", IntegerType(), True),
            StructField("start_vd_time", TimestampType(), True),
            StructField("end_vd_time", TimestampType(), True),
            StructField("duration_seconds", LongType(), True),
            StructField("confirmed_at", TimestampType(), True),
            StructField("event_ts", TimestampType(), True),
        ]
    )

    VD_DEBUG_SCHEMA = StructType(
        [
            StructField("timestamp", TimestampType(), True),
            StructField("tank_id", IntegerType(), True),
            StructField("vol", DoubleType(), True),
            StructField("flow", DoubleType(), True),
            StructField("flow_active", BooleanType(), True),
            StructField("flow_rise", BooleanType(), True),
            StructField("flow_fall", BooleanType(), True),
            StructField("vd_session_id", LongType(), True),
            StructField("vol_active", BooleanType(), True),
            StructField("vol_rise", BooleanType(), True),
            StructField("vol_fall", BooleanType(), True),
            StructField("deep_vacuum_session_id", LongType(), True),
        ]
    )


    spark = spark_df_vd_pims.sparkSession

    # ----------------------------
    # PIMS columns
    # ----------------------------
    TS = "timestamp"

    V1_AR = "ACI@VD_VASO1_VOLUMEAR"
    V1_N2 = "ACI@VD_VASO1_VOLUMEN2"
    V2_AR = "ACI@VD_VASO2_VOLUMEAR"
    V2_N2 = "ACI@VD_VASO2_VOLUMEN2"

    F1_STRONG = "ACI@VD_VASO1_VZFORTE"
    F1_WEAK = "ACI@VD_VASO1_VZFRACA"
    F2_STRONG = "ACI@VD_VASO2_VZFORTE"
    F2_WEAK = "ACI@VD_VASO2_VZFRACA"

    PRESSURE = "ACI@VD_PRESSAO_DO_VACUO"

    flow_cols = [F1_STRONG, F1_WEAK, F2_STRONG, F2_WEAK]

    columns_to_fill = [
        V1_AR,
        V1_N2,
        V2_AR,
        V2_N2,
        F1_STRONG,
        F1_WEAK,
        F2_STRONG,
        F2_WEAK,
        PRESSURE,
    ]

    # ----------------------------
    # 1) Preprocess: filter + cast + stable order + forward-fill (no future)
    #
    # We read some history before start_dt to avoid artificial edges at the boundary.
    # ----------------------------
    preprocess_start_dt = start_dt
    if startup_lookback_seconds > 0:
        preprocess_start_dt = _shift_dt(start_dt, -startup_lookback_seconds)

    base = preprocess_pims_data_streaming(
        spark_df=spark_df_vd_pims,
        first_timestamp=preprocess_start_dt,
        last_timestamp=end_dt,
        timestamp_column=TS,
        columns_to_fill=columns_to_fill,
    )

    # Consolidate to 1 row per timestamp (defensive against duplicates)
    base_1p = (
        base.groupBy(F.col(TS).cast("timestamp").alias(TS))
        .agg(
            F.max(F.col(V1_AR)).alias(V1_AR),
            F.max(F.col(V1_N2)).alias(V1_N2),
            F.max(F.col(V2_AR)).alias(V2_AR),
            F.max(F.col(V2_N2)).alias(V2_N2),
            F.max(F.col(F1_STRONG)).alias(F1_STRONG),
            F.max(F.col(F1_WEAK)).alias(F1_WEAK),
            F.max(F.col(F2_STRONG)).alias(F2_STRONG),
            F.max(F.col(F2_WEAK)).alias(F2_WEAK),
            F.max(F.col(PRESSURE)).alias(PRESSURE),
        )
        .orderBy(F.col(TS).asc())
    )

    if base_1p.rdd.isEmpty():
        return (
            spark.createDataFrame([], schema=VD_FINAL_SCHEMA),
            spark.createDataFrame([], schema=VD_EVENTS_SCHEMA),
            spark.createDataFrame([], schema=VD_DEBUG_SCHEMA),
        )

    # ----------------------------
    # 2) Compute totals (volume and flow per tank) + keep pressure
    # ----------------------------
    df_tot = base_1p.select(
        F.col(TS),
        (F.col(V1_AR) + F.col(V1_N2)).alias("vol_total_v1"),
        (F.col(V2_AR) + F.col(V2_N2)).alias("vol_total_v2"),
        (F.col(F1_STRONG) + F.col(F1_WEAK)).alias("flow_total_v1"),
        (F.col(F2_STRONG) + F.col(F2_WEAK)).alias("flow_total_v2"),
        F.col(PRESSURE).alias("vacuum_pressure"),
    )

    # ----------------------------
    # 2.1) Anti-noise FIX: quantize flow tags
    # ----------------------------
    def quantize_flow(c: str) -> F.Column:
        return F.when(
            # F.col(c) < F.lit(0.2405333),  # 2025-12-09T12:37:40.000
            # F.col(c) < F.lit(0.2422475),  # 2025-12-09T12:38:51.000
            F.col(c) < F.lit(0.2406),  # 2025-12-09T12:37:42.000
            F.lit(0.0),
        ).otherwise(F.col(c))
        # ).otherwise(F.round(F.col(c)))

    for c in ["flow_total_v1", "flow_total_v2"]:
        df_tot = df_tot.withColumn(c, quantize_flow(c))

    # ----------------------------
    # 3) Unpivot to (tank_id, timestamp, vol, flow, pressure)
    # ----------------------------
    v1_struct = F.struct(
        F.lit(1).alias("tank_id"),
        F.col(TS).alias(TS),
        F.col("vol_total_v1").alias("vol"),
        F.col("flow_total_v1").alias("flow"),
        F.col("vacuum_pressure").alias("pressure"),
    )
    v2_struct = F.struct(
        F.lit(2).alias("tank_id"),
        F.col(TS).alias(TS),
        F.col("vol_total_v2").alias("vol"),
        F.col("flow_total_v2").alias("flow"),
        F.col("vacuum_pressure").alias("pressure"),
    )

    df_stacked = (
        df_tot.select(F.array(v1_struct, v2_struct).alias("data"))
        .select(F.explode("data").alias("d"))
        .select(
            F.col("d.tank_id").alias("tank_id"),
            F.col(f"d.{TS}").alias(TS),
            F.col("d.vol").alias("vol"),
            F.col("d.flow").alias("flow"),
            F.col("d.pressure").alias("pressure"),
        )
        .orderBy(F.col(TS).asc(), F.col("tank_id").asc())
    )

    # ----------------------------
    # 3.1) Streaming-safe glitch repair ("hold" short inactive drops)
    #
    # This replaces the previous lookahead-based repair_data_glitches approach.
    # It is causal: uses only the last active value/time and holds it for up to N seconds.
    # ----------------------------
    w_hold = (
        Window.partitionBy("tank_id")
        .orderBy(F.col(TS).asc())
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    def hold_short_inactive_drops(
        value_col: str,
        hold_seconds: int,
    ) -> F.Column:
        """
        Hold (carry forward) the last “active” value for a short period when the signal briefly drops
        below an activity threshold, preventing short inactive dips from being interpreted as real changes.
        
        Args:
            value_col (str):
                Name of the Spark DataFrame column containing the numeric signal to stabilize
                (e.g., vacuum pressure, flow, valve position).
            hold_seconds (int):
                Maximum duration (in seconds) for which a brief drop below the threshold should be ignored.
                If `hold_seconds <= 0`, the function returns the original column unchanged.

        Returns:
            pyspark.sql.column.Column:
                A Spark Column expression representing a “held” version of `value_col`, where short-lived
                dips below `thr` (up to `hold_seconds`) are replaced by the last active value.
        """
        if hold_seconds <= 0:
            return F.col(value_col)

        last_active_val = F.last(
            F.when(F.col(value_col) >= F.lit(thr), F.col(value_col)),
            ignorenulls=True,
        ).over(w_hold)

        last_active_ts = F.last(
            F.when(F.col(value_col) >= F.lit(thr), F.col(TS)),
            ignorenulls=True,
        ).over(w_hold)

        dt_s = F.col(TS).cast("long") - last_active_ts.cast("long")

        return F.when(
            (last_active_ts.isNotNull())
            & (F.col(value_col) < F.lit(thr))
            & (dt_s >= F.lit(0))
            & (dt_s <= F.lit(int(hold_seconds))),
            last_active_val,
        ).otherwise(F.col(value_col))

    df_stacked = (
        df_stacked.withColumn(
            "flow",
            hold_short_inactive_drops(
                value_col="flow",
                hold_seconds=hold_inactive_seconds_flow,
            ),
        )
        .withColumn(
            "vol",
            hold_short_inactive_drops(
                value_col="vol",
                hold_seconds=hold_inactive_seconds_volume,
            ),
        )
    )

    # ----------------------------
    # 4) Edge detection with lag() only (no future)
    #
    # Note: We treat "active" as >= threshold (important when flow can be 1).
    # ----------------------------
    w_tank = Window.partitionBy("tank_id").orderBy(F.col(TS).asc())

    df_flags = (
        df_stacked.withColumn("flow_active", F.col("flow") >= F.lit(thr))
        .withColumn("prev_flow_active", F.lag("flow_active", 1, False).over(w_tank))
        .withColumn("flow_rise", (~F.col("prev_flow_active")) & F.col("flow_active"))
        .withColumn("flow_fall", F.col("prev_flow_active") & (~F.col("flow_active")))
        .withColumn(
            "vd_session_id",
            F.sum(F.col("flow_rise").cast("long")).over(w_tank),
        )
        .withColumn("vol_active", F.col("vol") >= F.lit(thr))
        .withColumn("prev_vol_active", F.lag("vol_active", 1, False).over(w_tank))
        .withColumn("vol_rise", (~F.col("prev_vol_active")) & F.col("vol_active"))
        .withColumn("vol_fall", F.col("prev_vol_active") & (~F.col("vol_active")))
        .withColumn(
            "deep_vacuum_session_id",
            F.sum(F.col("vol_rise").cast("long")).over(w_tank),
        )
    )

    # ----------------------------
    # 5) VD session aggregation (FLOW-based) + arrival/exit confirmation
    #
    # Goal:
    #   - event_ts for arrival = start_vd_time (true rising edge)
    #   - confirmed_at for arrival = first timestamp where
    #       (ts - start_vd_time) >= min_vd_arrival_confirm_seconds
    #
    #   - event_ts for exit = end_vd_time (true falling edge)
    #   - confirmed_at for exit = first timestamp where FLOW remained inactive
    #     for at least min_vd_exit_confirm_seconds (0 means immediate)
    # ----------------------------
    w_vd_part = Window.partitionBy("tank_id", "vd_session_id")
    w_vd_full = (
        Window.partitionBy("tank_id", "vd_session_id")
        .orderBy(F.col(TS).asc())
        .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    )

    vd_enriched = (
        df_flags.filter(F.col("vd_session_id") > 0)
        .withColumn(
            "vd_start_time",
            F.min(F.when(F.col("flow_active"), F.col(TS))).over(w_vd_part),
        )
        .withColumn(
            "vd_duration_so_far_s",
            F.when(
                F.col("flow_active"),
                F.col(TS).cast("long") - F.col("vd_start_time").cast("long"),
            ),
        )
        .withColumn(
            "vd_arrival_confirmed_at",
            F.min(
                F.when(
                    F.col("vd_duration_so_far_s")
                    >= F.lit(int(min_vd_arrival_confirm_seconds)),
                    F.col(TS),
                )
            ).over(w_vd_part),
        )
        .withColumn(
            "vd_end_edge_time",
            F.min(F.when(F.col("flow_fall"), F.col(TS))).over(w_vd_part),
        )
        .withColumn(
            "vd_exit_inactive_s",
            F.when(
                (F.col("vd_end_edge_time").isNotNull())
                & (F.col(TS) >= F.col("vd_end_edge_time"))
                & (~F.col("flow_active")),
                F.col(TS).cast("long") - F.col("vd_end_edge_time").cast("long"),
            ),
        )
        .withColumn(
            "vd_exit_confirmed_at",
            F.when(
                F.lit(int(min_vd_exit_confirm_seconds)) == F.lit(0),
                F.col("vd_end_edge_time"),
            ).otherwise(
                F.min(
                    F.when(
                        F.col("vd_exit_inactive_s")
                        >= F.lit(int(min_vd_exit_confirm_seconds)),
                        F.col(TS),
                    )
                ).over(w_vd_part)
            ),
        )
        .withColumn(
            "vd_last_seen_time",
            F.max(F.when(F.col("flow_active"), F.col(TS))).over(w_vd_part),
        )
    )

    vd_sessions_raw = (
        vd_enriched.select(
            "tank_id",
            "vd_session_id",
            F.col("vd_start_time").alias("start_vd_time"),
            F.col("vd_end_edge_time").alias("end_vd_time"),
            F.col("vd_last_seen_time").alias("last_seen_vd_time"),
            F.col("vd_arrival_confirmed_at").alias("arrival_confirmed_at"),
            F.col("vd_exit_confirmed_at").alias("exit_confirmed_at"),
        )
        .dropDuplicates(["tank_id", "vd_session_id"])
        .filter(F.col("start_vd_time").isNotNull())
        .withColumn(
            "duration_seconds",
            (
                F.coalesce(F.col("end_vd_time"), F.col("last_seen_vd_time")).cast("long")
                - F.col("start_vd_time").cast("long")
            ),
        )
    )

    # Drop short false VD sessions BEFORE assigning corrida / emitting events.
    vd_sessions = vd_sessions_raw.filter(
        F.col("duration_seconds") >= F.lit(int(min_vd_duration_seconds))
    )

    # corrida sequencial por ordem global de início (batch/microbatch)
    w_global = Window.orderBy(F.col("start_vd_time").asc(), F.col("tank_id").asc())

    vd_sessions_with_corrida = (
        vd_sessions.withColumn("heat_increment", F.row_number().over(w_global) - F.lit(1))
        .withColumn("corrida", F.lit(int(initial_heat_number)) + F.col("heat_increment"))
        .withColumn("predicted_tank_input", F.lit(predicted_tank_input))
    )

    # max volume observed while inside the VD session
    max_vol_per_vd = (
        df_flags.filter(F.col("vd_session_id") > 0)
        .groupBy("tank_id", "vd_session_id")
        .agg(F.max(F.col("vol")).alias("max_volume"))
    )

    df_vd_final = (
        vd_sessions_with_corrida.join(
            max_vol_per_vd,
            on=["tank_id", "vd_session_id"],
            how="left",
        )
        .select(
            "corrida",
            "tank_id",
            "start_vd_time",
            "end_vd_time",
            F.col("duration_seconds").cast("long").alias("duration_seconds"),
            F.coalesce(F.col("max_volume"), F.lit(0.0)).alias("max_volume"),
            "predicted_tank_input",
            "arrival_confirmed_at",
            "exit_confirmed_at",
        )
        .orderBy(F.col("start_vd_time").asc(), F.col("tank_id").asc())
    )

    # ----------------------------
    # 6) Deep vacuum detection (VOLUME-based) with "confirm_at"
    #
    # Start (VOLUME-based) with confirm_at:
    #   - event_ts (true start): deep_vacuum_start_time = rising edge timestamp (vol
    #                  becomes active)
    #   - confirmed_at: first timestamp where (current_ts - deep_vacuum_start_time)
    #                  reaches min_deep_vac_start_confirm_seconds
    #
    # This allows emitting DEEP_VAC_START as append-only (no retract):
    #   - when confirmed_at happens, we emit the event with event_ts = real start.
    #
    # For DEEP_VAC_END (PRESSURE-based) with sustain above threshold:
    #   - event_ts: first pressure crossing above deep_vac_end_pressure_threshold
    #   - confirmed_at: first ts where pressure stayed continuously above threshold
    #                  for >= min_deep_vac_end_confirm_seconds
    # ----------------------------
    w_dv_part = Window.partitionBy("tank_id", "deep_vacuum_session_id")
    w_dv_order = Window.partitionBy("tank_id", "deep_vacuum_session_id").orderBy(
        F.col(TS).asc()
    )
    w_dv_order_rows = w_dv_order.rowsBetween(Window.unboundedPreceding, Window.currentRow)

    w_dv_full = (
        Window.partitionBy("tank_id", "deep_vacuum_session_id")
        .orderBy(F.col(TS).asc())
        .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    )

    dv_enriched = (
        df_flags.filter(F.col("deep_vacuum_session_id") > 0)
        .withColumn(
            "deep_vacuum_start_time",
            F.min(F.when(F.col("vol_active"), F.col(TS))).over(w_dv_part),
        )
        .withColumn(
            "deep_vacuum_duration_so_far_s",
            F.when(
                F.col("vol_active"),
                F.col(TS).cast("long")
                - F.col("deep_vacuum_start_time").cast("long"),
            ),
        )
        .withColumn(
            "deep_vacuum_start_confirmed_at",
            F.min(
                F.when(
                    F.col("deep_vacuum_duration_so_far_s")
                    >= F.lit(int(min_deep_vac_start_confirm_seconds)),
                    F.col(TS),
                )
            ).over(w_dv_part),
        )
        # Which VD session were we in when deep vacuum started?
        .withColumn(
            "vd_session_at_deep_vacuum_start",
            F.first(
                F.when(F.col("vol_rise"), F.col("vd_session_id")),
                ignorenulls=True,
            ).over(w_dv_full),
        )
        # ----------------------------
        # DEEP_VAC_END via PRESSURE rising above threshold with sustain confirm
        # ----------------------------
        .withColumn("pressure_above_thr", F.col("pressure") >= F.lit(pressure_thr))
        .withColumn(
            "prev_pressure_above_thr",
            F.lag("pressure_above_thr", 1, False).over(w_dv_order),
        )
        .withColumn(
            "pressure_rise_thr",
            (~F.col("prev_pressure_above_thr")) & F.col("pressure_above_thr"),
        )
        # Run id increments on each rise above threshold (continuous above segments)
        .withColumn(
            "pressure_above_run_id",
            F.sum(F.col("pressure_rise_thr").cast("long")).over(w_dv_order_rows),
        )
    )

    # Per-run windows (continuous pressure-above segments)
    w_run = Window.partitionBy(
        "tank_id",
        "deep_vacuum_session_id",
        "pressure_above_run_id",
    )

    dv_enriched = dv_enriched.withColumn(
        "pressure_run_start_time",
        F.min(F.when(F.col("pressure_above_thr"), F.col(TS))).over(w_run),
    ).withColumn(
        "pressure_run_duration_so_far_s",
        F.when(
            F.col("pressure_above_thr"),
            F.col(TS).cast("long") - F.col("pressure_run_start_time").cast("long"),
        ),
    )

    if int(min_deep_vac_end_confirm_seconds) == 0:
        dv_enriched = dv_enriched.withColumn(
            "pressure_run_confirmed_at",
            F.col("pressure_run_start_time"),
        )
    else:
        dv_enriched = dv_enriched.withColumn(
            "pressure_run_confirmed_at",
            F.min(
                F.when(
                    F.col("pressure_run_duration_so_far_s")
                    >= F.lit(int(min_deep_vac_end_confirm_seconds)),
                    F.col(TS),
                )
            ).over(w_run),
        )

    # Pick the earliest valid pressure-above run AFTER deep vacuum start
    valid_run = (
        (F.col("pressure_above_run_id") > F.lit(0))
        & F.col("pressure_run_start_time").isNotNull()
        & F.col("pressure_run_confirmed_at").isNotNull()
        & (F.col("pressure_run_start_time") >= F.col("deep_vacuum_start_time"))
    )

    best_end_struct = F.min(
        F.when(
            valid_run,
            F.struct(
                F.col("pressure_run_start_time").alias("end_time"),
                F.col("pressure_run_confirmed_at").alias("confirmed_at"),
            ),
        )
    ).over(w_dv_part)

    dv_enriched = dv_enriched.withColumn(
        "deep_vacuum_end_time",
        best_end_struct["end_time"],
    ).withColumn(
        "deep_vacuum_end_confirmed_at",
        best_end_struct["confirmed_at"],
    )

    deep_vacuum_sessions = (
        dv_enriched.select(
            "tank_id",
            "deep_vacuum_session_id",
            "deep_vacuum_start_time",
            "deep_vacuum_start_confirmed_at",
            "deep_vacuum_end_time",
            "deep_vacuum_end_confirmed_at",
            "vd_session_at_deep_vacuum_start",
        )
        .dropDuplicates(["tank_id", "deep_vacuum_session_id"])
        # Keep only deep vacuum sessions that (a) happened inside a VD session and
        # (b) reached the minimal duration to be considered valid (start confirmed).
        .filter(F.col("vd_session_at_deep_vacuum_start") > 0)
        .filter(F.col("deep_vacuum_start_confirmed_at").isNotNull())
    )

    # Only deep vacuum sessions that belong to a VD session we kept
    # (i.e., VD duration >= min_vd_duration_seconds).
    dv_with_corrida = (
        deep_vacuum_sessions.join(
            vd_sessions_with_corrida.select(
                "tank_id",
                "vd_session_id",
                "corrida",
                "start_vd_time",
                "end_vd_time",
                "duration_seconds",
            ),
            on=(
                (deep_vacuum_sessions["tank_id"] == vd_sessions_with_corrida["tank_id"])
                & (
                    deep_vacuum_sessions["vd_session_at_deep_vacuum_start"]
                    == vd_sessions_with_corrida["vd_session_id"]
                )
            ),
            how="inner",
        )
        .select(
            F.col("corrida"),
            deep_vacuum_sessions["tank_id"].alias("tank_id"),
            F.col("start_vd_time"),
            F.col("end_vd_time"),
            F.col("duration_seconds").cast("long").alias("duration_seconds"),
            F.col("deep_vacuum_start_time"),
            F.col("deep_vacuum_end_time"),
            F.col("deep_vacuum_start_confirmed_at"),
            F.col("deep_vacuum_end_confirmed_at"),
        )
    )

    # ----------------------------
    # 7) Build df_vd_events (append-only)
    #
    # We emit events only after their confirmation timestamp is available.
    # This ensures no retract is required downstream.
    # ----------------------------
    arrival_events = (
        df_vd_final.filter(F.col("arrival_confirmed_at").isNotNull())
        .select(
            F.col("corrida"),
            F.lit("VD_ARRIVAL").alias("status"),
            F.col("tank_id"),
            F.col("start_vd_time"),
            F.col("end_vd_time"),
            F.col("duration_seconds").cast("long").alias("duration_seconds"),
            F.col("arrival_confirmed_at").alias("confirmed_at"),
            F.col("start_vd_time").alias("event_ts"),
        )
    )

    exit_events = (
        df_vd_final.filter(F.col("end_vd_time").isNotNull())
        .filter(F.col("exit_confirmed_at").isNotNull())
        .select(
            F.col("corrida"),
            F.lit("VD_EXIT").alias("status"),
            F.col("tank_id"),
            F.col("start_vd_time"),
            F.col("end_vd_time"),
            F.col("duration_seconds").cast("long").alias("duration_seconds"),
            F.col("exit_confirmed_at").alias("confirmed_at"),
            F.col("end_vd_time").alias("event_ts"),
        )
    )

    deep_vac_start_events = dv_with_corrida.select(
        F.col("corrida"),
        F.lit("DEEP_VAC_START").alias("status"),
        F.col("tank_id"),
        F.col("start_vd_time"),
        F.col("end_vd_time"),
        F.col("duration_seconds").cast("long").alias("duration_seconds"),
        F.col("deep_vacuum_start_confirmed_at").alias("confirmed_at"),
        F.col("deep_vacuum_start_time").alias("event_ts"),
    )

    deep_vac_end_events = (
        dv_with_corrida.filter(F.col("deep_vacuum_end_time").isNotNull())
        .filter(F.col("deep_vacuum_end_confirmed_at").isNotNull())
        .select(
            F.col("corrida"),
            F.lit("DEEP_VAC_END").alias("status"),
            F.col("tank_id"),
            F.col("start_vd_time"),
            F.col("end_vd_time"),
            F.col("duration_seconds").cast("long").alias("duration_seconds"),
            F.col("deep_vacuum_end_confirmed_at").alias("confirmed_at"),
            F.col("deep_vacuum_end_time").alias("event_ts"),
        )
    )

    df_vd_events = (
        arrival_events.unionByName(deep_vac_start_events)
        .unionByName(deep_vac_end_events)
        .unionByName(exit_events)
        .orderBy(F.col("event_ts").asc(), F.col("tank_id").asc())
    )

    # ----------------------------
    # 7.1) Respect original [start_dt, end_dt] window in outputs
    #
    # We used lookback for context; now we filter outputs to avoid "early" events.
    # ----------------------------
    start_lit = F.lit(start_dt)
    end_lit = F.lit(end_dt)

    df_vd_final = df_vd_final.filter(F.col("start_vd_time") >= start_lit)
    df_vd_events = df_vd_events.filter(
        (F.col("event_ts") >= start_lit) & (F.col("event_ts") <= end_lit)
    )

    # Optional startup grace: suppress events confirmed too soon after start.
    if startup_grace_seconds > 0:
        grace_cutoff_dt = _shift_dt(start_dt, startup_grace_seconds)
        df_vd_events = df_vd_events.filter(F.col("confirmed_at") >= F.lit(grace_cutoff_dt))

    # Remove helper columns from df_vd_final
    df_vd_final = df_vd_final.select(
        "corrida",
        "tank_id",
        "start_vd_time",
        "end_vd_time",
        "duration_seconds",
        "max_volume",
        "predicted_tank_input",
    )

    # ----------------------------
    # 8) Debug output (optional)
    # ----------------------------
    if debug:
        df_vd_debug = (
            dv_enriched.select(
                F.col(TS).alias("timestamp"),
                "tank_id",
                "vol",
                "flow",
                "pressure",
                "flow_active",
                "flow_rise",
                "flow_fall",
                "vd_session_id",
                "vol_active",
                "vol_rise",
                "vol_fall",
                "deep_vacuum_session_id",
                "pressure_above_thr",
                "pressure_rise_thr",
            )
            .orderBy(F.col("timestamp").asc(), F.col("tank_id").asc())
        )
    else:
        df_vd_debug = spark.createDataFrame([], schema=VD_DEBUG_SCHEMA)

    return df_vd_final, df_vd_events, df_vd_debug

# COMMAND ----------

# DBTITLE 1,DETECT_LC
# ============================================================
# CÉLULA 5 — FUNÇÃO LC (PIMS peso torre)
# ============================================================

def detect_lc_events(
    spark_df: pyspark.sql.DataFrame,
    timestamp_col: str,
    peso_col: str,
    start_ts: str,
    end_ts: str,
    initial_corrida: int,

    THR_UP: float = 120,
    THR_DOWN: float = 80,

    SMOOTH_WINDOW_S: int = 15,
    SLOPE_WINDOW_S: int = 30,

    MIN_POS_SLOPE: float = 10,        # start clássico
    MIN_ALT_POS_SLOPE: float = 30,    # start alternativo (subida forte)

    MIN_NEG_SLOPE: float = -10,       # confirmar corrida (começou a cair)

    MIN_DROP_CONFIRM: float = 40,     # queda acumulada mínima
    MAX_TIME_TO_DESCEND_S: int = 4 * 60,

    LOOKBACK_LOW_S: int = 5 * 60,

    LOW_WEIGHT: float = 50,
    STABLE_SECONDS: int = 60,
    END_SLOPE_EPS: float = -1,        # fim só se "não está descendo" (mais restritivo que -5)
):
    """
    Retorna Pandas:
      corrida | start_detected | end_detected
    """

    # -------- Spark prep: timestamp_sp e forward-fill
    df_lc = (
        spark_df
        .withColumn("timestamp_sp", F.col(timestamp_col))
        .withColumn("peso_raw", F.regexp_replace(F.col(peso_col), ",", ".").cast("double"))
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

    df_pd["timestamp"] = pd.to_datetime(df_pd["timestamp_sp"], errors="coerce")
    df = (
        df_pd
        .drop(columns=["timestamp_sp"])
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # -------- Smooth & slope
    w = int(SMOOTH_WINDOW_S)
    k = int(SLOPE_WINDOW_S)
    lb = int(LOOKBACK_LOW_S)

    df["peso_smooth"] = (
        df["peso"]
        .rolling(window=w, min_periods=max(3, w // 3))
        .median()
    )

    df["slope"] = df["peso_smooth"] - df["peso_smooth"].shift(k)

    # -------- low recent (histerese)
    df["was_low_recent"] = (
        (df["peso_smooth"] < THR_DOWN)
        .rolling(window=lb, min_periods=max(10, lb // 5))
        .max()
        .fillna(0)
        .astype(bool)
    )

    # -------- START candidates
    df["above_up"] = df["peso_smooth"] >= THR_UP
    df["cross_up"] = df["above_up"] & (~df["above_up"].shift(1, fill_value=False))

    df["start_classic"] = (
        df["cross_up"] &
        (df["slope"] >= MIN_POS_SLOPE) &
        (df["was_low_recent"])
    )

    df["start_alt"] = (
        (df["peso_smooth"] >= THR_UP) &
        (df["slope"] >= MIN_ALT_POS_SLOPE)
    )

    df["start_candidate"] = df["start_classic"] | df["start_alt"]

    # -------- END candidates (baixo + não descendo)
    df["is_low"] = df["peso_smooth"] <= LOW_WEIGHT
    df["end_not_descending"] = df["slope"] >= END_SLOPE_EPS
    df["end_candidate"] = df["is_low"] & df["end_not_descending"]

    # end sustentado (rolling)
    df["end_sustained"] = (
        df["end_candidate"]
        .rolling(window=STABLE_SECONDS, min_periods=STABLE_SECONDS)
        .mean()
        == 1.0
    )

    # -------- Máquina de estados
    events = []
    corrida = int(initial_corrida)

    state = "IDLE"
    candidate_start_time = None
    candidate_start_weight = None
    start_time = None

    n = len(df)

    for i in range(n):
        row = df.iloc[i]
        ts = row["timestamp"]

        if state == "IDLE":
            if bool(row["start_candidate"]):
                state = "POSSIBLE_START"
                candidate_start_time = ts
                candidate_start_weight = float(row["peso_smooth"]) if pd.notna(row["peso_smooth"]) else float(row["peso"])
            continue

        if state == "POSSIBLE_START":
            elapsed = (ts - candidate_start_time).total_seconds()
            drop = (float(row["peso_smooth"]) if pd.notna(row["peso_smooth"]) else float(row["peso"])) - candidate_start_weight

            # confirma corrida se começou a cair (ou já caiu o suficiente)
            if (pd.notna(row["slope"]) and float(row["slope"]) <= MIN_NEG_SLOPE) or (drop <= -MIN_DROP_CONFIRM):
                state = "RUNNING"
                start_time = candidate_start_time
                continue

            # timeout descarta falso positivo
            if elapsed >= MAX_TIME_TO_DESCEND_S:
                state = "IDLE"
                candidate_start_time = None
                candidate_start_weight = None
                start_time = None
                continue

        if state == "RUNNING":
            if bool(row["end_sustained"]):
                # end_time deve ser o PRIMEIRO instante do bloco sustentado (não o final da janela)
                j0 = max(0, i - STABLE_SECONDS + 1)
                end_time = df.iloc[j0]["timestamp"]

                events.append({
                    "corrida": corrida,
                    "start_detected": start_time,
                    "end_detected": end_time,
                })

                corrida += 1
                state = "IDLE"
                candidate_start_time = None
                candidate_start_weight = None
                start_time = None

    return pd.DataFrame(events, columns=["corrida", "start_detected", "end_detected"])


# COMMAND ----------

# DBTITLE 1,Classes Event e Corrida + Wrapper FEA
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import pandas as pd

# ============================================================
# CLASSES PARA SIMULAÇÃO DE FILAS
# ============================================================

@dataclass
class Event:
    """Evento temporal para simulação de filas."""
    ts: pd.Timestamp
    kind: str  # "FEA_START", "FEA_END", "FP_START", "FP_END", "VD_START", "VD_END", "LC_START", "LC_END"
    payload: dict


@dataclass
class Corrida:
    """Estado completo de uma corrida através das etapas."""
    corrida: int
    inicio_fea: Optional[pd.Timestamp] = None
    final_fea: Optional[pd.Timestamp] = None
    inicio_fp: Optional[pd.Timestamp] = None
    final_fp: Optional[pd.Timestamp] = None
    fp_carro: Optional[int] = None
    inicio_vd: Optional[pd.Timestamp] = None
    final_vd: Optional[pd.Timestamp] = None
    vd_tanque: Optional[int] = None
    inicio_lc: Optional[pd.Timestamp] = None
    final_lc: Optional[pd.Timestamp] = None


# ============================================================
# WRAPPER FEA: PROCESSA EVENTOS E RETORNA CORRIDAS
# ============================================================

def detect_fea_wrapper(
    df_fea_raw: pd.DataFrame,
    start_ts: str,
    end_ts: str,
    initial_heat: int,  # TODO: Futuramente pegar de outra função
    fea_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Wrapper para detect_fea_events_fsm_energia.
    
    Entrada:
        - df_fea_raw: DataFrame Pandas com coluna 'Time' e 'ACI@FEA_ELET_ENERGIA'
        - start_ts: timestamp inicial (string)
        - end_ts: timestamp final (string)
        - initial_heat: número inicial da corrida (ex: 135222)
                        TODO: Futuramente será obtido de outra função
        - fea_params: parâmetros opcionais para o detector FEA
    
    Saída:
        - DataFrame Pandas com colunas:
          corrida | inicio_fea_sys | final_fea_sys | close_reason
    """
    
    if fea_params is None:
        fea_params = {}
    
    # Preparar DataFrame para o detector
    df_prep = df_fea_raw.copy()
    
    # Verificar se precisa renomear Time para timestamp
    if 'Time' in df_prep.columns and 'timestamp' not in df_prep.columns:
        df_prep = df_prep.rename(columns={'Time': 'timestamp'})
    
    # Filtrar janela temporal
    df_prep['timestamp'] = pd.to_datetime(df_prep['timestamp'], errors='coerce')
    df_prep = df_prep[
        (df_prep['timestamp'] >= start_ts) & 
        (df_prep['timestamp'] <= end_ts)
    ].sort_values('timestamp').reset_index(drop=True)
    
    # Verificar se DataFrame está vazio
    if df_prep.empty:
        print("   ⚠️ Aviso: DataFrame FEA vazio após filtro temporal")
        return pd.DataFrame(columns=['corrida', 'inicio_fea_sys', 'final_fea_sys', 'close_reason'])
    
    # Chamar detector FEA
    df_results, df_events, df_debug, df_features = detect_fea_events_fsm_energia(
        df=df_prep,
        timestamp_col='timestamp',
        tag_col='ACI@FEA_ELET_ENERGIA',
        initial_corrida=initial_heat,
        **fea_params
    )
    
    # Verificar se há resultados
    if df_results.empty:
        print("   ⚠️ Aviso: Nenhum evento FEA detectado")
        return pd.DataFrame(columns=['corrida', 'inicio_fea_sys', 'final_fea_sys', 'close_reason'])
    
    # Retornar apenas os resultados (corridas fechadas)
    # Garantir que as colunas existem
    required_cols = ['corrida', 'inicio_fea_sys', 'final_fea_sys', 'close_reason']
    available_cols = [col for col in required_cols if col in df_results.columns]
    
    return df_results[available_cols].copy()


print("✓ Classes Event e Corrida definidas")
print("✓ Wrapper FEA criado: detect_fea_wrapper()")

# COMMAND ----------

# DBTITLE 1,build_events_with_fea
def build_events_with_fea(
    df_fea: pd.DataFrame,
    df_fp: pd.DataFrame,
    df_vd: pd.DataFrame,
    df_lc: pd.DataFrame,
    debug: bool = False
) -> List[Event]:
    """
    Constrói lista de eventos temporais incluindo FEA.
    
    FEA_START cria a corrida (payload contém número da corrida).
    FP, VD, LC não criam corrida (payload vazio ou só metadados).
    
    Entrada:
        - df_fea: DataFrame com colunas: corrida, inicio_fea_sys, final_fea_sys
        - df_fp: DataFrame com colunas: inicio_fp, final_fp, carro_fp (SEM corrida)
        - df_vd: DataFrame com colunas: inicio_vd, final_vd, vd_tanque (SEM corrida)
        - df_lc: DataFrame com colunas: inicio_lc, final_lc (SEM corrida)
    
    Saída:
        - Lista de Event ordenada por timestamp
    """
    events: List[Event] = []

    # =========================
    # FEA — CRIA A CORRIDA
    # =========================
    for _, r in df_fea.iterrows():
        c = int(r["corrida"])

        if pd.notna(r.get("inicio_fea_sys")):
            events.append(Event(
                ts=pd.to_datetime(r["inicio_fea_sys"]),
                kind="FEA_START",
                payload={"corrida": c}
            ))

        if pd.notna(r.get("final_fea_sys")):
            events.append(Event(
                ts=pd.to_datetime(r["final_fea_sys"]),
                kind="FEA_END",
                payload={"corrida": c}
            ))

    # =========================
    # FP — NÃO CRIA CORRIDA
    # =========================
    for _, r in df_fp.iterrows():
        carro = r.get("carro_fp")
        
        if pd.notna(r.get("inicio_fp")):
            events.append(Event(
                ts=pd.to_datetime(r["inicio_fp"]),
                kind="FP_START",
                payload={"carro": carro} if pd.notna(carro) else {}
            ))

        if pd.notna(r.get("final_fp")):
            events.append(Event(
                ts=pd.to_datetime(r["final_fp"]),
                kind="FP_END",
                payload={"carro": carro} if pd.notna(carro) else {}
            ))

    # =========================
    # VD — NÃO CRIA CORRIDA
    # =========================
    for _, r in df_vd.iterrows():
        tanque = r.get("vd_tanque")

        if pd.notna(r.get("inicio_vd")):
            events.append(Event(
                ts=pd.to_datetime(r["inicio_vd"]),
                kind="VD_START",
                payload={"tanque": tanque} if pd.notna(tanque) else {}
            ))

        if pd.notna(r.get("final_vd")):
            events.append(Event(
                ts=pd.to_datetime(r["final_vd"]),
                kind="VD_END",
                payload={"tanque": tanque} if pd.notna(tanque) else {}
            ))

    # =========================
    # LC — NÃO CRIA CORRIDA
    # =========================
    for _, r in df_lc.iterrows():
        if pd.notna(r.get("inicio_lc")):
            events.append(Event(
                ts=pd.to_datetime(r["inicio_lc"]),
                kind="LC_START",
                payload={}
            ))

        if pd.notna(r.get("final_lc")):
            events.append(Event(
                ts=pd.to_datetime(r["final_lc"]),
                kind="LC_END",
                payload={}
            ))

    # Ordenar eventos por timestamp
    events.sort(key=lambda e: (e.ts, e.kind))

    if debug:
        print(f"[INIT] {len(events)} eventos carregados")
        print(f"  - FEA: {sum(1 for e in events if 'FEA' in e.kind)}")
        print(f"  - FP:  {sum(1 for e in events if 'FP' in e.kind)}")
        print(f"  - VD:  {sum(1 for e in events if 'VD' in e.kind)}")
        print(f"  - LC:  {sum(1 for e in events if 'LC' in e.kind)}")

    return events


print("✓ Função build_events_with_fea() criada")

# COMMAND ----------

# DBTITLE 1,simulate_corridas_with_fea
def simulate_corridas_with_fea(
    df_fea: pd.DataFrame,
    df_fp: pd.DataFrame,
    df_vd: pd.DataFrame,
    df_lc: pd.DataFrame,
    delay_fea_fp_minutes: int = 3,
    delay_fp_vd_minutes: int = 5,
    delay_vd_lc_minutes: int = 2,
    debug: bool = True
) -> pd.DataFrame:
    """
    Simula filas causais: FEA → FP → VD → LC.
    
    Lógica:
        1. FEA_START cria corrida
        2. FEA_END entra na fila_fp (após delay)
        3. FP_START consome fila_fp e herda corrida
        4. FP_END entra na fila_vd (após delay)
        5. VD_START consome fila_vd e herda corrida
        6. VD_END entra na fila_lc (após delay)
        7. LC_START consome fila_lc e herda corrida
        8. LC_END finaliza corrida
    
    Entrada:
        - df_fea: eventos FEA com corridas
        - df_fp: eventos FP sem corridas (só timestamps e carro)
        - df_vd: eventos VD sem corridas (só timestamps e tanque)
        - df_lc: eventos LC sem corridas (só timestamps)
        - delays em minutos
    
    Saída:
        - DataFrame Pandas wide com todas as etapas por corrida
    """
    
    # Construir eventos
    events = build_events_with_fea(df_fea, df_fp, df_vd, df_lc, debug=debug)
    
    # Estruturas de controle
    corridas: Dict[int, Corrida] = {}
    
    fila_fp: List[int] = []  # corridas aguardando FP
    fila_vd: List[int] = []  # corridas aguardando VD
    fila_lc: List[int] = []  # corridas aguardando LC
    
    fp_busy = {1: None, 2: None}  # qual corrida está em cada carro
    vd_busy = {1: None, 2: None}  # qual corrida está em cada tanque
    lc_busy: Optional[int] = None  # qual corrida está no LC
    
    # Anchor times (liberação de filas)
    fp_anchor_time: Optional[pd.Timestamp] = None
    vd_anchor_time: Optional[pd.Timestamp] = None
    lc_anchor_time: Optional[pd.Timestamp] = None
    
    # Delays
    DELAY_FEA_FP = pd.Timedelta(minutes=delay_fea_fp_minutes)
    DELAY_FP_VD = pd.Timedelta(minutes=delay_fp_vd_minutes)
    DELAY_VD_LC = pd.Timedelta(minutes=delay_vd_lc_minutes)
    
    def get_corrida(c: int) -> Corrida:
        if c not in corridas:
            corridas[c] = Corrida(corrida=c)
        return corridas[c]
    
    # Processar eventos
    for ev in events:
        ts, kind = ev.ts, ev.kind
        
        # ===========================
        # FEA_START: CRIA CORRIDA
        # ===========================
        if kind == "FEA_START":
            c = ev.payload["corrida"]
            get_corrida(c).inicio_fea = ts
            
            if debug:
                print(f"[{ts}] FEA_START → corrida {c} criada")
            continue
        
        # ===========================
        # FEA_END: ENTRA NA FILA FP
        # ===========================
        if kind == "FEA_END":
            c = ev.payload["corrida"]
            get_corrida(c).final_fea = ts
            fila_fp.append(c)
            
            # Libera FP após delay
            if fp_anchor_time is None:
                fp_anchor_time = ts + DELAY_FEA_FP
                if debug:
                    print(f"[{ts}] FEA_END corrida {c} → FP libera após {fp_anchor_time}")
            
            if debug:
                print(f"[{ts}] FEA_END → fila_FP {fila_fp}")
            continue
        
        # ===========================
        # FP_START: CONSOME FILA FP
        # ===========================
        if kind == "FP_START":
            # Verifica se FP já está liberado
            if fp_anchor_time is None or ts < fp_anchor_time:
                continue
            
            carro = ev.payload.get("carro")
            if carro is None:
                continue
            
            # Verifica se carro está livre e há corrida na fila
            if fp_busy.get(carro) is not None or not fila_fp:
                continue
            
            # Consome fila e herda corrida
            c = fila_fp.pop(0)
            fp_busy[carro] = c
            get_corrida(c).inicio_fp = ts
            get_corrida(c).fp_carro = carro
            
            if debug:
                print(f"[{ts}] FP_START(carro{carro}) → corrida {c}")
            continue
        
        # ===========================
        # FP_END: ENTRA NA FILA VD
        # ===========================
        if kind == "FP_END":
            carro = ev.payload.get("carro")
            if carro is None:
                continue
            
            c = fp_busy.get(carro)
            if c is None:
                continue
            
            get_corrida(c).final_fp = ts
            fila_vd.append(c)
            fp_busy[carro] = None
            
            # Libera VD após delay
            if vd_anchor_time is None:
                vd_anchor_time = ts + DELAY_FP_VD
                if debug:
                    print(f"[{ts}] FP_END corrida {c} → VD libera após {vd_anchor_time}")
            
            if debug:
                print(f"[{ts}] FP_END(carro{carro}) → fila_VD {fila_vd}")
            continue
        
        # ===========================
        # VD_START: CONSOME FILA VD
        # ===========================
        if kind == "VD_START":
            # Verifica se VD já está liberado
            if vd_anchor_time is None or ts < vd_anchor_time:
                continue
            
            tanque = ev.payload.get("tanque")
            if tanque is None:
                continue
            
            # Verifica se tanque está livre e há corrida na fila
            if vd_busy.get(tanque) is not None or not fila_vd:
                continue
            
            # Consome fila e herda corrida
            c = fila_vd.pop(0)
            vd_busy[tanque] = c
            get_corrida(c).inicio_vd = ts
            get_corrida(c).vd_tanque = tanque
            
            if debug:
                print(f"[{ts}] VD_START(tanque{tanque}) → corrida {c}")
            continue
        
        # ===========================
        # VD_END: ENTRA NA FILA LC
        # ===========================
        if kind == "VD_END":
            tanque = ev.payload.get("tanque")
            if tanque is None:
                continue
            
            c = vd_busy.get(tanque)
            if c is None:
                continue
            
            get_corrida(c).final_vd = ts
            fila_lc.append(c)
            vd_busy[tanque] = None
            
            # Libera LC após delay
            if lc_anchor_time is None:
                lc_anchor_time = ts + DELAY_VD_LC
                if debug:
                    print(f"[{ts}] VD_END corrida {c} → LC libera após {lc_anchor_time}")
            
            if debug:
                print(f"[{ts}] VD_END(tanque{tanque}) → fila_LC {fila_lc}")
            continue
        
        # ===========================
        # LC_START: CONSOME FILA LC
        # ===========================
        if kind == "LC_START":
            # Verifica se LC já está liberado
            if lc_anchor_time is None or ts < lc_anchor_time:
                continue
            
            # Verifica se LC está livre e há corrida na fila
            if lc_busy is not None or not fila_lc:
                continue
            
            # Consome fila e herda corrida
            c = fila_lc.pop(0)
            lc_busy = c
            get_corrida(c).inicio_lc = ts
            
            if debug:
                print(f"[{ts}] LC_START → corrida {c}")
            continue
        
        # ===========================
        # LC_END: FINALIZA CORRIDA
        # ===========================
        if kind == "LC_END":
            if lc_busy is None:
                continue
            
            get_corrida(lc_busy).final_lc = ts
            
            if debug:
                print(f"[{ts}] LC_END → corrida {lc_busy} finalizada")
            
            lc_busy = None
            continue
    
    # Converter para DataFrame
    out = pd.DataFrame([vars(c) for c in corridas.values()]).sort_values("corrida")
    
    # Status atual
    def status(r):
        if pd.notna(r.inicio_lc) and pd.isna(r.final_lc):
            return "LC"
        if pd.notna(r.inicio_vd) and pd.isna(r.final_vd):
            return "VD"
        if pd.notna(r.inicio_fp) and pd.isna(r.final_fp):
            return "FP"
        if pd.notna(r.inicio_fea) and pd.isna(r.final_fea):
            return "FEA"
        return "FINALIZADA"
    
    out["status_atual"] = out.apply(status, axis=1)
    
    return out.reset_index(drop=True)


print("✓ Função simulate_corridas_with_fea() criada")

# COMMAND ----------

# DBTITLE 1,run_unified_detector
def run_unified_detector(
    df_fea_raw: pd.DataFrame,
    spark_fp_raw: pyspark.sql.DataFrame,
    spark_vd_raw: pyspark.sql.DataFrame,
    spark_lc_raw: pyspark.sql.DataFrame,
    
    start_ts: str,
    end_ts: str,
    initial_heat: int,  # TODO: Futuramente pegar de outra função
    
    fea_params: Optional[Dict[str, Any]] = None,
    fp_params: Optional[Dict[str, Any]] = None,
    vd_params: Optional[Dict[str, Any]] = None,
    lc_params: Optional[Dict[str, Any]] = None,
    
    delay_fea_fp_minutes: int = 3,
    delay_fp_vd_minutes: int = 5,
    delay_vd_lc_minutes: int = 2,
    
    debug: bool = True,
) -> pd.DataFrame:
    """
    Função unificada que orquestra detecção de todas as etapas + simulação de filas.
    
    Fluxo:
        1. Detecta eventos FEA (Pandas) → cria corridas sequenciais
        2. Detecta eventos FP (Spark) → SEM corrida, só timestamps
        3. Detecta eventos VD (Spark) → SEM corrida, só timestamps
        4. Detecta eventos LC (Spark) → SEM corrida, só timestamps
        5. Simula filas: FEA → FP → VD → LC (corridas herdadas)
    
    Entrada:
        - df_fea_raw: DataFrame Pandas com dados brutos FEA
        - spark_*_raw: DataFrames Spark com dados brutos FP/VD/LC
        - start_ts, end_ts: janela temporal
        - initial_heat: número inicial da corrida (ex: 135222)
        - *_params: parâmetros para cada detector
        - delays: delays entre etapas (minutos)
    
    Saída:
        - DataFrame Pandas wide com todas as etapas por corrida
    """
    
    if fea_params is None:
        fea_params = {}
    if fp_params is None:
        fp_params = {}
    if vd_params is None:
        vd_params = {}
    if lc_params is None:
        lc_params = {}
    
    if debug:
        print("="*80)
        print("▶️ Rodando detectores unificados...")
        print(f"   Janela: {start_ts} → {end_ts}")
        print(f"   Corrida inicial: {initial_heat}")
        print("="*80)
    
    # ========================================
    # 1. DETECTAR FEA (Pandas) → CRIA CORRIDAS
    # ========================================
    if debug:
        print("\n[1/4] Detectando FEA...")
    
    df_fea = detect_fea_wrapper(
        df_fea_raw=df_fea_raw,
        start_ts=start_ts,
        end_ts=end_ts,
        initial_heat=initial_heat,
        fea_params=fea_params,
    )
    
    if debug:
        print(f"   ✓ {len(df_fea)} eventos FEA detectados")
        if not df_fea.empty:
            print(f"   Corridas: {df_fea['corrida'].min()} → {df_fea['corrida'].max()}")
    
    # ========================================
    # 2. DETECTAR FP (Spark) → SEM CORRIDA
    # ========================================
    if debug:
        print("\n[2/4] Detectando FP...")
    
    # IMPORTANTE: NÃO mexer na função detect_fp_events_gas_based_streaming
    # Ela retorna eventos SEM número de corrida (só timestamps e carro)
    # ATUALIZAÇÃO: Agora retorna 3 DataFrames (final, events, heat)
    df_fp_final_spark, df_fp_events_spark, df_heat_spark = detect_fp_events_gas_based_streaming(
        spark_df=spark_fp_raw,
        initial_heat_number=0,  # dummy, não usado (FP não cria corrida)
        initial_timestamp=start_ts,
        last_timestamp=end_ts,
        **fp_params
    )
    
    # Converter para Pandas e renomear colunas
    df_fp = df_fp_final_spark.toPandas()
    if not df_fp.empty:
        df_fp = df_fp.rename(columns={'fp_start': 'inicio_fp', 'fp_end': 'final_fp', 'carro_fp': 'carro_fp'})
        df_fp = to_pandas_ts(df_fp, ['inicio_fp', 'final_fp'])
    
    if debug:
        print(f"   ✓ {len(df_fp)} eventos FP detectados")
    
    # ========================================
    # 3. DETECTAR VD (Spark) → SEM CORRIDA
    # ========================================
    if debug:
        print("\n[3/4] Detectando VD...")
    
    df_vd_spark = detect_vd_events_gas_based(
        spark_df_vd_pims=spark_vd_raw,
        start_ts=start_ts,
        end_ts=end_ts,
        initial_heat_number=0,  # dummy, não usado (VD não cria corrida)
        **vd_params
    )
    
    df_vd = df_vd_spark.toPandas()
    if not df_vd.empty:
        df_vd = to_pandas_ts(df_vd, ['inicio_vd', 'final_vd'])
        # Remover coluna 'corrida' se existir (não é usada)
        if 'corrida' in df_vd.columns:
            df_vd = df_vd.drop(columns=['corrida'])
    
    if debug:
        print(f"   ✓ {len(df_vd)} eventos VD detectados")
    
    # ========================================
    # 4. DETECTAR LC (Spark) → SEM CORRIDA
    # ========================================
    if debug:
        print("\n[4/4] Detectando LC...")
    
    df_lc = detect_lc_events(
        spark_df=spark_lc_raw,
        timestamp_col="timestamp",
        peso_col="ACI@LC_TORRE_PESOREAL",
        start_ts=start_ts,
        end_ts=end_ts,
        initial_corrida=0,  # dummy, não usado (LC não cria corrida)
        **lc_params
    )
    
    if not df_lc.empty:
        df_lc = to_pandas_ts(df_lc, ['start_detected', 'end_detected'])
        df_lc = df_lc.rename(columns={'start_detected': 'inicio_lc', 'end_detected': 'final_lc'})
        # Remover coluna 'corrida' se existir
        if 'corrida' in df_lc.columns:
            df_lc = df_lc.drop(columns=['corrida'])
    
    if debug:
        print(f"   ✓ {len(df_lc)} eventos LC detectados")
    
    # ========================================
    # 5. SIMULAR FILAS: FEA → FP → VD → LC
    # ========================================
    if debug:
        print("\n" + "="*80)
        print("▶️ Iniciando simulação de filas...")
        print(f"   Delays: FEA→FP={delay_fea_fp_minutes}min | FP→VD={delay_fp_vd_minutes}min | VD→LC={delay_vd_lc_minutes}min")
        print("="*80)
    
    df_result = simulate_corridas_with_fea(
        df_fea=df_fea,
        df_fp=df_fp,
        df_vd=df_vd,
        df_lc=df_lc,
        delay_fea_fp_minutes=delay_fea_fp_minutes,
        delay_fp_vd_minutes=delay_fp_vd_minutes,
        delay_vd_lc_minutes=delay_vd_lc_minutes,
        debug=debug,
    )
    
    if debug:
        print("\n" + "="*80)
        print("✓ Detecção unificada concluída!")
        print(f"   Total de corridas: {len(df_result)}")
        if not df_result.empty:
            print(f"   Status: {df_result['status_atual'].value_counts().to_dict()}")
        print("="*80)
    
    return df_result


print("✓ Função run_unified_detector() criada")

# COMMAND ----------

# DBTITLE 1,PARAMETROS
# ============================================================
# PARÂMETROS CENTRALIZADOS - DETECTOR UNIFICADO
# ============================================================

# Período temporal (ajustado para dados FEA disponíveis: nov 2025)
START_TS = "2025-11-15 00:00:00"
END_TS   = "2025-11-15 23:59:59"

# Número inicial da corrida
# TODO: Futuramente será obtido de outra função
INITIAL_HEAT = 130183

# Parâmetros legados (manter para compatibilidade)
INITIAL_HEAT_FP = 130475
INITIAL_HEAT_VD = 130474
INITIAL_HEAT_LC = 130473

# ============================================================
# PARÂMETROS FEA
# ============================================================
fea_params = dict(
    SMOOTH_WINDOW_S=5,
    ZERO_MAX=1000,
    ZERO_HOLD_S=1,
    START_TS_MODE="prev_zero",
    START_MIN=0,
    HIGH_MIN=6000.0,
    END_MAX=200.0,
    DROP_WINDOW_S=5,
    DROP_DELTA=3000.0,
    END_TS_MODE="prev",
    DEBUG=False,
)

# ============================================================
# PARÂMETROS FP (detect_fp_events_gas_based_streaming)
# ATUALIZADO: Nova versão só aceita 4 parâmetros
# ============================================================
fp_params_streaming = dict(
    min_peak_gas=100.0,              # pico mínimo de gás para validar corrida
    min_peak_window_s=300,           # janela para validar pico (5 min)
    energy_delta_eps=1.5,            # delta mínimo de energia para HEAT_ON
    reopen_merge_threshold=100.0,    # threshold para reabrir corrida em END_PENDING
)

# Parâmetros FP legados (função antiga - manter para compatibilidade)
fp_params = dict(
    gas_threshold=100.0,
    min_peak_total_gas=3000.0,
    minimal_duration_seconds=60,
    end_fraction_of_peak=0.90,
    min_start_rel_increase=0.05,
    debug=False,
)

# ============================================================
# PARÂMETROS VD
# ============================================================
vd_params = dict(
    reset_threshold=0.99,
    minimal_duration_seconds=10,
    debug=False,
)

# ============================================================
# PARÂMETROS LC
# ============================================================
lc_params = dict(
    THR_UP=120,
    THR_DOWN=80,
    SMOOTH_WINDOW_S=15,
    SLOPE_WINDOW_S=30,
    MIN_POS_SLOPE=10,
    MIN_ALT_POS_SLOPE=30,
    MIN_NEG_SLOPE=-5,
    MIN_DROP_CONFIRM=40,
    MAX_TIME_TO_DESCEND_S=6*60,
    LOOKBACK_LOW_S=5*60,
    LOW_WEIGHT=50,
    STABLE_SECONDS=60,
    END_SLOPE_EPS=-1, 
)

# ============================================================
# DELAYS ENTRE ETAPAS (minutos)
# ============================================================
DELAY_FEA_FP = 3
DELAY_FP_VD = 5
DELAY_VD_LC = 2

print("✓ Parâmetros carregados:")
print(f"  - Período: {START_TS} → {END_TS}")
print(f"  - Corrida inicial: {INITIAL_HEAT}")
print(f"  - Delays: FEA→FP={DELAY_FEA_FP}min | FP→VD={DELAY_FP_VD}min | VD→LC={DELAY_VD_LC}min")

# COMMAND ----------

def build_status_corrida_table(
    spark_fp_raw: pyspark.sql.DataFrame,
    spark_vd_raw: pyspark.sql.DataFrame,
    spark_lc_raw: pyspark.sql.DataFrame,

    start_ts: str,
    end_ts: str,

    initial_heat_fp: int,
    initial_heat_vd: int,
    initial_heat_lc: int,

    fp_params: Dict[str, Any],
    vd_params: Dict[str, Any],
    lc_params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Retorna DataFrame PANDAS wide (1 linha por corrida):
      corrida |
      inicio_fea | final_fea |
      inicio_fp | final_fp | fp_carro |
      inicio_vd | final_vd | vd_tanque |
      inicio_lc | final_lc |
      status_atual
    """

    # -----------------------
    # FP
    # -----------------------
    df_fp = (
        detect_fp_events_gas_based(
            spark_df=spark_fp_raw,
            start_ts=start_ts,
            end_ts=end_ts,
            initial_heat_number=initial_heat_fp,
            **fp_params
        )
        .toPandas()
    )
    df_fp = to_pandas_ts(df_fp, ["inicio_fp", "final_fp"])

    # -----------------------
    # VD
    # -----------------------
    df_vd = (
        detect_vd_events_gas_based(
            spark_df_vd_pims=spark_vd_raw,
            start_ts=start_ts,
            end_ts=end_ts,
            initial_heat_number=initial_heat_vd,
            **vd_params
        )
        .toPandas()
    )
    df_vd = to_pandas_ts(df_vd, ["inicio_vd", "final_vd"])

    # -----------------------
    # LC
    # -----------------------
    df_lc = detect_lc_events(
        spark_df=spark_lc_raw,
        timestamp_col="timestamp",
        peso_col="ACI@LC_TORRE_PESOREAL",
        start_ts=start_ts,
        end_ts=end_ts,
        initial_corrida=initial_heat_lc,
        **lc_params
    )
    df_lc = to_pandas_ts(df_lc, ["start_detected", "end_detected"])
    df_lc = df_lc.rename(
        columns={
            "start_detected": "inicio_lc",
            "end_detected": "final_lc",
        }
    )

    # -----------------------
    # Base de corridas (união)
    # -----------------------
    corridas = sorted(set(
        (df_fp["corrida"].dropna().astype(int).tolist() if not df_fp.empty else []) +
        (df_vd["corrida"].dropna().astype(int).tolist() if not df_vd.empty else []) +
        (df_lc["corrida"].dropna().astype(int).tolist() if not df_lc.empty else [])
    ))

    out = pd.DataFrame({"corrida": corridas}).astype({"corrida": "int64"})

    # -----------------------
    # merges wide
    # -----------------------
    if not df_fp.empty:
        out = out.merge(df_fp, on="corrida", how="left")
    else:
        out["inicio_fp"] = pd.NaT
        out["final_fp"] = pd.NaT
        out["fp_carro"] = pd.NA

    if not df_vd.empty:
        out = out.merge(df_vd, on="corrida", how="left")
    else:
        out["inicio_vd"] = pd.NaT
        out["final_vd"] = pd.NaT
        out["vd_tanque"] = pd.NA

    if not df_lc.empty:
        out = out.merge(df_lc, on="corrida", how="left")
    else:
        out["inicio_lc"] = pd.NaT
        out["final_lc"] = pd.NaT

    # -----------------------
    # FEA placeholder
    # -----------------------
    out["inicio_fea"] = pd.NaT
    out["final_fea"] = pd.NaT

    # -----------------------
    # status atual
    # prioridade: LC > VD > FP
    # -----------------------
    def _status(row):
        if pd.notna(row.get("inicio_lc")) and pd.isna(row.get("final_lc")):
            return "LC"
        if pd.notna(row.get("inicio_vd")) and pd.isna(row.get("final_vd")):
            return "VD"
        if pd.notna(row.get("inicio_fp")) and pd.isna(row.get("final_fp")):
            return "FP"
        return "FINALIZADA"

    out["status_atual"] = out.apply(_status, axis=1)

    # -----------------------
    # ordenação final
    # -----------------------
    out = out[[
        "corrida",
        "inicio_fea", "final_fea",
        "inicio_fp", "final_fp", "fp_carro",
        "inicio_vd", "final_vd", "vd_tanque",
        "inicio_lc", "final_lc",
        "status_atual",
    ]].sort_values("corrida").reset_index(drop=True)

    return out


# COMMAND ----------

# DBTITLE 1,READ TABLES
# ============================================================
# CÉLULA 7 — LEITURA DAS TABELAS (PIMS RAW)
# ============================================================
# Read the saved table from Unity Catalog
df_fea_raw  = spark.table("industrial_iatemperaturascha_refined_dev.computed_features.hist_tags_IBA_novembro")

df_fp_raw = spark.table("industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_fp") \
    .withColumn("timestamp", F.col("timestamp") - F.expr("INTERVAL 3 HOURS"))
df_vd_raw = spark.table("industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_vd") \
    .withColumn("timestamp", F.col("timestamp") - F.expr("INTERVAL 3 HOURS"))
df_lc_raw = spark.table("industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_lc") \
    .withColumn("timestamp", F.col("timestamp") - F.expr("INTERVAL 3 HOURS"))


# COMMAND ----------

# DBTITLE 1,Converter FEA para Pandas e processar
# Convert Spark DataFrame to Pandas
print("Converting Spark DataFrame to Pandas...")
print("="*80)

# Primeiro converter para Pandas
df_fea_raw = df_fea_raw.toPandas()

print(f"✓ Converted to Pandas DataFrame with {len(df_fea_raw)} rows")

# Parse Time column to datetime
print("\nParsing Time column to datetime...")
df_fea_raw['Time'] = pd.to_datetime(df_fea_raw['Time'], format='%d.%m.%Y %H:%M:%S.%f', errors='coerce')

# Convert all other columns to numeric (replacing comma with dot for decimal)
print("Converting numeric columns...")
for col in df_fea_raw.columns:
    if col != 'Time':
        try:
            df_fea_raw[col] = pd.to_numeric(df_fea_raw[col].astype(str).str.replace(',', '.'), errors='coerce')
        except:
            pass

print(f"\n✓ Conversion completed!")
print(f"\nData types after conversion:")
print(df_fea_raw.dtypes)
print(f"\nDataFrame shape: {df_fea_raw.shape}")
print(f"\nFirst few rows:")
display(df_fea_raw.head())

# COMMAND ----------

# DBTITLE 1,OPERADOR
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

df_final = df_final.filter(
    F.col("inicio_fp").between(START_TS, END_TS) |
    F.col("inicio_vd").between(START_TS, END_TS) |
    F.col("inicio_lc").between(START_TS, END_TS) |
    F.col("inicio_fea").between(START_TS, END_TS)
)

display(df_final.orderBy(F.desc("corrida")))

# COMMAND ----------

# DBTITLE 1,Executar detector unificado FEA+FP+VD+LC
# ============================================================
# EXECUÇÃO DO DETECTOR UNIFICADO FEA + FP + VD + LC
# ============================================================
# Todos os parâmetros estão centralizados na célula 11 (PARAMETROS)
# ============================================================

print("\n" + "="*80)
print("🚀 INICIANDO DETECTOR UNIFICADO FEA + FP + VD + LC")
print("="*80)
print(f"Período: {START_TS} → {END_TS}")
print(f"Corrida inicial: {INITIAL_HEAT}")
print(f"Delays: FEA→FP={DELAY_FEA_FP}min | FP→VD={DELAY_FP_VD}min | VD→LC={DELAY_VD_LC}min")
print("="*80)

df_unified = run_unified_detector(
    df_fea_raw=df_fea_raw,  # Pandas DataFrame (processado na célula 14)
    spark_fp_raw=df_fp_raw,  # Spark DataFrame
    spark_vd_raw=df_vd_raw,  # Spark DataFrame
    spark_lc_raw=df_lc_raw,  # Spark DataFrame
    
    start_ts=START_TS,
    end_ts=END_TS,
    initial_heat=INITIAL_HEAT,
    
    fea_params=fea_params,
    fp_params=fp_params_streaming,  # Usar parâmetros streaming
    vd_params=vd_params,
    lc_params=lc_params,
    
    delay_fea_fp_minutes=DELAY_FEA_FP,
    delay_fp_vd_minutes=DELAY_FP_VD,
    delay_vd_lc_minutes=DELAY_VD_LC,
    
    debug=True,
)

print("\n" + "="*80)
print("✅ DETECTOR UNIFICADO CONCLUÍDO!")
print("="*80)

display(df_unified)

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F


def comparacao_operador(
    df_sys_pd: pd.DataFrame,
    df_op_sp,
    spark,
):
    """
    Compara tempos do SISTEMA (pandas) vs OPERADOR (spark),
    retornando um DataFrame Spark com colunas _op, _sys e diffs (minutos),
    para todas as etapas existentes no SYS (FEA, FP, VD, LC).
    """

    # ======================================================
    # 1) SYS (pandas) -> Spark
    # ======================================================
    df_sys = df_sys_pd.copy()

    df_sys["corrida"] = (
        pd.to_numeric(df_sys["corrida"], errors="coerce")
        .astype("Int64")
    )

    # Incluir FEA nas colunas de timestamp
    ts_cols = [
        "inicio_fea", "final_fea",
        "inicio_fp", "final_fp",
        "inicio_vd", "final_vd",
        "inicio_lc", "final_lc",
    ]

    for c in ts_cols:
        if c in df_sys.columns:
            df_sys[c] = pd.to_datetime(df_sys[c], errors="coerce")
            try:
                df_sys[c] = df_sys[c].dt.tz_localize(None)
            except Exception:
                pass

    df_sys_sp = spark.createDataFrame(df_sys)

    # ======================================================
    # 2) JOIN OP x SYS
    # ======================================================
    df_join = (
        df_op_sp.alias("op")
        .join(df_sys_sp.alias("sys"), on="corrida", how="left")
    )

    # ======================================================
    # 3) Função diff (minutos)
    # ======================================================
    def diff_min(sys_col, op_col):
        return F.when(
            F.col(sys_col).isNull() | F.col(op_col).isNull(),
            F.lit(None).cast("double")
        ).otherwise(
            (F.col(sys_col).cast("long") - F.col(op_col).cast("long")) / 60.0
        )

    # ======================================================
    # 4) Construção dinâmica das colunas (incluindo FEA)
    # ======================================================
    select_cols = [F.col("corrida")]

    # Incluir FEA na lista de etapas
    etapas = ["fea", "fp", "vd", "lc"]

    for etapa in etapas:
        ini = f"inicio_{etapa}"
        fim = f"final_{etapa}"

        if ini in df_sys.columns:
            select_cols.extend([
                F.col(f"op.{ini}").alias(f"{ini}_op"),
                F.col(f"sys.{ini}").alias(f"{ini}_sys"),
                diff_min(f"sys.{ini}", f"op.{ini}").alias(f"diff_{ini}"),
            ])

        if fim in df_sys.columns:
            select_cols.extend([
                F.col(f"op.{fim}").alias(f"{fim}_op"),
                F.col(f"sys.{fim}").alias(f"{fim}_sys"),
                diff_min(f"sys.{fim}", f"op.{fim}").alias(f"diff_{fim}"),
            ])

    df_compare = df_join.select(*select_cols)

    return df_compare


print("✓ Função comparacao_operador() atualizada (inclui FEA)")

# COMMAND ----------

# DBTITLE 1,Comparar resultados unificados vs operador
# ============================================================
# COMPARAÇÃO: DETECTOR UNIFICADO vs OPERADOR
# ============================================================

print("\n" + "="*80)
print("🔍 COMPARANDO RESULTADOS: SISTEMA vs OPERADOR")
print("="*80)

df_compare_unified = comparacao_operador(
    df_sys_pd=df_unified,
    df_op_sp=df_final,
    spark=spark
)

print("\n✓ Comparação concluída!")
print("\nResultados ordenados por corrida (decrescente):\n")

display(df_compare_unified.orderBy(F.desc("corrida")))

# COMMAND ----------

# DBTITLE 1,Função de visualização para debug de detecção
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def plot_detection_debug(
    df_raw: pd.DataFrame,
    df_compare: pd.DataFrame,
    etapa: str,
    col_signals: list,  # Lista de colunas para plotar
    start_ts: str,
    window_hours: int = 6,
    figsize: tuple = (22, 8),
    title: str = None,
):
    """
    Plota série temporal com detecções do operador vs sistema para debug.
    SIMPLES: apenas plota as linhas sem pré-processamento.
    """
    
    # Converter comparação para Pandas
    df_comp_pd = df_compare.toPandas()
    
    # Preparar dados brutos
    df_plot = df_raw.copy()
    
    # Garantir timestamp
    if 'timestamp' not in df_plot.columns and 'Time' in df_plot.columns:
        df_plot['timestamp'] = pd.to_datetime(df_plot['Time'], errors='coerce')
    else:
        df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'], errors='coerce')
    
    # Filtrar janela temporal
    start_dt = pd.to_datetime(start_ts)
    end_dt = start_dt + timedelta(hours=window_hours)
    
    df_plot = df_plot[
        (df_plot['timestamp'] >= start_dt) & 
        (df_plot['timestamp'] <= end_dt)
    ].sort_values('timestamp').reset_index(drop=True)
    
    if df_plot.empty:
        print(f"⚠️ Nenhum dado no período {start_dt} → {end_dt}")
        return
    
    # Converter timestamps na comparação
    ini_col_op = f"inicio_{etapa}_op"
    fim_col_op = f"final_{etapa}_op"
    ini_col_sys = f"inicio_{etapa}_sys"
    fim_col_sys = f"final_{etapa}_sys"
    
    for col in [ini_col_op, fim_col_op, ini_col_sys, fim_col_sys]:
        if col in df_comp_pd.columns:
            df_comp_pd[col] = pd.to_datetime(df_comp_pd[col], errors='coerce')
    
    # Filtrar corridas na janela
    mask = (
        (df_comp_pd[ini_col_op].notna() & (df_comp_pd[ini_col_op] >= start_dt) & (df_comp_pd[ini_col_op] <= end_dt)) |
        (df_comp_pd[fim_col_op].notna() & (df_comp_pd[fim_col_op] >= start_dt) & (df_comp_pd[fim_col_op] <= end_dt)) |
        (df_comp_pd[ini_col_sys].notna() & (df_comp_pd[ini_col_sys] >= start_dt) & (df_comp_pd[ini_col_sys] <= end_dt)) |
        (df_comp_pd[fim_col_sys].notna() & (df_comp_pd[fim_col_sys] >= start_dt) & (df_comp_pd[fim_col_sys] <= end_dt))
    )
    
    df_corridas = df_comp_pd[mask].copy()
    
    # Criar figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plotar múltiplas colunas de sinal
    for col_signal in col_signals:
        if col_signal in df_plot.columns:
            # Converter para numérico
            df_plot[col_signal] = pd.to_numeric(df_plot[col_signal], errors='coerce')
            ax.plot(df_plot['timestamp'], df_plot[col_signal], 
                   linewidth=1.5, alpha=0.7, label=col_signal)
    
    # Plotar eventos
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(df_corridas))))
    
    for idx, (_, row) in enumerate(df_corridas.iterrows()):
        corrida = int(row['corrida'])
        color = colors[idx % len(colors)]
        
        # OPERADOR (linhas sólidas)
        if pd.notna(row.get(ini_col_op)):
            ax.axvline(row[ini_col_op], color=color, linestyle='-', linewidth=2, alpha=0.8)
            ax.text(row[ini_col_op], ax.get_ylim()[1] * 0.95, f"{corrida}", 
                   rotation=90, verticalalignment='top', fontsize=9, color=color, fontweight='bold')
        
        if pd.notna(row.get(fim_col_op)):
            ax.axvline(row[fim_col_op], color=color, linestyle='-', linewidth=2, alpha=0.8)
        
        # SISTEMA (linhas pontilhadas)
        if pd.notna(row.get(ini_col_sys)):
            ax.axvline(row[ini_col_sys], color=color, linestyle='--', linewidth=2, alpha=0.6)
            ax.text(row[ini_col_sys], ax.get_ylim()[1] * 0.85, f"{corrida}*", 
                   rotation=90, verticalalignment='top', fontsize=8, color=color, style='italic')
        
        if pd.notna(row.get(fim_col_sys)):
            ax.axvline(row[fim_col_sys], color=color, linestyle='--', linewidth=2, alpha=0.6)
    
    # Formatação
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(rotation=45)
    
    ax.set_xlabel('Tempo (HH:MM)', fontsize=12)
    ax.set_ylabel('Valor', fontsize=12)
    
    if title is None:
        title = f"Debug {etapa.upper()} - {start_dt.strftime('%Y-%m-%d %H:%M')} ({window_hours}h)"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Estatísticas
    print(f"\n📊 Estatísticas:")
    print(f"  - Período: {start_dt} → {end_dt}")
    print(f"  - Amostras: {len(df_plot)}")
    print(f"  - Corridas: {len(df_corridas)}")
    if not df_corridas.empty:
        print(f"  - IDs: {df_corridas['corrida'].tolist()}")

print("✓ Função plot_detection_debug() criada")

# COMMAND ----------

# DBTITLE 1,Debug FEA - Visualização
# ============================================================
# DEBUG FEA - DETECÇÃO ISOLADA
# ============================================================

print("\n" + "="*80)
print("🔍 DEBUG FEA - Energia Elétrica (detecção isolada)")
print("="*80)

# Definir janela temporal
start_dt = pd.to_datetime(START_TS)
end_dt = start_dt + timedelta(hours=6)

print(f"Período: {start_dt} → {end_dt}")

# CHAMAR DETECTOR FEA ISOLADAMENTE
CORRIDA_INICIAL_FEA = 130183

df_fea_detected = detect_fea_wrapper(
    df_fea_raw=df_fea_raw,
    start_ts=str(start_dt),
    end_ts=str(end_dt),
    initial_heat=CORRIDA_INICIAL_FEA,
    fea_params=fea_params,
)

print(f"✓ FEA detectado: {len(df_fea_detected)} corridas")
if not df_fea_detected.empty:
    print(f"  - Corridas: {df_fea_detected['corrida'].min()} → {df_fea_detected['corrida'].max()}")

# Preparar dados para plot
df_fea_prep = df_fea_raw.copy()
if 'Time' in df_fea_prep.columns and 'timestamp' not in df_fea_prep.columns:
    df_fea_prep = df_fea_prep.rename(columns={'Time': 'timestamp'})

df_fea_prep['timestamp'] = pd.to_datetime(df_fea_prep['timestamp'], errors='coerce')
df_fea_prep = df_fea_prep[
    (df_fea_prep['timestamp'] >= start_dt) & 
    (df_fea_prep['timestamp'] <= end_dt)
].sort_values('timestamp').reset_index(drop=True)

# Obter dados do operador
df_op_fea = df_final.filter(
    (F.col("inicio_fea") >= str(start_dt)) & 
    (F.col("inicio_fea") <= str(end_dt))
).select("corrida", "inicio_fea", "final_fea").toPandas()

df_op_fea['inicio_fea'] = pd.to_datetime(df_op_fea['inicio_fea'], errors='coerce')
df_op_fea['final_fea'] = pd.to_datetime(df_op_fea['final_fea'], errors='coerce')

print(f"✓ Operador: {len(df_op_fea)} corridas")

# PLOTAR
fig, ax = plt.subplots(figsize=(22, 8))

# Plotar energia (LINHA, sem fill)
ax.plot(df_fea_prep['timestamp'], df_fea_prep['ACI@FEA_ELET_ENERGIA'], 
        linewidth=1.5, alpha=0.8, color='blue', label='Energia Elétrica')

# Plotar eventos OPERADOR (linhas sólidas)
colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(df_op_fea))))
for idx, (_, row) in enumerate(df_op_fea.iterrows()):
    corrida = int(row['corrida'])
    color = colors[idx % len(colors)]
    
    if pd.notna(row['inicio_fea']):
        ax.axvline(row['inicio_fea'], color=color, linestyle='-', linewidth=2, alpha=0.8)
        ax.text(row['inicio_fea'], ax.get_ylim()[1] * 0.95, f"{corrida}", 
               rotation=90, verticalalignment='top', fontsize=9, color=color, fontweight='bold')
    
    if pd.notna(row['final_fea']):
        ax.axvline(row['final_fea'], color=color, linestyle='-', linewidth=2, alpha=0.8)

# Plotar eventos SISTEMA (linhas pontilhadas)
for idx, (_, row) in enumerate(df_fea_detected.iterrows()):
    corrida = int(row['corrida'])
    color = colors[idx % len(colors)]
    
    if pd.notna(row['inicio_fea_sys']):
        ax.axvline(row['inicio_fea_sys'], color=color, linestyle='--', linewidth=2, alpha=0.6)
        ax.text(row['inicio_fea_sys'], ax.get_ylim()[1] * 0.85, f"{corrida}*", 
               rotation=90, verticalalignment='top', fontsize=8, color=color, style='italic')
    
    if pd.notna(row['final_fea_sys']):
        ax.axvline(row['final_fea_sys'], color=color, linestyle='--', linewidth=2, alpha=0.6)

# Formatação
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.xticks(rotation=45)

ax.set_xlabel('Tempo (HH:MM)', fontsize=12)
ax.set_ylabel('Energia Elétrica (kW)', fontsize=12)
ax.set_title(f"Debug FEA - Energia Elétrica - {START_TS[:10]}", fontsize=14, fontweight='bold')

ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()

print(f"\n📊 Resumo:")
print(f"  - Operador: {len(df_op_fea)} corridas")
print(f"  - Sistema: {len(df_fea_detected)} corridas")

# COMMAND ----------

# DBTITLE 1,EDA - Preparação dos dados
# ============================================================
# EDA - PREPARAÇÃO DOS DADOS
# ============================================================

print("\n" + "="*80)
print("📊 EDA - ANÁLISE EXPLORATÓRIA DE DADOS")
print("="*80)

# Converter para Pandas
df_eda = df_compare_unified.toPandas()

print(f"\n✓ Total de corridas: {len(df_eda)}")
print(f"✓ Período: {df_eda['corrida'].min()} → {df_eda['corrida'].max()}")

# Converter timestamps para datetime
timestamp_cols = [
    'inicio_fea_op', 'final_fea_op', 'inicio_fea_sys', 'final_fea_sys',
    'inicio_fp_op', 'final_fp_op', 'inicio_fp_sys', 'final_fp_sys',
    'inicio_vd_op', 'final_vd_op', 'inicio_vd_sys', 'final_vd_sys',
    'inicio_lc_op', 'final_lc_op', 'inicio_lc_sys', 'final_lc_sys'
]

for col in timestamp_cols:
    if col in df_eda.columns:
        df_eda[col] = pd.to_datetime(df_eda[col], errors='coerce')

# ============================================================
# CALCULAR DURAÇÕES DE CADA ETAPA (em minutos)
# ============================================================

# FEA
df_eda['duracao_fea_op'] = (df_eda['final_fea_op'] - df_eda['inicio_fea_op']).dt.total_seconds() / 60
df_eda['duracao_fea_sys'] = (df_eda['final_fea_sys'] - df_eda['inicio_fea_sys']).dt.total_seconds() / 60

# FP
df_eda['duracao_fp_op'] = (df_eda['final_fp_op'] - df_eda['inicio_fp_op']).dt.total_seconds() / 60
df_eda['duracao_fp_sys'] = (df_eda['final_fp_sys'] - df_eda['inicio_fp_sys']).dt.total_seconds() / 60

# VD
df_eda['duracao_vd_op'] = (df_eda['final_vd_op'] - df_eda['inicio_vd_op']).dt.total_seconds() / 60
df_eda['duracao_vd_sys'] = (df_eda['final_vd_sys'] - df_eda['inicio_vd_sys']).dt.total_seconds() / 60

# LC
df_eda['duracao_lc_op'] = (df_eda['final_lc_op'] - df_eda['inicio_lc_op']).dt.total_seconds() / 60
df_eda['duracao_lc_sys'] = (df_eda['final_lc_sys'] - df_eda['inicio_lc_sys']).dt.total_seconds() / 60

# ============================================================
# CALCULAR DURAÇÃO TOTAL DA CORRIDA (FEA início → LC fim)
# ============================================================

df_eda['duracao_total_op'] = (df_eda['final_lc_op'] - df_eda['inicio_fea_op']).dt.total_seconds() / 60
df_eda['duracao_total_sys'] = (df_eda['final_lc_sys'] - df_eda['inicio_fea_sys']).dt.total_seconds() / 60

# ============================================================
# CALCULAR TEMPO ENTRE CORRIDAS (por etapa)
# ============================================================

# Ordenar por corrida
df_eda = df_eda.sort_values('corrida').reset_index(drop=True)

# Tempo entre corridas = início da corrida N - fim da corrida N-1
for etapa in ['fea', 'fp', 'vd', 'lc']:
    # Operador
    df_eda[f'intervalo_{etapa}_op'] = (
        df_eda[f'inicio_{etapa}_op'] - df_eda[f'final_{etapa}_op'].shift(1)
    ).dt.total_seconds() / 60
    
    # Sistema
    df_eda[f'intervalo_{etapa}_sys'] = (
        df_eda[f'inicio_{etapa}_sys'] - df_eda[f'final_{etapa}_sys'].shift(1)
    ).dt.total_seconds() / 60

# ============================================================
# CALCULAR TEMPO ENTRE ETAPAS (transições)
# ============================================================

# FEA → FP
df_eda['transicao_fea_fp_op'] = (df_eda['inicio_fp_op'] - df_eda['final_fea_op']).dt.total_seconds() / 60
df_eda['transicao_fea_fp_sys'] = (df_eda['inicio_fp_sys'] - df_eda['final_fea_sys']).dt.total_seconds() / 60

# FP → VD
df_eda['transicao_fp_vd_op'] = (df_eda['inicio_vd_op'] - df_eda['final_fp_op']).dt.total_seconds() / 60
df_eda['transicao_fp_vd_sys'] = (df_eda['inicio_vd_sys'] - df_eda['final_fp_sys']).dt.total_seconds() / 60

# VD → LC
df_eda['transicao_vd_lc_op'] = (df_eda['inicio_lc_op'] - df_eda['final_vd_op']).dt.total_seconds() / 60
df_eda['transicao_vd_lc_sys'] = (df_eda['inicio_lc_sys'] - df_eda['final_vd_sys']).dt.total_seconds() / 60

print("\n✓ Cálculos concluídos:")
print("  - Durações de cada etapa (FEA, FP, VD, LC)")
print("  - Duração total das corridas")
print("  - Intervalos entre corridas")
print("  - Tempos de transição entre etapas")

print(f"\n📋 Dimensões do dataset: {df_eda.shape}")
print(f"📋 Colunas calculadas: {len([c for c in df_eda.columns if 'duracao' in c or 'intervalo' in c or 'transicao' in c])}")

# COMMAND ----------

# DBTITLE 1,EDA - Estatísticas descritivas
# ============================================================
# EDA - ESTATÍSTICAS DESCRITIVAS
# ============================================================

print("\n" + "="*80)
print("📊 ESTATÍSTICAS DESCRITIVAS")
print("="*80)

# Função para exibir estatísticas
def print_stats(df, col_op, col_sys, label):
    print(f"\n{label}:")
    print("-" * 60)
    
    # Operador
    if col_op in df.columns:
        op_data = df[col_op].dropna()
        if len(op_data) > 0:
            print(f"  OPERADOR:")
            print(f"    Média: {op_data.mean():.2f} min | Mediana: {op_data.median():.2f} min")
            print(f"    Desvio: {op_data.std():.2f} min | Min: {op_data.min():.2f} min | Max: {op_data.max():.2f} min")
            print(f"    Dados válidos: {len(op_data)}/{len(df)} ({100*len(op_data)/len(df):.1f}%)")
    
    # Sistema
    if col_sys in df.columns:
        sys_data = df[col_sys].dropna()
        if len(sys_data) > 0:
            print(f"  SISTEMA:")
            print(f"    Média: {sys_data.mean():.2f} min | Mediana: {sys_data.median():.2f} min")
            print(f"    Desvio: {sys_data.std():.2f} min | Min: {sys_data.min():.2f} min | Max: {sys_data.max():.2f} min")
            print(f"    Dados válidos: {len(sys_data)}/{len(df)} ({100*len(sys_data)/len(df):.1f}%)")
    
    # Diferença
    if col_op in df.columns and col_sys in df.columns:
        diff = df[col_sys] - df[col_op]
        diff_valid = diff.dropna()
        if len(diff_valid) > 0:
            print(f"  DIFERENÇA (Sistema - Operador):")
            print(f"    Média: {diff_valid.mean():.2f} min | Mediana: {diff_valid.median():.2f} min")
            print(f"    Desvio: {diff_valid.std():.2f} min")

# ============================================================
# DURAÇÕES POR ETAPA
# ============================================================

print("\n" + "="*80)
print("🕒 DURAÇÕES POR ETAPA")
print("="*80)

print_stats(df_eda, 'duracao_fea_op', 'duracao_fea_sys', '🔥 FEA (Forno Elétrico a Arco)')
print_stats(df_eda, 'duracao_fp_op', 'duracao_fp_sys', '🏭 FP (Forno Panela)')
print_stats(df_eda, 'duracao_vd_op', 'duracao_vd_sys', '💨 VD (Desgaseificação a Vácuo)')
print_stats(df_eda, 'duracao_lc_op', 'duracao_lc_sys', '🏭 LC (Lingotamento Contínuo)')

# ============================================================
# DURAÇÃO TOTAL
# ============================================================

print("\n" + "="*80)
print("⏱️ DURAÇÃO TOTAL DA CORRIDA (FEA → LC)")
print("="*80)

print_stats(df_eda, 'duracao_total_op', 'duracao_total_sys', 'Duração Total')

# ============================================================
# INTERVALOS ENTRE CORRIDAS
# ============================================================

print("\n" + "="*80)
print("⏳ INTERVALOS ENTRE CORRIDAS (por etapa)")
print("="*80)

print_stats(df_eda, 'intervalo_fea_op', 'intervalo_fea_sys', 'Intervalo FEA')
print_stats(df_eda, 'intervalo_fp_op', 'intervalo_fp_sys', 'Intervalo FP')
print_stats(df_eda, 'intervalo_vd_op', 'intervalo_vd_sys', 'Intervalo VD')
print_stats(df_eda, 'intervalo_lc_op', 'intervalo_lc_sys', 'Intervalo LC')

# ============================================================
# TRANSIÇÕES ENTRE ETAPAS
# ============================================================

print("\n" + "="*80)
print("➡️ TRANSIÇÕES ENTRE ETAPAS")
print("="*80)

print_stats(df_eda, 'transicao_fea_fp_op', 'transicao_fea_fp_sys', 'FEA → FP')
print_stats(df_eda, 'transicao_fp_vd_op', 'transicao_fp_vd_sys', 'FP → VD')
print_stats(df_eda, 'transicao_vd_lc_op', 'transicao_vd_lc_sys', 'VD → LC')

# ============================================================
# DIFERENÇAS (DIFFS) ENTRE SISTEMA E OPERADOR
# ============================================================

print("\n" + "="*80)
print("🔄 DIFERENÇAS ENTRE SISTEMA E OPERADOR (minutos)")
print("="*80)

diff_cols = [c for c in df_eda.columns if c.startswith('diff_')]

for col in diff_cols:
    data = df_eda[col].dropna()
    if len(data) > 0:
        etapa = col.replace('diff_', '').replace('_', ' ').upper()
        print(f"\n{etapa}:")
        print(f"  Média: {data.mean():.2f} min | Mediana: {data.median():.2f} min")
        print(f"  Desvio: {data.std():.2f} min | Min: {data.min():.2f} min | Max: {data.max():.2f} min")
        print(f"  Dados válidos: {len(data)}/{len(df_eda)} ({100*len(data)/len(df_eda):.1f}%)")

# COMMAND ----------

# DBTITLE 1,EDA - Visualizações: Durações por Etapa
# ============================================================
# EDA - VISUALIZAÇÕES: DURAÇÕES POR ETAPA
# ============================================================

print("\n" + "="*80)
print("📊 VISUALIZAÇÕES: DURAÇÕES POR ETAPA")
print("="*80)

fig, axes = plt.subplots(4, 2, figsize=(18, 16))
fig.suptitle('Durações por Etapa: Operador vs Sistema', fontsize=16, fontweight='bold', y=0.995)

etapas = [
    ('FEA', 'duracao_fea_op', 'duracao_fea_sys', '🔥 FEA (Forno Elétrico)'),
    ('FP', 'duracao_fp_op', 'duracao_fp_sys', '🏭 FP (Forno Panela)'),
    ('VD', 'duracao_vd_op', 'duracao_vd_sys', '💨 VD (Desgaseificação)'),
    ('LC', 'duracao_lc_op', 'duracao_lc_sys', '🏭 LC (Lingotamento)')
]

for idx, (etapa, col_op, col_sys, label) in enumerate(etapas):
    # Histograma
    ax_hist = axes[idx, 0]
    
    data_op = df_eda[col_op].dropna()
    data_sys = df_eda[col_sys].dropna()
    
    if len(data_op) > 0:
        ax_hist.hist(data_op, bins=20, alpha=0.6, color='blue', label='Operador', edgecolor='black')
    if len(data_sys) > 0:
        ax_hist.hist(data_sys, bins=20, alpha=0.6, color='orange', label='Sistema', edgecolor='black')
    
    ax_hist.set_xlabel('Duração (minutos)', fontsize=10)
    ax_hist.set_ylabel('Frequência', fontsize=10)
    ax_hist.set_title(f'{label} - Histograma', fontsize=11, fontweight='bold')
    ax_hist.legend(loc='best')
    ax_hist.grid(True, alpha=0.3)
    
    # Boxplot
    ax_box = axes[idx, 1]
    
    box_data = []
    box_labels = []
    
    if len(data_op) > 0:
        box_data.append(data_op)
        box_labels.append('Operador')
    if len(data_sys) > 0:
        box_data.append(data_sys)
        box_labels.append('Sistema')
    
    if len(box_data) > 0:
        bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True, 
                            showmeans=True, meanline=True)
        
        # Colorir boxes
        colors = ['lightblue', 'lightsalmon']
        for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(color)
    
    ax_box.set_ylabel('Duração (minutos)', fontsize=10)
    ax_box.set_title(f'{label} - Boxplot', fontsize=11, fontweight='bold')
    ax_box.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("✓ Gráficos de durações gerados")

# COMMAND ----------

# DBTITLE 1,EDA - Scatter Plots: Operador vs Sistema
# ============================================================
# EDA - SCATTER PLOTS: OPERADOR VS SISTEMA
# ============================================================

print("\n" + "="*80)
print("📊 SCATTER PLOTS: OPERADOR VS SISTEMA")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Comparação Operador vs Sistema: Durações', fontsize=16, fontweight='bold', y=0.995)

etapas = [
    ('FEA', 'duracao_fea_op', 'duracao_fea_sys', '🔥 FEA'),
    ('FP', 'duracao_fp_op', 'duracao_fp_sys', '🏭 FP'),
    ('VD', 'duracao_vd_op', 'duracao_vd_sys', '💨 VD'),
    ('LC', 'duracao_lc_op', 'duracao_lc_sys', '🏭 LC')
]

for idx, (etapa, col_op, col_sys, label) in enumerate(etapas):
    ax = axes[idx // 2, idx % 2]
    
    # Filtrar dados válidos
    df_valid = df_eda[[col_op, col_sys]].dropna()
    
    if len(df_valid) > 0:
        x = df_valid[col_op]
        y = df_valid[col_sys]
        
        # Scatter plot
        ax.scatter(x, y, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
        
        # Linha de referência (y=x)
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Linha Ideal (y=x)', alpha=0.7)
        
        # Calcular correlação
        corr = x.corr(y)
        
        # Calcular MAE e RMSE
        mae = np.abs(y - x).mean()
        rmse = np.sqrt(((y - x) ** 2).mean())
        
        # Adicionar texto com métricas
        textstr = f'Correlação: {corr:.3f}\nMAE: {mae:.2f} min\nRMSE: {rmse:.2f} min\nN: {len(df_valid)}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        ax.set_xlabel('Operador (minutos)', fontsize=11)
        ax.set_ylabel('Sistema (minutos)', fontsize=11)
        ax.set_title(f'{label} - Duração', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Sem dados válidos', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{label} - Duração', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

print("✓ Scatter plots gerados")

# COMMAND ----------

# DBTITLE 1,EDA - Análise de Diferenças (Diffs)
# ============================================================
# EDA - ANÁLISE DE DIFERENÇAS (SISTEMA - OPERADOR)
# ============================================================

print("\n" + "="*80)
print("🔄 ANÁLISE DE DIFERENÇAS (Sistema - Operador)")
print("="*80)

fig, axes = plt.subplots(4, 2, figsize=(18, 16))
fig.suptitle('Diferenças entre Sistema e Operador (minutos)', fontsize=16, fontweight='bold', y=0.995)

diff_pairs = [
    ('diff_inicio_fea', 'diff_final_fea', '🔥 FEA'),
    ('diff_inicio_fp', 'diff_final_fp', '🏭 FP'),
    ('diff_inicio_vd', 'diff_final_vd', '💨 VD'),
    ('diff_inicio_lc', 'diff_final_lc', '🏭 LC')
]

for idx, (col_inicio, col_final, label) in enumerate(diff_pairs):
    # Histograma - Início
    ax_inicio = axes[idx, 0]
    
    data_inicio = df_eda[col_inicio].dropna()
    
    if len(data_inicio) > 0:
        ax_inicio.hist(data_inicio, bins=25, alpha=0.7, color='steelblue', edgecolor='black')
        ax_inicio.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero (perfeito)')
        ax_inicio.axvline(data_inicio.mean(), color='green', linestyle='-', linewidth=2, label=f'Média: {data_inicio.mean():.2f}')
        
        ax_inicio.set_xlabel('Diferença (minutos)', fontsize=10)
        ax_inicio.set_ylabel('Frequência', fontsize=10)
        ax_inicio.set_title(f'{label} - Diff Início', fontsize=11, fontweight='bold')
        ax_inicio.legend(loc='best', fontsize=9)
        ax_inicio.grid(True, alpha=0.3)
        
        # Adicionar texto com estatísticas
        textstr = f'Média: {data_inicio.mean():.2f}\nMediana: {data_inicio.median():.2f}\nStd: {data_inicio.std():.2f}'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax_inicio.text(0.98, 0.97, textstr, transform=ax_inicio.transAxes, fontsize=9,
                      verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Histograma - Final
    ax_final = axes[idx, 1]
    
    data_final = df_eda[col_final].dropna()
    
    if len(data_final) > 0:
        ax_final.hist(data_final, bins=25, alpha=0.7, color='coral', edgecolor='black')
        ax_final.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero (perfeito)')
        ax_final.axvline(data_final.mean(), color='green', linestyle='-', linewidth=2, label=f'Média: {data_final.mean():.2f}')
        
        ax_final.set_xlabel('Diferença (minutos)', fontsize=10)
        ax_final.set_ylabel('Frequência', fontsize=10)
        ax_final.set_title(f'{label} - Diff Final', fontsize=11, fontweight='bold')
        ax_final.legend(loc='best', fontsize=9)
        ax_final.grid(True, alpha=0.3)
        
        # Adicionar texto com estatísticas
        textstr = f'Média: {data_final.mean():.2f}\nMediana: {data_final.median():.2f}\nStd: {data_final.std():.2f}'
        props = dict(boxstyle='round', facecolor='lightsalmon', alpha=0.8)
        ax_final.text(0.98, 0.97, textstr, transform=ax_final.transAxes, fontsize=9,
                     verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.show()

print("✓ Gráficos de diferenças gerados")

# COMMAND ----------

# DBTITLE 1,EDA - Intervalos entre Corridas
# ============================================================
# EDA - INTERVALOS ENTRE CORRIDAS
# ============================================================

print("\n" + "="*80)
print("⏳ INTERVALOS ENTRE CORRIDAS (Tempo entre fim de uma e início da próxima)")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Intervalos entre Corridas por Etapa', fontsize=16, fontweight='bold', y=0.995)

etapas = [
    ('FEA', 'intervalo_fea_op', 'intervalo_fea_sys', '🔥 FEA'),
    ('FP', 'intervalo_fp_op', 'intervalo_fp_sys', '🏭 FP'),
    ('VD', 'intervalo_vd_op', 'intervalo_vd_sys', '💨 VD'),
    ('LC', 'intervalo_lc_op', 'intervalo_lc_sys', '🏭 LC')
]

for idx, (etapa, col_op, col_sys, label) in enumerate(etapas):
    ax = axes[idx // 2, idx % 2]
    
    data_op = df_eda[col_op].dropna()
    data_sys = df_eda[col_sys].dropna()
    
    # Boxplot comparativo
    box_data = []
    box_labels = []
    
    if len(data_op) > 0:
        box_data.append(data_op)
        box_labels.append('Operador')
    if len(data_sys) > 0:
        box_data.append(data_sys)
        box_labels.append('Sistema')
    
    if len(box_data) > 0:
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, 
                       showmeans=True, meanline=True, widths=0.6)
        
        # Colorir boxes
        colors = ['lightblue', 'lightsalmon']
        for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(color)
        
        # Adicionar estatísticas
        stats_text = ""
        if len(data_op) > 0:
            stats_text += f"Operador:\n  Média: {data_op.mean():.1f} min\n  Mediana: {data_op.median():.1f} min\n"
        if len(data_sys) > 0:
            stats_text += f"Sistema:\n  Média: {data_sys.mean():.1f} min\n  Mediana: {data_sys.median():.1f} min"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax.set_ylabel('Intervalo (minutos)', fontsize=11)
    ax.set_title(f'{label} - Intervalo entre Corridas', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("✓ Gráficos de intervalos entre corridas gerados")

# COMMAND ----------

# DBTITLE 1,EDA - Transições entre Etapas
# ============================================================
# EDA - TRANSIÇÕES ENTRE ETAPAS
# ============================================================

print("\n" + "="*80)
print("➡️ TRANSIÇÕES ENTRE ETAPAS (Tempo entre fim de uma etapa e início da próxima)")
print("="*80)

fig, axes = plt.subplots(3, 2, figsize=(18, 14))
fig.suptitle('Transições entre Etapas', fontsize=16, fontweight='bold', y=0.995)

transicoes = [
    ('FEA→FP', 'transicao_fea_fp_op', 'transicao_fea_fp_sys'),
    ('FP→VD', 'transicao_fp_vd_op', 'transicao_fp_vd_sys'),
    ('VD→LC', 'transicao_vd_lc_op', 'transicao_vd_lc_sys')
]

for idx, (label, col_op, col_sys) in enumerate(transicoes):
    # Histograma
    ax_hist = axes[idx, 0]
    
    data_op = df_eda[col_op].dropna()
    data_sys = df_eda[col_sys].dropna()
    
    if len(data_op) > 0:
        ax_hist.hist(data_op, bins=20, alpha=0.6, color='blue', label='Operador', edgecolor='black')
    if len(data_sys) > 0:
        ax_hist.hist(data_sys, bins=20, alpha=0.6, color='orange', label='Sistema', edgecolor='black')
    
    ax_hist.set_xlabel('Tempo de Transição (minutos)', fontsize=10)
    ax_hist.set_ylabel('Frequência', fontsize=10)
    ax_hist.set_title(f'{label} - Histograma', fontsize=11, fontweight='bold')
    ax_hist.legend(loc='best')
    ax_hist.grid(True, alpha=0.3)
    
    # Boxplot
    ax_box = axes[idx, 1]
    
    box_data = []
    box_labels = []
    
    if len(data_op) > 0:
        box_data.append(data_op)
        box_labels.append('Operador')
    if len(data_sys) > 0:
        box_data.append(data_sys)
        box_labels.append('Sistema')
    
    if len(box_data) > 0:
        bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True, 
                           showmeans=True, meanline=True, widths=0.6)
        
        # Colorir boxes
        colors = ['lightblue', 'lightsalmon']
        for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(color)
        
        # Adicionar estatísticas
        stats_text = ""
        if len(data_op) > 0:
            stats_text += f"Operador:\n  Média: {data_op.mean():.1f} min\n  Mediana: {data_op.median():.1f} min\n  N: {len(data_op)}\n"
        if len(data_sys) > 0:
            stats_text += f"Sistema:\n  Média: {data_sys.mean():.1f} min\n  Mediana: {data_sys.median():.1f} min\n  N: {len(data_sys)}"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax_box.text(0.98, 0.97, stats_text, transform=ax_box.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax_box.set_ylabel('Tempo de Transição (minutos)', fontsize=11)
    ax_box.set_title(f'{label} - Boxplot', fontsize=11, fontweight='bold')
    ax_box.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("✓ Gráficos de transições entre etapas gerados")

# COMMAND ----------

# DBTITLE 1,EDA - Duração Total da Corrida
# ============================================================
# EDA - DURAÇÃO TOTAL DA CORRIDA (FEA → LC)
# ============================================================

print("\n" + "="*80)
print("⏱️ DURAÇÃO TOTAL DA CORRIDA (FEA início → LC fim)")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Duração Total da Corrida: Operador vs Sistema', fontsize=16, fontweight='bold', y=0.995)

# Preparar dados
data_op = df_eda['duracao_total_op'].dropna()
data_sys = df_eda['duracao_total_sys'].dropna()

# 1. Histograma
ax1 = axes[0, 0]
if len(data_op) > 0:
    ax1.hist(data_op, bins=20, alpha=0.6, color='blue', label='Operador', edgecolor='black')
if len(data_sys) > 0:
    ax1.hist(data_sys, bins=20, alpha=0.6, color='orange', label='Sistema', edgecolor='black')

ax1.set_xlabel('Duração Total (minutos)', fontsize=11)
ax1.set_ylabel('Frequência', fontsize=11)
ax1.set_title('Histograma - Duração Total', fontsize=12, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# 2. Boxplot
ax2 = axes[0, 1]
box_data = []
box_labels = []

if len(data_op) > 0:
    box_data.append(data_op)
    box_labels.append('Operador')
if len(data_sys) > 0:
    box_data.append(data_sys)
    box_labels.append('Sistema')

if len(box_data) > 0:
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, 
                    showmeans=True, meanline=True, widths=0.6)
    
    colors = ['lightblue', 'lightsalmon']
    for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
        patch.set_facecolor(color)
    
    # Estatísticas
    stats_text = ""
    if len(data_op) > 0:
        stats_text += f"Operador:\n  Média: {data_op.mean():.1f} min\n  Mediana: {data_op.median():.1f} min\n  Std: {data_op.std():.1f} min\n"
    if len(data_sys) > 0:
        stats_text += f"Sistema:\n  Média: {data_sys.mean():.1f} min\n  Mediana: {data_sys.median():.1f} min\n  Std: {data_sys.std():.1f} min"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)

ax2.set_ylabel('Duração Total (minutos)', fontsize=11)
ax2.set_title('Boxplot - Duração Total', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Scatter Plot
ax3 = axes[1, 0]
df_valid = df_eda[['duracao_total_op', 'duracao_total_sys']].dropna()

if len(df_valid) > 0:
    x = df_valid['duracao_total_op']
    y = df_valid['duracao_total_sys']
    
    ax3.scatter(x, y, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    
    # Linha y=x
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Linha Ideal (y=x)', alpha=0.7)
    
    # Métricas
    corr = x.corr(y)
    mae = np.abs(y - x).mean()
    rmse = np.sqrt(((y - x) ** 2).mean())
    
    textstr = f'Correlação: {corr:.3f}\nMAE: {mae:.2f} min\nRMSE: {rmse:.2f} min\nN: {len(df_valid)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    ax3.set_xlabel('Operador (minutos)', fontsize=11)
    ax3.set_ylabel('Sistema (minutos)', fontsize=11)
    ax3.set_title('Scatter Plot - Operador vs Sistema', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)

# 4. Diferença (Sistema - Operador)
ax4 = axes[1, 1]
if len(df_valid) > 0:
    diff = df_valid['duracao_total_sys'] - df_valid['duracao_total_op']
    
    ax4.hist(diff, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero (perfeito)')
    ax4.axvline(diff.mean(), color='green', linestyle='-', linewidth=2, label=f'Média: {diff.mean():.2f}')
    
    textstr = f'Média: {diff.mean():.2f} min\nMediana: {diff.median():.2f} min\nStd: {diff.std():.2f} min'
    props = dict(boxstyle='round', facecolor='lavender', alpha=0.8)
    ax4.text(0.98, 0.97, textstr, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax4.set_xlabel('Diferença (Sistema - Operador) em minutos', fontsize=11)
    ax4.set_ylabel('Frequência', fontsize=11)
    ax4.set_title('Diferença na Duração Total', fontsize=12, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Gráficos de duração total gerados")

# COMMAND ----------

# DBTITLE 1,EDA - Evolução Temporal das Durações
# ============================================================
# EDA - EVOLUÇÃO TEMPORAL DAS DURAÇÕES
# ============================================================

print("\n" + "="*80)
print("📈 EVOLUÇÃO TEMPORAL DAS DURAÇÕES (por corrida)")
print("="*80)

fig, axes = plt.subplots(3, 2, figsize=(20, 14))
fig.suptitle('Evolução das Durações ao Longo das Corridas', fontsize=16, fontweight='bold', y=0.995)

# Ordenar por corrida
df_sorted = df_eda.sort_values('corrida')

visualizacoes = [
    ('duracao_fea_op', 'duracao_fea_sys', '🔥 FEA'),
    ('duracao_fp_op', 'duracao_fp_sys', '🏭 FP'),
    ('duracao_vd_op', 'duracao_vd_sys', '💨 VD'),
    ('duracao_lc_op', 'duracao_lc_sys', '🏭 LC'),
    ('duracao_total_op', 'duracao_total_sys', '⏱️ Total'),
]

for idx, (col_op, col_sys, label) in enumerate(visualizacoes):
    if idx >= 5:  # Apenas 5 gráficos (3x2 grid, usando 5)
        break
    
    ax = axes[idx // 2, idx % 2]
    
    # Plotar operador
    df_op_valid = df_sorted[['corrida', col_op]].dropna()
    if len(df_op_valid) > 0:
        ax.plot(df_op_valid['corrida'], df_op_valid[col_op], 
               marker='o', linestyle='-', linewidth=2, markersize=6, 
               alpha=0.7, label='Operador', color='blue')
    
    # Plotar sistema
    df_sys_valid = df_sorted[['corrida', col_sys]].dropna()
    if len(df_sys_valid) > 0:
        ax.plot(df_sys_valid['corrida'], df_sys_valid[col_sys], 
               marker='s', linestyle='-', linewidth=2, markersize=6, 
               alpha=0.7, label='Sistema', color='orange')
    
    ax.set_xlabel('Número da Corrida', fontsize=11)
    ax.set_ylabel('Duração (minutos)', fontsize=11)
    ax.set_title(f'{label} - Evolução', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Rotacionar labels do eixo x se necessário
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# Remover subplot extra
if len(visualizacoes) < 6:
    fig.delaxes(axes[2, 1])

plt.tight_layout()
plt.show()

print("✓ Gráficos de evolução temporal gerados")

# COMMAND ----------

# DBTITLE 1,EDA - Matriz de Correlação
# ============================================================
# EDA - MATRIZ DE CORRELAÇÃO
# ============================================================

print("\n" + "="*80)
print("🔗 MATRIZ DE CORRELAÇÃO ENTRE MÉTRICAS")
print("="*80)

# Selecionar colunas numéricas relevantes
cols_duracao = [
    'duracao_fea_op', 'duracao_fp_op', 'duracao_vd_op', 'duracao_lc_op',
    'duracao_fea_sys', 'duracao_fp_sys', 'duracao_vd_sys', 'duracao_lc_sys',
    'duracao_total_op', 'duracao_total_sys'
]

cols_transicao = [
    'transicao_fea_fp_op', 'transicao_fp_vd_op', 'transicao_vd_lc_op',
    'transicao_fea_fp_sys', 'transicao_fp_vd_sys', 'transicao_vd_lc_sys'
]

# Criar figura com 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(22, 10))
fig.suptitle('Matrizes de Correlação', fontsize=16, fontweight='bold', y=0.98)

# 1. Correlação entre Durações
ax1 = axes[0]
df_duracao = df_eda[cols_duracao].dropna(how='all')
corr_duracao = df_duracao.corr()

im1 = ax1.imshow(corr_duracao, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax1.set_xticks(range(len(corr_duracao.columns)))
ax1.set_yticks(range(len(corr_duracao.columns)))
ax1.set_xticklabels([c.replace('duracao_', '').replace('_op', ' (OP)').replace('_sys', ' (SYS)') 
                     for c in corr_duracao.columns], rotation=45, ha='right', fontsize=9)
ax1.set_yticklabels([c.replace('duracao_', '').replace('_op', ' (OP)').replace('_sys', ' (SYS)') 
                     for c in corr_duracao.columns], fontsize=9)
ax1.set_title('Correlação entre Durações', fontsize=13, fontweight='bold', pad=10)

# Adicionar valores na matriz
for i in range(len(corr_duracao)):
    for j in range(len(corr_duracao)):
        text = ax1.text(j, i, f'{corr_duracao.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=7)

plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# 2. Correlação entre Transições
ax2 = axes[1]
df_transicao = df_eda[cols_transicao].dropna(how='all')
corr_transicao = df_transicao.corr()

im2 = ax2.imshow(corr_transicao, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax2.set_xticks(range(len(corr_transicao.columns)))
ax2.set_yticks(range(len(corr_transicao.columns)))
ax2.set_xticklabels([c.replace('transicao_', '').replace('_op', ' (OP)').replace('_sys', ' (SYS)') 
                     for c in corr_transicao.columns], rotation=45, ha='right', fontsize=9)
ax2.set_yticklabels([c.replace('transicao_', '').replace('_op', ' (OP)').replace('_sys', ' (SYS)') 
                     for c in corr_transicao.columns], fontsize=9)
ax2.set_title('Correlação entre Transições', fontsize=13, fontweight='bold', pad=10)

# Adicionar valores na matriz
for i in range(len(corr_transicao)):
    for j in range(len(corr_transicao)):
        text = ax2.text(j, i, f'{corr_transicao.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

print("✓ Matrizes de correlação geradas")

# Imprimir correlações mais fortes
print("\n" + "="*80)
print("🔍 CORRELAÇÕES MAIS FORTES (|r| > 0.7)")
print("="*80)

for name, corr_matrix in [('Durações', corr_duracao), ('Transições', corr_transicao)]:
    print(f"\n{name}:")
    # Pegar apenas triângulo superior
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_upper = corr_matrix.where(mask)
    
    # Encontrar correlações fortes
    strong_corr = []
    for col in corr_upper.columns:
        for idx in corr_upper.index:
            val = corr_upper.loc[idx, col]
            if pd.notna(val) and abs(val) > 0.7:
                strong_corr.append((idx, col, val))
    
    if strong_corr:
        for var1, var2, corr_val in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {var1} ↔ {var2}: {corr_val:.3f}")
    else:
        print("  Nenhuma correlação forte encontrada")

# COMMAND ----------

# DBTITLE 1,EDA - Relatório Final e Insights
# ============================================================
# EDA - RELATÓRIO FINAL E INSIGHTS
# ============================================================

print("\n" + "="*80)
print("📊 RELATÓRIO FINAL - INSIGHTS DA ANÁLISE EXPLORATÓRIA")
print("="*80)

print("\n" + "="*80)
print("📈 1. COBERTURA DE DADOS")
print("="*80)

# Calcular cobertura por etapa
etapas_check = [
    ('FEA', 'inicio_fea_op', 'inicio_fea_sys'),
    ('FP', 'inicio_fp_op', 'inicio_fp_sys'),
    ('VD', 'inicio_vd_op', 'inicio_vd_sys'),
    ('LC', 'inicio_lc_op', 'inicio_lc_sys')
]

for etapa, col_op, col_sys in etapas_check:
    n_op = df_eda[col_op].notna().sum()
    n_sys = df_eda[col_sys].notna().sum()
    pct_op = 100 * n_op / len(df_eda)
    pct_sys = 100 * n_sys / len(df_eda)
    
    print(f"\n{etapa}:")
    print(f"  Operador: {n_op}/{len(df_eda)} corridas ({pct_op:.1f}%)")
    print(f"  Sistema:  {n_sys}/{len(df_eda)} corridas ({pct_sys:.1f}%)")
    print(f"  Diferença: {n_op - n_sys} corridas ({pct_op - pct_sys:+.1f}%)")

print("\n" + "="*80)
print("🎯 2. ACURÁCIA DO SISTEMA (vs Operador)")
print("="*80)

# Calcular MAE e RMSE para cada etapa
metricas_acuracia = []

for etapa, col_op, col_sys, label in [
    ('FEA', 'duracao_fea_op', 'duracao_fea_sys', 'FEA'),
    ('FP', 'duracao_fp_op', 'duracao_fp_sys', 'FP'),
    ('VD', 'duracao_vd_op', 'duracao_vd_sys', 'VD'),
    ('LC', 'duracao_lc_op', 'duracao_lc_sys', 'LC'),
    ('Total', 'duracao_total_op', 'duracao_total_sys', 'Total')
]:
    df_valid = df_eda[[col_op, col_sys]].dropna()
    
    if len(df_valid) > 0:
        diff = df_valid[col_sys] - df_valid[col_op]
        mae = np.abs(diff).mean()
        rmse = np.sqrt((diff ** 2).mean())
        mape = 100 * np.abs(diff / df_valid[col_op]).mean()
        
        metricas_acuracia.append({
            'Etapa': label,
            'MAE (min)': mae,
            'RMSE (min)': rmse,
            'MAPE (%)': mape,
            'N': len(df_valid)
        })

df_metricas = pd.DataFrame(metricas_acuracia)
print("\nMétricas de Erro (Sistema vs Operador):")
print(df_metricas.to_string(index=False))

print("\n" + "="*80)
print("⏱️ 3. TEMPOS MÉDIOS POR ETAPA")
print("="*80)

for etapa, col_op, col_sys, label in [
    ('FEA', 'duracao_fea_op', 'duracao_fea_sys', '🔥 FEA'),
    ('FP', 'duracao_fp_op', 'duracao_fp_sys', '🏭 FP'),
    ('VD', 'duracao_vd_op', 'duracao_vd_sys', '💨 VD'),
    ('LC', 'duracao_lc_op', 'duracao_lc_sys', '🏭 LC')
]:
    op_mean = df_eda[col_op].mean()
    sys_mean = df_eda[col_sys].mean()
    
    print(f"\n{label}:")
    print(f"  Operador: {op_mean:.1f} min ({op_mean/60:.2f} horas)")
    print(f"  Sistema:  {sys_mean:.1f} min ({sys_mean/60:.2f} horas)")
    print(f"  Diferença: {sys_mean - op_mean:+.1f} min")

print("\n" + "="*80)
print("➡️ 4. TEMPOS DE TRANSIÇÃO MÉDIOS")
print("="*80)

for label, col_op, col_sys in [
    ('FEA → FP', 'transicao_fea_fp_op', 'transicao_fea_fp_sys'),
    ('FP → VD', 'transicao_fp_vd_op', 'transicao_fp_vd_sys'),
    ('VD → LC', 'transicao_vd_lc_op', 'transicao_vd_lc_sys')
]:
    op_mean = df_eda[col_op].mean()
    sys_mean = df_eda[col_sys].mean()
    
    print(f"\n{label}:")
    print(f"  Operador: {op_mean:.1f} min")
    print(f"  Sistema:  {sys_mean:.1f} min")
    print(f"  Diferença: {sys_mean - op_mean:+.1f} min")

print("\n" + "="*80)
print("💡 5. PRINCIPAIS INSIGHTS")
print("="*80)

# Calcular insights automáticos
insights = []

# 1. Etapa com maior erro
if len(df_metricas) > 0:
    etapa_maior_erro = df_metricas.loc[df_metricas['MAE (min)'].idxmax()]
    insights.append(f"⚠️ Maior erro em {etapa_maior_erro['Etapa']}: MAE = {etapa_maior_erro['MAE (min)']:.2f} min")

# 2. Etapa com melhor acurácia
if len(df_metricas) > 0:
    etapa_menor_erro = df_metricas.loc[df_metricas['MAE (min)'].idxmin()]
    insights.append(f"✅ Melhor acurácia em {etapa_menor_erro['Etapa']}: MAE = {etapa_menor_erro['MAE (min)']:.2f} min")

# 3. Cobertura
cobertura_sys = df_eda['inicio_fea_sys'].notna().sum() / len(df_eda) * 100
insights.append(f"📊 Cobertura do sistema: {cobertura_sys:.1f}% das corridas")

# 4. Duração total média
duracao_total_media = df_eda['duracao_total_op'].mean()
insights.append(f"⏱️ Duração média total (Operador): {duracao_total_media:.1f} min ({duracao_total_media/60:.2f} horas)")

# 5. Variabilidade
for etapa, col in [('FEA', 'duracao_fea_op'), ('FP', 'duracao_fp_op'), ('VD', 'duracao_vd_op'), ('LC', 'duracao_lc_op')]:
    cv = df_eda[col].std() / df_eda[col].mean() * 100
    if cv > 20:
        insights.append(f"📉 Alta variabilidade em {etapa}: CV = {cv:.1f}%")

print("\n")
for i, insight in enumerate(insights, 1):
    print(f"{i}. {insight}")

print("\n" + "="*80)
print("✅ ANÁLISE EXPLORATÓRIA CONCLUÍDA!")
print("="*80)

# COMMAND ----------

# DBTITLE 1,Debug FP - Visualização
# ============================================================
# DEBUG FP - DETECÇÃO ISOLADA
# ============================================================

print("\n" + "="*80)
print("🔍 DEBUG FP - Volume de Gás (detecção isolada)")
print("="*80)

# Definir janela temporal
start_dt = pd.to_datetime(START_TS)
end_dt = start_dt + timedelta(hours=6)

print(f"Período: {start_dt} → {end_dt}")

# CHAMAR DETECTOR FP ISOLADAMENTE
CORRIDA_INICIAL_FP = 130183

# ATUALIZAÇÃO: Agora retorna 3 DataFrames (final, events, heat)
df_fp_final_spark, df_fp_events_spark, df_heat_spark = detect_fp_events_gas_based_streaming(
    spark_df=df_fp_raw,
    initial_heat_number=CORRIDA_INICIAL_FP,
    initial_timestamp=str(start_dt),
    last_timestamp=str(end_dt),
    **fp_params_streaming
)

df_fp_detected = df_fp_final_spark.toPandas()
df_fp_detected = df_fp_detected.rename(columns={'fp_start': 'inicio_fp', 'fp_end': 'final_fp'})
df_fp_detected['inicio_fp'] = pd.to_datetime(df_fp_detected['inicio_fp'], errors='coerce')
df_fp_detected['final_fp'] = pd.to_datetime(df_fp_detected['final_fp'], errors='coerce')

print(f"✓ FP detectado: {len(df_fp_detected)} eventos")

# Obter dados do operador
df_op_fp = df_final.filter(
    (F.col("inicio_fp") >= str(start_dt)) & 
    (F.col("inicio_fp") <= str(end_dt))
).select("corrida", "inicio_fp", "final_fp").toPandas()

df_op_fp['inicio_fp'] = pd.to_datetime(df_op_fp['inicio_fp'], errors='coerce')
df_op_fp['final_fp'] = pd.to_datetime(df_op_fp['final_fp'], errors='coerce')

print(f"✓ Operador: {len(df_op_fp)} corridas")

# PRÉ-PROCESSAR DADOS (mesmo do detector)
COL_AR_C1 = "ACI@FP_Volume_Argonio_Carro_1"
COL_N2_C1 = "ACI@FP_Volume_Nitrogenio_Carro_1"
COL_AR_C2 = "ACI@FP_Volume_Argonio_Carro_2"
COL_N2_C2 = "ACI@FP_Volume_Nitrogenio_Carro_2"

df_fp_processed = preprocess_pims_data_streaming(
    spark_df=df_fp_raw,
    first_timestamp=str(start_dt),
    last_timestamp=str(end_dt),
    timestamp_column="timestamp",
    columns_to_fill=[COL_AR_C1, COL_N2_C1, COL_AR_C2, COL_N2_C2],
)

df_fp_viz = df_fp_processed.toPandas()
df_fp_viz['timestamp'] = pd.to_datetime(df_fp_viz['timestamp'], errors='coerce')

print(f"✓ Dados processados: {len(df_fp_viz)} amostras")

# PLOTAR (APENAS LINHAS, SEM FILL)
fig, ax = plt.subplots(figsize=(22, 8))

# Plotar 4 linhas de gás
ax.plot(df_fp_viz['timestamp'], df_fp_viz[COL_AR_C1], linewidth=1.5, alpha=0.7, label='Ar Carro 1')
ax.plot(df_fp_viz['timestamp'], df_fp_viz[COL_N2_C1], linewidth=1.5, alpha=0.7, label='N2 Carro 1')
ax.plot(df_fp_viz['timestamp'], df_fp_viz[COL_AR_C2], linewidth=1.5, alpha=0.7, label='Ar Carro 2')
ax.plot(df_fp_viz['timestamp'], df_fp_viz[COL_N2_C2], linewidth=1.5, alpha=0.7, label='N2 Carro 2')

# Plotar eventos OPERADOR (linhas sólidas)
colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(df_op_fp))))
for idx, (_, row) in enumerate(df_op_fp.iterrows()):
    corrida = int(row['corrida'])
    color = colors[idx % len(colors)]
    
    if pd.notna(row['inicio_fp']):
        ax.axvline(row['inicio_fp'], color=color, linestyle='-', linewidth=2, alpha=0.8)
        ax.text(row['inicio_fp'], ax.get_ylim()[1] * 0.95, f"{corrida}", 
               rotation=90, verticalalignment='top', fontsize=9, color=color, fontweight='bold')
    
    if pd.notna(row['final_fp']):
        ax.axvline(row['final_fp'], color=color, linestyle='-', linewidth=2, alpha=0.8)

# Plotar eventos SISTEMA (linhas pontilhadas)
for idx, (_, row) in enumerate(df_fp_detected.iterrows()):
    carro = int(row['carro_fp']) if pd.notna(row.get('carro_fp')) else 0
    color = colors[idx % len(colors)]
    
    if pd.notna(row['inicio_fp']):
        ax.axvline(row['inicio_fp'], color=color, linestyle='--', linewidth=2, alpha=0.6)
        ax.text(row['inicio_fp'], ax.get_ylim()[1] * 0.85, f"C{carro}*", 
               rotation=90, verticalalignment='top', fontsize=8, color=color, style='italic')
    
    if pd.notna(row['final_fp']):
        ax.axvline(row['final_fp'], color=color, linestyle='--', linewidth=2, alpha=0.6)

# Formatação
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.xticks(rotation=45)

ax.set_xlabel('Tempo (HH:MM)', fontsize=12)
ax.set_ylabel('Volume de Gás', fontsize=12)
ax.set_title(f"Debug FP - Volumes de Gás por Carro - {START_TS[:10]}", fontsize=14, fontweight='bold')

ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()

print(f"\n📊 Resumo:")
print(f"  - Operador: {len(df_op_fp)} corridas")
print(f"  - Sistema: {len(df_fp_detected)} eventos")

# COMMAND ----------

# DBTITLE 1,Debug VD - Visualização
# ============================================================
# DEBUG VD - DETECÇÃO ISOLADA
# ============================================================

print("\n" + "="*80)
print("🔍 DEBUG VD - Volume de Gás (detecção isolada)")
print("="*80)

# Definir janela temporal
start_dt = pd.to_datetime(START_TS)
end_dt = start_dt + timedelta(hours=6)

print(f"Período: {start_dt} → {end_dt}")

# CHAMAR DETECTOR VD ISOLADAMENTE
CORRIDA_INICIAL_VD = 130183

df_vd_detected_spark = detect_vd_events_gas_based(
    spark_df_vd_pims=df_vd_raw,
    start_ts=str(start_dt),
    end_ts=str(end_dt),
    initial_heat_number=CORRIDA_INICIAL_VD,
    **vd_params
)

df_vd_detected = df_vd_detected_spark.toPandas()
df_vd_detected['inicio_vd'] = pd.to_datetime(df_vd_detected['inicio_vd'], errors='coerce')
df_vd_detected['final_vd'] = pd.to_datetime(df_vd_detected['final_vd'], errors='coerce')

print(f"✓ VD detectado: {len(df_vd_detected)} eventos")

# Obter dados do operador
df_op_vd = df_final.filter(
    (F.col("inicio_vd") >= str(start_dt)) & 
    (F.col("inicio_vd") <= str(end_dt))
).select("corrida", "inicio_vd", "final_vd").toPandas()

df_op_vd['inicio_vd'] = pd.to_datetime(df_op_vd['inicio_vd'], errors='coerce')
df_op_vd['final_vd'] = pd.to_datetime(df_op_vd['final_vd'], errors='coerce')

print(f"✓ Operador: {len(df_op_vd)} corridas")

# PRÉ-PROCESSAR DADOS (mesmo do detector, mas SEM fillna para evitar fill visual)
df_vd_processed = (
    df_vd_raw
    .filter(
        (F.col("timestamp") >= str(start_dt)) &
        (F.col("timestamp") <= str(end_dt))
    )
    .orderBy("timestamp")
)

df_vd_viz = df_vd_processed.toPandas()
df_vd_viz['timestamp'] = pd.to_datetime(df_vd_viz['timestamp'], errors='coerce')

# Converter para numérico (manter NaN ao invés de zero para evitar fill)
for col in ['ACI@VD_VASO1_VOLUMEAR', 'ACI@VD_VASO1_VOLUMEN2', 'ACI@VD_VASO2_VOLUMEAR', 'ACI@VD_VASO2_VOLUMEN2']:
    if col in df_vd_viz.columns:
        df_vd_viz[col] = pd.to_numeric(df_vd_viz[col], errors='coerce')
        # Substituir zeros por NaN para quebrar linhas
        df_vd_viz[col] = df_vd_viz[col].replace(0, np.nan)

print(f"✓ Dados processados: {len(df_vd_viz)} amostras")

# PLOTAR (APENAS LINHAS, SEM FILL)
fig, ax = plt.subplots(figsize=(22, 8))

# Plotar 4 linhas de gás (NaN quebra as linhas, evitando fill)
ax.plot(df_vd_viz['timestamp'], df_vd_viz['ACI@VD_VASO1_VOLUMEAR'], 
        linewidth=1.5, alpha=0.7, label='Ar Vaso 1')
ax.plot(df_vd_viz['timestamp'], df_vd_viz['ACI@VD_VASO1_VOLUMEN2'], 
        linewidth=1.5, alpha=0.7, label='N2 Vaso 1')
ax.plot(df_vd_viz['timestamp'], df_vd_viz['ACI@VD_VASO2_VOLUMEAR'], 
        linewidth=1.5, alpha=0.7, label='Ar Vaso 2')
ax.plot(df_vd_viz['timestamp'], df_vd_viz['ACI@VD_VASO2_VOLUMEN2'], 
        linewidth=1.5, alpha=0.7, label='N2 Vaso 2')

# Plotar eventos OPERADOR (linhas sólidas)
colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(df_op_vd))))
for idx, (_, row) in enumerate(df_op_vd.iterrows()):
    corrida = int(row['corrida'])
    color = colors[idx % len(colors)]
    
    if pd.notna(row['inicio_vd']):
        ax.axvline(row['inicio_vd'], color=color, linestyle='-', linewidth=2, alpha=0.8)
        ax.text(row['inicio_vd'], ax.get_ylim()[1] * 0.95, f"{corrida}", 
               rotation=90, verticalalignment='top', fontsize=9, color=color, fontweight='bold')
    
    if pd.notna(row['final_vd']):
        ax.axvline(row['final_vd'], color=color, linestyle='-', linewidth=2, alpha=0.8)

# Plotar eventos SISTEMA (linhas pontilhadas)
for idx, (_, row) in enumerate(df_vd_detected.iterrows()):
    tanque = int(row['vd_tanque']) if pd.notna(row.get('vd_tanque')) else 0
    corrida = int(row['corrida']) if 'corrida' in row and pd.notna(row.get('corrida')) else idx
    color = colors[idx % len(colors)]
    
    if pd.notna(row['inicio_vd']):
        ax.axvline(row['inicio_vd'], color=color, linestyle='--', linewidth=2, alpha=0.6)
        ax.text(row['inicio_vd'], ax.get_ylim()[1] * 0.85, f"T{tanque}*", 
               rotation=90, verticalalignment='top', fontsize=8, color=color, style='italic')
    
    if pd.notna(row['final_vd']):
        ax.axvline(row['final_vd'], color=color, linestyle='--', linewidth=2, alpha=0.6)

# Formatação
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.xticks(rotation=45)

ax.set_xlabel('Tempo (HH:MM)', fontsize=12)
ax.set_ylabel('Volume de Gás', fontsize=12)
ax.set_title(f"Debug VD - Volumes de Gás por Vaso - {START_TS[:10]}", fontsize=14, fontweight='bold')

ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()

print(f"\n📊 Resumo:")
print(f"  - Operador: {len(df_op_vd)} corridas")
print(f"  - Sistema: {len(df_vd_detected)} eventos")

# COMMAND ----------

# DBTITLE 1,Debug LC - Visualização
# ============================================================
# DEBUG LC - DETECÇÃO ISOLADA
# ============================================================

print("\n" + "="*80)
print("🔍 DEBUG LC - Peso da Torre (detecção isolada)")
print("="*80)

# Definir janela temporal
start_dt = pd.to_datetime(START_TS)
end_dt = start_dt + timedelta(hours=6)

print(f"Período: {start_dt} → {end_dt}")

# CHAMAR DETECTOR LC ISOLADAMENTE
CORRIDA_INICIAL_LC = 130183

df_lc_detected = detect_lc_events(
    spark_df=df_lc_raw,
    timestamp_col="timestamp",
    peso_col="ACI@LC_TORRE_PESOREAL",
    start_ts=str(start_dt),
    end_ts=str(end_dt),
    initial_corrida=CORRIDA_INICIAL_LC,
    **lc_params
)

df_lc_detected['inicio_lc'] = pd.to_datetime(df_lc_detected['start_detected'], errors='coerce')
df_lc_detected['final_lc'] = pd.to_datetime(df_lc_detected['end_detected'], errors='coerce')

print(f"✓ LC detectado: {len(df_lc_detected)} eventos")

# Obter dados do operador
df_op_lc = df_final.filter(
    (F.col("inicio_lc") >= str(start_dt)) & 
    (F.col("inicio_lc") <= str(end_dt))
).select("corrida", "inicio_lc", "final_lc").toPandas()

df_op_lc['inicio_lc'] = pd.to_datetime(df_op_lc['inicio_lc'], errors='coerce')
df_op_lc['final_lc'] = pd.to_datetime(df_op_lc['final_lc'], errors='coerce')

print(f"✓ Operador: {len(df_op_lc)} corridas")

# PRÉ-PROCESSAR DADOS (mesmo do detector)
df_lc_prep = (
    df_lc_raw
    .withColumn("timestamp_sp", F.col("timestamp"))
    .withColumn("peso_raw", F.regexp_replace(F.col("ACI@LC_TORRE_PESOREAL"), ",", ".").cast("double"))
)

# Forward-fill
from pyspark.sql.window import Window
w_ffill = Window.orderBy("timestamp_sp").rowsBetween(Window.unboundedPreceding, 0)
df_lc_prep = df_lc_prep.withColumn("peso", F.last("peso_raw", ignorenulls=True).over(w_ffill))

df_lc_prep = df_lc_prep.filter(
    (F.col("timestamp_sp") >= str(start_dt)) & 
    (F.col("timestamp_sp") <= str(end_dt))
).orderBy("timestamp_sp")

df_lc_viz = df_lc_prep.select("timestamp_sp", "peso_raw", "peso").toPandas()
df_lc_viz['timestamp'] = pd.to_datetime(df_lc_viz['timestamp_sp'], errors='coerce')

# Smooth
SMOOTH_WINDOW_S = lc_params.get('SMOOTH_WINDOW_S', 15)
df_lc_viz['peso_smooth'] = df_lc_viz['peso'].rolling(window=SMOOTH_WINDOW_S, min_periods=max(3, SMOOTH_WINDOW_S // 3)).median()

print(f"✓ Dados processados: {len(df_lc_viz)} amostras")

# PLOTAR (APENAS LINHAS, SEM FILL)
fig, ax = plt.subplots(figsize=(22, 8))

# Plotar peso raw e smooth
ax.plot(df_lc_viz['timestamp'], df_lc_viz['peso_raw'], linewidth=1, alpha=0.4, color='gray', label='Peso Raw')
ax.plot(df_lc_viz['timestamp'], df_lc_viz['peso_smooth'], linewidth=1.5, alpha=0.8, color='blue', label='Peso Smooth')

# Plotar eventos OPERADOR (linhas sólidas)
colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(df_op_lc))))
for idx, (_, row) in enumerate(df_op_lc.iterrows()):
    corrida = int(row['corrida'])
    color = colors[idx % len(colors)]
    
    if pd.notna(row['inicio_lc']):
        ax.axvline(row['inicio_lc'], color=color, linestyle='-', linewidth=2, alpha=0.8)
        ax.text(row['inicio_lc'], ax.get_ylim()[1] * 0.95, f"{corrida}", 
               rotation=90, verticalalignment='top', fontsize=9, color=color, fontweight='bold')
    
    if pd.notna(row['final_lc']):
        ax.axvline(row['final_lc'], color=color, linestyle='-', linewidth=2, alpha=0.8)

# Plotar eventos SISTEMA (linhas pontilhadas)
for idx, (_, row) in enumerate(df_lc_detected.iterrows()):
    corrida = int(row['corrida']) if 'corrida' in row and pd.notna(row.get('corrida')) else idx
    color = colors[idx % len(colors)]
    
    if pd.notna(row['inicio_lc']):
        ax.axvline(row['inicio_lc'], color=color, linestyle='--', linewidth=2, alpha=0.6)
        ax.text(row['inicio_lc'], ax.get_ylim()[1] * 0.85, f"{corrida}*", 
               rotation=90, verticalalignment='top', fontsize=8, color=color, style='italic')
    
    if pd.notna(row['final_lc']):
        ax.axvline(row['final_lc'], color=color, linestyle='--', linewidth=2, alpha=0.6)

# Formatação
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.xticks(rotation=45)

ax.set_xlabel('Tempo (HH:MM)', fontsize=12)
ax.set_ylabel('Peso (ton)', fontsize=12)
ax.set_title(f"Debug LC - Peso da Torre - {START_TS[:10]}", fontsize=14, fontweight='bold')

ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()

print(f"\n📊 Resumo:")
print(f"  - Operador: {len(df_op_lc)} corridas")
print(f"  - Sistema: {len(df_lc_detected)} eventos")

# COMMAND ----------

# DBTITLE 1,comparacao_operador (com FEA)
import pandas as pd
from pyspark.sql import functions as F


def comparacao_operador(
    df_sys_pd: pd.DataFrame,
    df_op_sp,
    spark,
):
    """
    Compara tempos do SISTEMA (pandas) vs OPERADOR (spark),
    retornando um DataFrame Spark com colunas _op, _sys e diffs (minutos),
    para todas as etapas existentes no SYS (FEA, FP, VD, LC).
    """

    # ======================================================
    # 1) SYS (pandas) -> Spark
    # ======================================================
    df_sys = df_sys_pd.copy()

    df_sys["corrida"] = (
        pd.to_numeric(df_sys["corrida"], errors="coerce")
        .astype("Int64")
    )

    # Incluir FEA nas colunas de timestamp
    ts_cols = [
        "inicio_fea", "final_fea",
        "inicio_fp", "final_fp",
        "inicio_vd", "final_vd",
        "inicio_lc", "final_lc",
    ]

    for c in ts_cols:
        if c in df_sys.columns:
            df_sys[c] = pd.to_datetime(df_sys[c], errors="coerce")
            try:
                df_sys[c] = df_sys[c].dt.tz_localize(None)
            except Exception:
                pass

    df_sys_sp = spark.createDataFrame(df_sys)

    # ======================================================
    # 2) JOIN OP x SYS
    # ======================================================
    df_join = (
        df_op_sp.alias("op")
        .join(df_sys_sp.alias("sys"), on="corrida", how="left")
    )

    # ======================================================
    # 3) Função diff (minutos)
    # ======================================================
    def diff_min(sys_col, op_col):
        return F.when(
            F.col(sys_col).isNull() | F.col(op_col).isNull(),
            F.lit(None).cast("double")
        ).otherwise(
            (F.col(sys_col).cast("long") - F.col(op_col).cast("long")) / 60.0
        )

    # ======================================================
    # 4) Construção dinâmica das colunas (incluindo FEA)
    # ======================================================
    select_cols = [F.col("corrida")]

    # Incluir FEA na lista de etapas
    etapas = ["fea", "fp", "vd", "lc"]

    for etapa in etapas:
        ini = f"inicio_{etapa}"
        fim = f"final_{etapa}"

        if ini in df_sys.columns:
            select_cols.extend([
                F.col(f"op.{ini}").alias(f"{ini}_op"),
                F.col(f"sys.{ini}").alias(f"{ini}_sys"),
                diff_min(f"sys.{ini}", f"op.{ini}").alias(f"diff_{ini}"),
            ])

        if fim in df_sys.columns:
            select_cols.extend([
                F.col(f"op.{fim}").alias(f"{fim}_op"),
                F.col(f"sys.{fim}").alias(f"{fim}_sys"),
                diff_min(f"sys.{fim}", f"op.{fim}").alias(f"diff_{fim}"),
            ])

    df_compare = df_join.select(*select_cols)

    return df_compare


print("✓ Função comparacao_operador() criada (inclui FEA)")

# COMMAND ----------

def build_events(df_fp, df_vd, df_lc, debug=False) -> List[Event]:
    events: List[Event] = []

    # =========================
    # FP — NASCE A CORRIDA
    # =========================
    for _, r in df_fp.iterrows():
        c = int(r["ACI@FP_NUMERO_CORRIDA"])

        if pd.notna(r.get("inicio_fp")):
            events.append(Event(
                ts=pd.to_datetime(r["inicio_fp"]),
                kind="FP_START",
                payload={"corrida": c}
            ))

        if pd.notna(r.get("final_fp")):
            events.append(Event(
                ts=pd.to_datetime(r["final_fp"]),
                kind="FP_END",
                payload={"corrida": c}
            ))

    # =========================
    # VD — NÃO CRIA CORRIDA
    # =========================
    for _, r in df_vd.iterrows():
        tanque = r["vd_tanque"]

        if pd.notna(r.get("inicio_vd")):
            events.append(Event(
                ts=pd.to_datetime(r["inicio_vd"]),
                kind="VD_START",
                payload={"tanque": tanque}
            ))

        if pd.notna(r.get("final_vd")):
            events.append(Event(
                ts=pd.to_datetime(r["final_vd"]),
                kind="VD_END",
                payload={"tanque": tanque}
            ))

    # =========================
    # LC — NÃO CRIA CORRIDA
    # =========================
    for _, r in df_lc.iterrows():
        if pd.notna(r.get("inicio_lc")):
            events.append(Event(
                ts=pd.to_datetime(r["inicio_lc"]),
                kind="LC_START",
                payload={}
            ))

        if pd.notna(r.get("final_lc")):
            events.append(Event(
                ts=pd.to_datetime(r["final_lc"]),
                kind="LC_END",
                payload={}
            ))

    events.sort(key=lambda e: (e.ts, e.kind))

    if debug:
        print(f"[INIT] {len(events)} eventos carregados")

    return events


# COMMAND ----------

def simulate_corridas(
    df_fp: pd.DataFrame,
    df_vd: pd.DataFrame,
    df_lc: pd.DataFrame,
    debug: bool = True
) -> pd.DataFrame:

    events = build_events(df_fp, df_vd, df_lc, debug)

    corridas: Dict[int, Corrida] = {}

    fila_vd: List[int] = []
    fila_lc: List[int] = []

    vd_busy = {1: None, 2: None}
    lc_busy: Optional[int] = None

    vd_anchor_time: Optional[pd.Timestamp] = None
    lc_anchor_time: Optional[pd.Timestamp] = None

    def get_corrida(c):
        if c not in corridas:
            corridas[c] = Corrida(corrida=c)
        return corridas[c]

    started = False

    for ev in events:
        ts, kind = ev.ts, ev.kind

        # ---------------------------
        # START GLOBAL
        # ---------------------------
        if not started:
            if kind != "FP_START":
                continue

            c = ev.payload["corrida"]
            started = True
            get_corrida(c).inicio_fp = ts

            if debug:
                print(f"[{ts}] 🚀 START → corrida {c}")
            continue

        # ---------------------------
        # FP
        # ---------------------------
        if kind == "FP_START":
            c = ev.payload["corrida"]
            get_corrida(c).inicio_fp = ts
            if debug:
                print(f"[{ts}] FP_START → corrida {c}")
            continue

        if kind == "FP_END":
            c = ev.payload["corrida"]
            get_corrida(c).final_fp = ts
            fila_vd.append(c)

            if vd_anchor_time is None:
                vd_anchor_time = ts + DELAY_FP_VD
                if debug:
                    print(f"[{ts}] FP_END corrida {c} → VD libera após {vd_anchor_time}")

            if debug:
                print(f"[{ts}] FP_END → fila_VD {fila_vd}")
            continue

        # ---------------------------
        # VD
        # ---------------------------
        if kind == "VD_START":
            if vd_anchor_time is None or ts < vd_anchor_time:
                continue

            tanque = ev.payload["tanque"]
            if vd_busy[tanque] is not None or not fila_vd:
                continue

            c = fila_vd.pop(0)
            vd_busy[tanque] = c
            get_corrida(c).inicio_vd = ts
            get_corrida(c).vd_tanque = tanque

            if debug:
                print(f"[{ts}] VD_START(t{tanque}) → corrida {c}")
            continue

        if kind == "VD_END":
            tanque = ev.payload["tanque"]
            c = vd_busy.get(tanque)

            if c is None:
                continue

            get_corrida(c).final_vd = ts
            fila_lc.append(c)
            vd_busy[tanque] = None

            if lc_anchor_time is None:
                lc_anchor_time = ts + DELAY_VD_LC
                if debug:
                    print(f"[{ts}] VD_END corrida {c} → LC libera após {lc_anchor_time}")

            if debug:
                print(f"[{ts}] VD_END(t{tanque}) → fila_LC {fila_lc}")
            continue

        # ---------------------------
        # LC
        # ---------------------------
        if kind == "LC_START":
            if lc_anchor_time is None or ts < lc_anchor_time:
                continue

            if lc_busy is not None or not fila_lc:
                continue

            c = fila_lc.pop(0)
            lc_busy = c
            get_corrida(c).inicio_lc = ts

            if debug:
                print(f"[{ts}] LC_START → corrida {c}")
            continue

        if kind == "LC_END":
            if lc_busy is None:
                continue

            get_corrida(lc_busy).final_lc = ts

            if debug:
                print(f"[{ts}] LC_END → corrida {lc_busy}")

            lc_busy = None
            continue

    out = pd.DataFrame([vars(c) for c in corridas.values()]).sort_values("corrida")

    def status(r):
        if pd.notna(r.inicio_lc) and pd.isna(r.final_lc):
            return "LC"
        if pd.notna(r.inicio_vd) and pd.isna(r.final_vd):
            return "VD"
        if pd.notna(r.inicio_fp) and pd.isna(r.final_fp):
            return "FP"
        return "FINALIZADA"

    out["status_atual"] = out.apply(status, axis=1)
    return out.reset_index(drop=True)


# COMMAND ----------

def run_corrida_simulator(
    spark_fp_raw,
    spark_vd_raw,
    spark_lc_raw,

    start_ts: str,
    end_ts: str,

    fp_params: Dict[str, Any],
    vd_params: Dict[str, Any],
    lc_params: Dict[str, Any],

    delay_fp_vd_minutes: int = 5,
    delay_vd_lc_minutes: int = 2,

    debug: bool = True,
) -> pd.DataFrame:

    if debug:
        print("▶️ Rodando detectores...")

# =========================
# ---------- FP ----------
# =========================

    # 1️⃣ Detecta eventos de FP (Spark → pandas)
    df_fp_detected = (
        detect_fp_events_gas_based(
            spark_df=spark_fp_raw,
            start_ts=start_ts,
            end_ts=end_ts,
            initial_heat_number=0,  # dummy
            **fp_params
        )
        .toPandas()
    )

    # garante datetime
    df_fp_detected["inicio_fp"] = pd.to_datetime(df_fp_detected["inicio_fp"])
    df_fp_detected["final_fp"] = pd.to_datetime(df_fp_detected["final_fp"])

    # 2️⃣ Carrega RAW do FP com número da corrida
    df_fp_raw_pd = (
        spark_fp_raw
        .select("timestamp", "ACI@FP_NUMERO_CORRIDA")
        .where(
            (spark_fp_raw.timestamp >= start_ts) &
            (spark_fp_raw.timestamp <= end_ts)
        )
        .toPandas()
    )

    df_fp_raw_pd["timestamp"] = pd.to_datetime(df_fp_raw_pd["timestamp"])
    df_fp_raw_pd = df_fp_raw_pd.sort_values("timestamp")

    # 3️⃣ Normaliza o sinal ACI@FP_NUMERO_CORRIDA
    def normalize_fp_corrida(x):
        if pd.isna(x):
            return pd.NA

        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return pd.NA

        try:
            x_int = int(x)
        except Exception:
            return pd.NA

        # zero ou negativo não é corrida válida
        if x_int <= 0:
            return pd.NA

        return x_int

    df_fp_raw_pd["fp_corrida_norm"] = (
        df_fp_raw_pd["ACI@FP_NUMERO_CORRIDA"]
        .apply(normalize_fp_corrida)
        .astype("Int64")  # inteiro nullable
    )

    # 4️⃣ Forward-fill do estado da corrida
    df_fp_raw_pd["fp_corrida_ffill"] = (
        df_fp_raw_pd["fp_corrida_norm"]
        .ffill()
    )

    # 5️⃣ Merge temporal robusto (estado mais recente válido)
    df_fp = pd.merge_asof(
        df_fp_detected.sort_values("inicio_fp"),
        df_fp_raw_pd[["timestamp", "fp_corrida_ffill"]].sort_values("timestamp"),
        left_on="inicio_fp",
        right_on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(seconds=30)  # AJUSTÁVEL
    )

    df_fp = (
        df_fp
        .rename(columns={"fp_corrida_ffill": "ACI@FP_NUMERO_CORRIDA"})
        .drop(columns=["timestamp"])
    )

    # 6️⃣ Validação correta (somente FP_START)
    mask_inicio = df_fp["inicio_fp"].notna()
    missing = df_fp.loc[mask_inicio, "ACI@FP_NUMERO_CORRIDA"].isna().sum()

    if debug:
        print(f"⚠️ FP_START sem número de corrida associado: {missing}")

    # # opcional: remove FP_START inválidos (recomendado)
    df_fp = df_fp[
        ~(
            df_fp["inicio_fp"].notna() &
            df_fp["ACI@FP_NUMERO_CORRIDA"].isna()
        )
    ]




    df_vd = (
        detect_vd_events_gas_based(
            spark_df_vd_pims=spark_vd_raw,
            start_ts=start_ts,
            end_ts=end_ts,
            initial_heat_number=0,
            **vd_params
        )
        .toPandas()
    )

    df_lc = (
        detect_lc_events(
            spark_df=spark_lc_raw,
            timestamp_col="timestamp",
            peso_col="ACI@LC_TORRE_PESOREAL",
            start_ts=start_ts,
            end_ts=end_ts,
            initial_corrida=0,
            **lc_params
        )
        .rename(columns={
            "start_detected": "inicio_lc",
            "end_detected": "final_lc",
        })
    )

    global DELAY_FP_VD, DELAY_VD_LC
    DELAY_FP_VD = pd.Timedelta(minutes=delay_fp_vd_minutes)
    DELAY_VD_LC = pd.Timedelta(minutes=delay_vd_lc_minutes)

    if debug:
        print(f"⏱️ DELAYS | FP→VD={DELAY_FP_VD} | VD→LC={DELAY_VD_LC}")
        print("▶️ Iniciando simulador causal...")

    return simulate_corridas(
        df_fp=df_fp,
        df_vd=df_vd,
        df_lc=df_lc,
        debug=debug
    )


# COMMAND ----------

df_status = run_corrida_simulator(
    spark_fp_raw=df_fp_raw,
    spark_vd_raw=df_vd_raw,
    spark_lc_raw=df_lc_raw,

    start_ts=START_TS,
    end_ts=END_TS,

    fp_params=fp_params,
    vd_params=vd_params,
    lc_params=lc_params,

    debug=True
)

display(df_status)


# COMMAND ----------

def build_events(df_fp, df_vd, df_lc, debug=False) -> List[Event]:
    events: List[Event] = []

    # =========================
    # FP — EVENTOS (não decide corrida)
    # =========================
    for _, r in df_fp.iterrows():
        if pd.notna(r.get("inicio_fp")):
            events.append(Event(
                ts=pd.to_datetime(r["inicio_fp"]),
                kind="FP_START",
                payload={}
            ))

        if pd.notna(r.get("final_fp")):
            events.append(Event(
                ts=pd.to_datetime(r["final_fp"]),
                kind="FP_END",
                payload={}
            ))

    # =========================
    # VD
    # =========================
    for _, r in df_vd.iterrows():
        tanque = r.get("vd_tanque")

        if pd.notna(r.get("inicio_vd")):
            events.append(Event(
                ts=pd.to_datetime(r["inicio_vd"]),
                kind="VD_START",
                payload={"tanque": tanque}
            ))

        if pd.notna(r.get("final_vd")):
            events.append(Event(
                ts=pd.to_datetime(r["final_vd"]),
                kind="VD_END",
                payload={"tanque": tanque}
            ))

    # =========================
    # LC
    # =========================
    for _, r in df_lc.iterrows():
        if pd.notna(r.get("inicio_lc")):
            events.append(Event(
                ts=pd.to_datetime(r["inicio_lc"]),
                kind="LC_START",
                payload={}
            ))

        if pd.notna(r.get("final_lc")):
            events.append(Event(
                ts=pd.to_datetime(r["final_lc"]),
                kind="LC_END",
                payload={}
            ))

    events.sort(key=lambda e: (e.ts, e.kind))

    if debug:
        print(f"[INIT] {len(events)} eventos carregados")

    return events


# COMMAND ----------

df_test = simulate_corridas_sequencial(
    df_fp=df_fp,
    df_vd=df_vd,
    df_lc=df_lc,
    corrida_inicial=130475,
    debug=True
)

display(df_test)


# COMMAND ----------

df_compare_test = comparacao_operador(
    df_sys_pd=df_test,
    df_op_sp=df_final,
    spark=spark
)

display(df_compare_test.orderBy("corrida"))
