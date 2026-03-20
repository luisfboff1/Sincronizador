# Databricks notebook source
# DBTITLE 1,Imports e Configurações
# ============================================================
# MVP v3 - PIPELINE UNIFICADA COM NOVA LÓGICA VD
# ============================================================
# Descrição: Pipeline completa de detecção de eventos FEA, FP, VD e LC
#            usando sistema de filas com a nova lógica do VD
# Data: 2026-01-28
# ============================================================

# Standard library imports
import math
import heapq
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

# Third party imports
import numpy as np
import pandas as pd

# PySpark imports
import pyspark
import pyspark.sql
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, DoubleType, StringType, TimestampType,
    BooleanType, LongType
)

# Configurações globais
TIMEZONE = "America/Sao_Paulo"

print("✓ Imports carregados")
print(f"✓ Timezone configurado: {TIMEZONE}")

# COMMAND ----------

# DBTITLE 1,Funções Auxiliares
# ============================================================
# FUNÇÕES AUXILIARES (PIMS preprocessing + helpers)
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



print("✓ Funções auxiliares definidas")

# COMMAND ----------

# DBTITLE 1,DETECT_FEA


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

# DBTITLE 1,detectr_lc_new
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
        3. Detecta eventos VD (Spark) → SEM corrida, só timestamps (NOVA LÓGICA)
        4. Detecta eventos LC (Pandas) → SEM corrida, só timestamps (NOVA LÓGICA FSM)
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
    # 3. DETECTAR VD (Spark) → SEM CORRIDA (NOVA LÓGICA)
    # ========================================
    if debug:
        print("\n[3/4] Detectando VD (NOVA LÓGICA)...")
    
    # NOVA FUNÇÃO: detect_vd_events retorna 3 DataFrames
    # df_vd_final: corrida, tank_id, start_vd_time, end_vd_time, duration_seconds, max_volume
    # df_vd_events: corrida, status, tank_id, start_vd_time, end_vd_time, duration_seconds, confirmed_at, event_ts
    # df_vd_debug: debug trace (vazio se debug=False)
    df_vd_final_spark, df_vd_events_spark, df_vd_debug_spark = detect_vd_events(
        spark_df_vd_pims=spark_vd_raw,
        start_ts=start_ts,
        end_ts=end_ts,
        initial_heat_number=0,  # dummy, não usado (VD não cria corrida)
        **vd_params
    )
    
    # Converter para Pandas e renomear colunas para padrão da pipeline
    df_vd = df_vd_final_spark.toPandas()
    if not df_vd.empty:
        # Renomear colunas: start_vd_time → inicio_vd, end_vd_time → final_vd, tank_id → vd_tanque
        df_vd = df_vd.rename(columns={
            'start_vd_time': 'inicio_vd',
            'end_vd_time': 'final_vd',
            'tank_id': 'vd_tanque'  # <<< IMPORTANTE: renomear para vd_tanque
        })
        df_vd = to_pandas_ts(df_vd, ['inicio_vd', 'final_vd'])
        
        # Remover coluna 'corrida' (será atribuída pela simulação de filas)
        if 'corrida' in df_vd.columns:
            df_vd = df_vd.drop(columns=['corrida'])
    
    if debug:
        print(f"   ✓ {len(df_vd)} eventos VD detectados")
        if not df_vd.empty and 'vd_tanque' in df_vd.columns:
            print(f"   Tanques: {df_vd['vd_tanque'].unique().tolist()}")
    
    # ========================================
    # 4. DETECTAR LC (Pandas) → SEM CORRIDA (NOVA LÓGICA FSM)
    # ========================================
    if debug:
        print("\n[4/4] Detectando LC (NOVA LÓGICA FSM)...")
    
    # Pré-processar dados LC (Spark → preprocess → Pandas)
    lc_columns = [col for col in spark_lc_raw.columns if col.startswith("ACI@LC")]
    
    df_lc_processed = preprocess_pims_data(
        spark_df=spark_lc_raw,
        first_timestamp=start_ts,
        last_timestamp=end_ts,
        timestamp_column="timestamp",
        columns_to_fill=lc_columns,
    )
    
    # Converter para Pandas
    df_lc_pandas = df_lc_processed.select(["timestamp"] + lc_columns).toPandas()
    df_lc_pandas["timestamp"] = pd.to_datetime(df_lc_pandas["timestamp"])
    
    # Preparar parâmetros LC (sobrescrever initial_corrida)
    lc_params_run = lc_params.copy()
    lc_params_run['initial_corrida'] = initial_heat
    
    # NOVA FUNÇÃO: detect_lc_events_fsm_pesobra retorna 4 DataFrames
    # df_sys_lc_final: corrida, inicio_lc_sys, final_lc_sys, braco, close_reason
    # df_sys_lc_events: corrida, status, inicio_lc_sys, final_lc_sys, braco, event_ts
    # df_debug: debug trace
    # df_features: features por timestamp
    df_lc_final, df_lc_events, df_lc_debug, df_lc_features = detect_lc_events_fsm_pesobra(
        df=df_lc_pandas,
        **lc_params_run
    )
    
    # Renomear colunas para padrão da pipeline
    df_lc = df_lc_final.copy()
    if not df_lc.empty:
        df_lc = df_lc.rename(columns={
            'inicio_lc_sys': 'inicio_lc',
            'final_lc_sys': 'final_lc',
            'braco': 'lc_braco'
        })
        df_lc = to_pandas_ts(df_lc, ['inicio_lc', 'final_lc'])
        
        # Remover coluna 'corrida' (será atribuída pela simulação de filas)
        if 'corrida' in df_lc.columns:
            df_lc = df_lc.drop(columns=['corrida'])
    
    if debug:
        print(f"   ✓ {len(df_lc)} eventos LC detectados")
        if not df_lc.empty and 'lc_braco' in df_lc.columns:
            print(f"   Braços: {df_lc['lc_braco'].unique().tolist()}")
    
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


print("✓ Função run_unified_detector() atualizada (usando NOVA LÓGICA LC FSM com detect_lc_events_fsm_pesobra)")

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
# PARÂMETROS VD (NOVA LÓGICA - detect_vd_events)
# ============================================================
# A nova função detect_vd_events usa FLOW e VOLUME com confirmação de eventos
# Detecta 4 tipos de eventos: VD_ARRIVAL, DEEP_VAC_START, DEEP_VAC_END, VD_EXIT
vd_params = dict(
    reset_threshold=0.99,                      # threshold para atividade de flow/volume
    flow_activity_threshold=0.2406,            # quantização de flow (elimina ruído)
    min_vd_duration_seconds=60*4,                # duração mínima de sessão VD
    min_vd_arrival_confirm_seconds=20,          # confirmação de chegada (5s)
    min_vd_exit_confirm_seconds=20,             # confirmação de saída (5s)
    min_deep_vac_start_confirm_seconds=20,     # confirmação de início de vácuo profundo (30s)
    min_deep_vac_end_confirm_seconds=20,        # confirmação de fim de vácuo profundo (0s)
    deep_vac_end_pressure_threshold=750.0,     # pressão para fim de vácuo profundo (mbar)
    startup_lookback_seconds=60,              # lookback para evitar edges artificiais (3min)
    startup_grace_seconds=0,                   # grace period após startup
    hold_inactive_seconds_flow=60,              # hold para glitch repair em flow (3s)
    hold_inactive_seconds_volume=30,            # hold para glitch repair em volume (3s)
    debug=False,                               # debug trace
)


# Parâmetros VD legados (função antiga - manter para referência)
vd_params_old = dict(
    reset_threshold=0.99,
    minimal_duration_seconds=10,
    debug=False,
)

# ============================================================
# PARÂMETROS LC (NOVA LÓGICA - detect_lc_events_fsm_pesobra)
# ============================================================
# A nova função usa FSM com dual arms (peso_bra1, peso_bra2)
# Retorna 4 DataFrames: df_sys_lc_final, df_sys_lc_events, df_debug, df_features
lc_params = dict(
    # Configuração - Início

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

    initial_braco=1,

    DEBUG=True,
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
print(f"  - VD: NOVA LÓGICA (detect_vd_events) com {len(vd_params)} parâmetros")
print(f"  - LC: NOVA LÓGICA (detect_lc_events_fsm_pesobra) com {len(lc_params)} parâmetros")

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

# DBTITLE 1,LEITURA DAS TABELAS
# ============================================================
# LEITURA DAS TABELAS (PIMS RAW)
# ============================================================

print("Carregando tabelas...")
print("="*80)

# FEA: Tabela hist_tags IBA (novembro)
df_fea_raw = spark.table("industrial_iatemperaturascha_refined_dev.computed_features.hist_tags_IBA_novembro")
print(f"✓ FEA carregado: {df_fea_raw.count():,} registros")

# FP: Tabela OPC Aciaria FP (ajuste timezone)
df_fp_raw = spark.table("industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_fp") \
    .withColumn("timestamp", F.col("timestamp") - F.expr("INTERVAL 3 HOURS"))
print(f"✓ FP carregado: {df_fp_raw.count():,} registros")

# VD: NOVA TABELA hist_tags_complete (tem todas as colunas necessárias)
print("\nCarregando VD (hist_tags_complete)...")
tb_hist_tags = spark.read.table("industrial_iatemperaturascha_refined_dev.computed_features.hist_tags_complete")

# Preparar DataFrame VD com as colunas necessárias
df_vd_raw = (
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
    
    # Selecionar apenas as colunas necessárias
    .select(
        "timestamp",
        "ACI@VD_VASO1_VOLUMEAR",
        "ACI@VD_VASO1_VOLUMEN2",
        "ACI@VD_VASO1_VZFORTE",
        "ACI@VD_VASO1_VZFRACA",
        "ACI@VD_VASO2_VOLUMEAR",
        "ACI@VD_VASO2_VOLUMEN2",
        "ACI@VD_VASO2_VZFORTE",
        "ACI@VD_VASO2_VZFRACA",
        "ACI@VD_PRESSAO_DO_VACUO"
    )
)
print(f"✓ VD preparado: {df_vd_raw.count():,} registros")
print("  Colunas VD disponíveis:")
for col in df_vd_raw.columns:
    print(f"    - {col}")

# ============================================================
# LC: NOVA TABELA hist_tags_novembro_2025 (PIMS)
# ============================================================
print("\nCarregando LC (hist_tags_novembro_2025)...")
df_lc_pims_raw = spark.table("industrial_iatemperaturascha_refined_dev.computed_features.hist_tags_novembro_2025")
print(f"✓ LC PIMS carregado: {df_lc_pims_raw.count():,} registros")

# Converter coluna Tempo para timestamp
df_lc_raw = df_lc_pims_raw.withColumn(
    "timestamp",
    F.to_timestamp(F.col("Tempo"), "dd-MMM-yy HH:mm:ss.S")
).drop("Tempo")

# Identificar colunas ACI@LC
lc_columns_all = [col for col in df_lc_raw.columns if col.startswith("ACI@LC")]
print(f"  Colunas ACI@LC encontradas: {len(lc_columns_all)}")

# Filtrar colunas com dados não-nulos
lc_columns = []
for col in lc_columns_all:
    non_null_count = df_lc_raw.filter(F.col(col).isNotNull()).count()
    if non_null_count > 0:
        lc_columns.append(col)

print(f"  Colunas ACI@LC válidas (com dados): {len(lc_columns)}")

# Selecionar apenas timestamp + colunas LC válidas
df_lc_raw = df_lc_raw.select(["timestamp"] + lc_columns)

print("\n" + "="*80)
print("✓ Todas as tabelas carregadas com sucesso!")
print("="*80)

# COMMAND ----------

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

# COMMAND ----------





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

# DBTITLE 1,df_final - OP
df_final = df_final.filter(
    F.col("inicio_fp").between(START_TS, END_TS) |
    F.col("inicio_vd").between(START_TS, END_TS) |
    F.col("inicio_lc").between(START_TS, END_TS) |
    F.col("inicio_fea").between(START_TS, END_TS)
)

display(df_final.orderBy(F.desc("corrida")))

# COMMAND ----------

# DBTITLE 1,DF_UNIFIED - SYS
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

# DBTITLE 1,Análise: Corridas Simultâneas em VD e FP
# ============================================================
# ANÁLISE: CORRIDAS SIMULTÂNEAS EM VD E FP
# ============================================================
# Identifica momentos onde existem 2 corridas ao mesmo tempo:
# - VD: Uma no Tanque 1 e outra no Tanque 2
# - FP: Uma no Carro 1 e outra no Carro 2
# ============================================================

import pandas as pd
from datetime import datetime, timedelta

print("\n" + "="*80)
print("🔍 ANÁLISE: CORRIDAS SIMULTÂNEAS EM VD E FP")
print("="*80)

# Converter df_unified para pandas
df_unified_pd = df_unified.copy()

# Garantir que colunas de tempo sejam datetime
time_cols = ['inicio_fp', 'final_fp', 'inicio_vd', 'final_vd']
for col in time_cols:
    if col in df_unified_pd.columns:
        df_unified_pd[col] = pd.to_datetime(df_unified_pd[col], errors='coerce')

# Remover corridas sem dados completos
df_unified_pd = df_unified_pd.dropna(subset=['corrida'])

print(f"\nTotal de corridas analisadas: {len(df_unified_pd)}")
print(f"Período: {df_unified_pd['inicio_fea'].min()} até {df_unified_pd['final_lc'].max()}")

# ============================================================
# FUNÇÃO: DETECTAR SOBREPOSIÇÃO TEMPORAL
# ============================================================

def encontrar_sobreposicoes(df, col_inicio, col_fim, etapa_nome):
    """
    Encontra pares de corridas que se sobrepõem temporalmente em uma etapa.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame com dados das corridas
    col_inicio : str
        Nome da coluna de início da etapa
    col_fim : str
        Nome da coluna de fim da etapa
    etapa_nome : str
        Nome da etapa (para exibição)
    
    Returns:
    --------
    list of dict
        Lista de sobreposições encontradas
    """
    
    sobreposicoes = []
    
    # Filtrar corridas com dados válidos
    df_valido = df[[col_inicio, col_fim, 'corrida']].dropna()
    
    # Comparar cada par de corridas
    for i, row1 in df_valido.iterrows():
        for j, row2 in df_valido.iterrows():
            if row1['corrida'] >= row2['corrida']:  # Evitar duplicatas e auto-comparação
                continue
            
            inicio1 = row1[col_inicio]
            fim1 = row1[col_fim]
            inicio2 = row2[col_inicio]
            fim2 = row2[col_fim]
            
            # Verificar sobreposição temporal
            # Sobreposição ocorre se:
            # - inicio1 < fim2 AND inicio2 < fim1
            if inicio1 < fim2 and inicio2 < fim1:
                # Calcular período de sobreposição
                inicio_overlap = max(inicio1, inicio2)
                fim_overlap = min(fim1, fim2)
                duracao_overlap = (fim_overlap - inicio_overlap).total_seconds() / 60.0  # minutos
                
                sobreposicoes.append({
                    'etapa': etapa_nome,
                    'corrida_1': int(row1['corrida']),
                    'inicio_1': inicio1,
                    'fim_1': fim1,
                    'corrida_2': int(row2['corrida']),
                    'inicio_2': inicio2,
                    'fim_2': fim2,
                    'inicio_overlap': inicio_overlap,
                    'fim_overlap': fim_overlap,
                    'duracao_overlap_min': round(duracao_overlap, 2)
                })
    
    return sobreposicoes

# ============================================================
# ANÁLISE: VD (VÁCUO DESGASEIFICAÇÃO)
# ============================================================

print("\n" + "-"*80)
print("🟢 ANALISANDO VD (Vácuo Desgaseificação) - Tanque 1 e Tanque 2")
print("-"*80)

sobreposicoes_vd = encontrar_sobreposicoes(
    df=df_unified_pd,
    col_inicio='inicio_vd',
    col_fim='final_vd',
    etapa_nome='VD'
)

if sobreposicoes_vd:
    print(f"\n✅ ENCONTRADAS {len(sobreposicoes_vd)} SOBREPOSIÇÕES NO VD!")
    print("\nIsso significa que houve momentos com corridas simultâneas:")
    print("  - Uma corrida no Tanque 1")
    print("  - Outra corrida no Tanque 2")
    print("\n" + "="*80)
    
    # Criar DataFrame para melhor visualização
    df_vd_overlap = pd.DataFrame(sobreposicoes_vd)
    
    # Mostrar resumo
    print("\n📊 RESUMO DAS SOBREPOSIÇÕES NO VD:")
    print(f"  Total de sobreposições: {len(df_vd_overlap)}")
    print(f"  Duração média de sobreposição: {df_vd_overlap['duracao_overlap_min'].mean():.2f} minutos")
    print(f"  Duração máxima de sobreposição: {df_vd_overlap['duracao_overlap_min'].max():.2f} minutos")
    print(f"  Duração mínima de sobreposição: {df_vd_overlap['duracao_overlap_min'].min():.2f} minutos")
    
    # Mostrar detalhes
    print("\n📋 DETALHES DAS SOBREPOSIÇÕES NO VD:")
    print("="*80)
    
    for idx, overlap in df_vd_overlap.iterrows():
        print(f"\n🔹 Sobreposição #{idx + 1}:")
        print(f"  Corrida {overlap['corrida_1']}: {overlap['inicio_1'].strftime('%d/%m %H:%M')} → {overlap['fim_1'].strftime('%d/%m %H:%M')}")
        print(f"  Corrida {overlap['corrida_2']}: {overlap['inicio_2'].strftime('%d/%m %H:%M')} → {overlap['fim_2'].strftime('%d/%m %H:%M')}")
        print(f"  ⏱️  Período de sobreposição: {overlap['inicio_overlap'].strftime('%d/%m %H:%M')} → {overlap['fim_overlap'].strftime('%d/%m %H:%M')}")
        print(f"  ⏳ Duração: {overlap['duracao_overlap_min']:.2f} minutos")
        print("-"*80)
    
    # Exibir tabela completa
    print("\n📊 TABELA COMPLETA - SOBREPOSIÇÕES NO VD:")
    display(df_vd_overlap[['corrida_1', 'corrida_2', 'inicio_overlap', 'fim_overlap', 'duracao_overlap_min']])
    
else:
    print("\n⚠️ NÃO foram encontradas sobreposições no VD.")
    print("Isso significa que nunca houve 2 corridas simultâneas no VD (Tanque 1 e 2).")

# ============================================================
# ANÁLISE: FP (FORNO PANELA)
# ============================================================

print("\n" + "-"*80)
print("🔵 ANALISANDO FP (Forno Panela) - Carro 1 e Carro 2")
print("-"*80)

sobreposicoes_fp = encontrar_sobreposicoes(
    df=df_unified_pd,
    col_inicio='inicio_fp',
    col_fim='final_fp',
    etapa_nome='FP'
)

if sobreposicoes_fp:
    print(f"\n✅ ENCONTRADAS {len(sobreposicoes_fp)} SOBREPOSIÇÕES NO FP!")
    print("\nIsso significa que houve momentos com corridas simultâneas:")
    print("  - Uma corrida no Carro 1")
    print("  - Outra corrida no Carro 2")
    print("\n" + "="*80)
    
    # Criar DataFrame para melhor visualização
    df_fp_overlap = pd.DataFrame(sobreposicoes_fp)
    
    # Mostrar resumo
    print("\n📊 RESUMO DAS SOBREPOSIÇÕES NO FP:")
    print(f"  Total de sobreposições: {len(df_fp_overlap)}")
    print(f"  Duração média de sobreposição: {df_fp_overlap['duracao_overlap_min'].mean():.2f} minutos")
    print(f"  Duração máxima de sobreposição: {df_fp_overlap['duracao_overlap_min'].max():.2f} minutos")
    print(f"  Duração mínima de sobreposição: {df_fp_overlap['duracao_overlap_min'].min():.2f} minutos")
    
    # Mostrar detalhes
    print("\n📋 DETALHES DAS SOBREPOSIÇÕES NO FP:")
    print("="*80)
    
    for idx, overlap in df_fp_overlap.iterrows():
        print(f"\n🔹 Sobreposição #{idx + 1}:")
        print(f"  Corrida {overlap['corrida_1']}: {overlap['inicio_1'].strftime('%d/%m %H:%M')} → {overlap['fim_1'].strftime('%d/%m %H:%M')}")
        print(f"  Corrida {overlap['corrida_2']}: {overlap['inicio_2'].strftime('%d/%m %H:%M')} → {overlap['fim_2'].strftime('%d/%m %H:%M')}")
        print(f"  ⏱️  Período de sobreposição: {overlap['inicio_overlap'].strftime('%d/%m %H:%M')} → {overlap['fim_overlap'].strftime('%d/%m %H:%M')}")
        print(f"  ⏳ Duração: {overlap['duracao_overlap_min']:.2f} minutos")
        print("-"*80)
    
    # Exibir tabela completa
    print("\n📊 TABELA COMPLETA - SOBREPOSIÇÕES NO FP:")
    display(df_fp_overlap[['corrida_1', 'corrida_2', 'inicio_overlap', 'fim_overlap', 'duracao_overlap_min']])
    
else:
    print("\n⚠️ NÃO foram encontradas sobreposições no FP.")
    print("Isso significa que nunca houve 2 corridas simultâneas no FP (Carro 1 e 2).")

# ============================================================
# RESUMO FINAL
# ============================================================

print("\n" + "="*80)
print("📊 RESUMO FINAL DA ANÁLISE")
print("="*80)
print(f"\n🟢 VD (Vácuo Desgaseificação):")
print(f"  Sobreposições encontradas: {len(sobreposicoes_vd)}")
if sobreposicoes_vd:
    print(f"  Tempo total de operação simultânea: {sum([s['duracao_overlap_min'] for s in sobreposicoes_vd]):.2f} minutos")

print(f"\n🔵 FP (Forno Panela):")
print(f"  Sobreposições encontradas: {len(sobreposicoes_fp)}")
if sobreposicoes_fp:
    print(f"  Tempo total de operação simultânea: {sum([s['duracao_overlap_min'] for s in sobreposicoes_fp]):.2f} minutos")

print("\n" + "="*80)
print("✅ ANÁLISE CONCLUÍDA!")
print("="*80)

# Armazenar resultados para uso posterior
if sobreposicoes_vd:
    df_sobreposicoes_vd = pd.DataFrame(sobreposicoes_vd)
else:
    df_sobreposicoes_vd = pd.DataFrame()

if sobreposicoes_fp:
    df_sobreposicoes_fp = pd.DataFrame(sobreposicoes_fp)
else:
    df_sobreposicoes_fp = pd.DataFrame()

print("\n💾 Resultados armazenados em:")
print("  - df_sobreposicoes_vd: Sobreposições no VD")
print("  - df_sobreposicoes_fp: Sobreposições no FP")

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

# DBTITLE 1,COMP OP - YS
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

# DBTITLE 1,Função: Snapshot de Estados das Corridas
import pandas as pd
from datetime import datetime
from typing import Union

def snapshot_corridas(
    timestamp: Union[str, datetime],
    df_data: pd.DataFrame,
    fonte: str = "Sistema"
) -> pd.DataFrame:
    """
    Cria um snapshot mostrando onde cada corrida está em um instante específico.
    
    Parameters:
    -----------
    timestamp : str ou datetime
        Momento para tirar o snapshot (ex: '2024-01-15 12:30:00')
    df_data : pd.DataFrame
        DataFrame com colunas: corrida, inicio_fea, final_fea, inicio_fp, final_fp,
                               inicio_vd, final_vd, inicio_lc, final_lc
    fonte : str
        Nome da fonte dos dados ("Sistema" ou "Operador")
    
    Returns:
    --------
    pd.DataFrame com uma linha mostrando o estado de cada corrida no timestamp
    """
    
    # Converter timestamp para datetime se necessário
    if isinstance(timestamp, str):
        ts = pd.to_datetime(timestamp)
    else:
        ts = timestamp
    
    # Garantir que todas as colunas de tempo sejam datetime
    df = df_data.copy()
    time_cols = ['inicio_fea', 'final_fea', 'inicio_fp', 'final_fp', 
                 'inicio_vd', 'final_vd', 'inicio_lc', 'final_lc']
    
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Dicionário para armazenar os resultados
    resultado = {
        'Hora Considerada': ts.strftime('%d/%m/%Y %H:%M:%S'),
        'Fonte': fonte
    }
    
    # Estados possíveis
    estados = [
        ('Aguardando FEA', None, 'inicio_fea'),
        ('IN FEA', 'inicio_fea', 'final_fea'),
        ('Transporte FEA→FP', 'final_fea', 'inicio_fp'),
        ('IN FP', 'inicio_fp', 'final_fp'),
        ('Transporte FP→VD', 'final_fp', 'inicio_vd'),
        ('IN VD', 'inicio_vd', 'final_vd'),
        ('Transporte VD→LC', 'final_vd', 'inicio_lc'),
        ('IN LC', 'inicio_lc', 'final_lc'),
        ('Concluído', 'final_lc', None)
    ]
    
    # Para cada estado, encontrar qual corrida está nele
    for estado_nome, col_inicio, col_fim in estados:
        corrida_encontrada = None
        
        for _, row in df.iterrows():
            corrida = row['corrida']
            
            # Aguardando FEA: próxima corrida que ainda não iniciou FEA
            if estado_nome == 'Aguardando FEA':
                if pd.notna(row[col_fim]) and row[col_fim] > ts:
                    if corrida_encontrada is None or corrida < corrida_encontrada:
                        corrida_encontrada = corrida
            
            # Concluído: terminou LC antes do timestamp
            elif estado_nome == 'Concluído':
                if pd.notna(row[col_inicio]) and row[col_inicio] <= ts:
                    if corrida_encontrada is None or corrida > corrida_encontrada:
                        corrida_encontrada = corrida
            
            # Estados intermediários: iniciou mas não terminou
            elif col_fim is not None:
                inicio_ok = pd.notna(row[col_inicio]) and row[col_inicio] <= ts
                fim_ok = pd.isna(row[col_fim]) or row[col_fim] > ts
                
                if inicio_ok and fim_ok:
                    corrida_encontrada = corrida
                    break
        
        resultado[estado_nome] = corrida_encontrada if corrida_encontrada is not None else '-'
    
    return pd.DataFrame([resultado])


print("✓ Função snapshot_corridas() criada com sucesso!")
print("\nExemplo de uso:")
print("  snapshot_corridas('2024-01-15 12:30:00', df_unified, fonte='Sistema')")
print("  snapshot_corridas('2024-01-15 12:30:00', df_final.toPandas(), fonte='Operador')")

# COMMAND ----------

# DBTITLE 1,Função: Snapshot Detalhado com FP (Carro 1/2) e VD (Tanque 1/2)
def snapshot_corridas_detalhado(
    timestamp: Union[str, datetime],
    df_data: pd.DataFrame,
    fonte: str = "Sistema"
) -> pd.DataFrame:
    """
    Cria um snapshot DETALHADO mostrando onde cada corrida está,
    incluindo separação de FP (Carro 1/2) e VD (Tanque 1/2).
    
    Parameters:
    -----------
    timestamp : str ou datetime
        Momento para tirar o snapshot
    df_data : pd.DataFrame
        DataFrame com as colunas de tempo das corridas
    fonte : str
        Nome da fonte dos dados
    
    Returns:
    --------
    pd.DataFrame com uma linha mostrando o estado detalhado de cada corrida
    """
    
    # Converter timestamp
    if isinstance(timestamp, str):
        ts = pd.to_datetime(timestamp)
    else:
        ts = timestamp
    
    # Preparar dados
    df = df_data.copy()
    time_cols = ['inicio_fea', 'final_fea', 'inicio_fp', 'final_fp', 
                 'inicio_vd', 'final_vd', 'inicio_lc', 'final_lc']
    
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Ordenar por corrida
    df = df.sort_values('corrida')
    
    # Função auxiliar para encontrar corrida em estado
    def encontrar_corrida(col_inicio, col_fim, todas=False):
        corridas = []
        for _, row in df.iterrows():
            if col_inicio is None:  # Aguardando FEA
                if pd.notna(row[col_fim]) and row[col_fim] > ts:
                    corridas.append(row['corrida'])
            elif col_fim is None:  # Concluído
                if pd.notna(row[col_inicio]) and row[col_inicio] <= ts:
                    corridas.append(row['corrida'])
            else:  # Estado intermediário
                inicio_ok = pd.notna(row[col_inicio]) and row[col_inicio] <= ts
                fim_ok = pd.isna(row[col_fim]) or row[col_fim] > ts
                if inicio_ok and fim_ok:
                    corridas.append(row['corrida'])
        
        if todas:
            return corridas if corridas else ['-']
        else:
            return corridas[0] if corridas else '-'
    
    # Construir resultado detalhado
    resultado = {
        'Hora Considerada': ts.strftime('%d/%m/%Y %H:%M:%S'),
        'Fonte': fonte,
        
        # FEA
        'Aguardando FEA': encontrar_corrida(None, 'inicio_fea'),
        'IN FEA': encontrar_corrida('inicio_fea', 'final_fea'),
        'Transporte FEA→FP': encontrar_corrida('final_fea', 'inicio_fp'),
        
        # FP - Dividir em Carro 1 e Carro 2
        'IN FP - Carro 1': '-',  # Placeholder - lógica simplificada
        'IN FP - Carro 2': '-',  # Placeholder - lógica simplificada
        # 'IN FP (Geral)': encontrar_corrida('inicio_fp', 'final_fp'),
        'Transporte FP→VD': encontrar_corrida('final_fp', 'inicio_vd'),
        
        # VD - Dividir em Tanque 1 e Tanque 2
        'IN VD - Tanque 1': '-',  # Placeholder - lógica simplificada
        'IN VD - Tanque 2': '-',  # Placeholder - lógica simplificada
        # 'IN VD (Geral)': encontrar_corrida('inicio_vd', 'final_vd'),
        'Transporte VD→LC': encontrar_corrida('final_vd', 'inicio_lc'),
        
        # LC
        'IN LC': encontrar_corrida('inicio_lc', 'final_lc'),
        'Concluído': max(encontrar_corrida('final_lc', None, todas=True)) if encontrar_corrida('final_lc', None, todas=True) != ['-'] else '-',
    }
    
    # Lógica para FP Carro 1/2 e VD Tanque 1/2
    # Assumindo que corridas ímpares vão para Carro 1/Tanque 1 e pares para Carro 2/Tanque 2
    corridas_fp = encontrar_corrida('inicio_fp', 'final_fp', todas=True)
    if corridas_fp != ['-']:
        for corrida in corridas_fp:
            if corrida % 2 == 1:  # Ímpar -> Carro 1
                resultado['IN FP - Carro 1'] = corrida
            else:  # Par -> Carro 2
                resultado['IN FP - Carro 2'] = corrida
    
    corridas_vd = encontrar_corrida('inicio_vd', 'final_vd', todas=True)
    if corridas_vd != ['-']:
        for corrida in corridas_vd:
            if corrida % 2 == 1:  # Ímpar -> Tanque 1
                resultado['IN VD - Tanque 1'] = corrida
            else:  # Par -> Tanque 2
                resultado['IN VD - Tanque 2'] = corrida
    
    return pd.DataFrame([resultado])


print("✓ Função snapshot_corridas_detalhado() criada com sucesso!")
print("\nEsta versão inclui:")
print("  - FP dividido em Carro 1 e Carro 2")
print("  - VD dividido em Tanque 1 e Tanque 2")
print("  - Lógica: corridas ímpares → Carro/Tanque 1, pares → Carro/Tanque 2")

# COMMAND ----------

# DBTITLE 1,Visualização: Pipeline Visual das Corridas (Dashboard)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def visualizar_pipeline_corridas(
    snapshot_df: pd.DataFrame,
    titulo: str = "Pipeline de Corridas",
    figsize: tuple = (20, 8)
) -> plt.Figure:
    """
    Cria uma visualização tipo pipeline/fluxograma mostrando onde cada corrida está.
    
    Parameters:
    -----------
    snapshot_df : pd.DataFrame
        DataFrame retornado pela função snapshot_corridas_detalhado()
    titulo : str
        Título do gráfico
    figsize : tuple
        Tamanho da figura
    
    Returns:
    --------
    plt.Figure
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Extrair dados do snapshot
    row = snapshot_df.iloc[0]
    hora = row['Hora Considerada']
    fonte = row['Fonte']
    
    # Definir etapas e posições
    etapas_config = [
        # (nome_coluna, label_display, x, y, cor)
        ('Aguardando FEA', 'Aguardando\nFEA', 0.5, 4.5, '#FFE5B4'),  # Pêssego claro
        ('IN FEA', 'IN FEA', 0.5, 3.0, '#FF6B6B'),  # Vermelho
        ('Transporte FEA→FP', 'Transporte\nFEA→FP', 2.0, 3.75, '#E8E8E8'),  # Cinza claro
        ('IN FP - Carro 1', 'FP\nCarro 1', 3.5, 4.5, '#4ECDC4'),  # Turquesa
        ('IN FP - Carro 2', 'FP\nCarro 2', 3.5, 3.0, '#4ECDC4'),  # Turquesa
        ('Transporte FP→VD', 'Transporte\nFP→VD', 5.0, 3.75, '#E8E8E8'),  # Cinza claro
        ('IN VD - Tanque 1', 'VD\nTanque 1', 6.5, 4.5, '#95E1D3'),  # Verde água
        ('IN VD - Tanque 2', 'VD\nTanque 2', 6.5, 3.0, '#95E1D3'),  # Verde água
        ('Transporte VD→LC', 'Transporte\nVD→LC', 8.0, 3.75, '#E8E8E8'),  # Cinza claro
        ('IN LC', 'IN LC', 9.0, 3.75, '#F38181'),  # Rosa
        ('Concluído', 'Concluído', 9.0, 1.5, '#A8E6CF'),  # Verde claro
    ]
    
    # Desenhar caixas para cada etapa
    for col_name, label, x, y, cor in etapas_config:
        if col_name in row:
            corrida = row[col_name]
            
            # Texto da corrida
            if corrida == '-' or pd.isna(corrida):
                corrida_text = '-'
                alpha = 0.3
            else:
                corrida_text = f'{int(corrida)}'
                alpha = 1.0
            
            # Desenhar caixa
            box = FancyBboxPatch(
                (x - 0.4, y - 0.4), 0.8, 0.8,
                boxstyle="round,pad=0.05",
                edgecolor='black',
                facecolor=cor,
                alpha=alpha,
                linewidth=2
            )
            ax.add_patch(box)
            
            # Label da etapa
            ax.text(x, y + 0.55, label, 
                   ha='center', va='bottom', 
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Número da corrida
            ax.text(x, y, corrida_text,
                   ha='center', va='center',
                   fontsize=16 if corrida != '-' else 12,
                   fontweight='bold' if corrida != '-' else 'normal',
                   color='black' if corrida != '-' else 'gray')
    
    # Desenhar setas de fluxo - CORRIGIDO
    setas = [
        # Fluxo correto:
        # 1. Aguardando FEA -> IN FEA
        (0.9, 4.5, 0.5, 3.4),
        
        # 2. IN FEA -> Transporte FEA-FP
        (0.9, 3.0, 1.6, 3.75),
        
        # 3. Transporte FEA-FP -> FP Carro 1
        (2.4, 3.75, 3.1, 4.5),
        
        # 4. Transporte FEA-FP -> FP Carro 2
        (2.4, 3.75, 3.1, 3.0),
        
        # 5. FP Carro 1 -> Transporte FP-VD
        (3.9, 4.5, 4.6, 3.75),
        
        # 6. FP Carro 2 -> Transporte FP-VD
        (3.9, 3.0, 4.6, 3.75),
        
        # 7. Transporte FP-VD -> VD Tanque 1
        (5.4, 3.75, 6.1, 4.5),
        
        # 8. Transporte FP-VD -> VD Tanque 2
        (5.4, 3.75, 6.1, 3.0),
        
        # 9. VD Tanque 1 -> Transporte VD-LC
        (6.9, 4.5, 7.6, 3.75),
        
        # 10. VD Tanque 2 -> Transporte VD-LC
        (6.9, 3.0, 7.6, 3.75),
        
        # 11. Transporte VD-LC -> IN LC
        (8.4, 3.75, 8.6, 3.75),
        
        # 12. IN LC -> Concluído
        (9.0, 3.35, 9.0, 1.9),
    ]
    
    for x1, y1, x2, y2 in setas:
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->,head_width=0.3,head_length=0.3',
            color='gray',
            linewidth=1.5,
            alpha=0.5,
            connectionstyle="arc3,rad=0.1"
        )
        ax.add_patch(arrow)
    
    # Título e informações
    fig.suptitle(titulo, fontsize=18, fontweight='bold', y=0.98)
    ax.text(5, 0.3, f'Fonte: {fonte} | Horário: {hora}',
           ha='center', va='center', fontsize=12,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    return fig


print("✓ Função visualizar_pipeline_corridas() criada com sucesso!")
print("\nEsta função cria um diagrama visual tipo pipeline/fluxograma.")
print("Ideal para dashboards e apresentações!")

# COMMAND ----------

# DBTITLE 1,📊 DASHBOARD: Pipeline Interativo com Widgets
# ============================================================
# DASHBOARD INTERATIVO: PIPELINE DE CORRIDAS
# ============================================================
# Esta célula usa os widgets para gerar visualizações dinâmicas
# Ideal para modo Dashboard!
# ============================================================

import pandas as pd
from datetime import datetime

print("\n" + "="*80)
print("📊 DASHBOARD INTERATIVO: PIPELINE DE CORRIDAS")
print("="*80)

# ============================================================
# LER VALORES DOS WIDGETS
# ============================================================
try:
    # Ler parâmetros dos widgets
    data_snapshot = dbutils.widgets.get("snapshot_data")
    hora_snapshot = dbutils.widgets.get("snapshot_hora")
    minuto_snapshot = dbutils.widgets.get("snapshot_minuto")
    fonte_selecionada = dbutils.widgets.get("fonte_dados")
    
    # Construir timestamp completo
    timestamp_str = f"{data_snapshot} {hora_snapshot}:{minuto_snapshot}:00"
    timestamp_snapshot = pd.to_datetime(timestamp_str)
    
    print(f"\n🕹️ Parâmetros selecionados:")
    print(f"  Data: {data_snapshot}")
    print(f"  Hora: {hora_snapshot}:{minuto_snapshot}")
    print(f"  Fonte: {fonte_selecionada}")
    print(f"  Timestamp completo: {timestamp_snapshot}")
    
except Exception as e:
    # Se widgets não existirem (modo notebook normal), usar valores padrão
    print("⚠️ Widgets não encontrados. Usando valores padrão.")
    timestamp_snapshot = HORARIO_SNAPSHOT
    fonte_selecionada = "Ambos"
    print(f"  Timestamp padrão: {timestamp_snapshot}")

print("\n" + "-"*80)

# ============================================================
# GERAR SNAPSHOTS
# ============================================================

# Preparar dados do operador
df_final_pd = df_final.toPandas()

# Gerar snapshot do Sistema
snapshot_sys = snapshot_corridas_detalhado(
    timestamp=timestamp_snapshot,
    df_data=df_unified,
    fonte="Sistema"
)

# Gerar snapshot do Operador
snapshot_op = snapshot_corridas_detalhado(
    timestamp=timestamp_snapshot,
    df_data=df_final_pd,
    fonte="Operador"
)

print(f"✓ Snapshots gerados para {timestamp_snapshot}")
print("-"*80)

# ============================================================
# VISUALIZAR CONFORME FONTE SELECIONADA
# ============================================================

if fonte_selecionada == "Sistema":
    print("\n🔵 Exibindo: SISTEMA")
    print("="*80)
    
    # Mostrar tabela
    display(snapshot_sys)
    
    # Mostrar pipeline visual
    fig = visualizar_pipeline_corridas(
        snapshot_df=snapshot_sys,
        titulo=f"Pipeline de Corridas - SISTEMA | {timestamp_snapshot.strftime('%d/%m/%Y %H:%M')}",
        figsize=(20, 8)
    )
    plt.show()
    
elif fonte_selecionada == "Operador":
    print("\n🟢 Exibindo: OPERADOR")
    print("="*80)
    
    # Mostrar tabela
    display(snapshot_op)
    
    # Mostrar pipeline visual
    fig = visualizar_pipeline_corridas(
        snapshot_df=snapshot_op,
        titulo=f"Pipeline de Corridas - OPERADOR | {timestamp_snapshot.strftime('%d/%m/%Y %H:%M')}",
        figsize=(20, 8)
    )
    plt.show()
    
else:  # "Ambos"
    print("\n🔄 Exibindo: COMPARAÇÃO SISTEMA vs OPERADOR")
    print("="*80)
    
    # Mostrar tabelas lado a lado
    print("\n🔵 SISTEMA:")
    display(snapshot_sys)
    
    print("\n🟢 OPERADOR:")
    display(snapshot_op)
    
    # Mostrar pipeline comparativo
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14))
    
    # Função auxiliar para desenhar
    def desenhar_pipeline_em_eixo(ax, snapshot_df, titulo):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')
        
        row = snapshot_df.iloc[0]
        hora = row['Hora Considerada']
        fonte = row['Fonte']
        
        etapas_config = [
            ('Aguardando FEA', 'Aguardando\nFEA', 0.5, 4.5, '#FFE5B4'),
            ('IN FEA', 'IN FEA', 0.5, 3.0, '#FF6B6B'),
            ('Transporte FEA→FP', 'Transporte\nFEA→FP', 2.0, 3.75, '#E8E8E8'),
            ('IN FP - Carro 1', 'FP\nCarro 1', 3.5, 4.5, '#4ECDC4'),
            ('IN FP - Carro 2', 'FP\nCarro 2', 3.5, 3.0, '#4ECDC4'),
            ('Transporte FP→VD', 'Transporte\nFP→VD', 5.0, 3.75, '#E8E8E8'),
            ('IN VD - Tanque 1', 'VD\nTanque 1', 6.5, 4.5, '#95E1D3'),
            ('IN VD - Tanque 2', 'VD\nTanque 2', 6.5, 3.0, '#95E1D3'),
            ('Transporte VD→LC', 'Transporte\nVD→LC', 8.0, 3.75, '#E8E8E8'),
            ('IN LC', 'IN LC', 9.0, 3.75, '#F38181'),
            ('Concluído', 'Concluído', 9.0, 1.5, '#A8E6CF'),
        ]
        
        for col_name, label, x, y, cor in etapas_config:
            if col_name in row:
                corrida = row[col_name]
                
                if corrida == '-' or pd.isna(corrida):
                    corrida_text = '-'
                    alpha = 0.3
                else:
                    corrida_text = f'{int(corrida)}'
                    alpha = 1.0
                
                box = FancyBboxPatch(
                    (x - 0.4, y - 0.4), 0.8, 0.8,
                    boxstyle="round,pad=0.05",
                    edgecolor='black',
                    facecolor=cor,
                    alpha=alpha,
                    linewidth=2
                )
                ax.add_patch(box)
                
                ax.text(x, y + 0.55, label, 
                       ha='center', va='bottom', 
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                ax.text(x, y, corrida_text,
                       ha='center', va='center',
                       fontsize=16 if corrida != '-' else 12,
                       fontweight='bold' if corrida != '-' else 'normal',
                       color='black' if corrida != '-' else 'gray')
        
        # Setas
        setas = [
            (0.9, 4.5, 0.5, 3.4), (0.9, 3.0, 1.6, 3.75), (2.4, 3.75, 3.1, 4.5),
            (2.4, 3.75, 3.1, 3.0), (3.9, 4.5, 4.6, 3.75), (3.9, 3.0, 4.6, 3.75),
            (5.4, 3.75, 6.1, 4.5), (5.4, 3.75, 6.1, 3.0), (6.9, 4.5, 7.6, 3.75),
            (6.9, 3.0, 7.6, 3.75), (8.4, 3.75, 8.6, 3.75), (9.0, 3.35, 9.0, 1.9),
        ]
        
        for x1, y1, x2, y2 in setas:
            arrow = FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle='->,head_width=0.3,head_length=0.3',
                color='gray',
                linewidth=1.5,
                alpha=0.5,
                connectionstyle="arc3,rad=0.1"
            )
            ax.add_patch(arrow)
        
        ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20)
        ax.text(5, 0.3, f'Fonte: {fonte} | Horário: {hora}',
               ha='center', va='center', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    # Desenhar ambos os pipelines
    desenhar_pipeline_em_eixo(ax1, snapshot_sys, f"Pipeline - SISTEMA | {timestamp_snapshot.strftime('%d/%m/%Y %H:%M')}")
    desenhar_pipeline_em_eixo(ax2, snapshot_op, f"Pipeline - OPERADOR | {timestamp_snapshot.strftime('%d/%m/%Y %H:%M')}")
    
    plt.tight_layout()
    plt.show()

print("\n" + "="*80)
print("✅ DASHBOARD ATUALIZADO COM SUCESSO!")
print("="*80)
print("\n💡 Dica: Altere os widgets acima para atualizar o dashboard em tempo real!")
print("\n📊 LEGENDA:")
print("  🔴 Vermelho = FEA (Forno Elétrico a Arco)")
print("  🔵 Turquesa = FP (Forno Panela) - Carro 1 e 2")
print("  🟢 Verde Água = VD (Vácuo Desgaseificação) - Tanque 1 e 2")
print("  🔴 Rosa = LC (Lingotamento Contínuo)")
print("  ⚪ Cinza = Transporte entre etapas")
print("  🟢 Verde Claro = Concluído")

# COMMAND ----------

# DBTITLE 1,Parâmetros de Seleção de Corridas
# ============================================================
# PARÂMETROS PARA VISUALIZAÇÃO DE EVOLUÇÃO DAS CORRIDAS
# ============================================================

# Obter range de corridas disponíveis
corridas_disponiveis = df_final.select("corrida").distinct().orderBy(F.desc("corrida")).collect()
corridas_list = [row.corrida for row in corridas_disponiveis if row.corrida is not None]

if len(corridas_list) > 0:
    CORRIDA_INICIAL_VIZ = corridas_list[min(4, len(corridas_list)-1)]  # 5ª corrida mais recente
    CORRIDA_FINAL_VIZ = corridas_list[0]  # Corrida mais recente
    print(f"✓ Range de corridas disponíveis: {corridas_list[-1]} até {corridas_list[0]}")
    print(f"✓ Seleção padrão: {CORRIDA_INICIAL_VIZ} até {CORRIDA_FINAL_VIZ} ({min(5, len(corridas_list))} corridas)")
else:
    print("⚠️ Nenhuma corrida disponível para visualização")
    CORRIDA_INICIAL_VIZ = None
    CORRIDA_FINAL_VIZ = None

# COMMAND ----------

# DBTITLE 1,Preparar Dados para Visualização
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# ============================================================
# PREPARAR DADOS DO OPERADOR (df_final)
# ============================================================

df_op_filtered = df_final.filter(
    (F.col("corrida") >= CORRIDA_INICIAL_VIZ) & 
    (F.col("corrida") <= CORRIDA_FINAL_VIZ)
).orderBy("corrida")

df_op_pd = df_op_filtered.toPandas()

# ============================================================
# PREPARAR DADOS DO SISTEMA (df_unified)
# ============================================================

df_sys_filtered = df_unified[
    (df_unified["corrida"] >= CORRIDA_INICIAL_VIZ) & 
    (df_unified["corrida"] <= CORRIDA_FINAL_VIZ)
].copy()

df_sys_filtered = df_sys_filtered.sort_values("corrida")

print(f"✓ Dados preparados:")
print(f"  - Operador: {len(df_op_pd)} corridas")
print(f"  - Sistema: {len(df_sys_filtered)} corridas")
print(f"\nCorridas selecionadas: {sorted(df_op_pd['corrida'].unique())}")

# COMMAND ----------

# DBTITLE 1,Visualização: Evolução das Corridas por Etapa
# ============================================================
# VISUALIZAÇÃO: EVOLUÇÃO DAS CORRIDAS ATRAVÉS DAS ETAPAS
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

# Definir todas as etapas detalhadas
etapas_detalhadas = [
    'inicio_fea', 'final_fea',
    'inicio_fp', 'final_fp',
    'inicio_vd', 'final_vd',
    'inicio_lc', 'final_lc'
]

etapas_pos = {etapa: i for i, etapa in enumerate(etapas_detalhadas)}

# Labels para o eixo X
etapas_labels = [
    'Início\nFEA', 'Final\nFEA',
    'Início\nFP', 'Final\nFP',
    'Início\nVD', 'Final\nVD',
    'Início\nLC', 'Final\nLC'
]

# Cores para cada corrida
colors = plt.cm.tab10(range(10))

# ============================================================
# GRÁFICO 1: OPERADOR
# ============================================================
ax1.set_title('EVOLUÇÃO DAS CORRIDAS - OPERADOR', fontsize=14, fontweight='bold', pad=20)

for idx, (_, row) in enumerate(df_op_pd.iterrows()):
    corrida = row['corrida']
    color = colors[idx % 10]
    
    # Coletar todos os pontos da corrida em ordem
    x_points = []
    y_points = []
    
    for etapa_col in etapas_detalhadas:
        if etapa_col in row and pd.notna(row[etapa_col]):
            x_pos = etapas_pos[etapa_col]
            x_points.append(x_pos)
            y_points.append(row[etapa_col])
    
    # Plotar linha conectando todos os pontos
    if len(x_points) > 0:
        ax1.plot(x_points, y_points, color=color, linewidth=2, alpha=0.7, 
                marker='o', markersize=8, label=f'Corrida {corrida}')

ax1.set_xticks(range(len(etapas_detalhadas)))
ax1.set_xticklabels(etapas_labels, fontsize=9)
ax1.set_xlabel('Etapa', fontsize=12, fontweight='bold')
ax1.set_ylabel('Tempo', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.legend(loc='upper left', fontsize=9)

# Formatação do eixo Y (tempo)
ax1.yaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
ax1.yaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.setp(ax1.yaxis.get_majorticklabels(), rotation=0)

# ============================================================
# GRÁFICO 2: SISTEMA
# ============================================================
ax2.set_title('EVOLUÇÃO DAS CORRIDAS - SISTEMA', fontsize=14, fontweight='bold', pad=20)

for idx, (_, row) in enumerate(df_sys_filtered.iterrows()):
    corrida = row['corrida']
    color = colors[idx % 10]
    
    # Coletar todos os pontos da corrida em ordem
    x_points = []
    y_points = []
    
    for etapa_col in etapas_detalhadas:
        if etapa_col in row and pd.notna(row[etapa_col]):
            x_pos = etapas_pos[etapa_col]
            x_points.append(x_pos)
            y_points.append(row[etapa_col])
    
    # Plotar linha conectando todos os pontos
    if len(x_points) > 0:
        ax2.plot(x_points, y_points, color=color, linewidth=2, alpha=0.7,
                marker='o', markersize=8, label=f'Corrida {corrida}')

ax2.set_xticks(range(len(etapas_detalhadas)))
ax2.set_xticklabels(etapas_labels, fontsize=9)
ax2.set_xlabel('Etapa', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend(loc='upper left', fontsize=9)

# Formatação do eixo Y (tempo)
ax2.yaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
ax2.yaxis.set_major_locator(mdates.HourLocator(interval=2))

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("📊 LEGENDA:")
print("  ● Pontos = Timestamps de início e fim de cada etapa")
print("  ─ Linha contínua = Evolução completa da corrida")
print("  Sequência: inicio_fea → final_fea → inicio_fp → final_fp → inicio_vd → final_vd → inicio_lc → final_lc")
print("="*80)

# COMMAND ----------

# DBTITLE 1,Análise Individual dos Events de Cada Detector
# ============================================================
# ANÁLISE INDIVIDUAL DOS EVENTS DE CADA DETECTOR
# ============================================================
# Objetivo: Entender a estrutura de eventos de cada detector
# para criar uma função unificada de tracking em tempo real
# ============================================================

print("\n" + "="*80)
print("🔍 ANALISANDO EVENTS DE CADA DETECTOR INDIVIDUALMENTE")
print("="*80)

# ======================================================
# 1. FEA DETECTOR EVENTS
# ======================================================
print("\n📊 1. FEA DETECTOR EVENTS")
print("-" * 80)

# Preparar DataFrame (renomear Time -> timestamp)
df_fea_prep = df_fea_raw.copy()
if 'Time' in df_fea_prep.columns and 'timestamp' not in df_fea_prep.columns:
    df_fea_prep = df_fea_prep.rename(columns={'Time': 'timestamp'})

df_fea_prep['timestamp'] = pd.to_datetime(df_fea_prep['timestamp'], errors='coerce')
df_fea_prep = df_fea_prep[
    (df_fea_prep['timestamp'] >= START_TS) & 
    (df_fea_prep['timestamp'] <= END_TS)
].sort_values('timestamp').reset_index(drop=True)

# Chamar detector
df_fea_results, df_fea_events, df_fea_debug, df_fea_features = detect_fea_events_fsm_energia(
    df=df_fea_prep,
    timestamp_col='timestamp',
    tag_col='ACI@FEA_ELET_ENERGIA',
    initial_corrida=INITIAL_HEAT,
    **fea_params
)

print(f"Total de eventos FEA: {len(df_fea_events)}")
print(f"Colunas: {df_fea_events.columns.tolist()}")
print(f"\nStatus únicos: {df_fea_events['status'].unique().tolist()}")
print(f"\nPrimeiras 5 linhas:")
display(df_fea_events.head())

# ======================================================
# 2. FP DETECTOR EVENTS
# ======================================================
print("\n📊 2. FP DETECTOR EVENTS")
print("-" * 80)

df_fp_final, df_fp_events, df_fp_heat = detect_fp_events_gas_based_streaming(
    spark_df=df_fp_raw,
    initial_heat_number=INITIAL_HEAT,
    initial_timestamp=START_TS,
    last_timestamp=END_TS,
    **fp_params_streaming
)

df_fp_events_pd = df_fp_events.toPandas()
print(f"Total de eventos FP: {len(df_fp_events_pd)}")
print(f"Colunas: {df_fp_events_pd.columns.tolist()}")
print(f"\nStatus únicos: {df_fp_events_pd['status'].unique().tolist()}")
print(f"\nPrimeiras 5 linhas:")
display(df_fp_events_pd.head())

# ======================================================
# 3. VD DETECTOR EVENTS
# ======================================================
print("\n📊 3. VD DETECTOR EVENTS")
print("-" * 80)

df_vd_final, df_vd_events, df_vd_debug = detect_vd_events(
    spark_df_vd_pims=df_vd_raw,
    start_ts=START_TS,
    end_ts=END_TS,
    initial_heat_number=INITIAL_HEAT,
    **vd_params
)

df_vd_events_pd = df_vd_events.toPandas()
print(f"Total de eventos VD: {len(df_vd_events_pd)}")
print(f"Colunas: {df_vd_events_pd.columns.tolist()}")
print(f"\nStatus únicos: {df_vd_events_pd['status'].unique().tolist()}")
print(f"\nPrimeiras 5 linhas:")
display(df_vd_events_pd.head())

# ======================================================
# 4. LC DETECTOR EVENTS
# ======================================================
print("\n📊 4. LC DETECTOR EVENTS")
print("-" * 80)

# Converter Spark -> Pandas e preparar
df_lc_prep = df_lc_raw.toPandas()
if 'Time' in df_lc_prep.columns and 'timestamp' not in df_lc_prep.columns:
    df_lc_prep = df_lc_prep.rename(columns={'Time': 'timestamp'})

df_lc_final, df_lc_events, df_lc_debug, df_lc_features = detect_lc_events_fsm_pesobra(
    df=df_lc_prep,
    initial_corrida=INITIAL_HEAT,
    **lc_params
)

print(f"Total de eventos LC: {len(df_lc_events)}")
print(f"Colunas: {df_lc_events.columns.tolist()}")
print(f"\nStatus únicos: {df_lc_events['status'].unique().tolist()}")
print(f"\nPrimeiras 5 linhas:")
display(df_lc_events.head())

print("\n" + "="*80)
print("✅ ANÁLISE INDIVIDUAL CONCLUÍDA")
print("="*80)

# COMMAND ----------

# DBTITLE 1,Função Unificada de Events para Tracking em Tempo Real
# ============================================================
# FUNÇÃO UNIFICADA DE EVENTS PARA TRACKING EM TEMPO REAL
# ============================================================
# Cada linha = 1 evento (não 1 corrida)
# Permite rastrear onde cada corrida está em tempo real
# ============================================================

import pandas as pd
from pyspark.sql import functions as F


def build_unified_events_timeline(
    df_fea_events_pd: pd.DataFrame,
    df_fp_events_sp,
    df_vd_events_sp,
    df_lc_events_pd: pd.DataFrame,
    spark
):
    """
    Unifica eventos de todos os detectores em uma timeline única.
    
    Retorna DataFrame Spark com:
    - corrida: número da corrida
    - stage: etapa (FEA, FP, VD, LC)
    - event_type: tipo do evento (RUNNING, CLOSED, VD_ARRIVAL, etc)
    - event_ts: timestamp do evento
    - start_time: início da etapa (quando aplicável)
    - end_time: fim da etapa (quando aplicável)
    - metadata: informações adicionais (carro, tanque, braço, etc)
    - duration_seconds: duração (quando aplicável)
    
    Cada linha representa UM evento, permitindo tracking em tempo real.
    """
    
    print("\n🔧 Construindo timeline unificada de eventos...")
    
    # ======================================================
    # 1. PADRONIZAR FEA EVENTS
    # ======================================================
    df_fea_std = df_fea_events_pd.copy()
    df_fea_std['stage'] = 'FEA'
    df_fea_std['event_type'] = df_fea_std['status']
    df_fea_std['start_time'] = df_fea_std.get('inicio_fea_sys', None)
    df_fea_std['end_time'] = df_fea_std.get('final_fea_sys', None)
    df_fea_std['metadata'] = None
    df_fea_std['duration_seconds'] = None
    
    df_fea_std = df_fea_std[[
        'corrida', 'stage', 'event_type', 'event_ts', 
        'start_time', 'end_time', 'metadata', 'duration_seconds'
    ]]
    
    # ======================================================
    # 2. PADRONIZAR FP EVENTS
    # ======================================================
    df_fp_pd = df_fp_events_sp.toPandas()
    df_fp_std = df_fp_pd.copy()
    df_fp_std['stage'] = 'FP'
    df_fp_std['event_type'] = df_fp_std['status']
    df_fp_std['start_time'] = df_fp_std.get('fp_start', None)
    df_fp_std['end_time'] = df_fp_std.get('fp_end', None)
    df_fp_std['metadata'] = df_fp_std.apply(
        lambda row: f"carro={row.get('carro_fp', 'N/A')}, reason={row.get('reason', 'N/A')}",
        axis=1
    )
    df_fp_std['duration_seconds'] = (
        (pd.to_datetime(df_fp_std['end_time']) - pd.to_datetime(df_fp_std['start_time']))
        .dt.total_seconds()
        .where(df_fp_std['end_time'].notna() & df_fp_std['start_time'].notna())
    )
    
    df_fp_std = df_fp_std[[
        'corrida', 'stage', 'event_type', 'event_ts', 
        'start_time', 'end_time', 'metadata', 'duration_seconds'
    ]]
    
    # ======================================================
    # 3. PADRONIZAR VD EVENTS
    # ======================================================
    df_vd_pd = df_vd_events_sp.toPandas()
    df_vd_std = df_vd_pd.copy()
    df_vd_std['stage'] = 'VD'
    df_vd_std['event_type'] = df_vd_std['status']
    df_vd_std['start_time'] = df_vd_std.get('start_vd_time', None)
    df_vd_std['end_time'] = df_vd_std.get('end_vd_time', None)
    df_vd_std['metadata'] = df_vd_std.apply(
        lambda row: f"tank={row.get('tank_id', 'N/A')}",
        axis=1
    )
    df_vd_std['duration_seconds'] = df_vd_std.get('duration_seconds', None)
    
    df_vd_std = df_vd_std[[
        'corrida', 'stage', 'event_type', 'event_ts', 
        'start_time', 'end_time', 'metadata', 'duration_seconds'
    ]]
    
    # ======================================================
    # 4. PADRONIZAR LC EVENTS
    # ======================================================
    df_lc_std = df_lc_events_pd.copy()
    df_lc_std['stage'] = 'LC'
    df_lc_std['event_type'] = df_lc_std['status']
    df_lc_std['start_time'] = df_lc_std.get('inicio_lc_sys', None)
    df_lc_std['end_time'] = df_lc_std.get('final_lc_sys', None)
    df_lc_std['metadata'] = df_lc_std.apply(
        lambda row: f"braco={row.get('braco', 'N/A')}",
        axis=1
    )
    df_lc_std['duration_seconds'] = None
    
    df_lc_std = df_lc_std[[
        'corrida', 'stage', 'event_type', 'event_ts', 
        'start_time', 'end_time', 'metadata', 'duration_seconds'
    ]]
    
    # ======================================================
    # 5. UNIFICAR TODOS OS EVENTOS
    # ======================================================
    df_all_events = pd.concat([
        df_fea_std,
        df_fp_std,
        df_vd_std,
        df_lc_std
    ], ignore_index=True)
    
    # Converter para timestamp
    df_all_events['event_ts'] = pd.to_datetime(df_all_events['event_ts'], errors='coerce')
    df_all_events['start_time'] = pd.to_datetime(df_all_events['start_time'], errors='coerce')
    df_all_events['end_time'] = pd.to_datetime(df_all_events['end_time'], errors='coerce')
    
    # Ordenar por timestamp do evento
    df_all_events = df_all_events.sort_values(['event_ts', 'corrida']).reset_index(drop=True)
    
    # Converter para Spark
    df_unified_events_sp = spark.createDataFrame(df_all_events)
    
    print(f"✓ Timeline unificada criada: {df_all_events.shape[0]} eventos totais")
    print(f"  - FEA: {len(df_fea_std)} eventos")
    print(f"  - FP: {len(df_fp_std)} eventos")
    print(f"  - VD: {len(df_vd_std)} eventos")
    print(f"  - LC: {len(df_lc_std)} eventos")
    
    return df_unified_events_sp


def get_corrida_current_status(df_unified_events_sp, corrida_num):
    """
    Retorna o status atual de uma corrida específica baseado nos eventos.
    
    Retorna dict com:
    - corrida: número da corrida
    - current_stage: etapa atual (FEA, FP, VD, LC, COMPLETED, NOT_FOUND)
    - current_event: último evento registrado
    - last_event_ts: timestamp do último evento
    - all_events: lista de todos os eventos da corrida
    """
    
    df_corrida = df_unified_events_sp.filter(F.col("corrida") == corrida_num)
    
    if df_corrida.count() == 0:
        return {
            'corrida': corrida_num,
            'current_stage': 'NOT_FOUND',
            'current_event': None,
            'last_event_ts': None,
            'all_events': []
        }
    
    # Pegar último evento
    last_event = df_corrida.orderBy(F.desc("event_ts")).first()
    
    # Pegar todos os eventos
    all_events = df_corrida.orderBy("event_ts").collect()
    
    return {
        'corrida': corrida_num,
        'current_stage': last_event['stage'],
        'current_event': last_event['event_type'],
        'last_event_ts': last_event['event_ts'],
        'all_events': [
            {
                'stage': e['stage'],
                'event_type': e['event_type'],
                'event_ts': e['event_ts'],
                'metadata': e['metadata']
            }
            for e in all_events
        ]
    }


print("✓ Funções unificadas de events criadas:")
print("  - build_unified_events_timeline(): cria timeline unificada")
print("  - get_corrida_current_status(): retorna status atual de uma corrida")

# COMMAND ----------

# DBTITLE 1,Executar Função Unificada de Events
# ============================================================
# EXECUTAR FUNÇÃO UNIFICADA DE EVENTS
# ============================================================

print("\n" + "="*80)
print("🚀 CRIANDO TIMELINE UNIFICADA DE EVENTOS")
print("="*80)

# Construir timeline unificada
df_unified_events = build_unified_events_timeline(
    df_fea_events_pd=df_fea_events,
    df_fp_events_sp=df_fp_events,
    df_vd_events_sp=df_vd_events,
    df_lc_events_pd=df_lc_events,
    spark=spark
)

print("\n" + "="*80)
print("📊 VISUALIZAÇÃO DA TIMELINE UNIFICADA")
print("="*80)
print("Cada linha = 1 evento (não 1 corrida)")
print("Ordenado por timestamp do evento\n")

display(
    df_unified_events
    .orderBy(F.desc("event_ts"))
    .limit(100)
)

print("\n" + "="*80)
print("📈 ESTATÍSTICAS POR ETAPA")
print("="*80)

display(
    df_unified_events
    .groupBy("stage", "event_type")
    .count()
    .orderBy("stage", F.desc("count"))
)

# COMMAND ----------

display(df_unified_events.orderBy(F.col("event_ts").asc()))

# COMMAND ----------

# DBTITLE 1,Exemplo: Tracking de Corridas Específicas
# ============================================================
# EXEMPLO: TRACKING DE CORRIDAS ESPECÍFICAS EM TEMPO REAL
# ============================================================

print("\n" + "="*80)
print("🔍 EXEMPLO: TRACKING DE CORRIDAS ESPECÍFICAS")
print("="*80)

# Pegar as 5 corridas mais recentes
top_corridas = (
    df_unified_events
    .select("corrida")
    .distinct()
    .orderBy(F.desc("corrida"))
    .limit(5)
    .collect()
)

print(f"\nAnalisando as {len(top_corridas)} corridas mais recentes:\n")

for row in top_corridas:
    corrida_num = row['corrida']
    status = get_corrida_current_status(df_unified_events, corrida_num)
    
    print(f"\n{'='*60}")
    print(f"Corrida: {status['corrida']}")
    print(f"Etapa Atual: {status['current_stage']}")
    print(f"Evento Atual: {status['current_event']}")
    print(f"Último Update: {status['last_event_ts']}")
    print(f"\nHistórico de Eventos ({len(status['all_events'])} eventos):")
    
    for i, evt in enumerate(status['all_events'], 1):
        print(f"  {i}. [{evt['stage']}] {evt['event_type']} @ {evt['event_ts']}")
        if evt['metadata']:
            print(f"     └─ {evt['metadata']}")

print("\n" + "="*80)
print("✅ TRACKING CONCLUÍDO")
print("="*80)

# Visualizar timeline completa de uma corrida específica
if top_corridas:
    exemplo_corrida = top_corridas[0]['corrida']
    print(f"\n📊 Timeline completa da corrida {exemplo_corrida}:\n")
    
    display(
        df_unified_events
        .filter(F.col("corrida") == exemplo_corrida)
        .orderBy("event_ts")
    )

# COMMAND ----------

# DBTITLE 1,Análise Visual Completa: Diferenças, Tempos Totais e Entre Etapas
# ============================================================
# ANÁLISE VISUAL COMPLETA: HISTOGRAMAS E GRÁFICOS DE TEMPO
# ============================================================
# Gera todos os histogramas de diferenças, tempos totais e
# tempos entre etapas em uma única célula
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("\n" + "="*80)
print("📊 GERANDO ANÁLISES VISUAIS COMPLETAS")
print("="*80)

# Converter para Pandas para análise
df_comp_pd = df_compare_unified.toPandas()
df_op_pd = df_final.toPandas()  # Dados do operador

print(f"\nTotal de corridas analisadas:")
print(f"  - Comparação: {len(df_comp_pd)} corridas")
print(f"  - Operador: {len(df_op_pd)} corridas")
print(f"Período: {df_comp_pd['corrida'].min()} → {df_comp_pd['corrida'].max()}")

# ============================================================
# 1. HISTOGRAMAS DAS DIFERENÇAS (SYS - OP) EM MINUTOS
# ============================================================
print("\n📈 1. Gerando histogramas das diferenças...")

# Identificar colunas de diferença
diff_cols = [col for col in df_comp_pd.columns if col.startswith('diff_')]

if diff_cols:
    n_diffs = len(diff_cols)
    n_cols = 3
    n_rows = (n_diffs + n_cols - 1) // n_cols
    
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes1 = axes1.flatten() if n_diffs > 1 else [axes1]
    
    for idx, col in enumerate(diff_cols):
        ax = axes1[idx]
        data = df_comp_pd[col].dropna()
        
        if len(data) > 0:
            # Histograma
            ax.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            
            # Estatísticas
            mean_val = data.mean()
            median_val = data.median()
            std_val = data.std()
            
            # Linhas de referência
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Média: {mean_val:.2f} min')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Mediana: {median_val:.2f} min')
            ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Zero (perfeito)')
            
            # Título e labels
            etapa = col.replace('diff_', '').replace('_', ' ').upper()
            ax.set_title(f'Diferença: {etapa}\n(Sistema - Operador)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Diferença (minutos)', fontsize=10)
            ax.set_ylabel('Frequência', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Texto com estatísticas
            stats_text = f'N={len(data)}\nStd={std_val:.2f} min'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'Sem dados', ha='center', va='center', fontsize=12)
            ax.set_title(col, fontsize=12)
    
    # Remover subplots vazios
    for idx in range(n_diffs, len(axes1)):
        fig1.delaxes(axes1[idx])
    
    plt.tight_layout()
    plt.suptitle('HISTOGRAMAS DAS DIFERENÇAS (Sistema - Operador)', 
                 fontsize=16, fontweight='bold', y=1.002)
    plt.show()
    
    print(f"✓ {len(diff_cols)} histogramas de diferenças gerados")
else:
    print("⚠️ Nenhuma coluna de diferença encontrada")

# ============================================================
# 2. TEMPOS TOTAIS POR ETAPA (DURAÇÃO) - OPERADOR vs SISTEMA
# ============================================================
print("\n📈 2. Gerando análise de tempos totais por etapa (Operador vs Sistema)...")

etapas = ['fea', 'fp', 'vd', 'lc']
duracoes_comparacao = []

for etapa in etapas:
    etapa_data = {'etapa': etapa.upper()}
    
    # Sistema - buscar no df_comp_pd com sufixo _sys
    inicio_sys = f'inicio_{etapa}_sys'
    final_sys = f'final_{etapa}_sys'
    
    if inicio_sys in df_comp_pd.columns and final_sys in df_comp_pd.columns:
        df_temp = df_comp_pd[[inicio_sys, final_sys]].copy()
        df_temp[inicio_sys] = pd.to_datetime(df_temp[inicio_sys], errors='coerce')
        df_temp[final_sys] = pd.to_datetime(df_temp[final_sys], errors='coerce')
        
        duracao_sys = (df_temp[final_sys] - df_temp[inicio_sys]).dt.total_seconds() / 60.0
        etapa_data['sys'] = duracao_sys.dropna()
    
    # Operador - buscar no df_op_pd SEM sufixo _sys
    inicio_op = f'inicio_{etapa}'
    final_op = f'final_{etapa}'
    
    if inicio_op in df_op_pd.columns and final_op in df_op_pd.columns:
        df_temp = df_op_pd[[inicio_op, final_op]].copy()
        df_temp[inicio_op] = pd.to_datetime(df_temp[inicio_op], errors='coerce')
        df_temp[final_op] = pd.to_datetime(df_temp[final_op], errors='coerce')
        
        duracao_op = (df_temp[final_op] - df_temp[inicio_op]).dt.total_seconds() / 60.0
        etapa_data['op'] = duracao_op.dropna()
    
    if 'sys' in etapa_data or 'op' in etapa_data:
        duracoes_comparacao.append(etapa_data)

if duracoes_comparacao:
    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 14))
    axes2 = axes2.flatten()
    
    for idx, etapa_info in enumerate(duracoes_comparacao):
        if idx < 4:
            ax = axes2[idx]
            
            # Plotar Sistema
            if 'sys' in etapa_info:
                data_sys = etapa_info['sys']
                ax.hist(data_sys, bins=30, alpha=0.5, color='coral', edgecolor='black', label='Sistema')
                mean_sys = data_sys.mean()
                ax.axvline(mean_sys, color='red', linestyle='--', linewidth=2, 
                          label=f'Média SYS: {mean_sys:.1f} min')
            
            # Plotar Operador
            if 'op' in etapa_info:
                data_op = etapa_info['op']
                ax.hist(data_op, bins=30, alpha=0.5, color='steelblue', edgecolor='black', label='Operador')
                mean_op = data_op.mean()
                ax.axvline(mean_op, color='blue', linestyle='--', linewidth=2, 
                          label=f'Média OP: {mean_op:.1f} min')
            
            # Título e labels
            ax.set_title(f"Duração Total: {etapa_info['etapa']} (Sistema vs Operador)", 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Duração (minutos)', fontsize=10)
            ax.set_ylabel('Frequência', fontsize=10)
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Estatísticas
            stats_lines = []
            if 'sys' in etapa_info:
                stats_lines.append(f"SYS: N={len(etapa_info['sys'])}, Std={etapa_info['sys'].std():.1f}")
            if 'op' in etapa_info:
                stats_lines.append(f"OP: N={len(etapa_info['op'])}, Std={etapa_info['op'].std():.1f}")
            
            stats_text = '\n'.join(stats_lines)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Remover subplots vazios
    for idx in range(len(duracoes_comparacao), 4):
        fig2.delaxes(axes2[idx])
    
    plt.tight_layout()
    plt.suptitle('TEMPOS TOTAIS POR ETAPA - Operador vs Sistema', 
                 fontsize=16, fontweight='bold', y=1.002)
    plt.show()
    
    print(f"✓ {len(duracoes_comparacao)} comparações de duração geradas")
else:
    print("⚠️ Não foi possível calcular durações")

# ============================================================
# 3. TEMPOS ENTRE ETAPAS (DELAYS) - OPERADOR vs SISTEMA
# ============================================================
print("\n📈 3. Gerando análise de tempos entre etapas (Operador vs Sistema)...")

transicoes = [
    ('FEA→FP', 'final_fea', 'inicio_fp'),
    ('FP→VD', 'final_fp', 'inicio_vd'),
    ('VD→LC', 'final_vd', 'inicio_lc')
]

delays_comparacao = []

for nome, col_fim_base, col_inicio_base in transicoes:
    transicao_data = {'transicao': nome}
    
    # Sistema - buscar no df_comp_pd com sufixo _sys
    col_fim_sys = f'{col_fim_base}_sys'
    col_inicio_sys = f'{col_inicio_base}_sys'
    
    if col_fim_sys in df_comp_pd.columns and col_inicio_sys in df_comp_pd.columns:
        df_temp = df_comp_pd[[col_fim_sys, col_inicio_sys]].copy()
        df_temp[col_fim_sys] = pd.to_datetime(df_temp[col_fim_sys], errors='coerce')
        df_temp[col_inicio_sys] = pd.to_datetime(df_temp[col_inicio_sys], errors='coerce')
        
        delay_sys = (df_temp[col_inicio_sys] - df_temp[col_fim_sys]).dt.total_seconds() / 60.0
        transicao_data['sys'] = delay_sys.dropna()
    
    # Operador - buscar no df_op_pd SEM sufixo _sys
    if col_fim_base in df_op_pd.columns and col_inicio_base in df_op_pd.columns:
        df_temp = df_op_pd[[col_fim_base, col_inicio_base]].copy()
        df_temp[col_fim_base] = pd.to_datetime(df_temp[col_fim_base], errors='coerce')
        df_temp[col_inicio_base] = pd.to_datetime(df_temp[col_inicio_base], errors='coerce')
        
        delay_op = (df_temp[col_inicio_base] - df_temp[col_fim_base]).dt.total_seconds() / 60.0
        transicao_data['op'] = delay_op.dropna()
    
    if 'sys' in transicao_data or 'op' in transicao_data:
        delays_comparacao.append(transicao_data)

if delays_comparacao:
    fig3, axes3 = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, delay_info in enumerate(delays_comparacao):
        ax = axes3[idx]
        
        # Plotar Sistema
        if 'sys' in delay_info:
            data_sys = delay_info['sys']
            ax.hist(data_sys, bins=30, alpha=0.5, color='lightgreen', edgecolor='black', label='Sistema')
            mean_sys = data_sys.mean()
            ax.axvline(mean_sys, color='darkgreen', linestyle='--', linewidth=2, 
                      label=f'Média SYS: {mean_sys:.1f} min')
        
        # Plotar Operador
        if 'op' in delay_info:
            data_op = delay_info['op']
            ax.hist(data_op, bins=30, alpha=0.5, color='orange', edgecolor='black', label='Operador')
            mean_op = data_op.mean()
            ax.axvline(mean_op, color='darkorange', linestyle='--', linewidth=2, 
                      label=f'Média OP: {mean_op:.1f} min')
        
        # Título e labels
        ax.set_title(f"Tempo de Transição: {delay_info['transicao']}\n(Sistema vs Operador)", 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Delay (minutos)', fontsize=10)
        ax.set_ylabel('Frequência', fontsize=10)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Estatísticas
        stats_lines = []
        if 'sys' in delay_info:
            stats_lines.append(f"SYS: N={len(delay_info['sys'])}, Std={delay_info['sys'].std():.1f}")
        if 'op' in delay_info:
            stats_lines.append(f"OP: N={len(delay_info['op'])}, Std={delay_info['op'].std():.1f}")
        
        stats_text = '\n'.join(stats_lines)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    plt.suptitle('TEMPOS ENTRE ETAPAS (Delays) - Operador vs Sistema', 
                 fontsize=16, fontweight='bold', y=1.002)
    plt.show()
    
    print(f"✓ {len(delays_comparacao)} comparações de delay geradas")
else:
    print("⚠️ Não foi possível calcular delays entre etapas")

# ============================================================
# 4. RESUMO ESTATÍSTICO EM TABELA
# ============================================================
print("\n📊 4. Gerando resumo estatístico...")

# Criar tabela resumo
resumo_rows = []

# Diferenças
for col in diff_cols:
    data = df_comp_pd[col].dropna()
    if len(data) > 0:
        resumo_rows.append({
            'Métrica': col.replace('diff_', 'Diff ').replace('_', ' ').title(),
            'Tipo': 'Diferença (min)',
            'Fonte': 'SYS-OP',
            'N': len(data),
            'Média': f"{data.mean():.2f}",
            'Mediana': f"{data.median():.2f}",
            'Std': f"{data.std():.2f}",
            'Min': f"{data.min():.2f}",
            'Max': f"{data.max():.2f}"
        })

# Durações - Sistema e Operador
for etapa_info in duracoes_comparacao:
    if 'sys' in etapa_info:
        data = etapa_info['sys']
        resumo_rows.append({
            'Métrica': f"Duração {etapa_info['etapa']}",
            'Tipo': 'Tempo Total (min)',
            'Fonte': 'Sistema',
            'N': len(data),
            'Média': f"{data.mean():.2f}",
            'Mediana': f"{data.median():.2f}",
            'Std': f"{data.std():.2f}",
            'Min': f"{data.min():.2f}",
            'Max': f"{data.max():.2f}"
        })
    
    if 'op' in etapa_info:
        data = etapa_info['op']
        resumo_rows.append({
            'Métrica': f"Duração {etapa_info['etapa']}",
            'Tipo': 'Tempo Total (min)',
            'Fonte': 'Operador',
            'N': len(data),
            'Média': f"{data.mean():.2f}",
            'Mediana': f"{data.median():.2f}",
            'Std': f"{data.std():.2f}",
            'Min': f"{data.min():.2f}",
            'Max': f"{data.max():.2f}"
        })

# Delays - Sistema e Operador
for delay_info in delays_comparacao:
    if 'sys' in delay_info:
        data = delay_info['sys']
        resumo_rows.append({
            'Métrica': f"Delay {delay_info['transicao']}",
            'Tipo': 'Tempo Entre Etapas (min)',
            'Fonte': 'Sistema',
            'N': len(data),
            'Média': f"{data.mean():.2f}",
            'Mediana': f"{data.median():.2f}",
            'Std': f"{data.std():.2f}",
            'Min': f"{data.min():.2f}",
            'Max': f"{data.max():.2f}"
        })
    
    if 'op' in delay_info:
        data = delay_info['op']
        resumo_rows.append({
            'Métrica': f"Delay {delay_info['transicao']}",
            'Tipo': 'Tempo Entre Etapas (min)',
            'Fonte': 'Operador',
            'N': len(data),
            'Média': f"{data.mean():.2f}",
            'Mediana': f"{data.median():.2f}",
            'Std': f"{data.std():.2f}",
            'Min': f"{data.min():.2f}",
            'Max': f"{data.max():.2f}"
        })

if resumo_rows:
    df_resumo = pd.DataFrame(resumo_rows)
    
    print("\n" + "="*80)
    print("📋 RESUMO ESTATÍSTICO COMPLETO")
    print("="*80)
    display(df_resumo)
else:
    print("⚠️ Não foi possível gerar resumo estatístico")

print("\n" + "="*80)
print("✅ ANÁLISE VISUAL COMPLETA CONCLUÍDA!")
print("="*80)
print(f"\nTotal de gráficos gerados:")
print(f"  - Diferenças (Sistema - Operador): {len(diff_cols)} histogramas")
print(f"  - Tempos Totais por Etapa (OP vs SYS): {len(duracoes_comparacao)} comparações")
print(f"  - Tempos Entre Etapas/Delays (OP vs SYS): {len(delays_comparacao)} comparações")
print(f"  - Resumo Estatístico: 1 tabela com {len(resumo_rows)} métricas")

# COMMAND ----------

# DBTITLE 1,1. Preparação dos Dados - Tempos Entre Etapas (SISTEMA)
print("="*120)
print(" "*35 + "ANÁLISE ESTATÍSTICA DE TEMPOS ENTRE ETAPAS")
print(" "*40 + "DADOS DO SISTEMA (df_unified)")
print("="*120)
print("\n🎯 OBJETIVO: Identificar tempos típicos entre etapas, remover outliers e definir valores para previsão\n")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Usar df_unified (dados do SISTEMA)
df_sys_pd = df_unified.copy()

print("📊 FONTE DOS DADOS: SISTEMA (df_unified)")
print("-" * 120)
print(f"  Total de corridas: {len(df_sys_pd)}")
print(f"  Período: {df_sys_pd['inicio_fea'].min()} até {df_sys_pd['final_lc'].max()}")
print("-" * 120)

print("\n📊 ETAPAS ANALISADAS:")
print("-" * 120)
print("  1. FEA → FP: Tempo entre final do FEA e início do FP")
print("  2. FP → VD: Tempo entre final do FP e início do VD")
print("  3. VD → LC: Tempo entre final do VD e início do LC")
print("  4. FEA → LC: Tempo total do ciclo (FEA até início do LC)")
print("-" * 120)

# Calcular tempos entre etapas em MINUTOS
print("\n⏱️ Calculando tempos entre etapas...\n")

# Garantir que colunas são datetime
cols_datetime = ['inicio_fea', 'final_fea', 'inicio_fp', 'final_fp', 'inicio_vd', 'final_vd', 'inicio_lc', 'final_lc']
for col in cols_datetime:
    if col in df_sys_pd.columns:
        df_sys_pd[col] = pd.to_datetime(df_sys_pd[col], errors='coerce')

# Calcular tempos entre etapas
df_sys_pd['tempo_fea_fp'] = (df_sys_pd['inicio_fp'] - df_sys_pd['final_fea']).dt.total_seconds() / 60.0
df_sys_pd['tempo_fp_vd'] = (df_sys_pd['inicio_vd'] - df_sys_pd['final_fp']).dt.total_seconds() / 60.0
df_sys_pd['tempo_vd_lc'] = (df_sys_pd['inicio_lc'] - df_sys_pd['final_vd']).dt.total_seconds() / 60.0
df_sys_pd['tempo_ciclo_total'] = (df_sys_pd['inicio_lc'] - df_sys_pd['inicio_fea']).dt.total_seconds() / 60.0

# Remover valores negativos (inconsistências)
for col in ['tempo_fea_fp', 'tempo_fp_vd', 'tempo_vd_lc', 'tempo_ciclo_total']:
    antes = df_sys_pd[col].notna().sum()
    df_sys_pd.loc[df_sys_pd[col] < 0, col] = np.nan
    depois = df_sys_pd[col].notna().sum()
    removidos = antes - depois
    if removidos > 0:
        print(f"  ⚠️ {col}: {removidos} valores negativos removidos")

print("\n✅ Tempos calculados com sucesso!")
print("\n📈 Estatísticas Iniciais (com outliers):")
print("-" * 120)

tempos_cols = ['tempo_fea_fp', 'tempo_fp_vd', 'tempo_vd_lc', 'tempo_ciclo_total']
for col in tempos_cols:
    data = df_sys_pd[col].dropna()
    if len(data) > 0:
        print(f"\n{col.upper().replace('_', ' ')}:")
        print(f"  N = {len(data)} | Média = {data.mean():.2f} min | Mediana = {data.median():.2f} min")
        print(f"  Min = {data.min():.2f} min | Max = {data.max():.2f} min | Std = {data.std():.2f} min")

print("\n" + "="*120)

# COMMAND ----------

# DBTITLE 1,2. Identificação de Outliers - Método IQR e Z-Score
print("="*120)
print(" "*40 + "IDENTIFICAÇÃO DE OUTLIERS")
print("="*120)
print("\n🔍 MÉTODOS UTILIZADOS:")
print("  1. IQR (Interquartile Range): Outliers = valores fora de [Q1 - 1.5*IQR, Q3 + 1.5*IQR]")
print("  2. Z-Score: Outliers = valores com |Z-score| > 3")
print("  3. Combinado: Outlier se identificado por AMBOS os métodos (mais conservador)")
print("-" * 120)

def identificar_outliers(data, nome_coluna):
    """Identifica outliers usando IQR e Z-score"""
    data_clean = data.dropna()
    
    if len(data_clean) == 0:
        return pd.Series(False, index=data.index), {}
    
    # Método 1: IQR
    Q1 = data_clean.quantile(0.25)
    Q3 = data_clean.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_iqr = (data < lower_bound) | (data > upper_bound)
    
    # Método 2: Z-Score
    z_scores = np.abs(stats.zscore(data_clean))
    outliers_zscore = pd.Series(False, index=data.index)
    outliers_zscore.loc[data_clean.index] = z_scores > 3
    
    # Combinado: outlier se identificado por AMBOS
    outliers_combinado = outliers_iqr & outliers_zscore
    
    # Estatísticas
    stats_dict = {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'n_outliers_iqr': outliers_iqr.sum(),
        'n_outliers_zscore': outliers_zscore.sum(),
        'n_outliers_combinado': outliers_combinado.sum(),
        'perc_outliers': (outliers_combinado.sum() / len(data_clean)) * 100
    }
    
    return outliers_combinado, stats_dict

# Identificar outliers para cada tempo
outliers_info = {}
df_sys_pd_clean = df_sys_pd.copy()

print("\n📊 RESULTADOS DA IDENTIFICAÇÃO DE OUTLIERS:\n")

for col in tempos_cols:
    print(f"\n{'='*100}")
    print(f"  {col.upper().replace('_', ' ')}")
    print(f"{'='*100}")
    
    outliers_mask, stats_dict = identificar_outliers(df_sys_pd[col], col)
    outliers_info[col] = {'mask': outliers_mask, 'stats': stats_dict}
    
    # Mostrar estatísticas
    print(f"\n  📈 Limites IQR:")
    print(f"     Q1 = {stats_dict['Q1']:.2f} min | Q3 = {stats_dict['Q3']:.2f} min | IQR = {stats_dict['IQR']:.2f} min")
    print(f"     Limite Inferior = {stats_dict['lower_bound']:.2f} min")
    print(f"     Limite Superior = {stats_dict['upper_bound']:.2f} min")
    
    print(f"\n  🚨 Outliers Identificados:")
    print(f"     Método IQR: {stats_dict['n_outliers_iqr']} outliers")
    print(f"     Método Z-Score: {stats_dict['n_outliers_zscore']} outliers")
    print(f"     Método Combinado (IQR + Z-Score): {stats_dict['n_outliers_combinado']} outliers ({stats_dict['perc_outliers']:.2f}%)")
    
    # Criar coluna sem outliers
    col_clean = f"{col}_clean"
    df_sys_pd_clean[col_clean] = df_sys_pd[col].copy()
    df_sys_pd_clean.loc[outliers_mask, col_clean] = np.nan
    
    # Estatísticas após remoção
    data_clean = df_sys_pd_clean[col_clean].dropna()
    if len(data_clean) > 0:
        print(f"\n  ✅ Estatísticas APÓS remoção de outliers:")
        print(f"     N = {len(data_clean)} | Média = {data_clean.mean():.2f} min | Mediana = {data_clean.median():.2f} min")
        print(f"     Min = {data_clean.min():.2f} min | Max = {data_clean.max():.2f} min | Std = {data_clean.std():.2f} min")

print("\n" + "="*120)
print("✅ Identificação de outliers concluída!")
print("="*120)

# COMMAND ----------

# DBTITLE 1,3. Análise de Percentis (P5, P25, P50, P75, P95)
print("="*120)
print(" "*40 + "ANÁLISE DE PERCENTIS")
print("="*120)
print("\n📊 Percentis calculados: P5, P10, P25, P50 (mediana), P75, P90, P95")
print("   Úteis para definir tempos conservadores (P75/P90) ou otimistas (P25/P10) na previsão\n")
print("-" * 120)

# Calcular percentis
percentis = [5, 10, 25, 50, 75, 90, 95]
resultados_percentis = []

for col in tempos_cols:
    col_clean = f"{col}_clean"
    data = df_sys_pd_clean[col_clean].dropna()
    
    if len(data) > 0:
        row = {
            'Etapa': col.replace('tempo_', '').replace('_', ' → ').upper(),
            'N': len(data),
            'Média': f"{data.mean():.2f}",
            'Std': f"{data.std():.2f}"
        }
        
        for p in percentis:
            row[f'P{p}'] = f"{data.quantile(p/100):.2f}"
        
        resultados_percentis.append(row)

df_percentis = pd.DataFrame(resultados_percentis)

print("\n📋 TABELA DE PERCENTIS (em minutos):\n")
display(df_percentis)

print("\n" + "="*120)
print("💡 INTERPRETAÇÃO DOS PERCENTIS:")
print("="*120)
print("  • P5/P10: Tempos muito rápidos (5-10% das corridas são mais rápidas que isso)")
print("  • P25: Primeiro quartil (25% das corridas são mais rápidas)")
print("  • P50: Mediana (50% das corridas são mais rápidas) - VALOR TÍPICO")
print("  • P75: Terceiro quartil (75% das corridas são mais rápidas)")
print("  • P90/P95: Tempos conservadores (90-95% das corridas são mais rápidas)")
print("\n  🎯 RECOMENDAÇÃO PARA PREVISÃO:")
print("     - Usar P50 (mediana) para previsão TÍPICA")
print("     - Usar P75 para previsão CONSERVADORA (cobre 75% dos casos)")
print("     - Usar P90 para previsão MUITO CONSERVADORA (cobre 90% dos casos)")
print("="*120)

# COMMAND ----------

# DBTITLE 1,4. Visualização - Boxplots Antes e Depois da Remoção de Outliers
print("="*120)
print(" "*35 + "VISUALIZAÇÃO: BOXPLOTS COMPARATIVOS")
print("="*120)
print("\n📊 Comparando distribuições ANTES e DEPOIS da remoção de outliers\n")

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('ANÁLISE DE OUTLIERS - Boxplots Antes e Depois da Remoção (SISTEMA)', fontsize=16, fontweight='bold')

for idx, col in enumerate(tempos_cols):
    col_clean = f"{col}_clean"
    
    # Boxplot ANTES (com outliers)
    ax1 = axes[0, idx]
    data_antes = df_sys_pd[col].dropna()
    if len(data_antes) > 0:
        bp1 = ax1.boxplot(data_antes, vert=True, patch_artist=True, 
                          boxprops=dict(facecolor='lightcoral', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2),
                          flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5))
        ax1.set_title(f"{col.replace('tempo_', '').replace('_', ' → ').upper()}\nANTES (com outliers)", 
                     fontsize=11, fontweight='bold')
        ax1.set_ylabel('Tempo (minutos)', fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # Adicionar estatísticas
        stats_text = f"N={len(data_antes)}\nMédia={data_antes.mean():.1f}\nMediana={data_antes.median():.1f}"
        ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
                verticalalignment='top', horizontalalignment='right', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Boxplot DEPOIS (sem outliers)
    ax2 = axes[1, idx]
    data_depois = df_sys_pd_clean[col_clean].dropna()
    if len(data_depois) > 0:
        bp2 = ax2.boxplot(data_depois, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightgreen', alpha=0.7),
                          medianprops=dict(color='darkgreen', linewidth=2),
                          flierprops=dict(marker='o', markerfacecolor='green', markersize=4, alpha=0.5))
        ax2.set_title(f"{col.replace('tempo_', '').replace('_', ' → ').upper()}\nDEPOIS (sem outliers)", 
                     fontsize=11, fontweight='bold')
        ax2.set_ylabel('Tempo (minutos)', fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        
        # Adicionar estatísticas
        n_removidos = len(data_antes) - len(data_depois)
        stats_text = f"N={len(data_depois)}\nMédia={data_depois.mean():.1f}\nMediana={data_depois.median():.1f}\nRemovidos={n_removidos}"
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.tight_layout()
plt.show()

print("\n✅ Boxplots gerados com sucesso!")
print("="*120)

# COMMAND ----------

# DBTITLE 1,5. Visualização - Histogramas com Percentis Marcados
print("="*120)
print(" "*35 + "VISUALIZAÇÃO: HISTOGRAMAS COM PERCENTIS")
print("="*120)
print("\n📊 Histogramas dos tempos SEM outliers, com percentis P25, P50, P75, P90 marcados\n")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for idx, col in enumerate(tempos_cols):
    ax = axes[idx]
    col_clean = f"{col}_clean"
    data = df_sys_pd_clean[col_clean].dropna()
    
    if len(data) > 0:
        # Histograma
        n, bins, patches = ax.hist(data, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Calcular percentis
        p25 = data.quantile(0.25)
        p50 = data.quantile(0.50)
        p75 = data.quantile(0.75)
        p90 = data.quantile(0.90)
        media = data.mean()
        
        # Linhas de percentis
        ax.axvline(p25, color='green', linestyle='--', linewidth=2, label=f'P25: {p25:.1f} min')
        ax.axvline(p50, color='orange', linestyle='--', linewidth=2.5, label=f'P50 (Mediana): {p50:.1f} min')
        ax.axvline(p75, color='red', linestyle='--', linewidth=2, label=f'P75: {p75:.1f} min')
        ax.axvline(p90, color='darkred', linestyle='--', linewidth=2, label=f'P90: {p90:.1f} min')
        ax.axvline(media, color='blue', linestyle=':', linewidth=2, label=f'Média: {media:.1f} min')
        
        # Título e labels
        etapa_nome = col.replace('tempo_', '').replace('_', ' → ').upper()
        ax.set_title(f'{etapa_nome}\n(Sem Outliers - SISTEMA)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Tempo (minutos)', fontsize=11)
        ax.set_ylabel('Frequência', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Estatísticas
        stats_text = f"N = {len(data)}\nStd = {data.std():.2f} min"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
plt.suptitle('HISTOGRAMAS DE TEMPOS ENTRE ETAPAS (Sem Outliers - SISTEMA) - Com Percentis', 
             fontsize=14, fontweight='bold', y=1.002)
plt.show()

print("\n✅ Histogramas com percentis gerados com sucesso!")
print("="*120)

# COMMAND ----------

# DBTITLE 1,6. Tabela Resumo Estatístico Completo
print("="*120)
print(" "*35 + "TABELA RESUMO ESTATÍSTICO COMPLETO")
print("="*120)
print("\n📋 Comparação de estatísticas ANTES e DEPOIS da remoção de outliers\n")

resumo_completo = []

for col in tempos_cols:
    col_clean = f"{col}_clean"
    etapa_nome = col.replace('tempo_', '').replace('_', ' → ').upper()
    
    # Dados ANTES
    data_antes = df_sys_pd[col].dropna()
    
    # Dados DEPOIS
    data_depois = df_sys_pd_clean[col_clean].dropna()
    
    if len(data_antes) > 0:
        # Estatísticas ANTES
        resumo_completo.append({
            'Etapa': etapa_nome,
            'Condição': 'ANTES (com outliers)',
            'N': len(data_antes),
            'Média (min)': f"{data_antes.mean():.2f}",
            'Mediana (min)': f"{data_antes.median():.2f}",
            'Std (min)': f"{data_antes.std():.2f}",
            'Min (min)': f"{data_antes.min():.2f}",
            'Max (min)': f"{data_antes.max():.2f}",
            'P25 (min)': f"{data_antes.quantile(0.25):.2f}",
            'P75 (min)': f"{data_antes.quantile(0.75):.2f}",
            'P90 (min)': f"{data_antes.quantile(0.90):.2f}",
            'P95 (min)': f"{data_antes.quantile(0.95):.2f}"
        })
    
    if len(data_depois) > 0:
        # Estatísticas DEPOIS
        n_removidos = len(data_antes) - len(data_depois)
        perc_removidos = (n_removidos / len(data_antes)) * 100 if len(data_antes) > 0 else 0
        
        resumo_completo.append({
            'Etapa': etapa_nome,
            'Condição': f'DEPOIS (sem outliers) [-{n_removidos} ({perc_removidos:.1f}%)]',
            'N': len(data_depois),
            'Média (min)': f"{data_depois.mean():.2f}",
            'Mediana (min)': f"{data_depois.median():.2f}",
            'Std (min)': f"{data_depois.std():.2f}",
            'Min (min)': f"{data_depois.min():.2f}",
            'Max (min)': f"{data_depois.max():.2f}",
            'P25 (min)': f"{data_depois.quantile(0.25):.2f}",
            'P75 (min)': f"{data_depois.quantile(0.75):.2f}",
            'P90 (min)': f"{data_depois.quantile(0.90):.2f}",
            'P95 (min)': f"{data_depois.quantile(0.95):.2f}"
        })

df_resumo_completo = pd.DataFrame(resumo_completo)

print("\n📊 TABELA COMPARATIVA:\n")
display(df_resumo_completo)

print("\n" + "="*120)
print("💡 OBSERVAÇÕES:")
print("="*120)
print("  • A remoção de outliers reduz a média e o desvio padrão")
print("  • A mediana é mais robusta a outliers (muda menos)")
print("  • P75 e P90 são bons candidatos para previsão conservadora")
print("  • Min e Max após remoção representam limites mais realistas")
print("="*120)

# COMMAND ----------

# DBTITLE 1,7. Recomendações para Previsão de Ciclo Completo
print("="*120)
print(" "*30 + "🎯 RECOMENDAÇÕES PARA PREVISÃO DE CICLO COMPLETO")
print("="*120)
print("\n📋 Valores recomendados para cada etapa (em minutos) - DADOS DO SISTEMA:\n")

recomendacoes = []

for col in tempos_cols:
    col_clean = f"{col}_clean"
    data = df_sys_pd_clean[col_clean].dropna()
    
    if len(data) > 0:
        etapa_nome = col.replace('tempo_', '').replace('_', ' → ').upper()
        
        recomendacoes.append({
            'Etapa': etapa_nome,
            'Cenário Otimista (P25)': f"{data.quantile(0.25):.2f}",
            'Cenário Típico (P50)': f"{data.quantile(0.50):.2f}",
            'Cenário Conservador (P75)': f"{data.quantile(0.75):.2f}",
            'Cenário Muito Conservador (P90)': f"{data.quantile(0.90):.2f}",
            'Média': f"{data.mean():.2f}",
            'N': len(data)
        })

df_recomendacoes = pd.DataFrame(recomendacoes)

print("\n📊 TABELA DE RECOMENDAÇÕES:\n")
display(df_recomendacoes)

# Calcular tempos totais de ciclo para cada cenário
print("\n" + "="*120)
print("⏱️ TEMPO TOTAL DE CICLO ESTIMADO (FEA → LC):")
print("="*120)

# Extrair apenas as etapas intermediárias (não o ciclo total)
etapas_intermediarias = ['tempo_fea_fp', 'tempo_fp_vd', 'tempo_vd_lc']

for cenario, percentil in [('Otimista', 0.25), ('Típico', 0.50), ('Conservador', 0.75), ('Muito Conservador', 0.90)]:
    tempo_total = 0
    detalhes = []
    
    for col in etapas_intermediarias:
        col_clean = f"{col}_clean"
        data = df_sys_pd_clean[col_clean].dropna()
        if len(data) > 0:
            valor = data.quantile(percentil)
            tempo_total += valor
            etapa_nome = col.replace('tempo_', '').replace('_', ' → ')
            detalhes.append(f"{etapa_nome}: {valor:.2f} min")
    
    print(f"\n  🎯 Cenário {cenario} (P{int(percentil*100)}):")
    print(f"     Tempo Total: {tempo_total:.2f} minutos ({tempo_total/60:.2f} horas)")
    print(f"     Detalhamento: {' + '.join(detalhes)}")

print("\n" + "="*120)
print("💡 COMO USAR ESTAS RECOMENDAÇÕES:")
print("="*120)
print("\n  1. PREVISÃO TÍPICA (P50 - Mediana):")
print("     → Use para estimativa padrão do tempo de ciclo")
print("     → Cobre 50% dos casos (metade das corridas são mais rápidas)")

print("\n  2. PREVISÃO CONSERVADORA (P75):")
print("     → Use quando precisar de margem de segurança")
print("     → Cobre 75% dos casos")
print("     → Recomendado para planejamento operacional")

print("\n  3. PREVISÃO MUITO CONSERVADORA (P90):")
print("     → Use para situações críticas ou planejamento de capacidade")
print("     → Cobre 90% dos casos")
print("     → Inclui margem para imprevistos")

print("\n  4. PREVISÃO OTIMISTA (P25):")
print("     → Use apenas para cenários ideais")
print("     → Apenas 25% das corridas são mais rápidas que isso")
print("     → Não recomendado para planejamento operacional")

print("\n" + "="*120)
print("✅ ANÁLISE ESTATÍSTICA DE TEMPOS ENTRE ETAPAS CONCLUÍDA!")
print("="*120)
print("\n📊 Resumo:")
print(f"  • {len(tempos_cols)} etapas analisadas")
print(f"  • Outliers identificados e removidos usando IQR + Z-Score")
print(f"  • Percentis calculados (P5, P10, P25, P50, P75, P90, P95)")
print(f"  • Recomendações geradas para 4 cenários de previsão")
print("\n🎯 Use a tabela de recomendações para configurar seu modelo de previsão!")
print("="*120)

# COMMAND ----------

# DBTITLE 1,8. Comparação SISTEMA vs OPERADOR - Tempos Entre Etapas
print("="*120)
print(" "*30 + "🔄 COMPARAÇÃO: SISTEMA vs OPERADOR")
print("="*120)
print("\n🎯 OBJETIVO: Comparar os tempos entre etapas detectados pelo SISTEMA vs informados pelo OPERADOR\n")

# Preparar dados do OPERADOR
df_op_pd = df_final.toPandas()

# Garantir que colunas são datetime
for col in cols_datetime:
    if col in df_op_pd.columns:
        df_op_pd[col] = pd.to_datetime(df_op_pd[col], errors='coerce')

# Calcular tempos entre etapas do OPERADOR
df_op_pd['tempo_fea_fp'] = (df_op_pd['inicio_fp'] - df_op_pd['final_fea']).dt.total_seconds() / 60.0
df_op_pd['tempo_fp_vd'] = (df_op_pd['inicio_vd'] - df_op_pd['final_fp']).dt.total_seconds() / 60.0
df_op_pd['tempo_vd_lc'] = (df_op_pd['inicio_lc'] - df_op_pd['final_vd']).dt.total_seconds() / 60.0
df_op_pd['tempo_ciclo_total'] = (df_op_pd['inicio_lc'] - df_op_pd['inicio_fea']).dt.total_seconds() / 60.0

# Remover valores negativos
for col in tempos_cols:
    df_op_pd.loc[df_op_pd[col] < 0, col] = np.nan

print("📊 ESTATÍSTICAS COMPARATIVAS:\n")
print("-" * 120)

comparacao_stats = []

for col in tempos_cols:
    etapa_nome = col.replace('tempo_', '').replace('_', ' → ').upper()
    
    # Dados do SISTEMA (sem outliers)
    col_clean = f"{col}_clean"
    data_sys = df_sys_pd_clean[col_clean].dropna()
    
    # Dados do OPERADOR
    data_op = df_op_pd[col].dropna()
    
    if len(data_sys) > 0 and len(data_op) > 0:
        comparacao_stats.append({
            'Etapa': etapa_nome,
            'N_Sistema': len(data_sys),
            'N_Operador': len(data_op),
            'Média_Sistema': f"{data_sys.mean():.2f}",
            'Média_Operador': f"{data_op.mean():.2f}",
            'Diff_Média': f"{data_sys.mean() - data_op.mean():.2f}",
            'Mediana_Sistema': f"{data_sys.median():.2f}",
            'Mediana_Operador': f"{data_op.median():.2f}",
            'Diff_Mediana': f"{data_sys.median() - data_op.median():.2f}",
            'P75_Sistema': f"{data_sys.quantile(0.75):.2f}",
            'P75_Operador': f"{data_op.quantile(0.75):.2f}"
        })

df_comparacao = pd.DataFrame(comparacao_stats)

print("\n📋 TABELA COMPARATIVA - SISTEMA vs OPERADOR:\n")
display(df_comparacao)

print("\n" + "="*120)
print("🔍 INTERPRETAÇÃO DAS DIFERENÇAS:")
print("="*120)
print("  • Diff_Média/Mediana POSITIVA: Sistema detecta tempos MAIORES que o operador informa")
print("  • Diff_Média/Mediana NEGATIVA: Sistema detecta tempos MENORES que o operador informa")
print("  • Diferenças grandes (>5 min) podem indicar:")
print("    - Atrasos não registrados pelo operador")
print("    - Detecção imprecisa do sistema")
print("    - Diferenças nos critérios de início/fim de etapa")
print("="*120)

# COMMAND ----------

# DBTITLE 1,9. Visualização Comparativa - SISTEMA vs OPERADOR
print("="*120)
print(" "*30 + "VISUALIZAÇÃO COMPARATIVA: SISTEMA vs OPERADOR")
print("="*120)

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('COMPARAÇÃO DE TEMPOS: SISTEMA vs OPERADOR', fontsize=16, fontweight='bold')

for idx, col in enumerate(tempos_cols):
    ax = axes[idx // 2, idx % 2]
    
    col_clean = f"{col}_clean"
    etapa_nome = col.replace('tempo_', '').replace('_', ' → ').upper()
    
    # Dados
    data_sys = df_sys_pd_clean[col_clean].dropna()
    data_op = df_op_pd[col].dropna()
    
    if len(data_sys) > 0 and len(data_op) > 0:
        # Histogramas sobrepostos
        ax.hist(data_sys, bins=30, alpha=0.6, label=f'Sistema (N={len(data_sys)})', 
                color='steelblue', edgecolor='black')
        ax.hist(data_op, bins=30, alpha=0.6, label=f'Operador (N={len(data_op)})', 
                color='coral', edgecolor='black')
        
        # Medianas
        med_sys = data_sys.median()
        med_op = data_op.median()
        ax.axvline(med_sys, color='blue', linestyle='--', linewidth=2.5, 
                   label=f'Mediana Sistema: {med_sys:.1f} min')
        ax.axvline(med_op, color='red', linestyle='--', linewidth=2.5, 
                   label=f'Mediana Operador: {med_op:.1f} min')
        
        # Título e labels
        ax.set_title(f'{etapa_nome}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Tempo (minutos)', fontsize=11)
        ax.set_ylabel('Frequência', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Diferença de medianas
        diff = med_sys - med_op
        color_diff = 'green' if abs(diff) < 2 else 'orange' if abs(diff) < 5 else 'red'
        ax.text(0.02, 0.98, f'Diff Mediana: {diff:+.1f} min', 
                transform=ax.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor=color_diff, alpha=0.6),
                fontweight='bold')

plt.tight_layout()
plt.show()

print("\n✅ Visualizações comparativas geradas!")
print("="*120)

# COMMAND ----------

# DBTITLE 1,10. Análise de Diferenças - SISTEMA vs OPERADOR
print("="*120)
print(" "*30 + "ANÁLISE DE DIFERENÇAS: SISTEMA vs OPERADOR")
print("="*120)
print("\n🔍 Analisando as diferenças entre os tempos detectados pelo SISTEMA e informados pelo OPERADOR\n")

# Fazer merge dos dataframes por corrida
df_merge = df_sys_pd[['corrida'] + tempos_cols].merge(
    df_op_pd[['corrida'] + tempos_cols],
    on='corrida',
    how='inner',
    suffixes=('_sys', '_op')
)

print(f"🔗 Corridas em comum: {len(df_merge)}\n")

# Calcular diferenças
for col in tempos_cols:
    col_sys = f"{col}_sys"
    col_op = f"{col}_op"
    col_diff = f"{col}_diff"
    
    df_merge[col_diff] = df_merge[col_sys] - df_merge[col_op]

print("📊 ESTATÍSTICAS DAS DIFERENÇAS (Sistema - Operador):\n")
print("-" * 120)

diferencas_stats = []

for col in tempos_cols:
    col_diff = f"{col}_diff"
    etapa_nome = col.replace('tempo_', '').replace('_', ' → ').upper()
    
    data_diff = df_merge[col_diff].dropna()
    
    if len(data_diff) > 0:
        # Contar quantas são positivas, negativas, zero
        n_positivas = (data_diff > 0).sum()
        n_negativas = (data_diff < 0).sum()
        n_zero = (data_diff == 0).sum()
        
        diferencas_stats.append({
            'Etapa': etapa_nome,
            'N': len(data_diff),
            'Média_Diff': f"{data_diff.mean():.2f}",
            'Mediana_Diff': f"{data_diff.median():.2f}",
            'Std_Diff': f"{data_diff.std():.2f}",
            'Min_Diff': f"{data_diff.min():.2f}",
            'Max_Diff': f"{data_diff.max():.2f}",
            'Sys>Op': f"{n_positivas} ({n_positivas/len(data_diff)*100:.1f}%)",
            'Sys<Op': f"{n_negativas} ({n_negativas/len(data_diff)*100:.1f}%)",
            'Sys=Op': f"{n_zero} ({n_zero/len(data_diff)*100:.1f}%)"
        })

df_diferencas = pd.DataFrame(diferencas_stats)

print("\n📋 TABELA DE DIFERENÇAS:\n")
display(df_diferencas)

# Visualização das diferenças
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle('ANÁLISE DE DIFERENÇAS: SISTEMA - OPERADOR (minutos)', fontsize=14, fontweight='bold')

for idx, col in enumerate(tempos_cols):
    ax = axes[idx // 2, idx % 2]
    col_diff = f"{col}_diff"
    etapa_nome = col.replace('tempo_', '').replace('_', ' → ').upper()
    
    data_diff = df_merge[col_diff].dropna()
    
    if len(data_diff) > 0:
        # Histograma das diferenças
        ax.hist(data_diff, bins=30, alpha=0.7, color='purple', edgecolor='black')
        
        # Linha zero
        ax.axvline(0, color='black', linestyle='-', linewidth=2, label='Zero (Sistema = Operador)')
        
        # Mediana e média
        med = data_diff.median()
        mean = data_diff.mean()
        ax.axvline(med, color='orange', linestyle='--', linewidth=2, label=f'Mediana: {med:+.1f} min')
        ax.axvline(mean, color='red', linestyle=':', linewidth=2, label=f'Média: {mean:+.1f} min')
        
        # Título e labels
        ax.set_title(f'{etapa_nome}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Diferença (Sistema - Operador) em minutos', fontsize=10)
        ax.set_ylabel('Frequência', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        # Estatísticas
        n_pos = (data_diff > 0).sum()
        n_neg = (data_diff < 0).sum()
        stats_text = f"N={len(data_diff)}\nSys>Op: {n_pos}\nSys<Op: {n_neg}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
plt.show()

print("\n" + "="*120)
print("💡 INTERPRETAÇÃO:")
print("="*120)
print("  • Diferença POSITIVA: Sistema detecta tempo MAIOR que operador informa")
print("  • Diferença NEGATIVA: Sistema detecta tempo MENOR que operador informa")
print("  • Diferença próxima de ZERO: Boa concordância entre sistema e operador")
print("\n  🎯 QUAL USAR PARA PREVISÃO?")
print("     - Se diferenças são pequenas (<2 min): Ambos são confiáveis")
print("     - Se diferenças são grandes (>5 min): Investigar causa e usar SISTEMA (mais preciso)")
print("     - SISTEMA é baseado em sensores (mais objetivo)")
print("     - OPERADOR é baseado em registro manual (pode ter atrasos)")
print("="*120)