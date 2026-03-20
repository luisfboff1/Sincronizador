# Databricks notebook source
# DBTITLE 1,IMPORTS
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

# DBTITLE 1,FUNÇÕES AUXILIARES
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

# DBTITLE 1,FUNÇÃO FP
# ============================================================
# CÉLULA 3 — FUNÇÃO FP (PIMS gás total)
# ============================================================

def detect_fp_events_gas_based(
    spark_df: pyspark.sql.DataFrame,
    start_ts: str,
    end_ts: str,
    initial_heat_number: int,

    gas_threshold: float = 100.0,
    min_peak_total_gas: float = 800.0,
    minimal_duration_seconds: int = 60,
    end_fraction_of_peak: float = 0.90,
    min_start_rel_increase: float = 0.05,
    debug: bool = False,
) -> pyspark.sql.DataFrame:
    """
    Detecta FP via gás total (PIMS).

    Retorna Spark DF:
      corrida | inicio_fp | final_fp | fp_carro
    """

    TIMESTAMP_COL = "timestamp"
    COL_CORRIDA   = "ACI@FP_NUMERO_CORRIDA"

    COL_AR_C1     = "ACI@FP_Volume_Argonio_Carro_1"
    COL_N2_C1     = "ACI@FP_Volume_Nitrogenio_Carro_1"
    COL_AR_C2     = "ACI@FP_Volume_Argonio_Carro_2"
    COL_N2_C2     = "ACI@FP_Volume_Nitrogenio_Carro_2"

    cols_fill = [COL_AR_C1, COL_N2_C1, COL_AR_C2, COL_N2_C2]

    df_fp = preprocess_pims_data(
        spark_df=spark_df,
        first_timestamp=start_ts,
        last_timestamp=end_ts,
        timestamp_column=TIMESTAMP_COL,
        columns_to_fill=cols_fill,
    ).orderBy(TIMESTAMP_COL)

    # corrida_raw (quando existir) — pode ajudar depois, mas não é obrigatório
    df_fp = df_fp.withColumn(
        "corrida_raw",
        F.when(F.col(COL_CORRIDA).cast("int") > 0, F.col(COL_CORRIDA).cast("int"))
         .otherwise(F.lit(None).cast("int"))
    )

    # gás total e por carro
    df_fp = df_fp.select(
        "*",
        (F.col(COL_AR_C1).cast("double") + F.col(COL_N2_C1).cast("double")).alias("gas_carro1"),
        (F.col(COL_AR_C2).cast("double") + F.col(COL_N2_C2).cast("double")).alias("gas_carro2"),
    ).withColumn(
        "gas_total",
        F.coalesce(F.col("gas_carro1"), F.lit(0.0)) + F.coalesce(F.col("gas_carro2"), F.lit(0.0))
    )

    w = Window.orderBy(TIMESTAMP_COL)

    df_fp = df_fp.withColumn("prev_gas_total", F.lag("gas_total", 1, 0.0).over(w))
    df_fp = df_fp.withColumn("next_gas_total", F.lead("gas_total", 1, 0.0).over(w))

    df_fp = df_fp.withColumn(
        "rel_increase_prev",
        F.when(
            F.col("prev_gas_total") > 0,
            (F.col("gas_total") - F.col("prev_gas_total")) / F.col("prev_gas_total")
        ).otherwise(F.lit(1.0))
    ).withColumn(
        "rel_increase_next",
        F.when(
            F.col("gas_total") > 0,
            (F.col("next_gas_total") - F.col("gas_total")) / F.col("gas_total")
        ).otherwise(F.lit(0.0))
    )

    df_fp = df_fp.withColumn(
        "is_start_fp",
        (F.col("prev_gas_total") <= gas_threshold) &
        (F.col("gas_total") > gas_threshold) &
        (
            (F.col("rel_increase_prev") >= min_start_rel_increase) |
            (F.col("rel_increase_next") >= min_start_rel_increase)
        )
    )

    # session_id global
    df_fp = df_fp.withColumn("session_id", F.sum(F.col("is_start_fp").cast("long")).over(w))

    # só sessões > 0
    df_sess = df_fp.filter(F.col("session_id") > 0)

    w_sess = Window.partitionBy("session_id").orderBy(TIMESTAMP_COL).rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

    # pico total + pico por carro
    df_sess = df_sess.withColumn("peak_total_gas", F.max("gas_total").over(w_sess))
    df_sess = df_sess.withColumn("peak_gas_carro1", F.max("gas_carro1").over(w_sess))
    df_sess = df_sess.withColumn("peak_gas_carro2", F.max("gas_carro2").over(w_sess))

    # timestamp do pico (último pico)
    df_sess = df_sess.withColumn("is_peak", F.col("gas_total") == F.col("peak_total_gas"))
    df_sess = df_sess.withColumn("peak_ts", F.max(F.when(F.col("is_peak"), F.col(TIMESTAMP_COL))).over(w_sess))

    # nível dinâmico de fim
    df_sess = df_sess.withColumn("dynamic_end_level", F.col("peak_total_gas") * F.lit(end_fraction_of_peak))
    df_sess = df_sess.withColumn(
        "is_dynamic_end_candidate",
        (F.col(TIMESTAMP_COL) >= F.col("peak_ts")) &
        (F.col("gas_total") <= F.col("dynamic_end_level"))
    )

    df_sess = df_sess.withColumn("dynamic_end_ts", F.when(F.col("is_dynamic_end_candidate"), F.col(TIMESTAMP_COL)))

    agg_raw = (
        df_sess
        .groupBy("session_id")
        .agg(
            F.min(TIMESTAMP_COL).alias("inicio_fp"),
            F.max(TIMESTAMP_COL).alias("fp_last_seen"),
            F.max("peak_total_gas").alias("peak_total_gas"),
            F.max("peak_gas_carro1").alias("peak_gas_carro1"),
            F.max("peak_gas_carro2").alias("peak_gas_carro2"),
            F.min("dynamic_end_ts").alias("final_fp_dynamic"),
        )
        .withColumn("final_fp", F.coalesce(F.col("final_fp_dynamic"), F.col("fp_last_seen")))
        .withColumn("duration_seconds", F.col("final_fp").cast("long") - F.col("inicio_fp").cast("long"))
    )

    # filtra sessões válidas (FP real)
    agg_valid = (
        agg_raw
        .filter(
            (F.col("duration_seconds") >= minimal_duration_seconds) &
            (F.col("peak_total_gas") >= min_peak_total_gas)
        )
        .withColumn(
            "fp_carro",
            F.when(F.col("peak_gas_carro1") >= F.col("peak_gas_carro2"), F.lit(1)).otherwise(F.lit(2))
        )
    )

    # corrida crescente = initial_heat_number + índice das sessões válidas
    w_events = Window.orderBy("inicio_fp")

    agg = (
        agg_valid
        .withColumn("event_idx", (F.row_number().over(w_events) - 1).cast("int"))
        .withColumn("corrida", (F.lit(initial_heat_number) + F.col("event_idx")).cast("int"))
        .select("corrida", "inicio_fp", "final_fp", "fp_carro")
        .orderBy("corrida")
    )

    if debug:
        display(agg)

    return agg


# COMMAND ----------

import pandas as pd
import numpy as np
import pyspark.sql
from pyspark.sql import functions as F


# ============================================================
# FP — FSM streaming-ready (sem olhar para o futuro)
# ============================================================
def detect_fp_events_fsm_gas(
    spark_df: pyspark.sql.DataFrame,
    start_ts: str,
    end_ts: str,
    initial_heat_number: int,

    gas_threshold: float = 100.0,
    min_peak_total_gas: float = 800.0,
    minimal_duration_seconds: int = 60,
    end_fraction_of_peak: float = 0.90,
    min_start_rel_increase: float = 0.05,

    # confirmação "no futuro do candidato" (como no LC)
    start_confirm_window_s: int = 15,          # espera até Xs para validar o início
    start_confirm_min_sustain_s: int = 5,      # dentro da janela, exigir pelo menos Xs acima do threshold

    debug: bool = False,
):
    """
    FSM streaming-ready para detecção de FP via gás (PIMS), sem lead/janelas futuras.

    Retorna 3 Spark DFs:
      1) df_fp_final  : corrida | inicio_fp_sys | final_fp_sys | fp_carro
      2) df_fp_events : corrida | status | inicio_fp_sys | final_fp_sys | fp_carro | confirmed_at | event_ts
      3) df_fp_debug  : trace detalhado (estado/eventos)

    Observações:
      - "inicio_fp_sys" é o timestamp real do início (primeiro cruzamento validado).
      - "confirmed_at" é quando o sistema confirmou (após acumular evidência dentro da janela).
      - "final_fp_sys" é quando confirmou o fim (primeira queda <= fração do pico após o pico).
      - Não usa ACI@FP_NUMERO_CORRIDA para o ID da corrida: gera sequencial a partir de initial_heat_number.
    """

    # ----------------------------
    # Colunas
    # ----------------------------
    TIMESTAMP_COL = "timestamp"

    COL_AR_C1 = "ACI@FP_Volume_Argonio_Carro_1"
    COL_N2_C1 = "ACI@FP_Volume_Nitrogenio_Carro_1"
    COL_AR_C2 = "ACI@FP_Volume_Argonio_Carro_2"
    COL_N2_C2 = "ACI@FP_Volume_Nitrogenio_Carro_2"
    cols_fill = [COL_AR_C1, COL_N2_C1, COL_AR_C2, COL_N2_C2]

    # ----------------------------
    # 1) Pré-processamento Spark → pandas
    # ----------------------------
    df_fp_sp = preprocess_pims_data(
        spark_df=spark_df,
        first_timestamp=start_ts,
        last_timestamp=end_ts,
        timestamp_column=TIMESTAMP_COL,
        columns_to_fill=cols_fill,
    ).orderBy(TIMESTAMP_COL)

    df_fp_pd = (
        df_fp_sp
        .select(TIMESTAMP_COL, COL_AR_C1, COL_N2_C1, COL_AR_C2, COL_N2_C2)
        .toPandas()
    )

    if df_fp_pd.empty:
        # retorna DFs vazios, com schema consistente
        spark = spark_df.sql_ctx.sparkSession

        df_fp_final = spark.createDataFrame(
            pd.DataFrame(columns=["corrida", "inicio_fp_sys", "final_fp_sys", "fp_carro"])
        )
        df_fp_events = spark.createDataFrame(
            pd.DataFrame(columns=["corrida", "status", "inicio_fp_sys", "final_fp_sys", "fp_carro", "confirmed_at", "event_ts"])
        )
        df_fp_debug = spark.createDataFrame(
            pd.DataFrame(columns=["i", "timestamp", "state", "event", "gas_total", "gas_carro1", "gas_carro2", "peak_total", "peak_ts", "candidate_ts", "confirmed_at", "reason"])
        )
        return df_fp_final, df_fp_events, df_fp_debug

    df_fp_pd[TIMESTAMP_COL] = pd.to_datetime(df_fp_pd[TIMESTAMP_COL], errors="coerce")
    df_fp_pd = df_fp_pd.dropna(subset=[TIMESTAMP_COL]).sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # numéricos
    for c in [COL_AR_C1, COL_N2_C1, COL_AR_C2, COL_N2_C2]:
        df_fp_pd[c] = pd.to_numeric(df_fp_pd[c], errors="coerce")

    df_fp_pd["gas_carro1"] = df_fp_pd[COL_AR_C1].fillna(0.0) + df_fp_pd[COL_N2_C1].fillna(0.0)
    df_fp_pd["gas_carro2"] = df_fp_pd[COL_AR_C2].fillna(0.0) + df_fp_pd[COL_N2_C2].fillna(0.0)
    df_fp_pd["gas_total"] = df_fp_pd["gas_carro1"].fillna(0.0) + df_fp_pd["gas_carro2"].fillna(0.0)

    # prev (sem futuro)
    df_fp_pd["prev_gas_total"] = df_fp_pd["gas_total"].shift(1).fillna(0.0)
    df_fp_pd["rel_increase_prev"] = np.where(
        df_fp_pd["prev_gas_total"] > 0,
        (df_fp_pd["gas_total"] - df_fp_pd["prev_gas_total"]) / df_fp_pd["prev_gas_total"],
        1.0
    )

    # ----------------------------
    # 2) FSM
    # ----------------------------
    state = "IDLE"
    corrida_seq = int(initial_heat_number)

    # contexto do candidato
    cand_start_ts = None
    cand_confirm_deadline = None
    cand_sustain_s = 0.0
    cand_confirmed_at = None

    # contexto de corrida em RUNNING
    run_corrida = None
    run_start_ts = None
    run_confirmed_at = None

    peak_total = None
    peak_ts = None
    peak_c1 = None
    peak_c2 = None

    # controle de fim
    after_peak = False

    # outputs
    final_rows = []    # consolidado
    event_rows = []    # TEMP append-only (status changes)
    debug_rows = []    # trace

    def dbg(i, ts, event, reason=None):
        if not debug:
            return
        debug_rows.append({
            "i": int(i),
            "timestamp": ts,
            "state": state,
            "event": event,
            "gas_total": float(df_fp_pd.at[i, "gas_total"]),
            "gas_carro1": float(df_fp_pd.at[i, "gas_carro1"]),
            "gas_carro2": float(df_fp_pd.at[i, "gas_carro2"]),
            "peak_total": (float(peak_total) if peak_total is not None else None),
            "peak_ts": peak_ts,
            "candidate_ts": cand_start_ts,
            "confirmed_at": (run_confirmed_at if run_confirmed_at is not None else cand_confirmed_at),
            "reason": reason,
        })

    def finalize_open_run_if_any(last_ts):
        """Se acabar janela e ainda estiver RUNNING, mantém aberto (final null)."""
        nonlocal state, run_corrida, run_start_ts, run_confirmed_at
        nonlocal peak_total, peak_ts, peak_c1, peak_c2, after_peak

        if state != "RUNNING":
            return

        fp_carro = 1 if (peak_c1 is not None and peak_c2 is not None and peak_c1 >= peak_c2) else 2
        final_rows.append({
            "corrida": run_corrida,
            "inicio_fp_sys": run_start_ts,
            "final_fp_sys": None,
            "fp_carro": fp_carro,
        })
        dbg(len(df_fp_pd) - 1, last_ts, "RUN_LEFT_OPEN", reason="fim da janela sem detectar fechamento")
        state = "IDLE"
        run_corrida = None
        run_start_ts = None
        run_confirmed_at = None
        peak_total = None
        peak_ts = None
        peak_c1 = None
        peak_c2 = None
        after_peak = False

    n = len(df_fp_pd)

    for i in range(n):
        ts = df_fp_pd.at[i, TIMESTAMP_COL]
        gas = float(df_fp_pd.at[i, "gas_total"])
        prev_gas = float(df_fp_pd.at[i, "prev_gas_total"])
        rel_prev = float(df_fp_pd.at[i, "rel_increase_prev"])
        g1 = float(df_fp_pd.at[i, "gas_carro1"])
        g2 = float(df_fp_pd.at[i, "gas_carro2"])

        # delta t (para sustento e duração)
        if i == 0:
            dt_s = 0.0
        else:
            dt_s = (ts - df_fp_pd.at[i - 1, TIMESTAMP_COL]).total_seconds()
            if not np.isfinite(dt_s) or dt_s < 0:
                dt_s = 0.0

        # ----------------------------
        # IDLE: procura candidato de START
        # ----------------------------
        if state == "IDLE":
            is_cross = (prev_gas <= gas_threshold) and (gas > gas_threshold)
            is_rel_ok = (rel_prev >= min_start_rel_increase)

            if is_cross and is_rel_ok:
                cand_start_ts = ts
                cand_confirm_deadline = ts + pd.Timedelta(seconds=int(start_confirm_window_s))
                cand_sustain_s = 0.0
                cand_confirmed_at = None

                state = "START_CANDIDATE"
                dbg(i, ts, "START_CANDIDATE_OPEN", reason=f"cross>{gas_threshold} & rel_prev>={min_start_rel_increase}")
            else:
                dbg(i, ts, "IDLE_TICK")
            continue

        # ----------------------------
        # START_CANDIDATE: espera evidência (no "futuro do candidato")
        # ----------------------------
        if state == "START_CANDIDATE":
            # acumula sustain acima do threshold
            if gas > gas_threshold:
                cand_sustain_s += dt_s

            # aborta se voltou abaixo do threshold e ainda não sustentou nada relevante
            if gas <= gas_threshold and cand_sustain_s < float(start_confirm_min_sustain_s):
                dbg(i, ts, "START_CANDIDATE_ABORT", reason="voltou abaixo do threshold sem sustain suficiente")
                state = "IDLE"
                cand_start_ts = None
                cand_confirm_deadline = None
                cand_sustain_s = 0.0
                cand_confirmed_at = None
                continue

            # condição de confirmação do START:
            # - dentro da janela: atingiu sustain mínimo acima do threshold
            # - e "passou tempo mínimo" desde início (para não confirmar em 1 ponto)
            elapsed = (ts - cand_start_ts).total_seconds() if cand_start_ts is not None else 0.0

            if (cand_start_ts is not None) and (ts <= cand_confirm_deadline):
                if (cand_sustain_s >= float(start_confirm_min_sustain_s)) and (elapsed >= 1.0):
                    # confirma corrida e entra em RUNNING
                    run_corrida = corrida_seq
                    corrida_seq += 1

                    run_start_ts = cand_start_ts
                    run_confirmed_at = ts

                    # inicializa pico com o que temos até agora
                    peak_total = gas
                    peak_ts = ts
                    peak_c1 = g1
                    peak_c2 = g2
                    after_peak = False

                    # emite TEMP (RUNNING)
                    event_rows.append({
                        "corrida": run_corrida,
                        "status": "RUNNING",
                        "inicio_fp_sys": run_start_ts,
                        "final_fp_sys": None,
                        "fp_carro": None,
                        "confirmed_at": run_confirmed_at,
                        "event_ts": run_confirmed_at,
                    })

                    dbg(i, ts, "START_CONFIRMED", reason="sustain + janela de confirmação")

                    # limpa candidato
                    cand_start_ts = None
                    cand_confirm_deadline = None
                    cand_sustain_s = 0.0
                    cand_confirmed_at = None

                    state = "RUNNING"
                    continue
                else:
                    dbg(i, ts, "START_CANDIDATE_TICK", reason=f"sustain={cand_sustain_s:.1f}s elapsed={elapsed:.1f}s")
                    continue

            # passou da janela e não confirmou → descarta
            if (cand_start_ts is not None) and (ts > cand_confirm_deadline):
                dbg(i, ts, "START_CANDIDATE_EXPIRED", reason="janela de confirmação expirou")
                state = "IDLE"
                cand_start_ts = None
                cand_confirm_deadline = None
                cand_sustain_s = 0.0
                cand_confirmed_at = None
                continue

            dbg(i, ts, "START_CANDIDATE_TICK")
            continue

        # ----------------------------
        # RUNNING: atualiza pico e detecta fim sem olhar futuro
        # ----------------------------
        if state == "RUNNING":
            # atualiza pico (streaming)
            if (peak_total is None) or (gas >= float(peak_total)):
                peak_total = gas
                peak_ts = ts
                peak_c1 = g1
                peak_c2 = g2
                after_peak = False
                dbg(i, ts, "PEAK_UPDATE", reason="novo pico")
            else:
                # já estamos após o pico se o ts passou do peak_ts
                if peak_ts is not None and ts > peak_ts:
                    after_peak = True
                dbg(i, ts, "RUNNING_TICK", reason="sem novo pico")

            # só permite fim depois do pico
            if after_peak and peak_total is not None:
                dynamic_end_level = float(peak_total) * float(end_fraction_of_peak)

                # condição de fim: cruzou abaixo do nível dinâmico
                if gas <= dynamic_end_level:
                    end_ts = ts
                    fp_carro = 1 if (peak_c1 is not None and peak_c2 is not None and peak_c1 >= peak_c2) else 2

                    # consolida corrida
                    final_rows.append({
                        "corrida": run_corrida,
                        "inicio_fp_sys": run_start_ts,
                        "final_fp_sys": end_ts,
                        "fp_carro": fp_carro,
                    })

                    # emite TEMP (CLOSED)
                    event_rows.append({
                        "corrida": run_corrida,
                        "status": "CLOSED",
                        "inicio_fp_sys": run_start_ts,
                        "final_fp_sys": end_ts,
                        "fp_carro": fp_carro,
                        "confirmed_at": end_ts,
                        "event_ts": end_ts,
                    })

                    dbg(i, ts, "END_CONFIRMED", reason=f"gas<=peak*{end_fraction_of_peak} ({gas:.2f}<={dynamic_end_level:.2f})")

                    # reseta estado
                    state = "IDLE"
                    run_corrida = None
                    run_start_ts = None
                    run_confirmed_at = None
                    peak_total = None
                    peak_ts = None
                    peak_c1 = None
                    peak_c2 = None
                    after_peak = False
                    continue

            continue

        # fallback
        dbg(i, ts, "UNKNOWN_STATE", reason=str(state))
        state = "IDLE"

    # se terminou RUNNING, deixa aberto (final null)
    finalize_open_run_if_any(df_fp_pd.at[n - 1, TIMESTAMP_COL])

    # ----------------------------
    # 3) Monta DataFrames (pandas → spark)
    # ----------------------------
    df_fp_final_pd = pd.DataFrame(final_rows, columns=["corrida", "inicio_fp_sys", "final_fp_sys", "fp_carro"])
    df_fp_events_pd = pd.DataFrame(event_rows, columns=["corrida", "status", "inicio_fp_sys", "final_fp_sys", "fp_carro", "confirmed_at", "event_ts"])
    df_fp_debug_pd = pd.DataFrame(debug_rows, columns=[
        "i", "timestamp", "state", "event",
        "gas_total", "gas_carro1", "gas_carro2",
        "peak_total", "peak_ts",
        "candidate_ts", "confirmed_at",
        "reason"
    ])

    # garantir tipos
    for c in ["inicio_fp_sys", "final_fp_sys", "confirmed_at", "event_ts", "timestamp", "peak_ts", "candidate_ts"]:
        if c in df_fp_final_pd.columns:
            df_fp_final_pd[c] = pd.to_datetime(df_fp_final_pd[c], errors="coerce")
        if c in df_fp_events_pd.columns:
            df_fp_events_pd[c] = pd.to_datetime(df_fp_events_pd[c], errors="coerce")
        if c in df_fp_debug_pd.columns:
            df_fp_debug_pd[c] = pd.to_datetime(df_fp_debug_pd[c], errors="coerce")

    # Spark timestamp é tz-naive
    for _df in [df_fp_final_pd, df_fp_events_pd, df_fp_debug_pd]:
        for c in _df.columns:
            if "ts" in c or c in ["timestamp", "inicio_fp_sys", "final_fp_sys", "confirmed_at", "event_ts", "peak_ts", "candidate_ts"]:
                if c in _df.columns:
                    try:
                        _df[c] = pd.to_datetime(_df[c], errors="coerce").dt.tz_localize(None)
                    except Exception:
                        pass

    spark = spark_df.sql_ctx.sparkSession
    df_fp_final_sp = spark.createDataFrame(df_fp_final_pd)
    df_fp_events_sp = spark.createDataFrame(df_fp_events_pd)
    df_fp_debug_sp = spark.createDataFrame(df_fp_debug_pd)

    if debug:
        display(df_fp_final_sp.orderBy("corrida"))
        display(df_fp_events_sp.orderBy("event_ts"))
        display(df_fp_debug_sp.orderBy("timestamp"))

    return df_fp_final_sp, df_fp_events_sp, df_fp_debug_sp





# COMMAND ----------

# # ============================================================
# # EXEMPLO DE USO
# # ============================================================
# df_fp_new, df_fp_events, df_fp_debug = detect_fp_events_fsm_gas(
#     spark_df=df_fp_raw,
#     start_ts="2025-12-03 00:00:00",
#     end_ts="2025-12-03 23:59:59",
#     initial_heat_number=130475,
#     gas_threshold=100.0,
#     min_peak_total_gas=800.0,
#     minimal_duration_seconds=60,
#     end_fraction_of_peak=0.90,
#     min_start_rel_increase=0.05,
#     debug=True,
# )

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window


def compare_fp_old_vs_fsm(
    spark_fp_raw,

    start_ts: str,
    end_ts: str,
    initial_heat_number: int = 0,

    # ===== parâmetros FP =====
    gas_threshold: float = 100.0,
    min_peak_total_gas: float = 800.0,
    minimal_duration_seconds: int = 60,
    end_fraction_of_peak: float = 0.90,
    min_start_rel_increase: float = 0.05,

    debug: bool = False,
):
    """
    Executa:
      1) FP antigo (batch)
      2) FP novo (FSM)
      3) Cria tabela comparativa (início / fim / diffs)

    Retorna:
      df_compare (Spark)
    """

    # ======================================================
    # 1️⃣ FP ANTIGO (batch atual)
    # ======================================================
    df_fp_old = detect_fp_events_gas_based(
        spark_df=spark_fp_raw,
        start_ts=start_ts,
        end_ts=end_ts,
        initial_heat_number=initial_heat_number,
        gas_threshold=gas_threshold,
        min_peak_total_gas=min_peak_total_gas,
        minimal_duration_seconds=minimal_duration_seconds,
        end_fraction_of_peak=end_fraction_of_peak,
        min_start_rel_increase=min_start_rel_increase,
        debug=debug,
    ).withColumnRenamed("inicio_fp", "inicio_fp_old") \
     .withColumnRenamed("final_fp", "final_fp_old") \
     .withColumnRenamed("fp_carro", "fp_carro_old")

    # ======================================================
    # 2️⃣ FP NOVO (FSM)
    # ======================================================
    df_fp_new, df_fp_events = detect_fp_events_gas_based_fsm(
        spark_df=spark_fp_raw,
        start_ts=start_ts,
        end_ts=end_ts,
        initial_heat_number=initial_heat_number,
        gas_threshold=gas_threshold,
        min_peak_total_gas=min_peak_total_gas,
        minimal_duration_seconds=minimal_duration_seconds,
        end_fraction_of_peak=end_fraction_of_peak,
        min_start_rel_increase=min_start_rel_increase,
        debug=debug,
    )

    df_fp_new = (
        df_fp_new
        .withColumnRenamed("inicio_fp", "inicio_fp_new")
        .withColumnRenamed("final_fp", "final_fp_new")
        .withColumnRenamed("fp_carro", "fp_carro_new")
    )

    # ======================================================
    # 3️⃣ JOIN + DIFFS
    # ======================================================
    df_compare = (
        df_fp_old.alias("old")
        .join(df_fp_new.alias("new"), on="corrida", how="full")
        .select(
            F.col("corrida"),

            # -------- INÍCIO --------
            F.col("old.inicio_fp_old"),
            F.col("new.inicio_fp_new"),
            (
                F.col("new.inicio_fp_new").cast("long") -
                F.col("old.inicio_fp_old").cast("long")
            ).alias("diff_inicio_fp_sec"),

            # -------- FIM --------
            F.col("old.final_fp_old"),
            F.col("new.final_fp_new"),
            (
                F.col("new.final_fp_new").cast("long") -
                F.col("old.final_fp_old").cast("long")
            ).alias("diff_final_fp_sec"),

            # -------- CARRO --------
            F.col("old.fp_carro_old"),
            F.col("new.fp_carro_new"),
        )
        .orderBy("corrida")
    )

    return df_compare, df_fp_old, df_fp_new, df_fp_events


# COMMAND ----------

# df_fp_new, df_fp_events = detect_fp_events_fsm_gas(
#     spark_df=df_fp_raw,
#     start_ts="2025-12-03 00:00:00",
#     end_ts="2025-12-03 23:59:59",
#     initial_heat_number=130475,
#     gas_threshold=100.0,
#     min_peak_total_gas=800.0,
#     minimal_duration_seconds=60,
#     end_fraction_of_peak=0.90,
#     min_start_rel_increase=0.05,
#     debug=True,   # 🔥 IMPORTANTE
# )


# COMMAND ----------

# display(df_fp_events)


# COMMAND ----------

# df_compare, df_fp_old, df_fp_new, df_fp_events = compare_fp_old_vs_fsm(
#     spark_fp_raw=df_fp_raw,
#     start_ts="2025-12-03 00:00:00",
#     end_ts="2025-12-03 23:59:59",
#     initial_heat_number=130475,

#     gas_threshold=100.0,
#     min_peak_total_gas=800.0,
#     minimal_duration_seconds=60,
#     end_fraction_of_peak=0.90,
#     min_start_rel_increase=0.05,

#     debug=False,
# )

# display(df_compare)


# COMMAND ----------

# DBTITLE 1,FUNÇÃO VD
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

# DBTITLE 1,FUNÇÃO LC
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

# DBTITLE 1,FUNÇÃO ÚNICA
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

# DBTITLE 1,LER TABELAS
# ============================================================
# CÉLULA 7 — LEITURA DAS TABELAS (PIMS RAW)
# ============================================================

df_fp_raw = spark.table("industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_fp") \
    .withColumn("timestamp", F.col("timestamp") - F.expr("INTERVAL 3 HOURS"))
df_vd_raw = spark.table("industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_vd") \
    .withColumn("timestamp", F.col("timestamp") - F.expr("INTERVAL 3 HOURS"))
df_lc_raw = spark.table("industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_lc") \
    .withColumn("timestamp", F.col("timestamp") - F.expr("INTERVAL 3 HOURS"))


# COMMAND ----------

# DBTITLE 1,PARAMETROS
START_TS = "2025-12-03 00:00:00"
END_TS   = "2025-12-03 12:59:59"



INITIAL_HEAT_FP = 130475
INITIAL_HEAT_VD = 130474
INITIAL_HEAT_LC = 130473


fp_params = dict(
    gas_threshold=100.0,
    min_peak_total_gas=3000.0,
    minimal_duration_seconds=60,
    end_fraction_of_peak=0.90,
    min_start_rel_increase=0.05,
    debug=False,
)

vd_params = dict(
    reset_threshold=0.99,
    minimal_duration_seconds=10,
    debug=False,
)

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

# COMMAND ----------

# DBTITLE 1,TABELA ÚINICA
# ============================================================
# CÉLULA 8 — PARÂMETROS + EXECUÇÃO
# ============================================================



df_status = build_status_corrida_table(
    spark_fp_raw=df_fp_raw,
    spark_vd_raw=df_vd_raw,
    spark_lc_raw=df_lc_raw,

    start_ts=START_TS,
    end_ts=END_TS,

    initial_heat_fp=INITIAL_HEAT_FP,
    initial_heat_vd=INITIAL_HEAT_VD,
    initial_heat_lc=INITIAL_HEAT_LC,

    fp_params=fp_params,
    vd_params=vd_params,
    lc_params=lc_params,
)

display(df_status)



# COMMAND ----------

# DBTITLE 1,DEBUG DAS FUNÇÕES
# ============================================================
# DEBUG ISOLADO — FP / VD / LC (robusto a diferenças de assinatura)
# ============================================================

from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd


def _call_detect_fp(detector_func, spark_df, initial_heat, start_ts, end_ts, fp_params: dict):
    """
    Chama detect_fp_events_gas_based mesmo que a assinatura mude entre notebooks.
    Tenta:
      1) initial_timestamp/last_timestamp
      2) start_ts/end_ts
    """
    try:
        return detector_func(
            spark_df=spark_df,
            initial_heat_number=initial_heat,
            initial_timestamp=start_ts,
            last_timestamp=end_ts,
            **fp_params
        )
    except TypeError as e1:
        # tenta o outro padrão
        try:
            return detector_func(
                spark_df=spark_df,
                initial_heat_number=initial_heat,
                start_ts=start_ts,
                end_ts=end_ts,
                **fp_params
            )
        except TypeError as e2:
            raise TypeError(
                "Não consegui chamar detect_fp_events_gas_based com nenhuma assinatura conhecida.\n"
                f"Erro c/ initial_timestamp/last_timestamp: {e1}\n"
                f"Erro c/ start_ts/end_ts: {e2}\n"
                "=> Cole aqui a assinatura atual da sua função (primeiras linhas do def) que eu ajusto 100%."
            )


def debug_individual_detectors(
    spark_fp_raw,
    spark_vd_raw,
    spark_lc_raw,
    start_ts: str,
    end_ts: str,
    initial_heat: int,
    fp_params: dict,
    vd_params: dict,
    lc_params: dict,
):
    # ============================================================
    # FP
    # ============================================================
    print("=" * 80)
    print("DEBUG FP (isolado)")
    print("=" * 80)

    df_fp = _call_detect_fp(
        detector_func=detect_fp_events_gas_based,
        spark_df=spark_fp_raw,
        initial_heat=initial_heat,
        start_ts=start_ts,
        end_ts=end_ts,
        fp_params=fp_params
    ).orderBy("corrida")

    display(df_fp)

    print("FP — resumo:")
    df_fp.select(
        F.count("*").alias("n_corridas"),
        F.min("corrida").alias("min_corrida"),
        F.max("corrida").alias("max_corrida"),
        F.min("inicio_fp").alias("min_inicio_fp"),
        F.max("final_fp").alias("max_final_fp"),
    ).show(truncate=False)

    print("\nFP — gaps de corrida:")
    fp_gaps = (
        df_fp
        .select(
            "corrida",
            (F.col("corrida") - F.lag("corrida").over(Window.orderBy("corrida"))).alias("delta")
        )
        .filter(F.col("delta") > 1)
    )
    if fp_gaps.count() > 0:
        fp_gaps.show(truncate=False)
    else:
        print("FP sem gaps ✔️")

    # ============================================================
    # VD
    # ============================================================
    print("\n" + "=" * 80)
    print("DEBUG VD (isolado)")
    print("=" * 80)

    # OBS: seu VD detector normalmente não recebe end_ts, então filtramos depois
    df_vd = (
        detect_vd_events_gas_based(
            spark_df=spark_vd_raw,
            initial_heat_number=initial_heat,
            start_ts=start_ts,
            end_ts=end_ts,
            **vd_params
        )
        .filter((F.col("inicio_vd") >= start_ts) & (F.col("inicio_vd") <= end_ts))
        .orderBy("corrida")
    )

    display(df_vd)

    print("VD — resumo:")
    df_vd.select(
        F.count("*").alias("n_corridas"),
        F.min("corrida").alias("min_corrida"),
        F.max("corrida").alias("max_corrida"),
        F.min("inicio_vd").alias("min_inicio_vd"),
        F.max("final_vd").alias("max_final_vd"),
    ).show(truncate=False)

    print("\nVD — gaps de corrida:")
    vd_gaps = (
        df_vd
        .select(
            "corrida",
            (F.col("corrida") - F.lag("corrida").over(Window.orderBy("corrida"))).alias("delta")
        )
        .filter(F.col("delta") > 1)
    )
    if vd_gaps.count() > 0:
        vd_gaps.show(truncate=False)
    else:
        print("VD sem gaps ✔️")

    # ============================================================
    # LC (Pandas)
    # ============================================================
    print("\n" + "=" * 80)
    print("DEBUG LC (isolado)")
    print("=" * 80)

    df_lc = detect_lc_events(
        spark_df=spark_lc_raw,
        timestamp_col="timestamp",
        peso_col="ACI@LC_TORRE_PESOREAL",
        start_ts=start_ts,
        end_ts=end_ts,
        initial_corrida=initial_heat,
        **lc_params
    )

    display(df_lc)

    if df_lc.empty:
        print("LC — nenhuma corrida detectada")
    else:
        df_lc = df_lc.sort_values("corrida").reset_index(drop=True)
        df_lc["delta"] = df_lc["corrida"].diff()

        print("LC — resumo:")
        print({
            "n_corridas": len(df_lc),
            "min_corrida": int(df_lc["corrida"].min()),
            "max_corrida": int(df_lc["corrida"].max()),
            "min_inicio_lc": df_lc["start_detected"].min(),
            "max_final_lc": df_lc["end_detected"].max(),
        })

        print("\nLC — gaps de corrida:")
        gaps_lc = df_lc[df_lc["delta"] > 1][["corrida", "delta"]]
        if not gaps_lc.empty:
            display(gaps_lc)
        else:
            print("LC sem gaps ✔️")

    print("\nDEBUG FINALIZADO")


# COMMAND ----------

# DBTITLE 1,RODAR DEBUG
# debug_individual_detectors(
#     spark_fp_raw=df_fp_raw,
#     spark_vd_raw=df_vd_pre,   # aquele com volume_total
#     spark_lc_raw=df_lc_raw,
#     start_ts=START_TS,
#     end_ts=END_TS,
#     initial_heat=INITIAL_HEAT,
#     fp_params=fp_params,
#     vd_params=vd_params,
#     lc_params=lc_params,
# )


# COMMAND ----------

# DBTITLE 1,CRIAÇÃO DF_FINAL
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

# DBTITLE 1,df_final
df_final = df_final.filter(
    F.col("inicio_fp").between(START_TS, END_TS) |
    F.col("inicio_vd").between(START_TS, END_TS) |
    F.col("inicio_lc").between(START_TS, END_TS) |
    F.col("inicio_fea").between(START_TS, END_TS)
)

display(df_final.orderBy(F.desc("corrida")))

# COMMAND ----------

# DBTITLE 1,COMPARAÇÃO OP VS SYS TABELA
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
    apenas para as etapas existentes no SYS.
    """

    # ======================================================
    # 1) SYS (pandas) -> Spark
    # ======================================================
    df_sys = df_sys_pd.copy()

    df_sys["corrida"] = (
        pd.to_numeric(df_sys["corrida"], errors="coerce")
        .astype("Int64")
    )

    ts_cols = [
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
    # 4) Construção dinâmica das colunas
    # ======================================================
    select_cols = [F.col("corrida")]

    etapas = ["fp", "vd", "lc"]

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

df_compare = comparacao_operador(
    df_sys_pd=df_status,
    df_op_sp=df_final,
    spark=spark
)

display(df_compare.orderBy(F.desc("corrida")))



# COMMAND ----------

# DBTITLE 1,df_plot
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df_plot = df_compare.orderBy("corrida").toPandas()

# garantir datetime
cols_dt = [
    "inicio_fp_op","final_fp_op","inicio_fp_sys","final_fp_sys",
    "inicio_vd_op","final_vd_op","inicio_vd_sys","final_vd_sys",
    "inicio_lc_op","final_lc_op","inicio_lc_sys","final_lc_sys",
]
for c in cols_dt:
    if c in df_plot.columns:
        df_plot[c] = pd.to_datetime(df_plot[c], errors="coerce")


# COMMAND ----------

# DBTITLE 1,plot_intervals_op_sys
def plot_intervals_op_sys(
    df: pd.DataFrame,
    start_op: str,
    end_op: str,
    start_sys: str,
    end_sys: str,
    title: str,
    x_start: pd.Timestamp,
    x_end: pd.Timestamp,
    color_op: str = "#1f77b4",   # azul
    color_sys: str = "#ff7f0e",  # laranja
):
    """
    Plota intervalos OP vs SYS:
      - mesma cor para OP
      - mesma cor para SYS
      - apenas corridas com dados
      - eixo X fixo no período analisado
    """

    # --------------------------------------------------
    # 1. Filtrar apenas corridas com algum intervalo
    # --------------------------------------------------
    df_plot = df[
        (
            (df[start_op].notna() & df[end_op].notna()) |
            (df[start_sys].notna() & df[end_sys].notna())
        )
    ].copy()

    if df_plot.empty:
        print("⚠️ Nenhuma corrida com dados para plotar.")
        return

    df_plot = df_plot.sort_values("corrida").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 0.5 * len(df_plot) + 3))

    bar_height = 0.35
    y_positions = range(len(df_plot))

    for i, row in df_plot.iterrows():
        y = i
        corrida = int(row["corrida"])

        # OP (barra inferior)
        if pd.notna(row[start_op]) and pd.notna(row[end_op]):
            ax.barh(
                y - bar_height / 2,
                row[end_op] - row[start_op],
                left=row[start_op],
                height=bar_height,
                color=color_op,
            )

        # SYS (barra superior)
        if pd.notna(row[start_sys]) and pd.notna(row[end_sys]):
            ax.barh(
                y + bar_height / 2,
                row[end_sys] - row[start_sys],
                left=row[start_sys],
                height=bar_height,
                color=color_sys,
            )

    # --------------------------------------------------
    # 2. Eixos
    # --------------------------------------------------
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(df_plot["corrida"].astype(int))
    ax.invert_yaxis()

    ax.set_xlim(x_start, x_end)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    ax.set_xlabel("Tempo")
    ax.set_ylabel("Corrida")
    ax.set_title(title)

    ax.grid(True, axis="x")

    # --------------------------------------------------
    # 3. Legenda FORÇADA (fix do SYS sumir)
    # --------------------------------------------------
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=color_op, label="OP"),
        Patch(color=color_sys, label="SYS"),
    ]
    ax.legend(handles=legend_handles)

    plt.tight_layout()
    plt.show()


# COMMAND ----------

# DBTITLE 1,PLOT FP
x_start = pd.to_datetime(START_TS)
x_end   = pd.to_datetime(END_TS)

plot_intervals_op_sys(
    df=df_plot,
    start_op="inicio_fp_op",
    end_op="final_fp_op",
    start_sys="inicio_fp_sys",
    end_sys="final_fp_sys",
    title="FP — Operador (OP) vs Sistema (SYS)",
    x_start=x_start,
    x_end=x_end,
)


# COMMAND ----------

# DBTITLE 1,PLOT VD
plot_intervals_op_sys(
    df=df_plot,
    start_op="inicio_vd_op",
    end_op="final_vd_op",
    start_sys="inicio_vd_sys",
    end_sys="final_vd_sys",
    title="VD — Operador (OP) vs Sistema (SYS)",
    x_start=x_start,
    x_end=x_end,
)


# COMMAND ----------

# DBTITLE 1,PLOT LC
plot_intervals_op_sys(
    df=df_plot,
    start_op="inicio_lc_op",
    end_op="final_lc_op",
    start_sys="inicio_lc_sys",
    end_sys="final_lc_sys",
    title="LC — Operador (OP) vs Sistema (SYS)",
    x_start=x_start,
    x_end=x_end,
)


# COMMAND ----------

# DBTITLE 1,PLOT GERAL
plot_intervals_op_sys(
    df=df_plot,
    start_op="inicio_fp_op",
    end_op="final_lc_op",
    start_sys="inicio_fp_sys",
    end_sys="final_lc_sys",
    title="Corrida Completa — FP → LC | Operador vs Sistema",
    x_start=x_start,
    x_end=x_end,
)


# COMMAND ----------

# DBTITLE 1,build_long_etapas
import pandas as pd
import matplotlib.pyplot as plt

# ordem das etapas no eixo X
ETAPAS = [
    ("inicio_fp", "Início FP"),
    ("final_fp",  "Fim FP"),
    ("inicio_vd", "Início VD"),
    ("final_vd",  "Fim VD"),
    ("inicio_lc", "Início LC"),
    ("final_lc",  "Fim LC"),
]

def build_long_etapas(df, origem: str):
    """
    origem = 'op' ou 'sys'
    Retorna DataFrame longo:
      corrida | etapa | etapa_label | tempo
    """
    rows = []

    for _, r in df.iterrows():
        corrida = int(r["corrida"])

        for col, label in ETAPAS:
            c = f"{col}_{origem}"
            if c in df.columns and pd.notna(r[c]):
                rows.append({
                    "corrida": corrida,
                    "etapa": col,
                    "etapa_label": label,
                    "tempo": r[c],
                })

    return pd.DataFrame(rows)

def select_middle_corridas(corridas: list[int], max_corridas: int):
    if max_corridas is None or max_corridas >= len(corridas):
        return corridas

    mid = len(corridas) // 2
    half = max_corridas // 2

    start = max(mid - half, 0)
    end = start + max_corridas

    return corridas[start:end]


df_sys_long = build_long_etapas(df_plot, "sys")
df_op_long  = build_long_etapas(df_plot, "op")



# COMMAND ----------

# DBTITLE 1,plot_corridas_por_etapa_ordem
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

def plot_corridas_por_etapa_ordem(
    df: pd.DataFrame,
    origem: str,               # "op" ou "sys"
    title: str,
    max_corridas: int = 5,
):
    """
    Gráfico:
      - eixo X = etapas (ordem lógica)
      - eixo Y = tempo
      - cada corrida = uma cor
      - HH:MM anotado em cada ponto
    """

    # -------------------------
    # Seleção de corridas do meio
    # -------------------------
    corridas = sorted(df["corrida"].dropna().unique().astype(int).tolist())
    corridas_sel = select_middle_corridas(corridas, max_corridas)

    df_sel = df[df["corrida"].isin(corridas_sel)].copy()

    if df_sel.empty:
        print("⚠️ Nenhuma corrida para plotar.")
        return

    # -------------------------
    # Mapa de etapas
    # -------------------------
    etapa_pos = {label: i for i, (_, label) in enumerate(ETAPAS)}

    # -------------------------
    # Cores únicas por corrida
    # -------------------------
    cmap = cm.get_cmap("tab10", len(corridas_sel))
    color_map = {c: cmap(i) for i, c in enumerate(corridas_sel)}

    plt.figure(figsize=(14, 6))

    # -------------------------
    # Plot por corrida
    # -------------------------
    for corrida in corridas_sel:
        row = df_sel[df_sel["corrida"] == corrida].iloc[0]

        x = []
        y = []
        labels = []

        for col, label in ETAPAS:
            c = f"{col}_{origem}"
            if c in df_sel.columns and pd.notna(row[c]):
                ts = pd.to_datetime(row[c])
                x.append(etapa_pos[label])
                y.append(ts)
                labels.append(ts.strftime("%H:%M"))

        if len(x) >= 2:
            plt.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                color=color_map[corrida],
                label=f"Corrida {corrida}",
            )

            # -------------------------
            # Anotação HH:MM em cada ponto
            # -------------------------
            for xi, yi, txt in zip(x, y, labels):
                plt.text(
                    xi,
                    yi,
                    txt,
                    fontsize=9,
                    ha="center",
                    va="bottom",
                )

    # -------------------------
    # Eixos
    # -------------------------
    plt.xticks(
        ticks=range(len(ETAPAS)),
        labels=[label for _, label in ETAPAS],
        rotation=30,
    )

    plt.xlabel("Etapa (ordem lógica)")
    plt.ylabel("Tempo")
    plt.title(title)
    plt.grid(True)
    plt.legend(title="Corridas", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


# COMMAND ----------

# DBTITLE 1,PLOT CORRIDAS POR ETAPA PARA ORIGEM SYS
plot_corridas_por_etapa_ordem(
    df=df_plot,
    origem="sys",
    title="Linha do Tempo — SYS",
    max_corridas=5,
)


# COMMAND ----------

# DBTITLE 1,GRÁFICO DE CORRIDAS POR ETAPA E ORDEM OP
plot_corridas_por_etapa_ordem(
    df=df_plot,
    origem="op",
    title="Linha do Tempo — OP",
    max_corridas=5,
)

# COMMAND ----------

# DBTITLE 1,DEBUG FP
def debug_fp_period(
    spark_df_fp_raw,
    spark_df_fp_official,   # df_final (Spark)
    start_ts: str,
    end_ts: str,
    initial_heat_number: int,
    detect_kwargs: dict,
):
    """
    FP — Debug visual:
      - Gás total FP (PIMS tratado)
      - Início/fim oficial (operador)
      - Início/fim detectado (lógica)
    """

    # ==========================
    # 1. DETECÇÃO (lógica)
    # ==========================
    df_detected = detect_fp_events_gas_based(
        spark_df=spark_df_fp_raw,
        start_ts=start_ts,
        end_ts=end_ts,
        initial_heat_number=initial_heat_number,   # <-- FIX
        **detect_kwargs,
    ).toPandas()

    # ==========================
    # 2. PIMS TRATADO
    # ==========================
    df_pims = preprocess_pims_data(
        spark_df=spark_df_fp_raw,
        first_timestamp=start_ts,
        last_timestamp=end_ts,
        timestamp_column="timestamp",
        columns_to_fill=[
            "ACI@FP_Volume_Argonio_Carro_1",
            "ACI@FP_Volume_Nitrogenio_Carro_1",
            "ACI@FP_Volume_Argonio_Carro_2",
            "ACI@FP_Volume_Nitrogenio_Carro_2",
        ],
    )

    # ==========================
    # 3. SÉRIE TEMPORAL
    # ==========================
    df_plot = (
        df_pims
        .select(
            "timestamp",
            (
                F.col("ACI@FP_Volume_Argonio_Carro_1") +
                F.col("ACI@FP_Volume_Nitrogenio_Carro_1") +
                F.col("ACI@FP_Volume_Argonio_Carro_2") +
                F.col("ACI@FP_Volume_Nitrogenio_Carro_2")
            ).alias("gas_total")
        )
        .orderBy("timestamp")
        .toPandas()
    )
    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"], errors="coerce")

    # ==========================
    # 4. OFICIAL (operador)
    # ==========================
    df_official = (
        spark_df_fp_official
        .filter(
            (F.col("inicio_fp") >= start_ts) &
            (F.col("final_fp") <= end_ts)
        )
        .select("corrida", "inicio_fp", "final_fp")
        .toPandas()
    )
    df_official["inicio_fp"] = pd.to_datetime(df_official["inicio_fp"], errors="coerce")
    df_official["final_fp"]  = pd.to_datetime(df_official["final_fp"], errors="coerce")

    # ==========================
    # 5. PLOT
    # ==========================
    plt.figure(figsize=(15, 5))
    plt.plot(df_plot["timestamp"], df_plot["gas_total"], label="Gás total FP")

    y_max = float(df_plot["gas_total"].max()) if not df_plot.empty else 1.0

    # Operador
    for _, r in df_official.iterrows():
        c = int(r["corrida"])
        if pd.notna(r["inicio_fp"]):
            plt.axvline(r["inicio_fp"], color="green", linestyle="--")
            plt.text(r["inicio_fp"], y_max * 0.95, f"OP início {c}", rotation=90, color="green", ha="right")
        if pd.notna(r["final_fp"]):
            plt.axvline(r["final_fp"], color="red", linestyle="--")
            plt.text(r["final_fp"], y_max * 0.90, f"OP fim {c}", rotation=90, color="red", ha="left")

    # Detectado
    df_detected["inicio_fp"] = pd.to_datetime(df_detected["inicio_fp"], errors="coerce")
    df_detected["final_fp"]  = pd.to_datetime(df_detected["final_fp"], errors="coerce")

    for _, r in df_detected.iterrows():
        c = int(r["corrida"])
        if pd.notna(r["inicio_fp"]):
            plt.axvline(r["inicio_fp"], color="blue", linestyle=":")
            plt.text(r["inicio_fp"], y_max * 0.85, f"SYS início {c}", rotation=90, color="blue", ha="right")
        if pd.notna(r["final_fp"]):
            plt.axvline(r["final_fp"], color="orange", linestyle=":")
            plt.text(r["final_fp"], y_max * 0.80, f"SYS fim {c}", rotation=90, color="orange", ha="left")

    plt.title("DEBUG FP — Operador x Detectado")
    plt.grid(True)
    plt.legend()
    plt.show()

    return df_detected


# COMMAND ----------

# DBTITLE 1,R DEBUG FP
debug_fp_period(
    spark_df_fp_raw=df_fp_raw,
    spark_df_fp_official=df_final,
    start_ts=START_TS,
    end_ts=END_TS,
    initial_heat_number=INITIAL_HEAT_FP,   
    detect_kwargs=fp_params,
)


# COMMAND ----------

# DBTITLE 1,DEBUG VD
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F


def debug_vd_period(
    spark_df_vd_pims,          # PIMS (VD)
    spark_df_vd_official,      # tabela operador (inicio_vd / final_vd)
    start_ts: str,
    end_ts: str,
    initial_heat_number: int,
    detect_kwargs: dict,       # 🔑 MESMOS PARAMS DA LÓGICA
):
    """
    DEBUG VD:
      - Volume por vaso (PIMS tratado)
      - Início/Fim Oficial (Operador)
      - Início/Fim Detectado (Sistema)

    detect_kwargs deve conter, por exemplo:
      {
        "reset_threshold": 1.0,
        "minimal_duration_seconds": 30,
        "debug": False
      }
    """

    # ======================================================
    # 1. DETECÇÃO (USA A FUNÇÃO FINAL)
    # ======================================================
    df_detected = (
        detect_vd_events_gas_based(
            spark_df_vd_pims=spark_df_vd_pims,
            start_ts=start_ts,
            end_ts=end_ts,
            initial_heat_number=initial_heat_number,
            **detect_kwargs,   # 🔑 repasse direto
        )
        .toPandas()
    )

    if not df_detected.empty:
        df_detected["inicio_vd"] = pd.to_datetime(df_detected["inicio_vd"])
        df_detected["final_vd"]  = pd.to_datetime(df_detected["final_vd"])

    # ======================================================
    # 2. MESMO PRÉ-PROCESSAMENTO DA LÓGICA (PIMS)
    # ======================================================
    df_pims = (
        spark_df_vd_pims
        .filter(
            (F.col("timestamp") >= start_ts) &
            (F.col("timestamp") <= end_ts)
        )
        .fillna(0, subset=[
            "ACI@VD_VASO1_VOLUMEAR",
            "ACI@VD_VASO1_VOLUMEN2",
            "ACI@VD_VASO2_VOLUMEAR",
            "ACI@VD_VASO2_VOLUMEN2",
        ])
        .select(
            "timestamp",
            (
                F.col("ACI@VD_VASO1_VOLUMEAR") +
                F.col("ACI@VD_VASO1_VOLUMEN2")
            ).alias("vol_v1"),
            (
                F.col("ACI@VD_VASO2_VOLUMEAR") +
                F.col("ACI@VD_VASO2_VOLUMEN2")
            ).alias("vol_v2"),
        )
        .orderBy("timestamp")
        .toPandas()
    )

    df_pims["timestamp"] = pd.to_datetime(df_pims["timestamp"])

    # ======================================================
    # 3. VD OFICIAL (OPERADOR)
    # ======================================================
    df_official = (
        spark_df_vd_official
        .filter(
            (F.col("inicio_vd") >= start_ts) &
            (F.col("final_vd") <= end_ts)
        )
        .select("corrida", "inicio_vd", "final_vd")
        .toPandas()
    )

    if not df_official.empty:
        df_official["inicio_vd"] = pd.to_datetime(df_official["inicio_vd"])
        df_official["final_vd"]  = pd.to_datetime(df_official["final_vd"])

    # ======================================================
    # 4. PLOT
    # ======================================================
    plt.figure(figsize=(16, 6))

    plt.plot(
        df_pims["timestamp"],
        df_pims["vol_v1"],
        label="VD Vaso 1",
        linewidth=1.5,
    )

    plt.plot(
        df_pims["timestamp"],
        df_pims["vol_v2"],
        label="VD Vaso 2",
        linewidth=1.5,
    )

    y_max = max(df_pims["vol_v1"].max(), df_pims["vol_v2"].max()) * 1.05

    # ---------------- OFICIAL ----------------
    for _, r in df_official.iterrows():
        c = int(r["corrida"])

        plt.axvline(r["inicio_vd"], color="green", linestyle="--", linewidth=2)
        plt.text(
            r["inicio_vd"],
            y_max * 0.95,
            f"OP início {c}",
            rotation=90,
            color="green",
            ha="right",
        )

        plt.axvline(r["final_vd"], color="red", linestyle="--", linewidth=2)
        plt.text(
            r["final_vd"],
            y_max * 0.60,
            f"OP fim {c}",
            rotation=90,
            color="red",
            ha="left",
        )

    # ---------------- DETECTADO ----------------
    for _, r in df_detected.iterrows():
        c = int(r["corrida"])

        plt.axvline(r["inicio_vd"], color="blue", linestyle=":", linewidth=2)
        plt.text(
            r["inicio_vd"],
            y_max * 0.35,
            f"SYS início {c}",
            rotation=90,
            color="blue",
            ha="right",
        )

        if pd.notna(r["final_vd"]):
            plt.axvline(r["final_vd"], color="orange", linestyle=":", linewidth=2)
            plt.text(
                r["final_vd"],
                y_max * 0.05,
                f"SYS fim {c}",
                rotation=90,
                color="orange",
                ha="left",
            )

    plt.title("DEBUG VD — Volume por Vaso | Operador x Sistema")
    plt.xlabel("Tempo")
    plt.ylabel("Volume acumulado")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df_detected


# COMMAND ----------

# DBTITLE 1,R DEBUG VD
debug_vd_period(
    spark_df_vd_pims=df_vd_raw,
    spark_df_vd_official=df_final,
    start_ts=START_TS,
    end_ts=END_TS,
    initial_heat_number=INITIAL_HEAT_VD,
    detect_kwargs=vd_params,
)


# COMMAND ----------

# DBTITLE 1,DEBUG LC
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
    Debug visual do LC:
      - Peso torre (PIMS)
      - Início/Fim operador
      - Início/Fim detectado (lógica)
    """

    # =====================================================
    # 1. DETECÇÃO (lógica real do LC)
    # =====================================================
    df_detected = detect_lc_events(
        spark_df=spark_df_lc,
        timestamp_col=timestamp_col,
        peso_col=peso_col,
        start_ts=start_ts,
        end_ts=end_ts,
        initial_corrida=initial_corrida,
        **detect_params,
    )

    # Garantir datetime
    if not df_detected.empty:
        df_detected["start_detected"] = pd.to_datetime(df_detected["start_detected"])
        df_detected["end_detected"]   = pd.to_datetime(df_detected["end_detected"])

    # =====================================================
    # 2. SÉRIE PARA PLOT (MESMA DA LÓGICA)
    # =====================================================
    df_plot = (
        spark_df_lc
        .withColumn(
            "timestamp_sp",
            F.col(timestamp_col)
        )
        .withColumn(
            "peso",
            F.regexp_replace(F.col(peso_col), ",", ".").cast("double")
        )
        .filter(
            (F.col("timestamp_sp") >= start_ts) &
            (F.col("timestamp_sp") <= end_ts)
        )
        .orderBy("timestamp_sp")
        .toPandas()
    )

    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp_sp"])
    df_plot = df_plot.drop(columns=["timestamp_sp"])

    # =====================================================
    # 3. EVENTOS OFICIAIS (OPERADOR)
    # =====================================================
    df_official = (
        spark_df_official
        .filter(
            (F.col("INICIOLINGOTAMENTO") >= start_ts) &
            (F.col("FINALLINGOTAMENTO") <= end_ts)
        )
        .select("CORRIDA", "INICIOLINGOTAMENTO", "FINALLINGOTAMENTO")
        .toPandas()
    )

    if not df_official.empty:
        df_official["INICIOLINGOTAMENTO"] = pd.to_datetime(df_official["INICIOLINGOTAMENTO"])
        df_official["FINALLINGOTAMENTO"]  = pd.to_datetime(df_official["FINALLINGOTAMENTO"])

    # =====================================================
    # 4. PLOT
    # =====================================================
    plt.figure(figsize=(14, 5))
    ax = plt.gca()

    ax.plot(df_plot["timestamp"], df_plot["peso"], label="Peso Torre")

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    y_max = df_plot["peso"].max()

    # ---------- OPERADOR ----------
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
            f"Início SYS {c}",
            rotation=90,
            color="blue",
            ha="left",
        )

        ax.axvline(r["end_detected"], color="orange", linestyle=":")
        ax.text(
            r["end_detected"],
            y_max * 0.80,
            f"Fim SYS {c}",
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

# DBTITLE 1,R DEBUG LC
df_lc_detected = debug_lc_period(
    spark_df_lc=df_lc_raw,
    spark_df_official=lc_df,  # tabela operador LC
    timestamp_col="timestamp",
    peso_col="ACI@LC_TORRE_PESOREAL",
    start_ts=START_TS,
    end_ts=END_TS,
    initial_corrida=INITIAL_HEAT_LC,
    detect_params=lc_params,
)


# COMMAND ----------

# DBTITLE 1,IMPORTS E DEFINIÇÃO DE ATRASOS PADRÃO
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd

# Delays (default – sobrescritos pelo runner)
DELAY_FP_VD = pd.Timedelta(minutes=5)
DELAY_VD_LC = pd.Timedelta(minutes=2)


# COMMAND ----------

# DBTITLE 1,DEFINIÇÃO DAS CLASSES DE EVENTOS E CORRIDA
@dataclass
class Event:
    ts: pd.Timestamp
    kind: str
    payload: Dict[str, Any]


@dataclass
class Corrida:
    corrida: int
    inicio_fp: pd.Timestamp = pd.NaT
    final_fp: pd.Timestamp = pd.NaT
    inicio_vd: pd.Timestamp = pd.NaT
    final_vd: pd.Timestamp = pd.NaT
    vd_tanque: Any = pd.NA
    inicio_lc: pd.Timestamp = pd.NaT
    final_lc: pd.Timestamp = pd.NaT


# COMMAND ----------

# DBTITLE 1,FUNÇÃO PARA MONTAR EVENTOS INTEGRADOS FP VD LC
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

# DBTITLE 1,Simulação de eventos das corridas com filas e tempos
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

# DBTITLE 1,FUNÇÃO SIMULADOR DE CORRIDA INTEGRADA
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

# DBTITLE 1,RODAR SIMULADOR CORRIDA COM PARÂMETROS
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

# DBTITLE 1,GERAÇÃO DATAFRAME DETECÇÃO FALSOS POSITIVOS
df_fp_detected = (
    detect_fp_events_gas_based(
        spark_df=df_fp_raw,
        start_ts=START_TS,
        end_ts=END_TS,
        initial_heat_number=0,  # dummy
        **fp_params
    )
    .toPandas()
)

# COMMAND ----------

# DBTITLE 1,RESUMO DE EVENTOS FP DETECTADOS E INICIO FP UNICOS
print("Eventos FP detectados:", len(df_fp_detected))
print(
    "inicio_fp únicos:",
    df_fp_detected["inicio_fp"].nunique()
)



# COMMAND ----------

# DBTITLE 1,EXTRAÇÃO E CONVERSÃO DF_FP_RAW PARA PANDAS
df_fp_raw_pd = (
    df_fp_raw
    .select("timestamp", "ACI@FP_NUMERO_CORRIDA")
    .toPandas()
)

# COMMAND ----------

# DBTITLE 1,TRATAMENTO E PLOTAGEM CORRIDAS FP MONOTÔNICAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1️⃣ RAW FP — NORMALIZAÇÃO MONOTÔNICA
# =========================================================

df_fp_raw_pd = (
    df_fp_raw
    .select("timestamp", "ACI@FP_NUMERO_CORRIDA")
    .where(
        (df_fp_raw.timestamp >= START_TS) &
        (df_fp_raw.timestamp <= END_TS)
    )
    .toPandas()
)

df_fp_raw_pd["timestamp"] = pd.to_datetime(df_fp_raw_pd["timestamp"])
df_fp_raw_pd = df_fp_raw_pd.sort_values("timestamp").reset_index(drop=True)

# -------- parser ROBUSTO (baseado no dump real) --------
def parse_fp_corrida(x):
    if x is None:
        return pd.NA

    if isinstance(x, str):
        x = x.strip()
        if x == "" or x == "0":
            return pd.NA
        try:
            return int(x)
        except Exception:
            return pd.NA

    try:
        x = int(x)
        if x <= 0:
            return pd.NA
        return x
    except Exception:
        return pd.NA


df_fp_raw_pd["corrida_raw"] = (
    df_fp_raw_pd["ACI@FP_NUMERO_CORRIDA"]
    .apply(parse_fp_corrida)
    .astype("Int64")
)

# -------- lógica MONOTÔNICA --------
corrida_atual = pd.NA
corrida_monotonic = []

for v in df_fp_raw_pd["corrida_raw"]:
    if pd.notna(v):
        if pd.isna(corrida_atual) or v > corrida_atual:
            corrida_atual = v
    corrida_monotonic.append(corrida_atual)

df_fp_raw_pd["fp_corrida_monotonic"] = pd.Series(corrida_monotonic, dtype="Int64")

# =========================================================
# 2️⃣ EVENTOS FP (vindos do df_status)
# =========================================================

df_fp_events = df_final.toPandas()[[
    "corrida", "inicio_fp", "final_fp"
]].copy()

df_fp_events["inicio_fp"] = pd.to_datetime(df_fp_events["inicio_fp"])
df_fp_events["final_fp"] = pd.to_datetime(df_fp_events["final_fp"])

# =========================================================
# 3️⃣ PLOT FINAL (apenas corridas > 10000)
# =========================================================

mask = df_fp_raw_pd["fp_corrida_monotonic"] > 10000
df_fp_raw_pd_plot = df_fp_raw_pd[mask]

df_fp_events_plot = df_fp_events[df_fp_events["corrida"] > 10000]

plt.figure(figsize=(18, 6))

# matplotlib não aceita Int64 / pd.NA
plt.plot(
    df_fp_raw_pd_plot["timestamp"],
    df_fp_raw_pd_plot["fp_corrida_monotonic"].astype("float"),
    label="FP_NUMERO_CORRIDA (monotônico)",
    linewidth=1.5
)

# FP_START (verde) - adiciona valor da corrida
for _, r in df_fp_events_plot.iterrows():
    if pd.notna(r["inicio_fp"]):
        plt.axvline(
            r["inicio_fp"],
            color="green",
            alpha=0.35,
            linestyle="--"
        )
        plt.text(
            r["inicio_fp"],
            r["corrida"],
            f'{r["corrida"]}',
            color="green",
            fontsize=10,
            rotation=90,
            va='bottom',
            ha='right'
        )

# FP_END (vermelho) - adiciona valor da corrida
for _, r in df_fp_events_plot.iterrows():
    if pd.notna(r["final_fp"]):
        plt.axvline(
            r["final_fp"],
            color="red",
            alpha=0.35,
            linestyle=":"
        )
        plt.text(
            r["final_fp"],
            r["corrida"],
            f'{r["corrida"]}',
            color="red",
            fontsize=10,
            rotation=90,
            va='bottom',
            ha='right'
        )

plt.title("FP_NUMERO_CORRIDA (monotônico) vs Eventos FP_START / FP_END (corridas > 10000)")
plt.xlabel("Timestamp")
plt.ylabel("Número da Corrida (FP)")
plt.grid(True)
plt.legend()
plt.show()

# COMMAND ----------

# DBTITLE 1,NORMALIZAÇÃO MONOTÔNICA E DETECÇÃO DE EVENTOS FP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1️⃣ RAW FP — NORMALIZAÇÃO MONOTÔNICA
# =========================================================

df_fp_raw_pd = (
    df_fp_raw
    .select("timestamp", "ACI@FP_NUMERO_CORRIDA")
    .where(
        (df_fp_raw.timestamp >= START_TS) &
        (df_fp_raw.timestamp <= END_TS)
    )
    .toPandas()
)

df_fp_raw_pd["timestamp"] = pd.to_datetime(df_fp_raw_pd["timestamp"])
df_fp_raw_pd = df_fp_raw_pd.sort_values("timestamp").reset_index(drop=True)

# -------- parser ROBUSTO (baseado no dado real) --------
def parse_fp_corrida(x):
    if x is None:
        return pd.NA

    if isinstance(x, str):
        x = x.strip()
        if x == "" or x == "0":
            return pd.NA
        try:
            return int(x)
        except Exception:
            return pd.NA

    try:
        x = int(x)
        if x <= 0:
            return pd.NA
        return x
    except Exception:
        return pd.NA


df_fp_raw_pd["corrida_raw"] = (
    df_fp_raw_pd["ACI@FP_NUMERO_CORRIDA"]
    .apply(parse_fp_corrida)
    .astype("Int64")
)

# -------- lógica MONOTÔNICA --------
corrida_atual = pd.NA
corrida_monotonic = []

for v in df_fp_raw_pd["corrida_raw"]:
    if pd.notna(v):
        if pd.isna(corrida_atual) or v > corrida_atual:
            corrida_atual = v
    corrida_monotonic.append(corrida_atual)

df_fp_raw_pd["fp_corrida_monotonic"] = pd.Series(corrida_monotonic, dtype="Int64")

# =========================================================
# 2️⃣ DETECTOR FP — EVENTOS DETECTADOS
# =========================================================

df_fp_detected = (
    detect_fp_events_gas_based(
        spark_df=df_fp_raw,
        start_ts=START_TS,
        end_ts=END_TS,
        initial_heat_number=130475,
        **fp_params
    )
    .toPandas()
)

df_fp_detected["inicio_fp"] = pd.to_datetime(df_fp_detected["inicio_fp"])
df_fp_detected["final_fp"] = pd.to_datetime(df_fp_detected["final_fp"])

# =========================================================
# 3️⃣ FILTROS PARA VISUALIZAÇÃO
# =========================================================

mask_raw = df_fp_raw_pd["fp_corrida_monotonic"] > 10000
df_fp_raw_plot = df_fp_raw_pd[mask_raw]

df_fp_detected_plot = df_fp_detected[df_fp_detected["corrida"] > 10000]

# =========================================================
# 4️⃣ PLOT FINAL
# =========================================================

plt.figure(figsize=(18, 6))

# sinal real da corrida (estado FP)
plt.plot(
    df_fp_raw_plot["timestamp"],
    df_fp_raw_plot["fp_corrida_monotonic"].astype("float"),
    label="FP_NUMERO_CORRIDA (monotônico)",
    linewidth=1.5
)

# -----------------------------
# FP_START detectado (verde contínuo)
# -----------------------------
for _, r in df_fp_detected_plot.iterrows():
    if pd.notna(r["inicio_fp"]):
        plt.axvline(
            r["inicio_fp"],
            color="green",
            alpha=0.6,
            linewidth=2,
            linestyle="-"
        )
        plt.text(
            r["inicio_fp"],
            r["corrida"],
            f'{r["corrida"]}',
            color="green",
            fontsize=10,
            rotation=90,
            va="bottom",
            ha="right"
        )

# -----------------------------
# FP_END detectado (vermelho contínuo)
# -----------------------------
for _, r in df_fp_detected_plot.iterrows():
    if pd.notna(r["final_fp"]):
        plt.axvline(
            r["final_fp"],
            color="red",
            alpha=0.6,
            linewidth=2,
            linestyle="-"
        )
        plt.text(
            r["final_fp"],
            r["corrida"],
            f'{r["corrida"]}',
            color="red",
            fontsize=10,
            rotation=90,
            va="bottom",
            ha="right"
        )

plt.title("FP_NUMERO_CORRIDA (monotônico) vs FP_START / FP_END detectados")
plt.xlabel("Timestamp")
plt.ylabel("Número da Corrida (FP)")
plt.grid(True)
plt.legend()
plt.show()


# COMMAND ----------

# DBTITLE 1,INSPEÇÃO DE VALORES E TIPOS DA COLUNA NUMERO CORRIDA
# =========================
# INSPEÇÃO DO VALOR REAL
# =========================

print(df_fp_raw_pd["ACI@FP_NUMERO_CORRIDA"].head(20))

print("\nTipos Python:")
print(df_fp_raw_pd["ACI@FP_NUMERO_CORRIDA"].head(20).apply(type))

print("\nValores únicos (amostra):")
print(df_fp_raw_pd["ACI@FP_NUMERO_CORRIDA"].dropna().unique()[:20])


# COMMAND ----------

# DBTITLE 1,SIMULAÇÃO SEQUENCIAL DE CORRIDAS COM EVENTOS E FILAS
def simulate_corridas_sequencial(
    df_fp: pd.DataFrame,
    df_vd: pd.DataFrame,
    df_lc: pd.DataFrame,
    corrida_inicial: int = 1,
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
    corrida_seq = corrida_inicial

    # =========================
    # LOOP DE EVENTOS
    # =========================
    for ev in events:
        ts, kind = ev.ts, ev.kind

        # ---------------------------
        # START GLOBAL
        # ---------------------------
        if not started:
            if kind != "FP_START":
                continue

            c = corrida_seq
            corrida_seq += 1

            started = True
            get_corrida(c).inicio_fp = ts

            if debug:
                print(f"[{ts}] 🚀 START → corrida {c}")
            continue

        # ---------------------------
        # FP
        # ---------------------------
        if kind == "FP_START":
            c = corrida_seq
            corrida_seq += 1

            get_corrida(c).inicio_fp = ts

            if debug:
                print(f"[{ts}] FP_START → corrida {c}")
            continue

        if kind == "FP_END":
            # sempre fecha a corrida mais recente que saiu do FP
            c = max(corridas.keys())
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

    # =========================
    # SAÍDA
    # =========================
    out = pd.DataFrame([vars(c) for c in corridas.values()]).sort_values("corrida")

    def status(r):
        # ainda nem começou
        if pd.isna(r.inicio_fp):
            return "NAO_INICIADA"

        # LC
        if pd.notna(r.inicio_lc):
            if pd.isna(r.final_lc):
                return "LC"
            else:
                return "FINALIZADA"

        # VD
        if pd.notna(r.inicio_vd):
            if pd.isna(r.final_vd):
                return "VD"
            else:
                return "AGUARDANDO_LC"

        # FP
        if pd.notna(r.inicio_fp):
            if pd.isna(r.final_fp):
                return "FP"
            else:
                return "AGUARDANDO_VD"


    out["status_atual"] = out.apply(status, axis=1)
    return out.reset_index(drop=True)


# COMMAND ----------

# DBTITLE 1,PROCESSAMENTO DETECÇÃO EVENTOS FP VD E LC
df_fp = (
    detect_fp_events_gas_based(
        spark_df=df_fp_raw,
        start_ts=START_TS,
        end_ts=END_TS,
        initial_heat_number=0,
        **fp_params
    )
    .toPandas()
)

df_vd = (
    detect_vd_events_gas_based(
        spark_df_vd_pims=df_vd_raw,
        start_ts=START_TS,
        end_ts=END_TS,
        initial_heat_number=0,
        **vd_params
    )
    .toPandas()
)

df_lc = (
    detect_lc_events(
        spark_df=df_lc_raw,
        timestamp_col="timestamp",
        peso_col="ACI@LC_TORRE_PESOREAL",
        start_ts=START_TS,
        end_ts=END_TS,
        initial_corrida=0,
        **lc_params
    )
    .rename(columns={
        "start_detected": "inicio_lc",
        "end_detected": "final_lc",
    })
)

df_lc["inicio_lc"] = pd.to_datetime(df_lc["inicio_lc"])
df_lc["final_lc"] = pd.to_datetime(df_lc["final_lc"])



# COMMAND ----------

# DBTITLE 1,FUNÇÃO PARA MONTAR EVENTOS FP VD E LC
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

# DBTITLE 1,SIMULAÇÃO SEQUENCIAL COM DEBUG E EXIBIÇÃO
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


# COMMAND ----------

print("LC rows:", len(df_lc))
print(df_lc.head(20))

print("LC_START válidos:",
      df_lc["inicio_lc"].notna().sum()
      if "inicio_lc" in df_lc.columns else "coluna não existe")

print("LC_END válidos:",
      df_lc["final_lc"].notna().sum()
      if "final_lc" in df_lc.columns else "coluna não existe")
