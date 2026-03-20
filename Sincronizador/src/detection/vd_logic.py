# """
# vd_logic.py

# Detects usage cycles (events) for Tank 1 and Tank 2 based on cumulative gas volume data.

# """

# Standard library imports
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


# -----------------------------
# Preprocess (streaming-safe)
# -----------------------------
def streaming_ffill_last_valid(
    df: pyspark.sql.DataFrame,
    timestamp_column: str,
    columns: list[str],
) -> pyspark.sql.DataFrame:
    """
    Forward-fill "last valid" (last non-null) values using only past rows.

    Notes:
      - Implemented with a window "unboundedPreceding -> currentRow" (no future).
      - In Structured Streaming, this is typically executed inside foreachBatch.
    """
    w = (
        Window.orderBy(F.col(timestamp_column))
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    out = df
    for c in columns:
        out = out.withColumn(c, F.last(F.col(c), ignorenulls=True).over(w))
    return out


def _shift_iso_ts(ts: str, seconds: int) -> str:
    """
    Shifts an ISO-like timestamp string by N seconds and returns a string that Spark
    can parse. Best-effort parsing:
      - supports 'YYYY-mm-dd HH:MM:SS'
      - supports 'YYYY-mm-ddTHH:MM:SS'
      - supports timezone offsets and 'Z' (converted to '+00:00')
    """
    s = ts.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    dt2 = dt + timedelta(seconds=int(seconds))
    # Spark timestamps are tz-naive in most pipelines; drop tzinfo if present
    if dt2.tzinfo is not None:
        dt2 = dt2.replace(tzinfo=None)
    return dt2.isoformat(sep=" ")

def preprocess_pims_data_streaming(
    spark_df: pyspark.sql.DataFrame,
    first_timestamp: str,
    last_timestamp: str,
    timestamp_column: str,
    columns_to_fill: list[str],
) -> pyspark.sql.DataFrame:
    """
    Streaming-safe preprocessing:
      1) filter by [first_timestamp, last_timestamp]
      2) cast selected columns to double
      3) stable ordering (timestamp + tie-break)
      4) forward fill last valid values (no future look-ahead)
    """
    filtered = spark_df.filter(
        (F.col(timestamp_column) >= F.lit(first_timestamp))
        & (F.col(timestamp_column) <= F.lit(last_timestamp))
    )

    casted = filtered
    for c in columns_to_fill:
        casted = casted.withColumn(c, F.col(c).cast("double"))

    ordered = (
        casted.withColumn("_tie", F.monotonically_increasing_id())
        .orderBy(F.col(timestamp_column).asc(), F.col("_tie").asc())
        .drop("_tie")
    )

    return streaming_ffill_last_valid(
        df=ordered,
        timestamp_column=timestamp_column,
        columns=columns_to_fill,
    )






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


# def detect_vd_events(
#     spark_df_vd_pims: pyspark.sql.DataFrame,
#     start_ts: str,
#     end_ts: str,
#     initial_heat_number: int,
#     reset_threshold: float = 0.99,
#     min_event_confirm_seconds: Optional[int] = None,
#     min_vd_arrival_confirm_seconds: Optional[int] = 20,
#     min_vd_exit_confirm_seconds: Optional[int] = 20,
#     min_deep_vac_start_confirm_seconds: Optional[int] = 20,
#     min_deep_vac_end_confirm_seconds: Optional[int] = 20,
#     min_vd_duration_seconds: int = 8*60,
#     debug: bool = False,
#     predicted_tank_input: Optional[int] = None,
# ) -> tuple[
#     pyspark.sql.DataFrame, pyspark.sql.DataFrame, pyspark.sql.DataFrame
#     ]:
#     """
#     Detects VD (Vacuum Degassing) usage cycles and deep vacuum intervals using a
#     streaming-friendly, 100% Spark approach (no lead / no future windows).

#     The detector is built to support micro-batch streaming execution (e.g.,
#     in Databricks `foreachBatch`) and batch runs. It uses only past information
#     (via `lag()` and "unboundedPreceding -> currentRow" windows) to derive events.

#     Logical events emitted (append-only):
#       1) VD_ARRIVAL
#          - Definition: Rising edge of FLOW (VZFORTE + VZFRACA) for a given tank.
#          - event_ts: The first timestamp where flow becomes active.
#          - confirmed_at: The first timestamp where the VD session has remained active
#                          for at least `min_vd_arrival_confirm_seconds`.

#       2) DEEP_VAC_START
#          - Definition: Rising edge of VOLUME (VOLUMEAR + VOLUMEN2) for a given tank.
#          - event_ts: The first timestamp where volume becomes active.
#          - confirmed_at: The first timestamp where the deep vacuum interval has remained
#                          active for at least `min_deep_vac_start_confirm_seconds`.

#       3) DEEP_VAC_END
#          - Definition: Falling edge of VOLUME (VOLUMEAR + VOLUMEN2) for a given tank.
#          - event_ts: The falling edge timestamp (first point where volume becomes inactive).
#          - confirmed_at:
#              * If `min_deep_vac_end_confirm_seconds == 0`, equals event_ts (immediate).
#              * Otherwise, the first timestamp where volume has remained inactive for
#                at least `min_deep_vac_end_confirm_seconds` (no retract, optional).

#       4) VD_EXIT
#          - Definition: Falling edge of FLOW (VZFORTE + VZFRACA) for a given tank.
#          - event_ts: The falling edge timestamp (first point where flow becomes inactive).
#          - confirmed_at:
#              * If `min_vd_exit_confirm_seconds == 0`, equals event_ts (immediate).
#              * Otherwise, the first timestamp where flow has remained inactive for
#                at least `min_vd_exit_confirm_seconds` (no retract, optional).

#     Anti-noise protections:
#       - FLOW tag quantization (critical):
#           * values < 1.0 are forced to 0.0
#           * values >= 1.0 are rounded to the nearest integer
#         This avoids tiny float noise (e.g. 0.05) triggering false VD sessions when
#         thresholds are near 0/1.

#       - VD minimum duration filter:
#           * VD sessions shorter than `min_vd_duration_seconds` are dropped BEFORE
#             assigning `corrida` and emitting events.

#       - Event confirmation delays (sustain-based confirmation):
#           * VD_ARRIVAL is emitted only after `min_vd_arrival_confirm_seconds`
#           * VD_EXIT can be delayed by `min_vd_exit_confirm_seconds`
#           * DEEP_VAC_START is emitted only after `min_deep_vac_start_confirm_seconds`
#           * DEEP_VAC_END can be delayed by `min_deep_vac_end_confirm_seconds`

#       - Optional global confirm seconds:
#           * If `min_event_confirm_seconds` is provided, then any confirmation parameter
#             that is None will inherit its value. This allows a single knob for
#             confirmation defaults while keeping per-event overrides.

#     Args:
#         spark_df_vd_pims (DataFrame):
#             Input Spark DataFrame containing at least:
#               - timestamp column "timestamp"
#               - volume columns:
#                   "ACI@VD_VASO1_VOLUMEAR", "ACI@VD_VASO1_VOLUMEN2",
#                   "ACI@VD_VASO2_VOLUMEAR", "ACI@VD_VASO2_VOLUMEN2"
#               - flow columns:
#                   "ACI@VD_VASO1_VZFORTE", "ACI@VD_VASO1_VZFRACA",
#                   "ACI@VD_VASO2_VZFORTE", "ACI@VD_VASO2_VZFRACA"
#         start_ts (str):
#             Inclusive ISO timestamp to start analysis.
#         end_ts (str):
#             Inclusive ISO timestamp to end analysis.
#         initial_heat_number (int):
#             Base heat number used to assign sequential "corrida" IDs in the output,
#             ordered by VD start time (in the processed batch / micro-batch).
#         reset_threshold (float):
#             Threshold for considering FLOW/VOLUME "active".
#             Example: 0.99 means active if strictly > 0.99.
#         min_event_confirm_seconds (Optional[int]):
#             Optional global default for confirmation delays. Any per-event confirm
#             parameter that is None will inherit this value.
#         min_vd_arrival_confirm_seconds (Optional[int]):
#             Sustain time (seconds) required to confirm VD arrival (FLOW active).
#             If None and min_event_confirm_seconds is set, it inherits from it.
#         min_vd_exit_confirm_seconds (Optional[int]):
#             Sustain time (seconds) required to confirm VD exit (FLOW inactive).
#             If None and min_event_confirm_seconds is set, it inherits from it.
#             Set to 0 for immediate exit emission.
#         min_deep_vac_start_confirm_seconds (Optional[int]):
#             Sustain time (seconds) required to confirm deep vacuum start (VOLUME active).
#             If None and min_event_confirm_seconds is set, it inherits from it.
#         min_deep_vac_end_confirm_seconds (Optional[int]):
#             Sustain time (seconds) required to confirm deep vacuum end (VOLUME inactive).
#             If None and min_event_confirm_seconds is set, it inherits from it.
#             Set to 0 for immediate end emission.
#         min_vd_duration_seconds (int):
#             Minimum VD session duration (seconds) required to keep a VD session.
#             This filter is applied before assigning corrida and emitting events.
#         debug (bool):
#             If True, returns a third dataframe with debug/trace columns (flags and IDs).
#         predicted_tank_input (Optional[int]):
#             Optional metadata: the predicted tank (1 or 2) to be attached to df_vd_final.

#     Returns:
#         Tuple[DataFrame, DataFrame, DataFrame]:
#             (df_vd_final, df_vd_events, df_vd_debug)

#             1) df_vd_final (DataFrame):
#                 Consolidated per-VD-session table (upsert-friendly):
#                   - corrida (int): sequential heat number
#                   - tank_id (int): 1 or 2
#                   - start_vd_time (timestamp): VD arrival time (true event start)
#                   - end_vd_time (timestamp|null): VD exit time (true event end); null if open
#                   - duration_seconds (long): duration from start to end (or to last seen if open)
#                   - max_volume (double): max summed volume seen during the VD session
#                   - predicted_tank_input (int|null): metadata passthrough

#             2) df_vd_events (DataFrame):
#                 Append-only event table (streaming-friendly):
#                   - corrida (int)
#                   - status (string): VD_ARRIVAL / DEEP_VAC_START / DEEP_VAC_END / VD_EXIT
#                   - tank_id (int)
#                   - start_vd_time (timestamp)
#                   - end_vd_time (timestamp|null)
#                   - duration_seconds (long)
#                   - confirmed_at (timestamp): when the detector confirmed/emitted the event
#                   - event_ts (timestamp): true timestamp of the event (edge timestamp)

#             3) df_vd_debug (DataFrame):
#                 Debug/trace flags. Empty if debug=False.
#     """

#     # ----------------------------
#     # Helpers: accept str/datetime/pandas.Timestamp for start_ts/end_ts
#     # ----------------------------
#     def _to_py_datetime(x) -> datetime.datetime:
#         if isinstance(x, datetime.datetime):
#             dt = x
#         else:
#             # pandas.Timestamp has .to_pydatetime()
#             if hasattr(x, "to_pydatetime"):
#                 dt = x.to_pydatetime()
#             else:
#                 dt = datetime.datetime.fromisoformat(str(x).replace("Z", "+00:00"))
#         # Spark timestamps are usually tz-naive in pipelines
#         if dt.tzinfo is not None:
#             dt = dt.replace(tzinfo=None)
#         return dt

#     def _shift_dt(dt: datetime.datetime, seconds: int) -> datetime.datetime:
#         return dt + datetime.timedelta(seconds=int(seconds))

#     start_dt = _to_py_datetime(start_ts)
#     end_dt = _to_py_datetime(end_ts)

#     # ----------------------------
#     # Confirmation defaults wiring
#     #
#     # If min_event_confirm_seconds is provided:
#     #   - any per-event confirm param that is None inherits from it.
#     # ----------------------------
#     if min_event_confirm_seconds is not None:
#         if min_vd_arrival_confirm_seconds is None:
#             min_vd_arrival_confirm_seconds = int(min_event_confirm_seconds)
#         if min_vd_exit_confirm_seconds is None:
#             min_vd_exit_confirm_seconds = int(min_event_confirm_seconds)
#         if min_deep_vac_start_confirm_seconds is None:
#             min_deep_vac_start_confirm_seconds = int(min_event_confirm_seconds)
#         if min_deep_vac_end_confirm_seconds is None:
#             min_deep_vac_end_confirm_seconds = int(min_event_confirm_seconds)

#     # Ensure non-null ints for downstream expressions
#     min_vd_arrival_confirm_seconds = int(min_vd_arrival_confirm_seconds or 0)
#     min_vd_exit_confirm_seconds = int(min_vd_exit_confirm_seconds or 0)
#     min_deep_vac_start_confirm_seconds = int(min_deep_vac_start_confirm_seconds or 0)
#     min_deep_vac_end_confirm_seconds = int(min_deep_vac_end_confirm_seconds or 0)
#     min_vd_duration_seconds = int(min_vd_duration_seconds)

#     # ----------------------------
#     # Schemas for empty returns
#     # ----------------------------
#     VD_FINAL_SCHEMA = StructType(
#         [
#             StructField("corrida", IntegerType(), True),
#             StructField("tank_id", IntegerType(), True),
#             StructField("start_vd_time", TimestampType(), True),
#             StructField("end_vd_time", TimestampType(), True),
#             StructField("duration_seconds", LongType(), True),
#             StructField("max_volume", DoubleType(), True),
#             StructField("predicted_tank_input", IntegerType(), True),
#         ]
#     )

#     VD_EVENTS_SCHEMA = StructType(
#         [
#             StructField("corrida", IntegerType(), True),
#             StructField("status", StringType(), True),
#             StructField("tank_id", IntegerType(), True),
#             StructField("start_vd_time", TimestampType(), True),
#             StructField("end_vd_time", TimestampType(), True),
#             StructField("duration_seconds", LongType(), True),
#             StructField("confirmed_at", TimestampType(), True),
#             StructField("event_ts", TimestampType(), True),
#         ]
#     )

#     VD_DEBUG_SCHEMA = StructType(
#         [
#             StructField("timestamp", TimestampType(), True),
#             StructField("tank_id", IntegerType(), True),
#             StructField("vol", DoubleType(), True),
#             StructField("flow", DoubleType(), True),
#             StructField("flow_active", BooleanType(), True),
#             StructField("flow_rise", BooleanType(), True),
#             StructField("flow_fall", BooleanType(), True),
#             StructField("vd_session_id", LongType(), True),
#             StructField("vol_active", BooleanType(), True),
#             StructField("vol_rise", BooleanType(), True),
#             StructField("vol_fall", BooleanType(), True),
#             StructField("deep_vacuum_session_id", LongType(), True),
#         ]
#     )

#     spark = spark_df_vd_pims.sparkSession

#     # ----------------------------
#     # PIMS columns
#     # ----------------------------
#     TS = "timestamp"

#     V1_AR = "ACI@VD_VASO1_VOLUMEAR"
#     V1_N2 = "ACI@VD_VASO1_VOLUMEN2"
#     V2_AR = "ACI@VD_VASO2_VOLUMEAR"
#     V2_N2 = "ACI@VD_VASO2_VOLUMEN2"

#     F1_STRONG = "ACI@VD_VASO1_VZFORTE"
#     F1_WEAK = "ACI@VD_VASO1_VZFRACA"
#     F2_STRONG = "ACI@VD_VASO2_VZFORTE"
#     F2_WEAK = "ACI@VD_VASO2_VZFRACA"

#     flow_cols = [F1_STRONG, F1_WEAK, F2_STRONG, F2_WEAK]

#     columns_to_fill = [
#         V1_AR,
#         V1_N2,
#         V2_AR,
#         V2_N2,
#         F1_STRONG,
#         F1_WEAK,
#         F2_STRONG,
#         F2_WEAK,
#     ]

#     # ----------------------------
#     # 1) Preprocess: filter + cast + stable order + forward-fill (no future)
#     #
#     # We read some history before start_dt to avoid artificial edges at the boundary.
#     # ----------------------------

#     preprocess_start_ts = start_dt
#     startup_lookback_seconds = 6
#     if startup_lookback_seconds and startup_lookback_seconds > 0:
#         preprocess_start_ts = _shift_iso_ts(start_dt, -int(startup_lookback_seconds))

#     base = preprocess_pims_data_streaming(
#         spark_df=spark_df_vd_pims,
#         first_timestamp=preprocess_start_ts,
#         last_timestamp=end_dt,
#         timestamp_column=TS,
#         columns_to_fill=columns_to_fill,
#     )

#     # Consolidate to 1 row per timestamp (defensive against duplicates)
#     base_1p = (
#         base.groupBy(F.col(TS).cast("timestamp").alias(TS))
#         .agg(
#             # We use max() to collapse potential multiple rows at the same timestamp.
#             F.max(F.col(V1_AR)).alias(V1_AR),
#             F.max(F.col(V1_N2)).alias(V1_N2),
#             F.max(F.col(V2_AR)).alias(V2_AR),
#             F.max(F.col(V2_N2)).alias(V2_N2),
#             F.max(F.col(F1_STRONG)).alias(F1_STRONG),
#             F.max(F.col(F1_WEAK)).alias(F1_WEAK),
#             F.max(F.col(F2_STRONG)).alias(F2_STRONG),
#             F.max(F.col(F2_WEAK)).alias(F2_WEAK),
#         )
#         .orderBy(F.col(TS).asc())
#     )

#     if base_1p.rdd.isEmpty():
#         return (
#             spark.createDataFrame([], schema=VD_FINAL_SCHEMA),
#             spark.createDataFrame([], schema=VD_EVENTS_SCHEMA),
#             spark.createDataFrame([], schema=VD_DEBUG_SCHEMA),
#         )

#     # ----------------------------
#     # 1.1) Anti-noise: quantize flow tags to avoid tiny floats creating false sessions
#     #
#     # - Tiny values (< 1) can trigger false sessions if threshold is near 0/1.
#     # ----------------------------
#     def quantize_flow(c: str) -> F.Column:
#         return F.when(
#             F.col(c) < F.lit(1.0),
#             F.lit(0.0),
#         ).otherwise(F.round(F.col(c)))

#     for c in flow_cols:
#         base_1p = base_1p.withColumn(c, quantize_flow(c))

#     # ----------------------------
#     # 2) Compute totals (volume and flow per tank)
#     # ----------------------------
#     df_tot = base_1p.select(
#         F.col(TS),
#         (F.col(V1_AR) + F.col(V1_N2)).alias("vol_total_v1"),
#         (F.col(V2_AR) + F.col(V2_N2)).alias("vol_total_v2"),
#         (F.col(F1_STRONG) + F.col(F1_WEAK)).alias("flow_total_v1"),
#         (F.col(F2_STRONG) + F.col(F2_WEAK)).alias("flow_total_v2"),
#     )

#     # ----------------------------
#     # 3) Unpivot to (tank_id, timestamp, vol, flow)
#     #
#     # This avoids duplicating logic for tank (vessel) 1 and tank 2.
#     # ----------------------------
#     v1_struct = F.struct(
#         F.lit(1).alias("tank_id"),
#         F.col(TS).alias(TS),
#         F.col("vol_total_v1").alias("vol"),
#         F.col("flow_total_v1").alias("flow"),
#     )
#     v2_struct = F.struct(
#         F.lit(2).alias("tank_id"),
#         F.col(TS).alias(TS),
#         F.col("vol_total_v2").alias("vol"),
#         F.col("flow_total_v2").alias("flow"),
#     )

#     df_stacked = (
#         df_tot.select(F.array(v1_struct, v2_struct).alias("data"))
#         .select(F.explode("data").alias("d"))
#         .select(
#             F.col("d.tank_id").alias("tank_id"),
#             F.col(f"d.{TS}").alias(TS),
#             F.col("d.vol").alias("vol"),
#             F.col("d.flow").alias("flow"),
#         )
#         .orderBy(F.col(TS).asc(), F.col("tank_id").asc())
#     )

#     # ----------------------------
#     # 4) Edge detection with lag() only (streaming-safe, no future)
#     #
#     # - VD session id increments on each FLOW rising edge.
#     # - Deep vacuum session id increments on each VOLUME rising edge.
#     # ----------------------------
#     thr = float(reset_threshold)
#     w_tank = Window.partitionBy("tank_id").orderBy(F.col(TS).asc())

#     df_flags = (
#         df_stacked
#         .withColumn("flow_active", F.col("flow") > F.lit(thr))
#         .withColumn("prev_flow_active", F.lag("flow_active", 1, False).over(w_tank))
#         .withColumn("flow_rise", (~F.col("prev_flow_active")) & F.col("flow_active"))
#         .withColumn("flow_fall", F.col("prev_flow_active") & (~F.col("flow_active")))
#         # VD session id increments on each FLOW rising edge
#         .withColumn(
#             "vd_session_id",
#             F.sum(F.col("flow_rise").cast("long")).over(w_tank),
#         )
#         .withColumn("vol_active", F.col("vol") > F.lit(thr))
#         .withColumn("prev_vol_active", F.lag("vol_active", 1, False).over(w_tank))
#         .withColumn("vol_rise", (~F.col("prev_vol_active")) & F.col("vol_active"))
#         .withColumn("vol_fall", F.col("prev_vol_active") & (~F.col("vol_active")))
#         # Deep vacuum session id increments on each VOLUME rising edge
#         .withColumn(
#             "deep_vacuum_session_id",
#             F.sum(F.col("vol_rise").cast("long")).over(w_tank),
#         )
#     )

#     # ----------------------------
#     # 5) VD session aggregation (FLOW-based) + arrival/exit confirmation
#     #
#     # Goal:
#     #   - event_ts for arrival = start_vd_time (true rising edge)
#     #   - confirmed_at for arrival = first timestamp where
#     #       (ts - start_vd_time) >= min_vd_arrival_confirm_seconds
#     #
#     #   - event_ts for exit = end_vd_time (true falling edge)
#     #   - confirmed_at for exit = first timestamp where FLOW remained inactive
#     #     for at least min_vd_exit_confirm_seconds (optional; 0 means immediate)
#     #
#     #   - vd_last_seen_time = last timestamp where flow_active is true (used for open
#     #   duration)
#     # ----------------------------
#     w_vd_part = Window.partitionBy("tank_id", "vd_session_id")
#     w_vd_full = (
#         Window.partitionBy("tank_id", "vd_session_id")
#         .orderBy(F.col(TS).asc())
#         .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
#     )

#     vd_enriched = (
#         df_flags.filter(F.col("vd_session_id") > 0)
#         .withColumn(
#             "vd_start_time",
#             F.min(F.when(F.col("flow_active"), F.col(TS))).over(w_vd_part),
#         )
#         .withColumn(
#             "vd_duration_so_far_s",
#             F.when(
#                 F.col("flow_active"),
#                 F.col(TS).cast("long") - F.col("vd_start_time").cast("long"),
#             ),
#         )
#         .withColumn(
#             "vd_arrival_confirmed_at",
#             F.min(
#                 F.when(
#                     F.col("vd_duration_so_far_s")
#                     >= F.lit(int(min_vd_arrival_confirm_seconds)),
#                     F.col(TS),
#                 )
#             ).over(w_vd_part),
#         )
#         # Capture the true exit edge timestamp (first flow_fall in this session)
#         .withColumn(
#             "vd_end_edge_time",
#             F.min(F.when(F.col("flow_fall"), F.col(TS))).over(w_vd_part),
#         )
#         # After the fall edge, confirm exit after N seconds of sustained inactivity.
#         # If min_vd_exit_confirm_seconds == 0, confirm immediately at the edge time.
#         .withColumn(
#             "vd_exit_inactive_s",
#             F.when(
#                 (F.col("vd_end_edge_time").isNotNull())
#                 & (F.col(TS) >= F.col("vd_end_edge_time"))
#                 & (~F.col("flow_active")),
#                 F.col(TS).cast("long") - F.col("vd_end_edge_time").cast("long"),
#             ),
#         )
#         .withColumn(
#             "vd_exit_confirmed_at",
#             F.when(
#                 F.lit(int(min_vd_exit_confirm_seconds)) == F.lit(0),
#                 F.col("vd_end_edge_time"),
#             ).otherwise(
#                 F.min(
#                     F.when(
#                         F.col("vd_exit_inactive_s")
#                         >= F.lit(int(min_vd_exit_confirm_seconds)),
#                         F.col(TS),
#                     )
#                 ).over(w_vd_part)
#             ),
#         )
#         .withColumn(
#             "vd_last_seen_time",
#             F.max(F.when(F.col("flow_active"), F.col(TS))).over(w_vd_part),
#         )
#     )

#     vd_sessions_raw = (
#         vd_enriched.select(
#             "tank_id",
#             "vd_session_id",
#             F.col("vd_start_time").alias("start_vd_time"),
#             F.col("vd_end_edge_time").alias("end_vd_time"),
#             F.col("vd_last_seen_time").alias("last_seen_vd_time"),
#             F.col("vd_arrival_confirmed_at").alias("arrival_confirmed_at"),
#             F.col("vd_exit_confirmed_at").alias("exit_confirmed_at"),
#         )
#         .dropDuplicates(["tank_id", "vd_session_id"])
#         .filter(F.col("start_vd_time").isNotNull())
#         .withColumn(
#             "duration_seconds",
#             (
#                 F.coalesce(F.col("end_vd_time"), F.col("last_seen_vd_time")).cast("long")
#                 - F.col("start_vd_time").cast("long")
#             ),
#         )
#     )

#     # Drop short false VD sessions BEFORE assigning corrida / emitting events.
#     vd_sessions = vd_sessions_raw.filter(
#         F.col("duration_seconds") >= F.lit(int(min_vd_duration_seconds))
#     )

#     # corrida sequencial por ordem global de início (batch/microbatch)
#     w_global = Window.orderBy(F.col("start_vd_time").asc(), F.col("tank_id").asc())

#     vd_sessions_with_corrida = (
#         vd_sessions.withColumn(
#             "heat_increment", F.row_number().over(w_global) - F.lit(1)
#         )
#         .withColumn(
#             "corrida", F.lit(int(initial_heat_number)) + F.col("heat_increment")
#         )
#         .withColumn("predicted_tank_input", F.lit(predicted_tank_input))
#     )

#     # max volume observed while inside the VD session
#     max_vol_per_vd = (
#         df_flags.filter(F.col("vd_session_id") > 0)
#         .groupBy("tank_id", "vd_session_id")
#         .agg(F.max(F.col("vol")).alias("max_volume"))
#     )

#     df_vd_final = (
#         vd_sessions_with_corrida.join(
#             max_vol_per_vd,
#             on=["tank_id", "vd_session_id"],
#             how="left",
#         )
#         .select(
#             "corrida",
#             "tank_id",
#             "start_vd_time",
#             "end_vd_time",
#             F.col("duration_seconds").cast("long").alias("duration_seconds"),
#             F.coalesce(F.col("max_volume"), F.lit(0.0)).alias("max_volume"),
#             "predicted_tank_input",
#         )
#         .orderBy(F.col("start_vd_time").asc(), F.col("tank_id").asc())
#         .filter(F.col("start_vd_time") >= start_ts_col)
#     )

#     # ----------------------------
#     # 6) Deep vacuum detection (VOLUME-based) with "confirm_at"
#     #
#     # Goal:
#     #   - event_ts (true start): deep_vacuum_start_time = rising edge timestamp
#     #   - confirmed_at: first timestamp where (current_ts - deep_vacuum_start_time)
#     #                  reaches min_deep_vac_start_confirm_seconds
#     #
#     # This allows emitting DEEP_VAC_START as append-only (no retract):
#     #   - when confirmed_at happens, we emit the event with event_ts = real start.
#     #
#     # For DEEP_VAC_END:
#     #   - event_ts is the true falling edge
#     #   - confirmed_at can be immediate (0) or delayed by sustained inactivity
#     # ----------------------------
#     w_dv_part = Window.partitionBy("tank_id", "deep_vacuum_session_id")
#     w_dv_full = (
#         Window.partitionBy("tank_id", "deep_vacuum_session_id")
#         .orderBy(F.col(TS).asc())
#         .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
#     )

#     dv_enriched = (
#         df_flags.filter(F.col("deep_vacuum_session_id") > 0)
#         .withColumn(
#             "deep_vacuum_start_time",
#             F.min(F.when(F.col("vol_active"), F.col(TS))).over(w_dv_part),
#         )
#         .withColumn(
#             "deep_vacuum_end_edge_time",
#             F.min(F.when(F.col("vol_fall"), F.col(TS))).over(w_dv_part),
#         )
#         # duration so far at each active point (uses only past timestamps)
#         .withColumn(
#             "deep_vacuum_duration_so_far_s",
#             F.when(
#                 F.col("vol_active"),
#                 F.col(TS).cast("long")
#                 - F.col("deep_vacuum_start_time").cast("long"),
#             ),
#         )
#         # confirmed_at: first ts where duration_so_far >= minimal_duration_seconds
#         .withColumn(
#             "deep_vacuum_start_confirmed_at",
#             F.min(
#                 F.when(
#                     F.col("deep_vacuum_duration_so_far_s")
#                     >= F.lit(int(min_deep_vac_start_confirm_seconds)),
#                     F.col(TS),
#                 )
#             ).over(w_dv_part),
#         )
#         # Optional end confirmation:
#         # if end_confirm_seconds == 0 -> confirmed_at = end_edge
#         # else -> first ts after edge where inactive duration >= end_confirm_seconds
#         .withColumn(
#             "deep_vacuum_end_inactive_s",
#             F.when(
#                 (F.col("deep_vacuum_end_edge_time").isNotNull())
#                 & (F.col(TS) >= F.col("deep_vacuum_end_edge_time"))
#                 & (~F.col("vol_active")),
#                 F.col(TS).cast("long") - F.col("deep_vacuum_end_edge_time").cast("long"),
#             ),
#         )
#         .withColumn(
#             "deep_vacuum_end_confirmed_at",
#             F.when(
#                 F.lit(int(min_deep_vac_end_confirm_seconds)) == F.lit(0),
#                 F.col("deep_vacuum_end_edge_time"),
#             ).otherwise(
#                 F.min(
#                     F.when(
#                         F.col("deep_vacuum_end_inactive_s")
#                         >= F.lit(int(min_deep_vac_end_confirm_seconds)),
#                         F.col(TS),
#                     )
#                 ).over(w_dv_part)
#             ),
#         )
#         # Which VD session were we in when deep vacuum started?
#         .withColumn(
#             "vd_session_at_deep_vacuum_start",
#             F.first(
#                 F.when(F.col("vol_rise"), F.col("vd_session_id")),
#                 ignorenulls=True,
#             ).over(w_dv_full),
#         )
#     )

#     deep_vacuum_sessions = (
#         dv_enriched.select(
#             "tank_id",
#             "deep_vacuum_session_id",
#             "deep_vacuum_start_time",
#             "deep_vacuum_end_edge_time",
#             "deep_vacuum_start_confirmed_at",
#             "deep_vacuum_end_confirmed_at",
#             "vd_session_at_deep_vacuum_start",
#         )
#         .dropDuplicates(["tank_id", "deep_vacuum_session_id"])
#         # Keep only deep vacuum sessions that (a) happened inside a VD session and
#         # (b) reached the minimal duration to be considered valid (start confirmed).
#         .filter(F.col("vd_session_at_deep_vacuum_start") > 0)
#         .filter(F.col("deep_vacuum_start_confirmed_at").isNotNull())
#     )

#     # Join deep vacuum sessions back to corrida by matching VD session id
#     # Keep only deep vacuum sessions that belong to a VD session
#     # (i.e., VD duration >= min_vd_duration_seconds).
#     dv_with_corrida = (
#         deep_vacuum_sessions.join(
#             vd_sessions_with_corrida.select(
#                 "tank_id",
#                 "vd_session_id",
#                 "corrida",
#                 "start_vd_time",
#                 "end_vd_time",
#                 "duration_seconds",
#             ),
#             on=(
#                 (deep_vacuum_sessions["tank_id"] == vd_sessions_with_corrida["tank_id"])
#                 & (
#                     deep_vacuum_sessions["vd_session_at_deep_vacuum_start"]
#                     == vd_sessions_with_corrida["vd_session_id"]
#                 )
#             ),
#             how="inner",
#         )
#         .select(
#             F.col("corrida"),
#             deep_vacuum_sessions["tank_id"].alias("tank_id"),
#             F.col("start_vd_time"),
#             F.col("end_vd_time"),
#             F.col("duration_seconds").cast("long").alias("duration_seconds"),
#             F.col("deep_vacuum_start_time"),
#             F.col("deep_vacuum_end_edge_time").alias("deep_vacuum_end_time"),
#             F.col("deep_vacuum_start_confirmed_at"),
#             F.col("deep_vacuum_end_confirmed_at"),
#         )
#     )

#     # ----------------------------
#     # 7) Build df_vd_events (append-only)
#     #
#     # We emit events only after their confirmation timestamp is available.
#     # This ensures no retract is required downstream.
#     # ----------------------------

#     # VD_ARRIVAL: event_ts = start_vd_time, confirmed_at = start_vd_time (immediate)
#     arrival_events = (
#         vd_sessions_with_corrida.filter(F.col("arrival_confirmed_at").isNotNull())
#         .select(
#             F.col("corrida"),
#             F.lit("VD_ARRIVAL").alias("status"),
#             F.col("tank_id"),
#             F.col("start_vd_time"),
#             F.col("end_vd_time"),
#             F.col("duration_seconds").cast("long").alias("duration_seconds"),
#             F.col("arrival_confirmed_at").alias("confirmed_at"),
#             F.col("start_vd_time").alias("event_ts"),
#         )
#     )

#     # VD_EXIT: emitted only when we have end_vd_time
#     exit_events = (
#         vd_sessions_with_corrida.filter(F.col("end_vd_time").isNotNull())
#         .filter(F.col("exit_confirmed_at").isNotNull())
#         .select(
#             F.col("corrida"),
#             F.lit("VD_EXIT").alias("status"),
#             F.col("tank_id"),
#             F.col("start_vd_time"),
#             F.col("end_vd_time"),
#             F.col("duration_seconds").cast("long").alias("duration_seconds"),
#             F.col("exit_confirmed_at").alias("confirmed_at"),
#             F.col("end_vd_time").alias("event_ts"),
#         )
#     )

#     # DEEP_VAC_START:
#     #   - event_ts     = deep_vacuum_start_time (true start)
#     #   - confirmed_at = deep_vacuum_confirmed_at (first time duration>=min)
#     deep_vac_start_events = dv_with_corrida.select(
#         F.col("corrida"),
#         F.lit("DEEP_VAC_START").alias("status"),
#         F.col("tank_id"),
#         F.col("start_vd_time"),
#         F.col("end_vd_time"),
#         F.col("duration_seconds").cast("long").alias("duration_seconds"),
#         F.col("deep_vacuum_start_confirmed_at").alias("confirmed_at"),
#         F.col("deep_vacuum_start_time").alias("event_ts"),
#     )

#     # DEEP_VAC_END: emitted only when deep_vacuum_end_time exists
#     deep_vac_end_events = (
#         dv_with_corrida.filter(F.col("deep_vacuum_end_time").isNotNull())
#         .filter(F.col("deep_vacuum_end_confirmed_at").isNotNull())
#         .select(
#             F.col("corrida"),
#             F.lit("DEEP_VAC_END").alias("status"),
#             F.col("tank_id"),
#             F.col("start_vd_time"),
#             F.col("end_vd_time"),
#             F.col("duration_seconds").cast("long").alias("duration_seconds"),
#             F.col("deep_vacuum_end_confirmed_at").alias("confirmed_at"),
#             F.col("deep_vacuum_end_time").alias("event_ts"),
#         )
#     )

#     df_vd_events = (
#         arrival_events.unionByName(deep_vac_start_events)
#         .unionByName(deep_vac_end_events)
#         .unionByName(exit_events)
#         .orderBy(F.col("event_ts").asc(), F.col("tank_id").asc())
#         .filter(F.col("event_ts") >= start_ts_col)
#     )

#     # ----------------------------
#     # 8) Debug output (optional)
#     # ----------------------------
#     if debug:
#         df_vd_debug = df_flags.select(
#             F.col(TS).alias("timestamp"),
#             "tank_id",
#             "vol",
#             "flow",
#             "flow_active",
#             "flow_rise",
#             "flow_fall",
#             "vd_session_id",
#             "vol_active",
#             "vol_rise",
#             "vol_fall",
#             "deep_vacuum_session_id",
#         ).orderBy(F.col("timestamp").asc(), F.col("tank_id").asc())
#     else:
#         df_vd_debug = spark.createDataFrame([], schema=VD_DEBUG_SCHEMA)

#     return df_vd_final, df_vd_events, df_vd_debug



# def detect_tank_usage_events(
#     df: pyspark.sql.DataFrame, 
#     initial_heat_number: int, 
#     initial_timestamp: str, 
#     predicted_tank: Optional[int] = None,
#     minimal_duration: int = 10,
#     debug: bool = False,
#     treshold: float = 0.99
# ) -> pyspark.sql.DataFrame:
#     """
#     Detects usage cycles (events) for Tank 1 and Tank 2 based on cumulative gas volume data.
    
#     The function identifies the active periods of each vessel by detecting:
#     - Rising Edge: Transition from 0 to a positive volume (Start of usage).
#     - Falling Edge: Transition from a positive volume back to 0 (End of usage).
    
#     It consolidates Argon and Nitrogen volumes to determine the total tank activity.

#     Args:
#         df (pyspark.sql.DataFrame): Input PySpark DataFrame containing columns:
#                         'timestamp', 
#                         'ACI@VD_VASO1_VOLUMEAR', 'ACI@VD_VASO1_VOLUMEN2',
#                         'ACI@VD_VASO2_VOLUMEAR', 'ACI@VD_VASO2_VOLUMEN2'.
#         initial_heat_number (int): The identifier for the heat (corrida) to associate with these events.
#         initial_timestamp (str): ISO format timestamp string to filter the start of the analysis.
#         predicted_tank (Optional[int]): The expected tank to be used (1 or 2). Added as metadata to the output.
#         minimal_duration (int): Minimum duration in seconds for an event to be considered valid.
#         debug (bool): If True, prints additional information for debugging purposes.

#     Returns:
#         pyspark.sql.DataFrame: A DataFrame containing the detected events with the following schema:
#                    [tank_id, start_time, end_time, duration_seconds, max_volume, heat_number, predicted_tank]
#     """

#     # 1. Pre-processing: Filter by start time to optimize and fill nulls
#     # We explicitly select and rename columns to avoid syntax issues with special chars like '@'
#     # and handle nulls which would break the math.

#     clean_df = (
#         df
#         .filter(F.col("timestamp") >= F.lit(initial_timestamp))
#         .fillna(0, subset=[
#             "ACI@VD_VASO1_VOLUMEAR", "ACI@VD_VASO1_VOLUMEN2",
#             "ACI@VD_VASO2_VOLUMEAR", "ACI@VD_VASO2_VOLUMEN2"
#         ])
#     )

#     # 2. Calculate Total Volume per Tank
#     # Logic: Tank is active if either Ar or N2 volume is increasing.
#     df_vol = clean_df.select(
#         "*",
#         (F.col("ACI@VD_VASO1_VOLUMEAR") + F.col("ACI@VD_VASO1_VOLUMEN2")).alias("vol_total_v1"),
#         (F.col("ACI@VD_VASO2_VOLUMEAR") + F.col("ACI@VD_VASO2_VOLUMEN2")).alias("vol_total_v2"),
#     )

#     if debug:
#         displayHTML("df_vol")
#         display(df_vol)

#     # 3. Define Window for Time Series Analysis
#     w = pyspark.sql.Window.orderBy("timestamp")

#     # 4. Detect Edges using Lag (previous) and Lead (next)
#     # We calculate the previous value to detect starts, and look ahead to detect abrupt stops.
#     df_edges = df_vol.select(
#         "*",
#         (F.lag("vol_total_v1", 1, default=0).over(w)).alias("prev_vol_v1"),
#         (F.lag("vol_total_v2", 1, default=0).over(w)).alias("prev_vol_v2"),
#         # No default value to detect abrupt stop of data flow
#         (F.lead("vol_total_v1", 1, default=None).over(w)).alias("next_vol_v1"),
#         (F.lead("vol_total_v2", 1, default=None).over(w)).alias("next_vol_v2"),
#         # Add a global row number to identify the dataframe boundary
#         (F.row_number().over(w)).alias("global_row_number"),
#     )

#     if debug:
#         displayHTML("df_edges")
#         display(df_edges)

#     # 5. Flag Active Periods (State Calculation)
#     # Start Condition: Previous was 0 (or null treated as 0), Current > 0.
#     # End Condition: Current > 0, Next is 0 (Abrupt reset).
#     # We verify Vaso 1 and Vaso 2 separately.
    
#     # Unpivot (stack) the dataframe to handle Vaso 1 and Vaso 2 in a single generic logic
#     # This makes the code cleaner than duplicating logic for columns.
    
#     # Create a structure to stack: Tank ID, Timestamp, Current Vol, Prev Vol, Next Vol
#     v1_struct = F.struct(
#         F.lit(1).alias("tank_id"), 
#         F.col("timestamp"), 
#         F.col("vol_total_v1").alias("vol"),
#         F.col("prev_vol_v1").alias("prev"),
#         F.col("next_vol_v1").alias("next"),
#         F.col("global_row_number")
#     )
    
#     v2_struct = F.struct(
#         F.lit(2).alias("tank_id"), 
#         F.col("timestamp"), 
#         F.col("vol_total_v2").alias("vol"),
#         F.col("prev_vol_v2").alias("prev"),
#         F.col("next_vol_v2").alias("next"),
#         F.col("global_row_number")
#     )

#     # Explode array to get rows for Tank 1 and Tank 2
#     df_stacked = (
#         df_edges
#         .select(F.array(v1_struct, v2_struct).alias("data"))
#         .select(F.explode("data").alias("d"))
#         .select(
#             "d.tank_id", 
#             "d.timestamp", 
#             "d.vol", 
#             "d.prev", 
#             "d.next",
#             "d.global_row_number",
#         )
#     )

#     if debug:
#         displayHTML("df_stacked")
#         display(df_stacked)

#     # 6. Determine Start and End Flags based on stacked data
#     # Threshold is 1 to account for floating point noise, though 0 usually works for registers.
#     THRESHOLD = 0.99
    
#     df_flagged = df_stacked.select(
#         "*",
#         (
#             (F.col("prev") <= THRESHOLD) & (F.col("vol") > THRESHOLD) & (F.col("global_row_number") > 1)
#         ).alias("is_start"), 
#         (
#             (F.col("vol") > THRESHOLD)
#         ).alias("is_active"),
#         (
#             (F.col("vol") > THRESHOLD) & (F.col("next") <= THRESHOLD)
#         ).alias("is_clean_finish"),
#     )

#     if debug:
#         displayHTML("df_flagged")
#         display(df_flagged)

#     # 7. Create Session IDs
#     # We do a running sum of 'is_start' to assign a unique ID to each continuous block of activity per tank.
#     w_tank = pyspark.sql.Window.partitionBy("tank_id").orderBy("timestamp")
    
#     df_sessions = (
#         df_flagged
#         .filter(F.col("is_active"))
#         .withColumn("session_id", F.sum(F.col("is_start").cast("long")).over(w_tank))
#     )

#     if debug:
#         displayHTML("df_sessions")
#         display(df_sessions)

#     # 8. Aggregate to find Start and End of each session
#     agg_df = (
#         df_sessions
#         .filter(F.col("session_id") > 0)
#         .groupBy("tank_id", "session_id")
#         .agg(
#             F.min("timestamp").alias("start_time"),
#             F.max("timestamp").alias("last_seen_timestamp"), # This is the last timestamp WITH volume
#             F.max("vol").alias("max_volume"),
#             # If ANY row in the session has a clean finish, the session is complete.
#             F.max("is_clean_finish").alias("has_clean_finish") 
#         )
#         # .withColumn("duration_seconds", F.col("end_time").cast("long") - F.col("start_time").cast("long"))
#         # .withColumn("heat_number", F.lit(initial_heat_number))
#         # .withColumn("predicted_tank_input", F.lit(predicted_tank))
#     )

#     # 9. Filter and final Calculations: Duration, Null End Time, Heat Increment

#     # Window to order events globally (across both tanks) to assign incremental heat numbers
#     w_global_order = Window.orderBy("start_time", "tank_id")

#     events_df = agg_df.select(
#         "*",
#         (
#             F.when(
#                 F.col("has_clean_finish") == True,
#                 F.col("last_seen_timestamp")
#             ).otherwise(F.lit(None)) # Return None if it didn't finish cleanly
#         ).alias("end_time"),
#         (
#             F.col("last_seen_timestamp").cast("long") - F.col("start_time").cast("long")
#         ).alias("duration_seconds"),
#     )

#     events_df = (
#         events_df
#         # Filter to keep only events that are longer than the minimum threshold BEFORE incrementing
#         # heat numbers (to avoid incrementing for noise)
#         .filter(F.col("duration_seconds") >= minimal_duration)
#         .select(
#             "*",
#             # Increment heat numbers
#             (
#                 F.row_number().over(w_global_order) - 1
#             ).alias("heat_increment"), 
#             (
#                 F.lit(initial_heat_number) + F.col("heat_increment")
#             ).alias("corrida"), 
#             F.lit(predicted_tank).alias("predicted_tank_input")
#         )
#     )

#     # # 9. Filter to keep only events that are longer than the minimum threshold
#     # events_df = events_df.filter(F.col("duration_seconds") >= minimal_duration)

#     # Optional: Order by time for readability
#     return events_df.orderBy("start_time", "tank_id").select(
#         "corrida", "tank_id", "start_time", "end_time", "duration_seconds", "max_volume", "predicted_tank_input"
#     )