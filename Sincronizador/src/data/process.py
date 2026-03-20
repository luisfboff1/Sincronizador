"""
process.py

Pré-processamento e construção do DataFrame unificado (df_final).
Contém todas as funções de transformação reutilizáveis.

Uso:
    from src.data.process import build_df_final, preprocess_pims_data

    df_final = build_df_final(spark)
    df_processed = preprocess_pims_data(df_pims, start, end, ts_col, cols)
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyspark.sql
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.window import Window

from src.data.read import read_mes_tables


# ======================================================================
# FUNÇÕES AUXILIARES DE TRANSFORMAÇÃO
# ======================================================================

def convert_datetime(
    df: DataFrame,
    date_col: str,
    hour_col: str,
    new_col: str,
) -> DataFrame:
    """
    Combina uma coluna de data e uma de hora (HHmm ou HH:mm) em timestamp.

    Args:
        df: Spark DataFrame.
        date_col: Coluna com a data.
        hour_col: Coluna com a hora (formato HHmm ou HH:mm).
        new_col: Nome da nova coluna timestamp.

    Returns:
        DataFrame com a nova coluna timestamp criada.
    """
    return (
        df.withColumn("date_clean", F.to_date(F.col(date_col)))
        .withColumn(
            "hour_clean",
            F.when(
                F.length(F.col(hour_col)) == 4,
                F.concat_ws(
                    ":",
                    F.col(hour_col).substr(1, 2),
                    F.col(hour_col).substr(3, 2),
                ),
            ).otherwise(F.col(hour_col)),
        )
        .withColumn(
            new_col,
            F.to_timestamp(
                F.concat_ws(" ", F.col("date_clean"), F.col("hour_clean")),
                "yyyy-MM-dd HH:mm",
            ),
        )
        .drop("date_clean", "hour_clean")
    )


def fill_nulls_with_last_valid(
    df: DataFrame,
    timestamp_column: str,
    columns: List[str],
) -> DataFrame:
    """
    Forward-fill (PIMS): se valor veio null, propaga o último válido.
    """
    window_spec = Window.orderBy(timestamp_column).rowsBetween(
        Window.unboundedPreceding, 0
    )
    select_exprs = []
    target_set = set(columns)

    for col_name in df.columns:
        if col_name in target_set:
            select_exprs.append(
                F.last(F.col(col_name), ignorenulls=True)
                .over(window_spec)
                .alias(col_name)
            )
        else:
            select_exprs.append(F.col(col_name))

    return df.select(*select_exprs)


def fill_first_row_with_first_valid(
    df: DataFrame,
    columns: List[str],
) -> DataFrame:
    """
    Backfill apenas na primeira linha: garante que a série comece com valor válido.
    """
    select_exprs = []
    target_set = set(columns)
    w_first = Window.orderBy(F.monotonically_increasing_id())

    for col_name in df.columns:
        if col_name in target_set:
            first_valid_row = (
                df.filter(F.col(col_name).isNotNull()).select(col_name).first()
            )
            first_valid_value = first_valid_row[0] if first_valid_row else None
            select_exprs.append(
                F.when(
                    F.row_number().over(w_first) == 1,
                    F.lit(first_valid_value),
                )
                .otherwise(F.col(col_name))
                .alias(col_name)
            )
        else:
            select_exprs.append(F.col(col_name))

    return df.select(*select_exprs)


def repair_data_glitches(
    df: DataFrame,
    column: str,
    active_threshold: float = 1.0,
    lookahead_steps: int = 2,
    timestamp_col: str = "timestamp",
) -> DataFrame:
    """
    Repara glitches: queda rápida pra ~0 e volta rápido => forward-fill.
    """
    w_lag = Window.orderBy(timestamp_col)
    w_lookahead = Window.orderBy(timestamp_col).rowsBetween(1, lookahead_steps)
    w_ffill = Window.orderBy(timestamp_col).rowsBetween(
        Window.unboundedPreceding, Window.currentRow
    )

    is_glitch = (
        (F.col(column) < 1)
        & (F.lag(column, 1, 0).over(w_lag) > active_threshold)
        & (F.max(column).over(w_lookahead) > active_threshold)
    )

    col_temp = f"{column}_temp_clean"
    df_masked = df.select(
        "*",
        F.when(is_glitch, F.lit(None)).otherwise(F.col(column)).alias(col_temp),
    )
    df_fixed = df_masked.withColumn(
        column, F.last(col_temp, ignorenulls=True).over(w_ffill)
    ).drop(col_temp)

    return df_fixed


# ======================================================================
# PRÉ-PROCESSAMENTO PIMS
# ======================================================================

def preprocess_pims_data(
    spark_df: DataFrame,
    first_timestamp: str,
    last_timestamp: str,
    timestamp_column: str,
    columns_to_fill: List[str],
    repair_glitches: bool = True,
) -> DataFrame:
    """
    Pipeline completa de pré-processamento PIMS:
      1) Filtra janela temporal
      2) Forward-fill nulls
      3) Backfill primeira linha
      4) Repara glitches (opcional)

    Args:
        spark_df: DataFrame PIMS com timestamp.
        first_timestamp: Início da janela (string ISO).
        last_timestamp: Fim da janela (string ISO).
        timestamp_column: Nome da coluna de timestamp.
        columns_to_fill: Colunas para forward-fill.
        repair_glitches: Se True, repara glitches de leitura.

    Returns:
        DataFrame processado e filtrado.
    """
    # 1) Filtrar janela
    filtered = spark_df.filter(
        (F.col(timestamp_column) >= F.lit(first_timestamp))
        & (F.col(timestamp_column) <= F.lit(last_timestamp))
    )

    # 2) Forward-fill
    filled = fill_nulls_with_last_valid(
        filtered.orderBy(timestamp_column, ascending=True),
        timestamp_column=timestamp_column,
        columns=columns_to_fill,
    )

    # 3) Backfill primeira linha
    filled = fill_first_row_with_first_valid(filled, columns=columns_to_fill)

    # 4) Repair glitches
    if repair_glitches:
        for c in columns_to_fill:
            filled = repair_data_glitches(
                filled,
                column=c,
                active_threshold=1,
                lookahead_steps=2,
                timestamp_col=timestamp_column,
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


# ======================================================================
# CONSTRUÇÃO DO df_final (TABELA ÚNICA COM TODOS OS TEMPOS)
# ======================================================================

def build_df_final(
    spark: pyspark.sql.SparkSession,
    mes_tables: Optional[Dict[str, DataFrame]] = None,
) -> DataFrame:
    """
    Constrói o DataFrame unificado com timestamps de todas as etapas.

    Se `mes_tables` não for passado, lê as tabelas automaticamente.

    Colunas resultantes:
        corrida, inicio_fea, final_fea, inicio_fp_tttfp, inicio_fp,
        final_fp, inicio_vd, final_vd, inicio_lc, final_lc,
        ADICIONALTTTFP, ADICIONALTTTVD

    Returns:
        Spark DataFrame com todos os tempos por corrida.
    """
    if mes_tables is None:
        mes_tables = read_mes_tables(spark)

    fea_df = mes_tables["fea"]
    fp_df = mes_tables["fp"]
    vd_df = mes_tables["vd"]
    lc_df = mes_tables["lc"]

    # ------------------------------------------------------------------
    # FEA: hr_vaz_fea = DATA + HRVAZAMENTO; inicio_fea = hr_vaz - TTT
    # ------------------------------------------------------------------
    fea_df = fea_df.withColumn(
        "hr_vaz_fea",
        F.to_timestamp(
            F.concat_ws(" ", F.to_date("DATA"), F.col("HRVAZAMENTO")),
            "yyyy-MM-dd HH:mm",
        ),
    )
    fea_df = fea_df.withColumn(
        "TTT_min", F.col("TTT").cast("double")
    ).withColumn("TTT_seconds", (F.col("TTT_min") * 60).cast("long"))

    fea_df = fea_df.withColumn("final_fea", F.col("hr_vaz_fea")).withColumn(
        "inicio_fea",
        F.from_unixtime(
            F.col("hr_vaz_fea").cast("long") - F.col("TTT_seconds")
        ).cast("timestamp"),
    )

    # ------------------------------------------------------------------
    # FP: convert_datetime para chegada/saída + TTTFP
    # ------------------------------------------------------------------
    fp_df = convert_datetime(fp_df, "DATACHEGADAFP", "HORACHEGADAFP", "chegada_fp")
    fp_df = convert_datetime(fp_df, "DATASAIDAFP", "HORASAIDAFP", "saida_fp")
    fp_df = fp_df.withColumn("inicio_fp", F.col("chegada_fp")).withColumn(
        "final_fp", F.col("saida_fp")
    )
    fp_df = fp_df.withColumn(
        "TTTFP_min", F.col("TTTFP").cast("double")
    ).withColumn("TTTFP_seconds", (F.col("TTTFP_min") * 60).cast("long"))
    fp_df = fp_df.withColumn(
        "inicio_fp_tttfp",
        F.from_unixtime(
            F.col("saida_fp").cast("long") - F.col("TTTFP_seconds")
        ).cast("timestamp"),
    )

    # ------------------------------------------------------------------
    # VD: join com FP para trazer TTTVD
    # ------------------------------------------------------------------
    vd_joined = vd_df.alias("vd").join(
        fp_df.select("corrida", "TTTVD"), "corrida", "left"
    )
    vd_joined = (
        vd_joined.withColumn("final_vd", F.to_timestamp("datahorasaidavd"))
        .withColumn("TTTVD_min", F.col("TTTVD").cast("double"))
        .withColumn("TTTVD_seconds", (F.col("TTTVD_min") * 60).cast("long"))
        .withColumn(
            "inicio_vd",
            F.from_unixtime(
                F.col("datahorasaidavd").cast("long") - F.col("TTTVD_seconds")
            ).cast("timestamp"),
        )
    )

    # ------------------------------------------------------------------
    # LC: renomear para padrão
    # ------------------------------------------------------------------
    lc_df = lc_df.withColumn(
        "inicio_lc", F.to_timestamp("INICIOLINGOTAMENTO")
    ).withColumn("final_lc", F.to_timestamp("FINALLINGOTAMENTO"))

    # ------------------------------------------------------------------
    # JOIN FINAL
    # ------------------------------------------------------------------
    df_final = (
        fea_df.alias("fea")
        .join(fp_df.alias("fp"), "corrida", "left")
        .join(vd_joined.alias("vd"), "corrida", "left")
        .join(lc_df.alias("lc"), "corrida", "left")
        .select(
            "corrida",
            "inicio_fea",
            "final_fea",
            "inicio_fp_tttfp",
            "inicio_fp",
            "final_fp",
            "inicio_vd",
            "final_vd",
            "inicio_lc",
            "final_lc",
            "ADICIONALTTTFP",
            "ADICIONALTTTVD",
        )
    )

    return df_final


# ======================================================================
# PREPARAÇÃO PARA PANDAS (plot)
# ======================================================================

def prepare_lc_plot_data(
    df_pims: DataFrame,
    start_ts: str,
    end_ts: str,
    lc_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Pré-processa dados PIMS LC e converte para Pandas para plotagem.

    Args:
        df_pims: DataFrame PIMS com timestamp e colunas ACI@LC.
        start_ts: Início da janela.
        end_ts: Fim da janela.
        lc_columns: Lista de colunas LC. Se None, auto-detecta.

    Returns:
        Tupla (df_lc_plot: pd.DataFrame, lc_columns: List[str])
    """
    # Auto-detectar colunas LC se não fornecidas
    if lc_columns is None:
        all_lc = [c for c in df_pims.columns if c.startswith("ACI@LC")]
        # Filtrar colunas com dados
        lc_columns = []
        for col in all_lc:
            if df_pims.filter(F.col(col).isNotNull()).limit(1).count() > 0:
                lc_columns.append(col)

    # Pré-processar
    df_processed = preprocess_pims_data(
        spark_df=df_pims,
        first_timestamp=start_ts,
        last_timestamp=end_ts,
        timestamp_column="timestamp",
        columns_to_fill=lc_columns,
    )

    # Converter para Pandas
    df_plot = df_processed.select(["timestamp"] + lc_columns).toPandas()
    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])

    return df_plot, lc_columns
