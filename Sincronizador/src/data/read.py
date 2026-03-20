"""
read.py

Leitura centralizada de todas as tabelas do Sincronizador.
Cada função retorna um Spark DataFrame pronto para processamento.

Uso:
    from src.data.read import read_mes_tables, read_opc_table, read_pims_hist

    mes = read_mes_tables(spark)
    df_opc_lc = read_opc_table(spark, "lc")
    df_pims = read_pims_hist(spark)
"""

from typing import Dict, Optional

import pyspark.sql
from pyspark.sql import DataFrame, functions as F

from src.config import (
    TABLE_FEA_CORRIDAS,
    TABLE_FP_CORRIDA,
    TABLE_VD_CORRIDA_TEMPOS,
    TABLE_LC_CORRIDALINGOTAMENTO,
    TABLE_OPC_LC,
    TABLE_OPC_VD,
    TABLE_OPC_FP,
    TABLE_PIMS_HIST_TAGS,
    TABLE_PIMS_HIST_IBA,
    TABLE_ACI_RETORNO,
    UTC_OFFSET_HOURS,
    TABLE_CSV,
)


# ======================================================================
# MES (Registros do Operador)
# ======================================================================

def read_mes_tables(
    spark: pyspark.sql.SparkSession,
) -> Dict[str, DataFrame]:
    """
    Lê as 4 tabelas MES do operador e retorna um dicionário.

    Returns:
        dict com chaves: 'fea', 'fp', 'vd', 'lc'
    """
    fea_df = spark.table(TABLE_FEA_CORRIDAS)
    fp_df = spark.table(TABLE_FP_CORRIDA)
    vd_df = spark.table(TABLE_VD_CORRIDA_TEMPOS)
    lc_df = spark.table(TABLE_LC_CORRIDALINGOTAMENTO)

    return {
        "fea": fea_df,
        "fp": fp_df,
        "vd": vd_df,
        "lc": lc_df,
    }


# ======================================================================
# OPC (Sensores segundo-a-segundo)
# ======================================================================

_OPC_TABLE_MAP = {
    "lc": TABLE_OPC_LC,
    "vd": TABLE_OPC_VD,
    "fp": TABLE_OPC_FP,
}


def read_opc_table(
    spark: pyspark.sql.SparkSession,
    stage: str,
    convert_utc: bool = True,
    cast_values: bool = True,
) -> DataFrame:
    """
    Lê uma tabela OPC (segundo-a-segundo) para a etapa especificada.

    Args:
        spark: SparkSession ativa.
        stage: Etapa desejada ('lc', 'vd', 'fp').
        convert_utc: Se True, converte timestamp de UTC para UTC-3.
        cast_values: Se True, converte colunas ACI@ para double.

    Returns:
        Spark DataFrame com timestamp ajustado e colunas numéricas.
    """
    table_name = _OPC_TABLE_MAP.get(stage.lower())
    if table_name is None:
        raise ValueError(
            f"Etapa '{stage}' inválida. Use: {list(_OPC_TABLE_MAP.keys())}"
        )

    df = spark.read.table(table_name)

    # Converter UTC → UTC-3
    if convert_utc:
        df = df.withColumn(
            "timestamp",
            F.col("timestamp") - F.expr(f"INTERVAL {UTC_OFFSET_HOURS} HOURS"),
        )

    # Converter colunas ACI@ para double
    if cast_values:
        aci_cols = [c for c in df.columns if c.startswith("ACI@")]
        for c in aci_cols:
            df = df.withColumn(c, F.col(c).cast("double"))

    return df


# ======================================================================
# PIMS (Histórico de Tags)
# ======================================================================

def read_pims_hist(
    spark: pyspark.sql.SparkSession,
    source: str = "tags",
    timestamp_format: str = "dd-MMM-yy HH:mm:ss.S",
    tempo_col: str = "Tempo",
) -> DataFrame:
    """
    Lê tabela PIMS histórica e converte coluna 'Tempo' para timestamp.

    Args:
        spark: SparkSession ativa.
        source: 'tags' para hist_tags_novembro_2025,
                'iba' para hist_tags_IBA_novembro.
        timestamp_format: Formato da coluna Tempo.
        tempo_col: Nome da coluna de tempo original.

    Returns:
        Spark DataFrame com coluna 'timestamp' como TimestampType.
    """
    table_name = (
        TABLE_PIMS_HIST_TAGS if source.lower() == "tags" else TABLE_PIMS_HIST_IBA
    )

    df = spark.table(table_name)

    # Converter Tempo string → timestamp
    df = df.withColumn(
        "timestamp",
        F.to_timestamp(F.col(tempo_col), timestamp_format),
    )

    # Remover coluna original se diferente de 'timestamp'
    if tempo_col != "timestamp":
        df = df.drop(tempo_col)

    return df


def read_pims_with_opc(
    spark: pyspark.sql.SparkSession,
    stage: str = "lc",
    resample_to_minute: bool = True,
) -> DataFrame:
    """
    Lê PIMS histórico e faz JOIN com OPC (agrupado por minuto).
    Combina tags de ambas as fontes em um único DataFrame.

    Args:
        spark: SparkSession.
        stage: Etapa OPC ('lc', 'vd', 'fp').
        resample_to_minute: Se True, agrega OPC por minuto com média.

    Returns:
        DataFrame PIMS + OPC unificado.
    """
    df_pims = read_pims_hist(spark, source="tags")
    df_opc = read_opc_table(spark, stage=stage)

    # Identificar colunas de valores OPC
    opc_value_cols = [c for c in df_opc.columns if c.startswith("ACI@")]

    if resample_to_minute:
        # Truncar para minuto e agregar
        df_opc = df_opc.withColumn(
            "minute_ts", F.date_trunc("minute", "timestamp")
        )
        agg_exprs = [F.avg(F.col(c)).alias(c) for c in opc_value_cols]
        df_opc = (
            df_opc.groupBy("minute_ts")
            .agg(*agg_exprs)
            .orderBy("minute_ts")
        )

    return df_opc


# ======================================================================
# AUXILIARES
# ======================================================================

def read_retorno_heats(
    spark: pyspark.sql.SparkSession,
    min_heat: int = 0,
) -> list:
    """
    Retorna lista de corridas com retorno (para filtro/exclusão).
    """
    import pandas as pd

    df = (
        spark.table(TABLE_ACI_RETORNO)
        .withColumn("corrida", F.col("corrida").cast("int"))
        .filter(F.col("corrida") >= min_heat)
    )
    return list(df.toPandas().apply(pd.to_numeric, errors="ignore")["corrida"])


# ======================================================================
# CSV PIMS (Tags exportadas do PIMS em formato CSV)
# ======================================================================

def read_csv_pesobra(
    spark: pyspark.sql.SparkSession,
    mes_filtro: str = None,
    table: str = None,
) -> "pd.DataFrame":
    """
    Carrega PESOBRA1/2 da tabela CSV (tags PIMS exportadas).

    O CSV tem header embutido como linha (nomePIMS) e timestamps
    em formato dd.mm.yyyy HH:MM:SS.ffffff.

    Args:
        spark: SparkSession ativa.
        mes_filtro: Filtro de mês no formato "mm.yyyy" (ex: "02.2026").
                    Se None, carrega tudo.
        table: Nome da tabela CSV. Se None, usa TABLE_CSV do config.

    Returns:
        pandas DataFrame com colunas:
          timestamp, ACI@LC_TORRE_PESOBRA1, ACI@LC_TORRE_PESOBRA2
    """
    import pandas as pd

    tbl = table or TABLE_CSV
    df = spark.read.table(tbl).withColumn(
        "_row_id", F.monotonically_increasing_id()
    )
    first_col = df.columns[0]

    # Descobre header embutido
    hdr_id = (
        df.filter(F.col(first_col) == "nomePIMS")
        .select("_row_id").orderBy("_row_id").first()["_row_id"]
    )
    hdr = df.filter(F.col("_row_id") == hdr_id).drop("_row_id").first()

    new_cols = []
    for v in hdr:
        c = str(v).strip() if v else ""
        new_cols.append("timestamp" if c == "nomePIMS" else c)

    df_data = df.filter(F.col("_row_id") > hdr_id).drop("_row_id")
    for i, n in enumerate(new_cols):
        df_data = df_data.withColumnRenamed(df_data.columns[i], n)

    # Filtra período
    if mes_filtro:
        df_data = df_data.filter(
            F.col("timestamp").contains(f".{mes_filtro}")
        )

    pdf = df_data.select(
        "timestamp", "ACI@LC_TORRE_PESOBRA1", "ACI@LC_TORRE_PESOBRA2"
    ).toPandas()

    pdf["timestamp"] = pd.to_datetime(
        pdf["timestamp"], format="%d.%m.%Y %H:%M:%S.%f", errors="coerce"
    )
    pdf = (
        pdf.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    for col in ["ACI@LC_TORRE_PESOBRA1", "ACI@LC_TORRE_PESOBRA2"]:
        pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

    return pdf
