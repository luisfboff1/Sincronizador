"""
comparacao_operador.py

Funções de comparação entre timestamps detectados pelo sistema (FSM) e
registros manuais do operador (MES).

================================================================================
VISÃO GERAL
================================================================================

Este módulo é genérico: serve para qualquer etapa do Sincronizador (FEA, FP, VD, LC)
desde que os DataFrames tenham colunas de início/fim com sufixo adequado.

O match é feito por PROXIMIDADE TEMPORAL, não por número de corrida.
Para cada corrida do operador, encontra a corrida do sistema com timestamp
mais próximo, aceitando matches com delta < 15 minutos (configurável).

Os números de corrida podem diferir entre sistema e operador porque vêm de
fontes independentes.

================================================================================
USO
================================================================================

    from comparacao_operador import comparacao_operador

    # df_sys_lc: resultado da FSM (pandas)
    #   Colunas: corrida, inicio_lc_sys, final_lc_sys, braco

    # df_final: dados do operador (Spark ou Pandas)
    #   Colunas: corrida, inicio_lc, final_lc

    df_cmp = comparacao_operador(df_sys_lc, df_final)

    # Resultado:
    #   corrida          -> número da corrida do OPERADOR
    #   inicio_lc        -> timestamp do operador
    #   inicio_lc_sys    -> timestamp do sistema
    #   delta_inicio_s   -> diferença em segundos (negativo = sys antecipou)
    #   final_lc         -> timestamp do operador
    #   final_lc_sys     -> timestamp do sistema
    #   delta_fim_s      -> diferença em segundos
    #   braco_sys        -> braço detectado pelo sistema

================================================================================
"""

import pandas as pd


def comparacao_operador(
    df_sys: pd.DataFrame,
    df_operador,
    col_inicio_sys: str = "inicio_lc_sys",
    col_final_sys: str = "final_lc_sys",
    col_inicio_op: str = "inicio_lc",
    col_final_op: str = "final_lc",
    max_delta_minutes: int = 15,
) -> pd.DataFrame:
    """
    Compara corridas detectadas pelo sistema com registros do operador.

    O match é feito por PROXIMIDADE TEMPORAL (não por número de corrida).
    Para cada corrida do operador, encontra a corrida do sistema com
    inicio mais próximo.

    Args:
        df_sys: DataFrame pandas do sistema. Deve conter:
            - corrida (int)
            - col_inicio_sys (timestamp)
            - col_final_sys (timestamp)
            - braco (int, opcional)

        df_operador: DataFrame do operador (Spark ou Pandas). Deve conter:
            - corrida (int)
            - col_inicio_op (timestamp)
            - col_final_op (timestamp)

        col_inicio_sys: Nome da coluna de início no df do sistema.
        col_final_sys: Nome da coluna de fim no df do sistema.
        col_inicio_op: Nome da coluna de início no df do operador.
        col_final_op: Nome da coluna de fim no df do operador.
        max_delta_minutes: Delta máximo (minutos) para aceitar match.

    Returns:
        DataFrame com matches contendo:
            - corrida: número da corrida do OPERADOR
            - {col_inicio_op}: timestamp do operador
            - {col_inicio_sys}: timestamp do sistema
            - delta_inicio_s: diferença em segundos (negativo = sys antecipou)
            - {col_final_op}: timestamp do operador
            - {col_final_sys}: timestamp do sistema
            - delta_fim_s: diferença em segundos
            - braco_sys: braço detectado (se disponível)
    """

    # --------------------------------------------------------------------------
    # 1) Converter Spark para Pandas se necessário
    # --------------------------------------------------------------------------
    if hasattr(df_operador, "toPandas"):
        df_op = (
            df_operador
            .select("corrida", col_inicio_op, col_final_op)
            .toPandas()
        )
    else:
        df_op = df_operador[["corrida", col_inicio_op, col_final_op]].copy()

    if df_op.empty:
        print("AVISO: df_operador está vazio!")
        return pd.DataFrame()

    df_s = df_sys.copy()
    if df_s.empty:
        print("AVISO: df_sys está vazio!")
        return pd.DataFrame()

    # --------------------------------------------------------------------------
    # 2) Garantir datetime
    # --------------------------------------------------------------------------
    df_s[col_inicio_sys] = pd.to_datetime(df_s[col_inicio_sys])
    df_s[col_final_sys] = pd.to_datetime(df_s[col_final_sys])
    df_op[col_inicio_op] = pd.to_datetime(df_op[col_inicio_op])
    df_op[col_final_op] = pd.to_datetime(df_op[col_final_op])

    # --------------------------------------------------------------------------
    # 3) Match por proximidade temporal
    #    Para cada corrida do operador (ordenada por início), encontra a
    #    corrida do sistema com início mais próximo. Cada corrida do sistema
    #    só pode ser usada uma vez.
    # --------------------------------------------------------------------------
    matches = []
    sys_used = set()

    for _, op_row in df_op.sort_values(col_inicio_op).iterrows():
        best_idx = None
        best_delta = pd.Timedelta.max

        for sys_idx, sys_row in df_s.iterrows():
            if sys_idx in sys_used:
                continue
            delta = abs(sys_row[col_inicio_sys] - op_row[col_inicio_op])
            if delta < best_delta:
                best_delta = delta
                best_idx = sys_idx

        # Aceitar match se delta < max_delta_minutes
        if best_idx is not None and best_delta <= pd.Timedelta(minutes=max_delta_minutes):
            sys_used.add(best_idx)
            sys_row = df_s.loc[best_idx]
            matches.append({
                "corrida": op_row["corrida"],
                col_inicio_op: op_row[col_inicio_op],
                col_inicio_sys: sys_row[col_inicio_sys],
                "delta_inicio_s": (
                    sys_row[col_inicio_sys] - op_row[col_inicio_op]
                ).total_seconds(),
                col_final_op: op_row[col_final_op],
                col_final_sys: sys_row[col_final_sys],
                "delta_fim_s": (
                    sys_row[col_final_sys] - op_row[col_final_op]
                ).total_seconds(),
                "braco_sys": sys_row.get("braco", None),
            })

    # --------------------------------------------------------------------------
    # 4) Resultado
    # --------------------------------------------------------------------------
    if not matches:
        print(f"AVISO: Nenhum match encontrado (tolerância: {max_delta_minutes} min).")
        print(f"  Período operador: {df_op[col_inicio_op].min()} a {df_op[col_inicio_op].max()}")
        print(f"  Período sistema:  {df_s[col_inicio_sys].min()} a {df_s[col_inicio_sys].max()}")
        return pd.DataFrame()

    df_cmp = pd.DataFrame(matches)

    # Estatísticas resumidas
    print(f"Matches: {len(df_cmp)} de {len(df_op)} corridas do operador")
    print(
        f"Delta início - Média: {df_cmp['delta_inicio_s'].mean():.1f}s | "
        f"Mediana: {df_cmp['delta_inicio_s'].median():.1f}s | "
        f"Std: {df_cmp['delta_inicio_s'].std():.1f}s"
    )
    print(
        f"Delta fim    - Média: {df_cmp['delta_fim_s'].mean():.1f}s | "
        f"Mediana: {df_cmp['delta_fim_s'].median():.1f}s | "
        f"Std: {df_cmp['delta_fim_s'].std():.1f}s"
    )

    return df_cmp
