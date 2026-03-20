"""lc_pairing.py

Emparelhamento de eventos de chegada na torre com ciclos de peso real,
formando corridas completas com metricas de timing.

Uso:
    from src.analysis.lc_pairing import emparelhar_corridas
    df_corridas = emparelhar_corridas(df_bra_sys, df_real_sys)
"""

import numpy as np
import pandas as pd


def emparelhar_corridas(
    df_bra_sys: pd.DataFrame,
    df_real_sys: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Emparelha chegadas na torre com ciclos de peso real.

    Cada inicio_pesoreal se associa a chegada_torre mais recente
    ANTES dele (sem repetir).

    Args:
        df_bra_sys: colunas [braco, chegada_torre_ts, peso_chegada_t]
        df_real_sys: colunas [corrida_seq, inicio_pesoreal_ts,
                              fim_pesoreal_ts, duracao_s]
        verbose: se True, imprime resumo.

    Returns:
        DataFrame com corridas emparelhadas e metricas de timing:
          braco, chegada_torre_ts, inicio/fim_pesoreal_ts,
          duracao_vazamento_s/min, chegada_to_inicio_min,
          ciclo_total_min, fim_to_chegada_min, fim_to_inicio_min
    """
    bra = df_bra_sys.sort_values("chegada_torre_ts").reset_index(drop=True)
    real = df_real_sys.sort_values("inicio_pesoreal_ts").reset_index(drop=True)

    corridas = []
    used = set()

    for _, row in real.iterrows():
        t_ini = row["inicio_pesoreal_ts"]
        t_fim = row["fim_pesoreal_ts"]
        dur = row["duracao_s"]

        cands = bra[
            (bra["chegada_torre_ts"] < t_ini) & (~bra.index.isin(used))
        ]

        if len(cands) > 0:
            best = cands["chegada_torre_ts"].idxmax()
            used.add(best)
            corridas.append({
                "braco": int(cands.loc[best, "braco"]),
                "chegada_torre_ts": cands.loc[best, "chegada_torre_ts"],
                "inicio_pesoreal_ts": t_ini,
                "fim_pesoreal_ts": t_fim,
                "duracao_vazamento_s": dur,
            })
        else:
            corridas.append({
                "braco": np.nan,
                "chegada_torre_ts": pd.NaT,
                "inicio_pesoreal_ts": t_ini,
                "fim_pesoreal_ts": t_fim,
                "duracao_vazamento_s": dur,
            })

    df = pd.DataFrame(corridas)

    # --- Metricas de timing (minutos) ---
    df["chegada_to_inicio_min"] = (
        (df["inicio_pesoreal_ts"] - df["chegada_torre_ts"])
        .dt.total_seconds() / 60
    )
    df["duracao_vazamento_min"] = df["duracao_vazamento_s"] / 60
    df["ciclo_total_min"] = (
        (df["fim_pesoreal_ts"] - df["chegada_torre_ts"])
        .dt.total_seconds() / 60
    )

    # Intervalos entre corridas consecutivas
    df["fim_to_chegada_min"] = np.nan
    df["fim_to_inicio_min"] = np.nan
    for i in range(1, len(df)):
        prev_fim = df.at[i - 1, "fim_pesoreal_ts"]
        curr_cheg = df.at[i, "chegada_torre_ts"]
        curr_ini = df.at[i, "inicio_pesoreal_ts"]
        if pd.notna(prev_fim) and pd.notna(curr_cheg):
            df.at[i, "fim_to_chegada_min"] = (
                (curr_cheg - prev_fim).total_seconds() / 60
            )
        if pd.notna(prev_fim) and pd.notna(curr_ini):
            df.at[i, "fim_to_inicio_min"] = (
                (curr_ini - prev_fim).total_seconds() / 60
            )

    if verbose:
        n_cheg = df["chegada_torre_ts"].notna().sum()
        print(f"Emparelhadas: {len(df)} corridas "
              f"({n_cheg} com chegada, {len(df) - n_cheg} sem)")

    return df
