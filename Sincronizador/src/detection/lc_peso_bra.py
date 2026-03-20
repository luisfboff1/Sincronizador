"""
lc_peso_bra.py

Detecção de chegada de panela na torre do Lingotamento Contínuo (LC)
usando dados de peso dos braços (PESOBRA1 / PESOBRA2).

================================================================================
VISÃO GERAL
================================================================================

Detecta automaticamente quando uma panela chega na torre, identificando a
subida do peso do braço de ~0 para >50 toneladas.

Processa linha por linha (streaming-ready), sem lookahead.

================================================================================
ARQUITETURA DA FSM
================================================================================

    ┌──────┐   peso < LOW_T      ┌──────────┐   peso > HIGH_T    ┌───────────┐
    │ IDLE │ ──────────────────►  │ WAIT_LOW │ ─────────────────► │ CONFIRMED │
    └──────┘                      └──────────┘                    └───────────┘
        ▲                                                              │
        │                          emite evento, volta                 │
        └──────────────────────────────────────────────────────────────┘

Estados (por braço, independente):
    * IDLE:       Aguardando sinal ir para baixo (< LOW_T)
    * WAIT_LOW:   Sinal está baixo, aguardando subida (> HIGH_T)
    * CONFIRMED:  Chegada detectada, emite evento e volta a IDLE

================================================================================
ESCALA
================================================================================

Os dados brutos são convertidos para toneladas (/ 10) internamente.
Todos os thresholds são em toneladas.

================================================================================
USO
================================================================================

    from src.detection.lc_peso_bra import detect_lc_chegada_torre

    df_sys, df_events, df_debug = detect_lc_chegada_torre(
        df=df,  # timestamp, ACI@LC_TORRE_PESOBRA1, ACI@LC_TORRE_PESOBRA2
    )

================================================================================
"""

import pandas as pd
import numpy as np
from typing import Tuple


def detect_lc_chegada_torre(
    df: pd.DataFrame,

    # --- Suavização ---
    SMOOTH_WINDOW_S: int = 5,
    # Mediana móvel (segundos).

    # --- Thresholds de detecção (toneladas) ---
    LOW_T: float = 10.0,
    # Peso abaixo do qual o braço é considerado "vazio".

    HIGH_T: float = 50.0,
    # Peso acima do qual consideramos que a panela chegou.

    # --- Filtro de gap mínimo ---
    MIN_GAP_S: float = 120.0,
    # Gap mínimo (s) entre detecções no mesmo braço para evitar ruído.

    # --- Escala ---
    SCALE: float = 10.0,
    # Fator de conversão: valor_bruto / SCALE = toneladas.

    # --- Debug ---
    DEBUG: bool = False,

) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Detecta chegada de panela na torre para cada braço.

    Streaming-ready: processa linha por linha, sem lookahead.

    Args:
        df: DataFrame pandas com colunas:
            - timestamp: datetime
            - ACI@LC_TORRE_PESOBRA1: peso bruto do braço 1
            - ACI@LC_TORRE_PESOBRA2: peso bruto do braço 2

    Returns:
        df_sys:    braco | chegada_torre_ts | peso_chegada_t
        df_events: braco | status | chegada_torre_ts | event_ts
        df_debug:  trace da FSM (se DEBUG=True)
    """

    # =========================================================================
    # 0) PREPARAÇÃO
    # =========================================================================
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    # Converter para toneladas e suavizar
    w = max(1, int(SMOOTH_WINDOW_S))
    for bra in [1, 2]:
        col = f"ACI@LC_TORRE_PESOBRA{bra}"
        raw = pd.to_numeric(df[col], errors="coerce")
        df[f"p_bra{bra}"] = (raw / SCALE).rolling(w, min_periods=1).median()

    # =========================================================================
    # 1) FSM — independente por braço
    # =========================================================================
    results = []
    events = []
    debug_rows = []

    def dbg(event, **kw):
        if DEBUG:
            debug_rows.append({"event": event, **kw})

    for bra in [1, 2]:
        col_p = f"p_bra{bra}"
        state = "IDLE"  # começa sem saber o estado
        last_event_ts = None

        for i in range(len(df)):
            ts = df.at[i, "timestamp"]
            p = df.at[i, col_p]

            if pd.isna(p):
                continue

            # ==============================================================
            # IDLE: aguardando o sinal ir para zona baixa
            # ==============================================================
            if state == "IDLE":
                if p < LOW_T:
                    state = "WAIT_LOW"
                    dbg("WENT_LOW", braco=bra, i=i, ts=ts, p=round(p, 2))
                continue

            # ==============================================================
            # WAIT_LOW: sinal está baixo, aguardando subida
            # ==============================================================
            if state == "WAIT_LOW":
                if p > HIGH_T:
                    # Verifica gap mínimo
                    if last_event_ts is None or (ts - last_event_ts).total_seconds() > MIN_GAP_S:
                        results.append({
                            "braco": bra,
                            "chegada_torre_ts": ts,
                            "peso_chegada_t": round(float(p), 1),
                        })
                        events.append({
                            "braco": bra,
                            "status": "CHEGADA",
                            "chegada_torre_ts": ts,
                            "event_ts": ts,
                        })
                        last_event_ts = ts
                        dbg("CHEGADA", braco=bra, i=i, ts=ts, p=round(p, 2))

                    state = "IDLE"
                continue

    # =========================================================================
    # 2) RETORNOS
    # =========================================================================
    df_sys = pd.DataFrame(
        results,
        columns=["braco", "chegada_torre_ts", "peso_chegada_t"],
    )

    df_ev = pd.DataFrame(
        events,
        columns=["braco", "status", "chegada_torre_ts", "event_ts"],
    )

    df_dbg = pd.DataFrame(debug_rows) if debug_rows else pd.DataFrame()

    return df_sys, df_ev, df_dbg
