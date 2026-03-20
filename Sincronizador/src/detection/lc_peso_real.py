"""
lc_peso_real.py

Detecção de início e fim de vazamento no Lingotamento Contínuo (LC)
usando dados de peso real da torre (ACI@LC_TORRE_PESOREAL).

================================================================================
VISÃO GERAL
================================================================================

Detecta automaticamente:
    * Início do peso real: momento em que o peso sobe de ~0 para >20t
      (panela na posição de vazamento, torre girou)
    * Fim do peso real: momento em que o peso desce e ESTABILIZA
      (aço acabou na panela)

A detecção de fim usa estabilização (variação máxima em janela), não
um threshold fixo — funciona mesmo se o sinal estabilizar a 2.5t, 0.9t, etc.

Processa linha por linha (streaming-ready), sem lookahead.

================================================================================
ARQUITETURA DA FSM
================================================================================

    ┌──────┐   peso < LOW_T     ┌───────────┐   peso > HIGH_T    ┌─────────┐
    │ IDLE │ ─────────────────► │ WAIT_RISE │ ────────────────► │ CASTING │
    └──────┘                     └───────────┘                   └─────────┘
        ▲                                                            │
        │                                              peso < DESCENT_T
        │                                                            ▼
        │   estabilizou        ┌────────────┐
        └──────────────────── │  DRAINING  │  (monitora rolling range)
                               └────────────┘

Estados:
    * IDLE:       Sinal alto ou desconhecido, aguardando ir para zona baixa
    * WAIT_RISE:  Sinal está baixo (< LOW_T), aguardando subida (> HIGH_T)
    * CASTING:    Vazamento em andamento (peso alto), aguardando queda
    * DRAINING:   Peso caiu abaixo de DESCENT_T, monitorando estabilização

================================================================================
USO
================================================================================

    from src.detection.lc_peso_real import detect_lc_peso_real

    df_sys, df_events, df_debug = detect_lc_peso_real(
        df=df,  # timestamp, ACI@LC_TORRE_PESOREAL
    )

================================================================================
"""

import pandas as pd
import numpy as np
from typing import Tuple
from collections import deque


def detect_lc_peso_real(
    df: pd.DataFrame,

    # --- Suavização ---
    SMOOTH_WINDOW_S: int = 5,
    # Mediana móvel (segundos).

    # --- Thresholds de início (toneladas) ---
    LOW_T: float = 5.0,
    # Peso abaixo do qual a torre é considerada "vazia".

    HIGH_T: float = 20.0,
    # Peso acima do qual consideramos que o vazamento começou.

    # --- Thresholds de fim (toneladas) ---
    DESCENT_T: float = 5.0,
    # Peso abaixo do qual entramos na "zona de fim".

    STABLE_WINDOW_S: int = 60,
    # Janela (s) para verificar estabilização.

    MAX_RANGE_T: float = 0.3,
    # Variação máxima (t) na janela para considerar estável.

    # --- Filtro de gap mínimo ---
    MIN_GAP_S: float = 120.0,
    # Gap mínimo (s) entre detecções consecutivas.

    # --- Escala ---
    SCALE: float = 10.0,
    # Fator de conversão: valor_bruto / SCALE = toneladas.

    # --- Debug ---
    DEBUG: bool = False,

) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Detecta início e fim de vazamento usando peso real da torre.

    Streaming-ready: processa linha por linha com buffer circular
    para detecção de estabilização (sem lookahead).

    Args:
        df: DataFrame pandas com colunas:
            - timestamp: datetime
            - ACI@LC_TORRE_PESOREAL: peso bruto real da torre

    Returns:
        df_sys:    corrida_seq | inicio_pesoreal_ts | fim_pesoreal_ts |
                   peso_inicio_t | peso_fim_t | duracao_s
        df_events: corrida_seq | status | event_ts
        df_debug:  trace da FSM (se DEBUG=True)
    """

    # =========================================================================
    # 0) PREPARAÇÃO
    # =========================================================================
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    raw = pd.to_numeric(df["ACI@LC_TORRE_PESOREAL"], errors="coerce")
    w = max(1, int(SMOOTH_WINDOW_S))
    df["p_real"] = (raw / SCALE).rolling(w, min_periods=1).median()

    # =========================================================================
    # 1) FSM
    # =========================================================================
    state = "IDLE"
    corrida_seq = 0
    inicio_ts = None
    peso_inicio = None

    # Buffer circular para detecção de estabilização (streaming)
    drain_buffer = deque()  # (timestamp, value)

    results = []
    events = []
    debug_rows = []

    def dbg(event, **kw):
        if DEBUG:
            debug_rows.append({"event": event, "state": state, **kw})

    n = len(df)

    for i in range(n):
        ts = df.at[i, "timestamp"]
        p = df.at[i, "p_real"]

        if pd.isna(p):
            continue

        # ==================================================================
        # IDLE: aguardando sinal ir para zona baixa
        # ==================================================================
        if state == "IDLE":
            if p < LOW_T:
                state = "WAIT_RISE"
                dbg("WENT_LOW", i=i, ts=ts, p=round(p, 2))
            continue

        # ==================================================================
        # WAIT_RISE: sinal baixo, aguardando subida
        # ==================================================================
        if state == "WAIT_RISE":
            if p > HIGH_T:
                corrida_seq += 1
                inicio_ts = ts
                peso_inicio = round(float(p), 1)

                events.append({
                    "corrida_seq": corrida_seq,
                    "status": "INICIO",
                    "event_ts": ts,
                })
                dbg("INICIO", corrida_seq=corrida_seq, i=i, ts=ts, p=round(p, 2))

                state = "CASTING"
            elif p >= LOW_T:
                # Voltou acima de LOW_T mas não atingiu HIGH_T,
                # ainda está na zona intermediária, continua aguardando
                pass
            continue

        # ==================================================================
        # CASTING: vazamento em andamento, aguardando queda
        # ==================================================================
        if state == "CASTING":
            if p < DESCENT_T:
                state = "DRAINING"
                drain_buffer.clear()
                drain_buffer.append((ts, float(p)))
                dbg("DESCENT", corrida_seq=corrida_seq, i=i, ts=ts, p=round(p, 2))
            continue

        # ==================================================================
        # DRAINING: peso baixo, monitorando estabilização
        # ==================================================================
        if state == "DRAINING":
            # Se o peso voltar a subir acima de DESCENT_T, volta pra CASTING
            if p >= DESCENT_T:
                state = "CASTING"
                drain_buffer.clear()
                dbg("BACK_TO_CASTING", corrida_seq=corrida_seq, i=i, ts=ts, p=round(p, 2))
                continue

            # Adiciona ao buffer
            drain_buffer.append((ts, float(p)))

            # Remove amostras fora da janela
            while drain_buffer and (ts - drain_buffer[0][0]).total_seconds() > STABLE_WINDOW_S:
                drain_buffer.popleft()

            # Verifica estabilização: janela cheia E range < MAX_RANGE_T
            if len(drain_buffer) >= 2:
                buf_ts_first = drain_buffer[0][0]
                window_len = (ts - buf_ts_first).total_seconds()

                if window_len >= STABLE_WINDOW_S:
                    vals = [v for _, v in drain_buffer]
                    r = max(vals) - min(vals)

                    if r < MAX_RANGE_T:
                        # Estabilizou! O fim é o início da janela estável
                        fim_ts = buf_ts_first
                        peso_fim = round(vals[0], 2)
                        duracao = (fim_ts - inicio_ts).total_seconds() if inicio_ts else None

                        results.append({
                            "corrida_seq": corrida_seq,
                            "inicio_pesoreal_ts": inicio_ts,
                            "fim_pesoreal_ts": fim_ts,
                            "peso_inicio_t": peso_inicio,
                            "peso_fim_t": peso_fim,
                            "duracao_s": duracao,
                        })
                        events.append({
                            "corrida_seq": corrida_seq,
                            "status": "FIM",
                            "event_ts": fim_ts,
                        })
                        dbg("FIM", corrida_seq=corrida_seq, i=i, ts=ts,
                            fim_ts=fim_ts, p=round(p, 2), rng=round(r, 3),
                            duracao_s=duracao)

                        # Reset para próximo ciclo
                        state = "IDLE"
                        inicio_ts = None
                        peso_inicio = None
                        drain_buffer.clear()

            continue

    # =========================================================================
    # Corrida em andamento sem fim (RUNNING)
    # =========================================================================
    if state in ("CASTING", "DRAINING") and inicio_ts is not None:
        events.append({
            "corrida_seq": corrida_seq,
            "status": "RUNNING",
            "event_ts": df.at[len(df) - 1, "timestamp"],
        })

    # =========================================================================
    # 2) RETORNOS
    # =========================================================================
    df_sys = pd.DataFrame(
        results,
        columns=["corrida_seq", "inicio_pesoreal_ts", "fim_pesoreal_ts",
                 "peso_inicio_t", "peso_fim_t", "duracao_s"],
    )

    df_ev = pd.DataFrame(
        events,
        columns=["corrida_seq", "status", "event_ts"],
    )

    df_dbg = pd.DataFrame(debug_rows) if debug_rows else pd.DataFrame()

    return df_sys, df_ev, df_dbg
