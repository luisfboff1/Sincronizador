"""
lc_pesobra_logic.py

Detecção automática de eventos de início/fim de corridas de Lingotamento Contínuo (LC)
usando dados de peso dos braços da torre (PESOBRA1 / PESOBRA2).

================================================================================
VISÃO GERAL
================================================================================

Este módulo implementa uma Máquina de Estados Finitos (FSM) que processa dados
de peso dos braços da torre segundo a segundo e detecta automaticamente:

    * Início de corrida: momento em que o peso começa a cair (lingotamento iniciado)
    * Fim de corrida: momento em que o peso sobe após cruzar 60t (novo carregamento)

A detecção alterna automaticamente entre Braço 1 e Braço 2.

================================================================================
ARQUITETURA DA FSM
================================================================================

    ┌──────┐    edge detectado     ┌──────────────┐   gap > 180s    ┌─────────┐
    │ IDLE │ ──────────────────►   │ START_ARMED  │ ──────────────► │ RUNNING │
    └──────┘                       └──────────────┘                 └─────────┘
        ▲                                                               │
        │                     peso sobe 2t após cruzar 60t              │
        └───────────────────────────────────────────────────────────────┘

Estados:
    * IDLE: Aguardando candidato de início no braço atual
    * START_ARMED: Acumulando candidatos; fecha cluster após CLUSTER_GAP_S sem novos
    * RUNNING: Corrida em andamento; monitorando peso para detectar fim

================================================================================
PIPELINE DE FEATURES
================================================================================

    1. Smoothing: Mediana móvel 5s sobre PESOBRA
    2. R1 (Queda Brusca): p[i] - p[i-8] <= -2t, se esteve >100t nos últimos 60s
    3. R2 (Sequência): 5s consecutivos com diff(p) < -0.05
    4. Refractory: Bloqueia 80s após salto de carga (+6t em 5s)
    5. Faixa Válida: Candidato entre 80t e 115t
    6. Peak Gate: (Opcional) Exige pico >108t nos últimos 60s
    7. Edge Detection: Flanco de subida (False->True) nas flags R1|R2
    8. Score: Queda futura 10s * 2.0 + bônus R1(3.0) + bônus R2(2.0)

================================================================================
USO
================================================================================

    from lc_pesobra_logic import detect_lc_events_fsm_pesobra

    # df_lc_plot: DataFrame pandas com colunas:
    #   - timestamp: datetime
    #   - ACI@LC_TORRE_PESOBRA1: peso braço 1
    #   - ACI@LC_TORRE_PESOBRA2: peso braço 2

    df_sys, df_events, df_debug, df_feat = detect_lc_events_fsm_pesobra(
        df=df_lc_plot,
        initial_corrida=130268,
        initial_braco=1,
    )

Para comparar com operador, veja: comparacao_operador.py

================================================================================
"""

import pandas as pd
import numpy as np
from typing import Tuple


def detect_lc_events_fsm_pesobra(
    df: pd.DataFrame,

    # =========================================================================
    # PARÂMETROS DE DETECÇÃO DE INÍCIO
    # Os defaults abaixo são os valores CALIBRADOS que funcionaram melhor
    # nos testes com dados de novembro/2025.
    # =========================================================================

    # --- Suavização ---
    SMOOTH_WINDOW_S: int = 5,
    # Janela da mediana móvel (segundos). Suaviza ruído no sinal de peso.

    # --- Regra "esteve carregado" ---
    HIGH_WEIGHT: float = 100.0,
    # Peso mínimo (t) para considerar que o braço estava carregado.

    HIGH_LOOKBACK_S: int = 60,
    # Janela de lookback (s) para verificar se esteve acima de HIGH_WEIGHT.

    # --- R1: Queda Brusca ---
    DROP_WINDOW_S: int = 8,
    # Janela (s) para calcular a queda: p[i] - p[i - DROP_WINDOW_S]
    # Calibrado: 8s funcionou melhor que 5s (menos falsos positivos).

    DROP_TON: float = 2.0,
    # Queda mínima (t) para disparar R1.
    # Calibrado: 2.0t mais sensível que 6.0t original.

    # --- R2: Sequência de Quedas ---
    DP_SEQ_S: int = 5,
    # Segundos consecutivos com diff(p) < -DP_EPS para disparar R2.
    # Calibrado: 5s (era 8s).

    DP_EPS: float = 0.05,
    # Epsilon mínimo de queda por segundo para R2.
    # Calibrado: 0.05 (era 0.0, qualquer queda).

    # --- Período Refratário (evita falsos positivos durante carga) ---
    USE_REFRACTORY: bool = True,
    # Ativar bloqueio após salto de carga.

    LOAD_JUMP_WINDOW_S: int = 5,
    # Janela (s) para detectar salto de carga.

    LOAD_JUMP_TON: float = 6.0,
    # Salto mínimo (t) para considerar que houve carga.

    REFRACTORY_AFTER_LOAD_S: int = 80,
    # Segundos de bloqueio após carga.
    # Calibrado: 80s (era 40s). Valor maior evitou falsos durante recarga.

    # --- Peak Gate (opcional) ---
    USE_PEAK_GATE: bool = False,
    # Se True, exige pico recente antes do candidato.
    # Calibrado: False por enquanto. Ligar depois de validar.

    PEAK_MIN_WEIGHT: float = 108.0,
    # Peso mínimo do pico (t). Calibrado: 108t (era 112t).

    PEAK_LOOKBACK_S: int = 60,
    # Lookback (s) para buscar pico. Calibrado: 60s (era 180s).

    PEAK_AFTER_MIN_S: int = 0,
    # Candidato deve ocorrer pelo menos X segundos após o pico.

    PEAK_AFTER_MAX_S: int = 60,
    # Candidato deve ocorrer no máximo X segundos após o pico.
    # Calibrado: 60s (era 120s).

    # --- Faixa de Peso Válida ---
    MIN_CAND_WEIGHT: float = 80.0,
    # Peso mínimo (t) para aceitar candidato.

    MAX_CAND_WEIGHT: float = 115.0,
    # Peso máximo (t) para aceitar candidato.

    # --- Score (ranking de candidatos dentro do cluster) ---
    SCORE_FUTURE_WINDOW_S: int = 10,
    # Janela futura (s) para calcular queda e pontuar candidato.
    # Calibrado: 10s (era 60s).

    SCORE_DROP_WEIGHT: float = 2.0,
    # Peso da queda futura no score.

    SCORE_R1_BONUS: float = 3.0,
    # Bônus de score se R1 disparou.

    SCORE_R2_BONUS: float = 2.0,
    # Bônus de score se R2 disparou.

    # --- Clustering ---
    CLUSTER_GAP_S: int = 180,
    # Gap (s) sem novos candidatos para fechar o cluster e confirmar início.

    # =========================================================================
    # PARÂMETROS DE DETECÇÃO DE FIM
    # =========================================================================

    END_ARM_WEIGHT: float = 60.0,
    # Peso (t) que "arma" a detecção de fim.

    END_RISE_TON: float = 2.0,
    # Subida mínima (t) após cruzar END_ARM_WEIGHT para confirmar fim.

    END_RISE_WINDOW_S: int = 25,
    # Janela (s) máxima para a subida ocorrer. Se estoura, faz sliding reference.
    # Calibrado: 25s (era 20s).

    # =========================================================================
    # PARÂMETROS DE FLUXO
    # =========================================================================

    initial_corrida: int = 0,
    # Número da primeira corrida a ser detectada.

    initial_braco: int = 1,
    # Braço inicial (1 ou 2). Alterna automaticamente após cada corrida.

    DEBUG: bool = True,
    # Se True, retorna DataFrames de debug e features.

) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Detecta eventos de início/fim de corridas LC usando FSM sequencial.

    Streaming-ready: processa linha por linha, sem lookahead.

    Args:
        df: DataFrame pandas com colunas:
            - timestamp: datetime
            - ACI@LC_TORRE_PESOBRA1: peso do braço 1
            - ACI@LC_TORRE_PESOBRA2: peso do braço 2

    Returns:
        df_sys_lc:    corrida | inicio_lc_sys | final_lc_sys | braco | close_reason
        df_events:    corrida | status | inicio_lc_sys | final_lc_sys | braco | event_ts
        df_debug:     trace completo da FSM (se DEBUG=True)
        df_features:  DataFrame com todas as features calculadas
    """

    # ==========================================================================
    # 0) PREPARAÇÃO E CÁLCULO DE FEATURES
    # ==========================================================================

    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # Garantir datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    # Converter pesos para numérico
    df["peso_bra1"] = pd.to_numeric(df["ACI@LC_TORRE_PESOBRA1"], errors="coerce")
    df["peso_bra2"] = pd.to_numeric(df["ACI@LC_TORRE_PESOBRA2"], errors="coerce")

    def s2n(s: int) -> int:
        return max(1, int(s))

    # --------------------------------------------------------------------------
    # Suavização: mediana móvel
    # --------------------------------------------------------------------------
    w = s2n(SMOOTH_WINDOW_S)
    df["p1"] = df["peso_bra1"].rolling(w, min_periods=1).median()
    df["p2"] = df["peso_bra2"].rolling(w, min_periods=1).median()

    # --------------------------------------------------------------------------
    # Feature: "esteve alto" (carregado) nos últimos N segundos
    # --------------------------------------------------------------------------
    lb = s2n(HIGH_LOOKBACK_S)
    df["was_high_bra1"] = (
        df["p1"].rolling(lb, min_periods=1).max() >= HIGH_WEIGHT
    ).fillna(False).astype(bool)
    df["was_high_bra2"] = (
        df["p2"].rolling(lb, min_periods=1).max() >= HIGH_WEIGHT
    ).fillna(False).astype(bool)

    # --------------------------------------------------------------------------
    # Feature R1: queda brusca
    # --------------------------------------------------------------------------
    dw = s2n(DROP_WINDOW_S)
    df["drop1"] = df["p1"] - df["p1"].shift(dw)
    df["drop2"] = df["p2"] - df["p2"].shift(dw)
    df["r1_bra1"] = (
        df["was_high_bra1"] & (df["drop1"] <= -DROP_TON).fillna(False)
    ).fillna(False).astype(bool)
    df["r1_bra2"] = (
        df["was_high_bra2"] & (df["drop2"] <= -DROP_TON).fillna(False)
    ).fillna(False).astype(bool)

    # --------------------------------------------------------------------------
    # Feature R2: sequência de quedas consecutivas
    # --------------------------------------------------------------------------
    df["dp1"] = df["p1"].diff()
    df["dp2"] = df["p2"].diff()
    seq = s2n(DP_SEQ_S)
    fall1 = (df["dp1"] < -DP_EPS).fillna(False).rolling(seq, min_periods=seq).sum() == seq
    fall2 = (df["dp2"] < -DP_EPS).fillna(False).rolling(seq, min_periods=seq).sum() == seq
    df["fallseq1"] = fall1.fillna(False).astype(bool)
    df["fallseq2"] = fall2.fillna(False).astype(bool)
    df["r2_bra1"] = (df["was_high_bra1"] & df["fallseq1"]).fillna(False).astype(bool)
    df["r2_bra2"] = (df["was_high_bra2"] & df["fallseq2"]).fillna(False).astype(bool)

    # --------------------------------------------------------------------------
    # Feature: período refratário (bloqueia após salto de carga)
    # --------------------------------------------------------------------------
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
        df["ok_ref1"] = (
            df["last_load_idx1"].isna() | ((idx - df["last_load_idx1"]) >= ref)
        ).fillna(False).astype(bool)
        df["ok_ref2"] = (
            df["last_load_idx2"].isna() | ((idx - df["last_load_idx2"]) >= ref)
        ).fillna(False).astype(bool)

        # Aplicar refractory nas regras R1 e R2
        df["r1_bra1"] = (df["r1_bra1"] & df["ok_ref1"]).fillna(False).astype(bool)
        df["r1_bra2"] = (df["r1_bra2"] & df["ok_ref2"]).fillna(False).astype(bool)
        df["r2_bra1"] = (df["r2_bra1"] & df["ok_ref1"]).fillna(False).astype(bool)
        df["r2_bra2"] = (df["r2_bra2"] & df["ok_ref2"]).fillna(False).astype(bool)
    else:
        df["ok_ref1"] = True
        df["ok_ref2"] = True

    # --------------------------------------------------------------------------
    # Combinar: candidato de início (R1 ou R2)
    # --------------------------------------------------------------------------
    df["start_any_bra1"] = (df["r1_bra1"] | df["r2_bra1"]).fillna(False).astype(bool)
    df["start_any_bra2"] = (df["r1_bra2"] | df["r2_bra2"]).fillna(False).astype(bool)

    # Filtrar por faixa de peso válida
    df["cand_range1"] = (
        (df["p1"] >= MIN_CAND_WEIGHT) & (df["p1"] <= MAX_CAND_WEIGHT)
    ).fillna(False).astype(bool)
    df["cand_range2"] = (
        (df["p2"] >= MIN_CAND_WEIGHT) & (df["p2"] <= MAX_CAND_WEIGHT)
    ).fillna(False).astype(bool)
    df["start_any_bra1"] = (df["start_any_bra1"] & df["cand_range1"]).fillna(False).astype(bool)
    df["start_any_bra2"] = (df["start_any_bra2"] & df["cand_range2"]).fillna(False).astype(bool)

    # --------------------------------------------------------------------------
    # Peak Gate (opcional)
    # --------------------------------------------------------------------------
    if USE_PEAK_GATE:
        idx = np.arange(len(df))
        df["is_peak1"] = (df["p1"] >= PEAK_MIN_WEIGHT).fillna(False).astype(bool)
        df["is_peak2"] = (df["p2"] >= PEAK_MIN_WEIGHT).fillna(False).astype(bool)

        lastp1 = np.where(df["is_peak1"].to_numpy(), idx, np.nan)
        lastp2 = np.where(df["is_peak2"].to_numpy(), idx, np.nan)
        df["last_peak_idx1"] = pd.Series(lastp1).ffill()
        df["last_peak_idx2"] = pd.Series(lastp2).ffill()

        look = s2n(PEAK_LOOKBACK_S)
        df["has_peak_recent1"] = (
            df["is_peak1"].rolling(look, min_periods=1).max() > 0
        ).fillna(False).astype(bool)
        df["has_peak_recent2"] = (
            df["is_peak2"].rolling(look, min_periods=1).max() > 0
        ).fillna(False).astype(bool)

        amin = s2n(PEAK_AFTER_MIN_S)
        amax = s2n(PEAK_AFTER_MAX_S)

        df["after_peak1"] = (
            df["last_peak_idx1"].notna()
            & ((idx - df["last_peak_idx1"]) >= amin)
            & ((idx - df["last_peak_idx1"]) <= amax)
        ).fillna(False).astype(bool)
        df["after_peak2"] = (
            df["last_peak_idx2"].notna()
            & ((idx - df["last_peak_idx2"]) >= amin)
            & ((idx - df["last_peak_idx2"]) <= amax)
        ).fillna(False).astype(bool)

        df["gate1"] = (df["has_peak_recent1"] & df["after_peak1"]).fillna(False).astype(bool)
        df["gate2"] = (df["has_peak_recent2"] & df["after_peak2"]).fillna(False).astype(bool)

        df["start_any_bra1"] = (df["start_any_bra1"] & df["gate1"]).fillna(False).astype(bool)
        df["start_any_bra2"] = (df["start_any_bra2"] & df["gate2"]).fillna(False).astype(bool)
    else:
        df["gate1"] = True
        df["gate2"] = True

    # --------------------------------------------------------------------------
    # Edge Detection (flanco de subida: False -> True)
    # Evita disparar múltiplas vezes no mesmo evento contínuo.
    # --------------------------------------------------------------------------
    prev1 = df["start_any_bra1"].shift(1).fillna(False).astype(bool)
    prev2 = df["start_any_bra2"].shift(1).fillna(False).astype(bool)
    df["edge_bra1"] = (df["start_any_bra1"] & (~prev1)).fillna(False).astype(bool)
    df["edge_bra2"] = (df["start_any_bra2"] & (~prev2)).fillna(False).astype(bool)

    # --------------------------------------------------------------------------
    # Score (ranking de candidatos: queda futura + bônus R1/R2)
    # --------------------------------------------------------------------------
    fw = s2n(SCORE_FUTURE_WINDOW_S)
    fmin1 = df["p1"].shift(-1)[::-1].rolling(fw, min_periods=1).min()[::-1]
    fmin2 = df["p2"].shift(-1)[::-1].rolling(fw, min_periods=1).min()[::-1]
    df["future_drop1"] = df["p1"] - fmin1
    df["future_drop2"] = df["p2"] - fmin2

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

    # ==========================================================================
    # 1) MÁQUINA DE ESTADOS FINITOS (FSM)
    # ==========================================================================

    state = "IDLE"
    corrida = int(initial_corrida)
    braco = int(initial_braco)

    # Cluster de candidatos (acumula durante START_ARMED)
    cluster = []
    last_cand_ts = None

    # Contexto da corrida em andamento (RUNNING)
    running_start_ts = None
    running_start_i = None
    crossed_60 = False
    p_ref = None      # referência de peso para sliding reference
    t_ref = None       # timestamp da referência

    # Resultados
    results = []
    events = []        # tabela append-only de estados (RUNNING/CLOSED)
    debug_rows = []

    def dbg(event: str, **kwargs):
        """Append debug row se DEBUG=True."""
        if not DEBUG:
            return
        row = {"event": event, "state": state}
        row.update(kwargs)
        debug_rows.append(row)

    n = len(df)

    for i in range(n):
        ts = df.at[i, "timestamp"]

        # Peso do braço atual (para fim)
        p_run = df.at[i, "p1"] if braco == 1 else df.at[i, "p2"]

        # Flags de edge por braço (para início)
        edge1 = bool(df.at[i, "edge_bra1"])
        edge2 = bool(df.at[i, "edge_bra2"])

        # ======================================================================
        # ESTADO: IDLE
        # Aguardando edge de início no braço atual.
        # ======================================================================
        if state == "IDLE":
            is_edge = edge1 if braco == 1 else edge2

            if is_edge:
                score = float(df.at[i, "score1"]) if braco == 1 else float(df.at[i, "score2"])
                p_here = float(df.at[i, "p1"]) if braco == 1 else float(df.at[i, "p2"])
                r1 = bool(df.at[i, "r1_bra1"]) if braco == 1 else bool(df.at[i, "r1_bra2"])
                r2 = bool(df.at[i, "r2_bra1"]) if braco == 1 else bool(df.at[i, "r2_bra2"])

                cluster = [{
                    "i": int(i), "timestamp": ts, "score": score,
                    "p": p_here, "r1": r1, "r2": r2,
                }]
                last_cand_ts = ts
                state = "START_ARMED"

                dbg("CLUSTER_OPEN", i=i, timestamp=ts, braco=braco, p=p_here, score=score)

            continue

        # ======================================================================
        # ESTADO: START_ARMED
        # Acumulando edges em cluster. Fecha após CLUSTER_GAP_S sem novos.
        # Seleciona candidato com melhor score (empate: mais tardio).
        # ======================================================================
        if state == "START_ARMED":
            is_edge = edge1 if braco == 1 else edge2

            if is_edge:
                score = float(df.at[i, "score1"]) if braco == 1 else float(df.at[i, "score2"])
                p_here = float(df.at[i, "p1"]) if braco == 1 else float(df.at[i, "p2"])
                r1 = bool(df.at[i, "r1_bra1"]) if braco == 1 else bool(df.at[i, "r1_bra2"])
                r2 = bool(df.at[i, "r2_bra1"]) if braco == 1 else bool(df.at[i, "r2_bra2"])

                cluster.append({
                    "i": int(i), "timestamp": ts, "score": score,
                    "p": p_here, "r1": r1, "r2": r2,
                })
                last_cand_ts = ts

                dbg("EDGE_CAND", i=i, timestamp=ts, braco=braco, p=p_here, score=score)

            # Fecha cluster ao atingir gap
            if last_cand_ts is not None and (ts - last_cand_ts).total_seconds() > CLUSTER_GAP_S:
                # Melhor score; empate -> timestamp mais tarde
                best = sorted(cluster, key=lambda x: (x["score"], x["timestamp"]))[-1]

                running_start_i = int(best["i"])
                running_start_ts = best["timestamp"]

                dbg(
                    "RUNNING_START", corrida=corrida, braco=braco,
                    i=running_start_i, timestamp=running_start_ts, confirmed_at=ts,
                )
                events.append({
                    "corrida": corrida,
                    "status": "RUNNING",
                    "inicio_lc_sys": running_start_ts,
                    "final_lc_sys": None,
                    "braco": braco,
                    "event_ts": ts,
                })

                # Transita para RUNNING
                state = "RUNNING"
                crossed_60 = False
                p_ref = None
                t_ref = None

            continue

        # ======================================================================
        # ESTADO: RUNNING
        # Corrida em andamento. Detecção de fim:
        #   Fase 1: Aguarda peso cruzar END_ARM_WEIGHT (60t) para baixo.
        #   Fase 2: Sliding reference — se peso sobe END_RISE_TON (2t) em
        #           END_RISE_WINDOW_S (25s) → fim. Se estoura janela, atualiza
        #           a referência e reinicia contagem.
        # ======================================================================
        if state == "RUNNING":
            if pd.isna(p_run):
                continue

            # Fase 1: cruzou 60t?
            if (not crossed_60) and (p_run <= END_ARM_WEIGHT):
                crossed_60 = True
                p_ref = float(p_run)
                t_ref = ts
                dbg(
                    "CROSSED_60", corrida=corrida, braco=braco,
                    i=i, timestamp=ts, p=p_ref, armed_at=ts,
                )
                continue

            # Fase 2: monitorar subidas com sliding reference
            if crossed_60:
                old_p_ref = p_ref
                dt = (ts - t_ref).total_seconds()
                dp = float(p_run) - float(p_ref)

                dbg(
                    "END_EVAL", corrida=corrida, braco=braco,
                    i=i, timestamp=ts, p=float(p_run), p_ref=p_ref,
                    delta_p=dp, dt_s=dt,
                    within_window=(dt <= END_RISE_WINDOW_S),
                    enough_rise=(dp >= END_RISE_TON),
                )

                # Subiu o suficiente dentro da janela? -> FIM
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
                    dbg(
                        "RUN_CLOSE", corrida=corrida, braco=braco,
                        i=i, timestamp=end_ts, p=float(p_run),
                        delta_p=dp, dt_s=dt,
                    )

                    # Avança corrida e alterna braço
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

                # Passou janela sem subir: atualiza referência (sliding reference)
                if dt > END_RISE_WINDOW_S:
                    p_ref = float(p_run)
                    t_ref = ts
                    dbg(
                        "END_REF_UPDATE", corrida=corrida, braco=braco,
                        i=i, timestamp=ts, p=float(p_run),
                        p_ref_before=old_p_ref, p_ref_after=p_ref,
                        dt_s=(ts - t_ref).total_seconds(),
                    )

            continue

    # ==========================================================================
    # 2) RETORNOS — sempre com colunas fixas
    # ==========================================================================
    df_sys_lc = pd.DataFrame(
        results,
        columns=["corrida", "inicio_lc_sys", "final_lc_sys", "braco", "close_reason"],
    )

    df_sys_events = pd.DataFrame(
        events,
        columns=["corrida", "status", "inicio_lc_sys", "final_lc_sys", "braco", "event_ts"],
    )

    df_debug = pd.DataFrame(debug_rows)

    return df_sys_lc, df_sys_events, df_debug, df
