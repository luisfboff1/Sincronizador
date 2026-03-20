"""
fea_energia.py

Deteccao de ciclos de aquecimento do Forno Eletrico a Arco (FEA)
usando dados de energia acumulada (ACI@FEA_ELET_ENERGIA).

================================================================================
VISAO GERAL
================================================================================

Detecta automaticamente:
    * Inicio do aquecimento: momento em que a energia comeca a subir
      (taxa de variacao sustentada acima de RATE_RISE_T por RISE_CONFIRM_S)
    * Pausas intermediarias: momentos em que os arcos sao desligados
      brevemente (taxa cai para ~0 por FLAT_CONFIRM_S no meio do ciclo)
    * Retomadas: momento em que os arcos religam apos uma pausa
    * Fim do ciclo: momento em que a energia cai abruptamente para ~0
      (reset do contador no vazamento)

O sinal de energia eh um contador acumulativo (kWh) que:
    - Parte de ~0 no inicio de cada corrida
    - Sobe em rampa (~12 unidades/s) quando os arcos estao ligados
    - Fica lateral (rate ~0) quando os arcos sao desligados (pausas)
    - Reseta para 0 no momento do vazamento (fim da corrida)

Processa linha por linha (streaming-ready), sem lookahead.

================================================================================
ARQUITETURA DA FSM
================================================================================

    +------+   rate > RISE_T    +---------+   rate ~ 0     +--------+
    | IDLE | ---- (E > END_T) -> | HEATING | ------------> | PAUSED |
    +------+   (RISE_CONFIRM_S)  +---------+  (FLAT_CONFIRM) +--------+
        ^                            |    ^                     |
        |                            |    |   rate > RISE_T     |
        |                            |    +---------------------+
        |        E < END_T           |        (RISE_CONFIRM_S)
        +----------------------------+
               (FIM do ciclo)

Estados:
    * IDLE:     Energia baixa ou flat, aguardando inicio de aquecimento
    * HEATING:  Arcos ligados, energia subindo (rate > RATE_RISE_T)
    * PAUSED:   Arcos desligados temporariamente (rate ~ 0), energia lateral

Transicoes:
    * IDLE -> HEATING:   rate > RATE_RISE_T E energia > END_T
                         sustentado por RISE_CONFIRM_S segundos
    * HEATING -> PAUSED: |rate| < RATE_FLAT_T sustentado por FLAT_CONFIRM_S
    * PAUSED -> HEATING: rate > RATE_RISE_T sustentado por RISE_CONFIRM_S
    * HEATING -> IDLE:   energia cai abaixo de END_T (reset = vazamento)
    * PAUSED -> IDLE:    energia cai abaixo de END_T, ou pausa > MAX_PAUSE_S

================================================================================
SINAL E ESCALA
================================================================================

O sinal ACI@FEA_ELET_ENERGIA eh usado diretamente (sem divisao por escala).
Valores tipicos:
    - Idle/reset:  0
    - Pico normal: 25.000 - 30.000
    - Taxa rampa:  ~12 unidades/s (quando arcos ligados)
    - Pausas:      1 a 10 minutos, tipicamente 2-5 por corrida
    - Ciclo total: ~45 a 60 minutos

================================================================================
FILTRAGEM DE PAUSAS
================================================================================

Pausas curtas (< MIN_PAUSE_S, default 60s) sao descartadas.
Isso filtra:
    - Estabilizacao natural da energia perto do pico antes do vazamento
    - Breves hesitacoes do sinal que nao representam paradas reais dos arcos

Eventos sao reconstruidos no momento de fechar cada ciclo:
    - Somente pausas que passaram no filtro MIN_PAUSE_S geram eventos
    - Toda PAUSA_INI tem um PAUSA_FIM correspondente (sem orfas)
    - Ciclos invalidos (muito curtos ou pouca energia) nao geram eventos

================================================================================
USO
================================================================================

    from src.detection.fea_energia import detect_fea_energia

    df_sys, df_events, df_debug = detect_fea_energia(
        df=df,  # timestamp, ACI@FEA_ELET_ENERGIA
    )

    # df_sys: resumo de cada ciclo (inicio, fim, duracao, pausas)
    # df_events: todos os eventos (INICIO, PAUSA_INI, PAUSA_FIM, FIM)
    # df_debug: trace da FSM (somente se DEBUG=True)

================================================================================
VALIDACAO
================================================================================

Validado com dados de 04/02/2026 (dia completo, 86.401 registros):
    - 17 ciclos detectados
    - Duracao media: ~50 min (range 28-72 min)
    - 2-5 pausas por ciclo tipicamente
    - FIM coincide com HRVAZAMENTO do MES (+-4 min)
    - Corrida 131261 (TTT=236min): detectada como 2 ciclos
      (1a metade ate hold, 2a metade apos retomada)

Parametros default calibrados com dados de fevereiro/2026.

================================================================================
"""

import pandas as pd
import numpy as np
from typing import Tuple


def detect_fea_energia(
    df: pd.DataFrame,
    col_energia: str = "ACI@FEA_ELET_ENERGIA",

    # --- Rolling rate window ---
    RATE_WINDOW_S: int = 30,
    # Janela (s) para calculo da taxa de variacao suavizada.
    # O rate eh calculado como (E[i] - E[j]) / (t[i] - t[j])
    # onde j eh o indice mais antigo dentro da janela.
    # 30s suaviza ruido sem perder resolucao temporal.

    # --- Thresholds de taxa (unidades/s) ---
    RATE_RISE_T: float = 3.0,
    # Taxa acima da qual consideramos aquecimento ativo.
    # Rampas tipicas sao ~12 un/s, entao 3.0 eh conservador.

    RATE_FLAT_T: float = 1.0,
    # Taxa abaixo da qual consideramos "flat" (pausa ou idle).
    # Durante pausas reais a taxa eh exatamente 0.

    # --- Confirmacao (s) ---
    RISE_CONFIRM_S: float = 15.0,
    # Tempo minimo (s) com rate > RATE_RISE_T para confirmar
    # inicio de aquecimento.

    FLAT_CONFIRM_S: float = 30.0,
    # Tempo minimo (s) com |rate| < RATE_FLAT_T para confirmar
    # uma pausa.

    # --- Thresholds de energia ---
    END_T: float = 500.0,
    # Energia abaixo da qual consideramos que houve reset
    # (fim do ciclo / vazamento).
    # Tambem usado como filtro no IDLE: so inicia deteccao
    # quando E > END_T (evita falsos inicios na rampa 0->500).

    MIN_HEATING_ENERGY: float = 5000.0,
    # Energia minima acumulada (delta entre inicio e pico)
    # para considerar o ciclo valido.

    # --- Filtros ---
    MIN_CYCLE_S: float = 600.0,
    # Duracao minima (s) de um ciclo valido. 600s = 10 min.

    MAX_PAUSE_S: float = 1800.0,
    # Pausa maxima (s) antes de considerar que o ciclo
    # acabou por timeout. 1800s = 30 min.

    MIN_GAP_S: float = 60.0,
    # Gap minimo (s) entre fim de um ciclo e inicio do proximo.

    MIN_PAUSE_S: float = 60.0,
    # Duracao minima (s) de uma pausa para ser considerada real.
    # Pausas < 60s sao tipicamente estabilizacao perto do pico,
    # nao paradas reais dos arcos. Elimina micro-pausas e
    # garante que toda PAUSA_INI tenha PAUSA_FIM (sem orfas).

    # --- Debug ---
    DEBUG: bool = False,
    # Se True, retorna df_debug com trace completo da FSM.

) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Detecta ciclos de energia do FEA: inicio, pausas intermediarias, e fim.

    Streaming-ready: processa linha por linha com contadores de
    confirmacao temporal (sem lookahead).

    Args:
        df: DataFrame pandas com colunas:
            - timestamp: datetime
            - ACI@FEA_ELET_ENERGIA: energia acumulada (bruta)

    Returns:
        df_sys:    ciclo_seq | inicio_ts | fim_ts | energia_inicio |
                   energia_pico | energia_delta | duracao_s |
                   n_pausas | pausas_total_s
        df_events: ciclo_seq | event_type | event_ts | energia
                   event_type in {INICIO, PAUSA_INI, PAUSA_FIM, FIM}
        df_debug:  trace da FSM (se DEBUG=True), vazio caso contrario
    """

    # =====================================================================
    # 0) PREPARACAO
    # =====================================================================
    df = df[["timestamp", col_energia]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df[col_energia] = pd.to_numeric(df[col_energia], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    energia = df[col_energia].values
    timestamps = df["timestamp"].values
    n = len(df)

    # Pre-calcula rolling rate (delta energia / delta tempo em janela)
    rate = np.zeros(n, dtype=float)
    for i in range(1, n):
        j = i - 1
        while j > 0:
            dt_s = (
                (timestamps[i] - timestamps[j]) / np.timedelta64(1, 's')
            )
            if dt_s >= RATE_WINDOW_S:
                break
            j -= 1
        dt_s = (
            (timestamps[i] - timestamps[j]) / np.timedelta64(1, 's')
        )
        if dt_s > 0:
            rate[i] = (energia[i] - energia[j]) / dt_s

    # =====================================================================
    # 1) FSM
    # =====================================================================
    state = "IDLE"
    ciclo_seq = 0
    inicio_ts = None
    energia_inicio = 0.0
    energia_pico = 0.0
    pausa_inicio_ts = None
    last_fim_ts = None

    # Contadores de confirmacao temporal
    rise_count_s = 0.0
    flat_count_s = 0.0

    # Pausas do ciclo atual (somente as que passam MIN_PAUSE_S)
    current_pausas = []

    # Resultados
    results = []
    all_events = []
    debug_rows = []

    def dbg(event, **kw):
        if DEBUG:
            debug_rows.append({"event": event, "state": state, **kw})

    def _add_pause(ini_ts, fim_ts):
        """Adiciona pausa se duracao >= MIN_PAUSE_S, senao descarta."""
        dur = (fim_ts - ini_ts).total_seconds()
        if dur >= MIN_PAUSE_S:
            current_pausas.append({
                "pausa_ini_ts": ini_ts,
                "pausa_fim_ts": fim_ts,
                "duracao_s": dur,
            })
            return True
        else:
            dbg("PAUSA_CURTA_DESCARTADA", dur=round(dur, 1))
            return False

    def _emit_cycle(fim_ts, energia_fim):
        """Finaliza ciclo. Se valido, grava em results + events."""
        nonlocal state, inicio_ts, energia_inicio, energia_pico
        nonlocal current_pausas, last_fim_ts

        duracao = (
            (fim_ts - inicio_ts).total_seconds() if inicio_ts else 0
        )
        delta_e = energia_pico - energia_inicio
        pausas_total = sum(
            p.get("duracao_s", 0) for p in current_pausas
        )

        valid = (
            duracao >= MIN_CYCLE_S and delta_e >= MIN_HEATING_ENERGY
        )

        if valid:
            # Reconstroi eventos limpos a partir de current_pausas
            clean_events = []
            # INICIO
            clean_events.append({
                "ciclo_seq": ciclo_seq,
                "event_type": "INICIO",
                "event_ts": inicio_ts,
                "energia": round(energia_inicio, 0),
            })
            # PAUSAS (somente as validadas por MIN_PAUSE_S)
            for p in current_pausas:
                clean_events.append({
                    "ciclo_seq": ciclo_seq,
                    "event_type": "PAUSA_INI",
                    "event_ts": p["pausa_ini_ts"],
                    "energia": 0,
                })
                clean_events.append({
                    "ciclo_seq": ciclo_seq,
                    "event_type": "PAUSA_FIM",
                    "event_ts": p["pausa_fim_ts"],
                    "energia": 0,
                })
            # FIM
            clean_events.append({
                "ciclo_seq": ciclo_seq,
                "event_type": "FIM",
                "event_ts": fim_ts,
                "energia": round(energia_fim, 0),
            })

            results.append({
                "ciclo_seq": ciclo_seq,
                "inicio_ts": inicio_ts,
                "fim_ts": fim_ts,
                "energia_inicio": round(energia_inicio, 0),
                "energia_pico": round(energia_pico, 0),
                "energia_delta": round(delta_e, 0),
                "duracao_s": round(duracao, 1),
                "n_pausas": len(current_pausas),
                "pausas_total_s": round(pausas_total, 1),
            })
            all_events.extend(clean_events)
            last_fim_ts = fim_ts
            dbg("FIM_VALIDO", ciclo_seq=ciclo_seq,
                dur=round(duracao, 1), delta_e=round(delta_e, 0),
                n_pausas=len(current_pausas))
        else:
            dbg("DESCARTADO", ciclo_seq=ciclo_seq,
                dur=round(duracao, 1), delta_e=round(delta_e, 0))

        # Reset
        state = "IDLE"
        inicio_ts = None
        energia_inicio = 0.0
        energia_pico = 0.0
        current_pausas = []
        rise_count_s = 0.0

    # =====================================================================
    # 2) LOOP PRINCIPAL
    # =====================================================================
    prev_ts = None

    for i in range(n):
        ts = pd.Timestamp(timestamps[i])
        e = float(energia[i])
        r = float(rate[i])

        # Delta tempo desde amostra anterior
        dt = 0.0
        if prev_ts is not None:
            dt = (ts - prev_ts).total_seconds()
        prev_ts = ts

        # Track pico de energia
        if state in ("HEATING", "PAUSED") and e > energia_pico:
            energia_pico = e

        # ==============================================================
        # IDLE: aguardando inicio de aquecimento
        # ==============================================================
        if state == "IDLE":
            # So considera subida quando energia ja esta acima de END_T
            if r > RATE_RISE_T and e > END_T:
                rise_count_s += dt
                if rise_count_s >= RISE_CONFIRM_S:
                    # Verifica MIN_GAP_S desde ultimo fim
                    if last_fim_ts is not None:
                        gap = (ts - last_fim_ts).total_seconds()
                        if gap < MIN_GAP_S:
                            rise_count_s = 0.0
                            continue

                    ciclo_seq += 1
                    inicio_ts = ts - pd.Timedelta(
                        seconds=rise_count_s
                    )
                    energia_inicio = e - r * rise_count_s
                    energia_pico = e

                    dbg("INICIO", ciclo_seq=ciclo_seq,
                        ts=str(inicio_ts), e=round(e, 0),
                        rate=round(r, 1))

                    state = "HEATING"
                    rise_count_s = 0.0
                    flat_count_s = 0.0
            else:
                rise_count_s = 0.0
            continue

        # ==============================================================
        # HEATING: energia subindo, arcos ligados
        # ==============================================================
        if state == "HEATING":
            # Fim: energia caiu para ~0 (reset)
            if e < END_T:
                _emit_cycle(ts, e)
                continue

            # Pausa: rate caiu para ~0
            if abs(r) < RATE_FLAT_T:
                flat_count_s += dt
                if flat_count_s >= FLAT_CONFIRM_S:
                    pausa_inicio_ts = ts - pd.Timedelta(
                        seconds=flat_count_s
                    )
                    dbg("PAUSA_INI", ciclo_seq=ciclo_seq,
                        ts=str(pausa_inicio_ts), e=round(e, 0))
                    state = "PAUSED"
                    flat_count_s = 0.0
                    rise_count_s = 0.0
            else:
                flat_count_s = 0.0
            continue

        # ==============================================================
        # PAUSED: energia lateral, arcos desligados temporariamente
        # ==============================================================
        if state == "PAUSED":
            # Fim: energia caiu para ~0 (reset)
            if e < END_T:
                if pausa_inicio_ts is not None:
                    _add_pause(pausa_inicio_ts, ts)
                _emit_cycle(ts, e)
                continue

            # Retomada: rate voltou a subir
            if r > RATE_RISE_T:
                rise_count_s += dt
                if rise_count_s >= RISE_CONFIRM_S:
                    retomada_ts = ts - pd.Timedelta(
                        seconds=rise_count_s
                    )
                    if pausa_inicio_ts is not None:
                        _add_pause(pausa_inicio_ts, retomada_ts)
                        dbg("PAUSA_FIM", ciclo_seq=ciclo_seq,
                            ts=str(retomada_ts))
                    pausa_inicio_ts = None
                    state = "HEATING"
                    rise_count_s = 0.0
                    flat_count_s = 0.0
            else:
                rise_count_s = 0.0

            # Timeout de pausa
            if pausa_inicio_ts is not None:
                pausa_dur = (ts - pausa_inicio_ts).total_seconds()
                if pausa_dur > MAX_PAUSE_S:
                    _add_pause(pausa_inicio_ts, ts)
                    dbg("PAUSA_TIMEOUT", ciclo_seq=ciclo_seq,
                        dur=round(pausa_dur, 1))
                    _emit_cycle(ts, e)
            continue

    # =====================================================================
    # 3) MONTA DATAFRAMES DE SAIDA
    # =====================================================================
    df_sys = pd.DataFrame(results)
    df_events = pd.DataFrame(all_events)
    df_debug = (
        pd.DataFrame(debug_rows) if debug_rows else pd.DataFrame()
    )

    return df_sys, df_events, df_debug
