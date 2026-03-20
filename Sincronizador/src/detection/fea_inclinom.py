"""
fea_inclinom.py

Deteccao de ciclos de inclinacao do Forno Eletrico a Arco (FEA)
usando dados do inclinometro (ACI@FEA_GERAL_INCLINOM).

================================================================================
VISAO GERAL
================================================================================

Detecta automaticamente os ciclos de inclinacao (vazamento) do FEA.
Cada ciclo compreende:

    * Disturbio inicio:  angulo sai da estabilidade (~0) e comeca a variar
                         (pequenas oscilacoes entre 1.5 e 5 graus)
    * Inclinando:        angulo passa de 5 graus - comecou a inclinar para vazar
    * Pico maximo:       angulo atinge o maximo (~8-12 graus) e estabiliza
                         brevemente (plato no topo)
    * Queda:             angulo cai rapidamente do pico
    * Negativo:          angulo vai para valores negativos (~-5 a -7 graus)
    * Retorno:           angulo volta para ~0 (forno endireita)

================================================================================
SINAL E ESCALA
================================================================================

O sinal ACI@FEA_GERAL_INCLINOM mede o angulo de inclinacao do forno em graus.

    -  0 graus:   forno na posicao vertical (estavel)
    - ~8-12 graus: forno inclinado para vazamento (pico)
    - ~-5 a -7 graus: forno retornando (overshoot negativo)

Valores tipicos (Fev/2026):
    - Angulo pico medio:    ~8.6 graus
    - Duracao disturbio:    ~1-2 min (disturbio ate >5 graus)
    - Duracao pico:         ~80-100 s (acima de 5 graus)
    - Duracao negativo:     ~100-180 s
    - Amostragem:           1 segundo

================================================================================
ARQUITETURA - DETECCAO POR SEGMENTOS
================================================================================

Abordagem baseada em segmentos (mesma filosofia do peso carro panela):

    1) Suaviza o sinal com mediana movel (SMOOTH_WIN)
    2) Classifica cada ponto em zonas:
           NEG (<-3), IDLE (+-1.5), DISTURBIO (1.5-5), INCLINANDO (5-8),
           PICO (>8)
    3) Encontra segmentos sustentados acima de TH_INCLINANDO
       (duracao > MIN_INCL_S)
    4) Para cada segmento, faz lookback e lookforward para
       identificar os eventos do ciclo completo

Lookback (do inicio do segmento INCLINANDO para tras):
    - Define janela temporal MAX_LOOKBACK_WINDOW_S (default 1800s = 30 min)
    - Dentro dessa janela, busca a PRIMEIRA perturbacao sustentada
      (DISTURBIO com duracao >= MIN_DIST_SUSTAIN_S = 3s)
    - Essa abordagem captura perturbacoes pre-operacionais intermitentes
      separadas por gaps IDLE de qualquer duracao
    - Ignora picos isolados de < 3s (ruido)

Lookforward (do fim do segmento INCLINANDO para frente):
    - Encontra zona NEG (queda/negativo)
    - Encontra retorno a IDLE (volta a ~0)

================================================================================
METRICAS DETECTADAS POR CICLO
================================================================================

    Metrica                  Intervalo                    Descricao
    -----------------------  ---------------------------  ---------------------------
    dur_disturbio_s          DISTURBIO -> INCLINANDO      Oscilacao antes de inclinar
    dur_inclinado_s          INCLINANDO -> QUEDA          Tempo total acima de 5 graus
    dur_pico_s               PICO_MAX -> QUEDA            Tempo no plato do pico
    dur_negativo_s           NEGATIVO -> RETORNO          Tempo em angulo negativo
    dur_ciclo_total_s        DISTURBIO -> RETORNO         Ciclo completo

================================================================================
USO
================================================================================

    from src.detection.fea_inclinom import detect_fea_inclinom

    df_sys, df_events, df_debug = detect_fea_inclinom(
        df=df,  # timestamp, ACI@FEA_GERAL_INCLINOM
    )

================================================================================
VALIDACAO
================================================================================

Validado com dados de Fev/2026 (28 dias, 2.4M registros):
    - 455 ciclos detectados
    - Angulo pico medio: ~12.1 graus
    - Duracao pico (>5 graus): ~80-100s
    - Duracao negativo: ~100-180s
    - Queda do pico ao negativo: ~1-2s (muito rapida)
    - Perturbacoes pre-inclinacao detectadas dentro de janela de 30 min

Parametros default calibrados com dados de fevereiro/2026.

================================================================================
"""

import pandas as pd
import numpy as np
from typing import Tuple


def detect_fea_inclinom(
    df: pd.DataFrame,
    col_inc: str = "ACI@FEA_GERAL_INCLINOM",
    col_ts: str = "timestamp",

    # --- Suavizacao ---
    SMOOTH_WIN: int = 5,
    # Janela (amostras) para mediana movel.
    # 5 amostras = 5s com amostragem de 1s.

    # --- Limiares de zona (graus) ---
    TH_NEGATIVO: float = -3.0,
    # Angulo abaixo do qual consideramos zona negativa
    # (forno retornando com overshoot).

    TH_IDLE: float = 1.5,
    # Angulo dentro de +-TH_IDLE consideramos zona idle
    # (forno estavel, sem inclinacao significativa).

    TH_INCLINANDO: float = 5.0,
    # Angulo acima do qual consideramos que o forno
    # comecou a inclinar para vazar. Marca o inicio
    # da fase de vazamento efetivo.

    TH_PICO: float = 8.0,
    # Angulo acima do qual consideramos zona de pico
    # (inclinacao maxima, forno no topo do vazamento).

    # --- Duracao minima segmento INCLINANDO (segundos) ---
    MIN_INCL_S: float = 30.0,
    # Segmentos com angulo > TH_INCLINANDO por menos de MIN_INCL_S
    # sao descartados (ruido, micro-inclinacoes).

    # --- Calculo de taxa ---
    RATE_WINDOW_S: int = 15,
    # Janela (s) para calculo da taxa de variacao.

    # --- Lookback: janela temporal maxima (segundos) ---
    MAX_LOOKBACK_WINDOW_S: float = 1800.0,
    # Janela temporal maxima para buscar perturbacoes que
    # precedem o INCLINANDO. Default: 1800s = 30 min.
    # Dentro desta janela, o lookback procura a PRIMEIRA
    # perturbacao sustentada (DISTURBIO >= MIN_DIST_SUSTAIN_S).
    # Perturbacoes pre-operacionais podem ter gaps IDLE de
    # varios minutos entre pulsos (ex: 5-12 min), por isso
    # a busca e feita por janela temporal e nao por gap.

    # --- Lookback: duracao minima de perturbacao (segundos) ---
    MIN_DIST_SUSTAIN_S: float = 3.0,
    # Duracao minima de DISTURBIO consecutivo para considerar
    # como perturbacao real (nao ruido). Default: 3s.
    # Picos isolados menores que isto sao ignorados.

) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Detecta ciclos de inclinacao do FEA.

    Estrategia: encontra segmentos sustentados acima de TH_INCLINANDO
    (>5 graus, >30s), depois faz lookback/lookforward para identificar
    eventos do ciclo completo.

    Args:
        df: DataFrame pandas com colunas:
            - timestamp: datetime
            - ACI@FEA_GERAL_INCLINOM: angulo em graus

    Returns:
        df_sys:    resumo por ciclo (timestamps, duracoes, angulo_max)
        df_events: todos os eventos detectados
        df_debug:  sinal suavizado + zona + taxa de variacao
    """

    # =================================================================
    # 0) PREPARACAO
    # =================================================================
    d = df[[col_ts, col_inc]].dropna().sort_values(col_ts).reset_index(drop=True)
    n = len(d)
    if n == 0:
        return pd.DataFrame(), pd.DataFrame(), d

    ts_arr = d[col_ts].values
    a_arr = d[col_inc].values.astype(float)

    # Suavizar com mediana movel
    a_s = (
        pd.Series(a_arr)
        .rolling(SMOOTH_WIN, center=True, min_periods=1)
        .median()
        .values
    )

    # Taxa de variacao (graus/min)
    half = RATE_WINDOW_S // 2
    rate = np.zeros(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n - 1, i + half)
        dt_s = (ts_arr[hi] - ts_arr[lo]) / np.timedelta64(1, "s")
        if dt_s > 0:
            rate[i] = (a_s[hi] - a_s[lo]) / dt_s * 60

    # =================================================================
    # 1) CLASSIFICAR ZONAS
    # =================================================================
    zones = np.full(n, "DISTURBIO", dtype=object)
    for i in range(n):
        a = a_s[i]
        if a < TH_NEGATIVO:
            zones[i] = "NEG"
        elif abs(a) <= TH_IDLE:
            zones[i] = "IDLE"
        elif a >= TH_PICO:
            zones[i] = "PICO"
        elif a >= TH_INCLINANDO:
            zones[i] = "INCLINANDO"

    # =================================================================
    # 2) SEGMENTOS INCLINANDO+ SUSTENTADOS (>= TH_INCLINANDO)
    # =================================================================
    seg_incl = []
    in_seg = False
    seg_start = 0
    for i in range(n):
        above = a_s[i] >= TH_INCLINANDO
        if above and not in_seg:
            seg_start = i
            in_seg = True
        elif not above and in_seg:
            dur = (ts_arr[i - 1] - ts_arr[seg_start]) / np.timedelta64(1, "s")
            if dur >= MIN_INCL_S:
                seg_incl.append((seg_start, i - 1))
            in_seg = False
    if in_seg:
        dur = (ts_arr[n - 1] - ts_arr[seg_start]) / np.timedelta64(1, "s")
        if dur >= MIN_INCL_S:
            seg_incl.append((seg_start, n - 1))

    # =================================================================
    # 3) FUNCOES DE LOOKBACK / LOOKFORWARD
    # =================================================================
    def _lookback(cs, lb):
        """Busca ts_disturbio_inicio dentro de janela temporal.

        Varre a janela de MAX_LOOKBACK_WINDOW_S antes do INCLINANDO
        start (cs) e encontra a PRIMEIRA perturbacao sustentada
        (DISTURBIO por >= MIN_DIST_SUSTAIN_S consecutivos).

        Isso captura perturbacoes pre-operacionais separadas por
        gaps IDLE de qualquer duracao (5s, 5min, 12min...).
        """
        ts_cs = ts_arr[cs]

        # Calcula limite inferior efetivo (lb do ciclo anterior
        # OU limite da janela temporal, o que for mais recente)
        effective_lb = lb
        for j in range(cs - 1, lb - 1, -1):
            dt = (ts_cs - ts_arr[j]) / np.timedelta64(1, "s")
            if dt > MAX_LOOKBACK_WINDOW_S:
                effective_lb = j + 1
                break

        # Varre de effective_lb ate cs, procurando a PRIMEIRA
        # sequencia de DISTURBIO sustentada (>= MIN_DIST_SUSTAIN_S)
        first_dist_start = None
        dist_run_start = None

        for j in range(effective_lb, cs):
            if zones[j] == "DISTURBIO":
                if dist_run_start is None:
                    dist_run_start = j
                # Verifica se ja atingiu duracao minima
                run_dur = (
                    (ts_arr[j] - ts_arr[dist_run_start])
                    / np.timedelta64(1, "s")
                )
                if run_dur >= MIN_DIST_SUSTAIN_S:
                    first_dist_start = dist_run_start
                    break  # Encontrou: retorna a mais antiga
            else:
                dist_run_start = None  # Reset: saiu de DISTURBIO

        if first_dist_start is not None:
            return pd.Timestamp(ts_arr[first_dist_start])

        # Fallback: se nao encontrou sustentada, tenta qualquer
        # DISTURBIO (mesmo curto) -- melhor que nada
        for j in range(effective_lb, cs):
            if zones[j] == "DISTURBIO":
                return pd.Timestamp(ts_arr[j])

        return None

    def _lookforward(ce, lf):
        """Do INCLINANDO end, busca: queda, negativo, retorno."""
        ts_queda = pd.Timestamp(ts_arr[ce])
        ts_negativo = None
        ts_retorno = None

        found_neg = False
        for i in range(ce + 1, lf):
            if zones[i] == "NEG" and not found_neg:
                ts_negativo = pd.Timestamp(ts_arr[i])
                found_neg = True
            elif zones[i] == "IDLE" and found_neg:
                ts_retorno = pd.Timestamp(ts_arr[i])
                break

        return ts_queda, ts_negativo, ts_retorno

    # =================================================================
    # 4) MONTAR CICLOS
    # =================================================================
    ciclos = []
    events_list = []

    for seq_idx, (cs, ce) in enumerate(seg_incl):
        seq = seq_idx + 1
        ts_cs = pd.Timestamp(ts_arr[cs])
        ts_ce = pd.Timestamp(ts_arr[ce])
        angulo_max = float(a_s[cs : ce + 1].max())
        incl_dur = (ts_ce - ts_cs).total_seconds()

        # Indice do pico maximo dentro do segmento
        pico_idx = cs + int(np.argmax(a_s[cs : ce + 1]))
        ts_pico_max = pd.Timestamp(ts_arr[pico_idx])

        # Limites de busca
        lb = seg_incl[seq_idx - 1][1] if seq_idx > 0 else 0
        lf = (
            seg_incl[seq_idx + 1][0]
            if seq_idx < len(seg_incl) - 1
            else n - 1
        )

        # Eventos
        ts_dist = _lookback(cs, lb)
        ts_queda, ts_negativo, ts_retorno = _lookforward(ce, lf)

        # Duracoes
        ciclo = {
            "ciclo_seq": seq,
            "ts_disturbio_inicio": ts_dist,
            "ts_inclinando": ts_cs,
            "ts_pico_max": ts_pico_max,
            "ts_queda": ts_queda,
            "ts_negativo": ts_negativo,
            "ts_retorno": ts_retorno,
            "angulo_max": angulo_max,
            "dur_inclinado_s": incl_dur,
        }

        # Derivadas
        if ts_dist and ts_cs:
            ciclo["dur_disturbio_s"] = (ts_cs - ts_dist).total_seconds()
        else:
            ciclo["dur_disturbio_s"] = None

        ciclo["dur_pico_s"] = (
            (ts_queda - ts_pico_max).total_seconds()
            if ts_pico_max and ts_queda
            else None
        )

        if ts_negativo and ts_retorno:
            ciclo["dur_negativo_s"] = (
                (ts_retorno - ts_negativo).total_seconds()
            )
        else:
            ciclo["dur_negativo_s"] = None

        if ts_dist and ts_retorno:
            ciclo["dur_ciclo_total_s"] = (
                (ts_retorno - ts_dist).total_seconds()
            )
        else:
            ciclo["dur_ciclo_total_s"] = None

        ciclos.append(ciclo)

        # Registrar eventos
        for etype, ets in [
            ("DISTURBIO_INICIO", ts_dist),
            ("INCLINANDO", ts_cs),
            ("PICO_MAX", ts_pico_max),
            ("QUEDA", ts_queda),
            ("NEGATIVO", ts_negativo),
            ("RETORNO", ts_retorno),
        ]:
            if ets is not None:
                events_list.append(
                    {
                        "ciclo_seq": seq,
                        "event_type": etype,
                        "event_ts": ets,
                    }
                )

    # =================================================================
    # 5) MONTA DATAFRAMES DE SAIDA
    # =================================================================
    df_sys = pd.DataFrame(ciclos)
    df_events = pd.DataFrame(events_list)

    df_debug = d.copy()
    df_debug["inc_smooth"] = a_s
    df_debug["rate_deg_min"] = rate
    df_debug["zone"] = zones

    return df_sys, df_events, df_debug
