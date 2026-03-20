"""
fea_peso_carro_panela.py

Deteccao de ciclos do Peso Carro Panela do Forno Eletrico a Arco (FEA)
usando dados da tag PESO_CARRO_PANELA.

================================================================================
VISAO GERAL
================================================================================

Detecta automaticamente os ciclos de chegada, vazamento e saida da panela
no carro do FEA. Cada ciclo compreende:

    * Panela chegou:      peso sobe de ~0 para ~37t (panela vazia no carro)
    * Tara:               peso cai de ~37 para ~0 (balanca tarada)
    * Vazamento inicio:   peso comeca a subir de ~0 (aco sendo despejado)
    * Plato:              peso estabiliza em ~55-65t (vazamento concluido,
                          aguardando a ponte remover a panela)
    * Panela saiu:        peso despenca de ~60 para ~-37 (ponte removeu)
    * Tara reset:         peso retorna de ~-37 para ~0 (balanca re-tarada)

================================================================================
SINAL E ESCALA
================================================================================

O sinal PESO_CARRO_PANELA eh o peso medido na balanca do carro panela.
A balanca eh tarada com o peso da panela vazia (~37t), de modo que:

    -  0t:   balanca tarada, sem panela
    - ~37t:  panela vazia presente (antes da tara)
    - ~60t:  panela cheia de aco (apos vazamento)
    - ~-37t: apos remocao da panela (peso negativo porque foi tarada)

Valores tipicos (Fev/2026):
    - Peso max medio:     62.2 t
    - Espera panela:      29 min (mediana) -- tempo de aquecimento FEA
    - Tara a vazamento:   2.0 min
    - Duracao vazamento:  2.1 min
    - Espera ponte:       2.1 min (mediana)
    - Ciclo total:        40 min (mediana)
    - Amostragem:         1 segundo

================================================================================
ARQUITETURA - DETECCAO POR SEGMENTOS
================================================================================

Diferente da FSM streaming do fea_energia, este detector usa uma abordagem
baseada em segmentos:

    1) Suaviza o sinal com mediana movel (SMOOTH_WIN)
    2) Classifica cada ponto em zonas:
           NEG (<-20), ZERO (+-10), TRANS (10-25), PANELA (25-48), CHEIO (>48)
    3) Encontra segmentos sustentados na zona CHEIO (duracao > MIN_CHEIO_S)
    4) Para cada segmento CHEIO, faz lookback e lookforward para
       identificar os eventos do ciclo completo
    5) Detecta o inicio do plato dentro do segmento CHEIO
       (taxa de variacao estabiliza < PLATO_RATE_MAX)

Lookback (do inicio do CHEIO para tras):
    - Pula zona de subida (TRANS/PANELA durante vazamento)
    - Encontra zona ZERO (pos-tara, pre-vazamento) -> ts_vazamento_inicio
    - Encontra zona PANELA antes do ZERO -> ts_tara (transicao PANELA->ZERO)
    - Encontra inicio da zona PANELA -> ts_panela_chegou

Lookforward (do fim do CHEIO para frente):
    - Encontra zona NEG (panela removida)
    - Encontra zona ZERO apos NEG -> ts_tara_reset

================================================================================
METRICAS DETECTADAS POR CICLO
================================================================================

    Metrica                  Intervalo                Descricao
    -----------------------  -----------------------  ---------------------------
    dur_panela_espera_s      CHEGOU -> TARA           Panela no carro ate tara
    dur_tara_a_vaz_s         TARA -> VAZ_INICIO       Tara ate inicio vazamento
    dur_vazamento_s          VAZ_INICIO -> PLATO      Subida ate estabilizar
    plato_dur_s              PLATO -> PANELA_SAIU     Espera ponte (plato)
    dur_negativo_s           PANELA_SAIU -> RESET     Peso negativo ate re-tara

================================================================================
USO
================================================================================

    from src.detection.fea_peso_carro_panela import detect_peso_carro_panela

    df_sys, df_events, df_debug = detect_peso_carro_panela(
        df=df,  # timestamp, PESO_CARRO_PANELA
    )

    # df_sys:    resumo por ciclo (timestamps, duracoes, peso_max)
    # df_events: todos os eventos detectados
    # df_debug:  sinal suavizado + zona + taxa de variacao

================================================================================
VALIDACAO
================================================================================

Validado com dados de fevereiro/2026 (28 dias, 2.4M registros):
    - 468 ciclos detectados (brutos)
    - 329 ciclos padrao apos filtragem (fantasma + paradas + IQR)
    - 24 ciclos fantasma (peso_max < 55t)
    - 5 paradas (plato > 30 min)
    - Dia mais movimentado: 25/02 (28 ciclos)
    - Eventos alinham visualmente com o sinal

Parametros default calibrados com dados de fevereiro/2026.

================================================================================
"""

import pandas as pd
import numpy as np
from typing import Tuple


def detect_peso_carro_panela(
    df: pd.DataFrame,
    col_peso: str = "PESO_CARRO_PANELA",
    col_ts: str = "timestamp",

    # --- Suavizacao ---
    SMOOTH_WIN: int = 5,
    # Janela (amostras) para mediana movel.
    # 5 amostras = 5s com amostragem de 1s.
    # Suaviza ruido sem perder resolucao temporal.

    # --- Limiares de zona (toneladas) ---
    TH_NEGATIVO: float = -20.0,
    # Peso abaixo do qual consideramos zona negativa
    # (panela removida, balanca nao re-tarada).

    TH_ZERO: float = 10.0,
    # Peso dentro de +-TH_ZERO consideramos zona zero
    # (balanca tarada, sem carga ou em idle).

    TH_PANELA: float = 25.0,
    # Peso acima do qual consideramos panela presente
    # (panela vazia ~37t antes da tara).

    TH_CHEIO: float = 48.0,
    # Peso acima do qual consideramos panela cheia
    # (aco ja vazado, peso tipico 55-65t).

    # --- Duracao minima segmento CHEIO (segundos) ---
    MIN_CHEIO_S: float = 60.0,
    # Segmentos com peso > TH_CHEIO por menos de MIN_CHEIO_S
    # sao descartados (spikes, leituras parciais).
    # 60s filtra blips de 20-50s observados nos dados.

    # --- Deteccao de plato ---
    PLATO_RATE_MAX: float = 0.5,
    # Taxa maxima (tons/min) para considerar peso estavel.
    # Durante vazamento a taxa eh ~28 tons/min.
    # No plato a taxa eh ~0.

    PLATO_CONFIRM_S: float = 20.0,
    # Segundos minimos de estabilidade para confirmar plato.

    PLATO_WINDOW_S: int = 30,
    # Janela (s) para calculo da taxa de variacao.

) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Detecta ciclos do peso carro panela do FEA.

    Estrategia: encontra segmentos sustentados de peso CHEIO (>48t, >60s),
    depois faz lookback/lookforward para identificar eventos do ciclo
    completo.

    Args:
        df: DataFrame pandas com colunas:
            - timestamp: datetime
            - PESO_CARRO_PANELA: peso em toneladas

    Returns:
        df_sys:    ciclo_seq | ts_panela_chegou | ts_tara |
                   ts_vazamento_inicio | ts_plato_inicio |
                   ts_panela_saiu | ts_tara_reset | peso_max |
                   plato_dur_s | dur_cheio_total_s |
                   dur_panela_espera_s | dur_vazamento_s | dur_negativo_s
        df_events: ciclo_seq | event_type | event_ts
                   event_type in {PANELA_CHEGOU, TARA, VAZ_INICIO,
                                  PLATO_INICIO, PANELA_SAIU, TARA_RESET}
        df_debug:  sinal suavizado + zona + taxa de variacao
    """

    # =================================================================
    # 0) PREPARACAO
    # =================================================================
    d = df[[col_ts, col_peso]].dropna().sort_values(col_ts).reset_index(drop=True)
    n = len(d)
    if n == 0:
        return pd.DataFrame(), pd.DataFrame(), d

    ts_arr = d[col_ts].values   # datetime64
    w_arr = d[col_peso].values.astype(float)

    # Suavizar com mediana movel
    w_s = (
        pd.Series(w_arr)
        .rolling(SMOOTH_WIN, center=True, min_periods=1)
        .median()
        .values
    )

    # Taxa de variacao (tons/min)
    half = PLATO_WINDOW_S // 2
    rate = np.zeros(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n - 1, i + half)
        dt_s = (ts_arr[hi] - ts_arr[lo]) / np.timedelta64(1, "s")
        if dt_s > 0:
            rate[i] = (w_s[hi] - w_s[lo]) / dt_s * 60

    # =================================================================
    # 1) CLASSIFICAR ZONAS
    # =================================================================
    zones = np.full(n, "TRANS", dtype=object)
    for i in range(n):
        w = w_s[i]
        if w < TH_NEGATIVO:
            zones[i] = "NEG"
        elif abs(w) <= TH_ZERO:
            zones[i] = "ZERO"
        elif w >= TH_CHEIO:
            zones[i] = "CHEIO"
        elif w >= TH_PANELA:
            zones[i] = "PANELA"

    # =================================================================
    # 2) SEGMENTOS CHEIO SUSTENTADOS
    # =================================================================
    seg_cheio = []
    in_seg = False
    seg_start = 0
    for i in range(n):
        if zones[i] == "CHEIO" and not in_seg:
            seg_start = i
            in_seg = True
        elif zones[i] != "CHEIO" and in_seg:
            dur = (ts_arr[i - 1] - ts_arr[seg_start]) / np.timedelta64(1, "s")
            if dur >= MIN_CHEIO_S:
                seg_cheio.append((seg_start, i - 1))
            in_seg = False
    if in_seg:
        dur = (ts_arr[n - 1] - ts_arr[seg_start]) / np.timedelta64(1, "s")
        if dur >= MIN_CHEIO_S:
            seg_cheio.append((seg_start, n - 1))

    # =================================================================
    # 3) FUNCOES DE LOOKBACK / LOOKFORWARD
    # =================================================================
    def _lookback(cs, lb):
        """Do CHEIO start, busca: vazamento_inicio, tara, panela_chegou."""
        ts_vaz = ts_tara = ts_pan = None
        i = cs - 1

        # 1) Pula zona de subida (TRANS, PANELA durante vazamento)
        while i > lb and zones[i] in ("TRANS", "PANELA"):
            i -= 1

        # 2) Deve estar em ZERO (pos-tara, pre-vazamento)
        if i > lb and zones[i] == "ZERO":
            ts_vaz = pd.Timestamp(ts_arr[i + 1])

            # 3) Atravessa zona ZERO para achar borda
            while i > lb and zones[i] == "ZERO":
                i -= 1

            # 4) Se encontra PANELA ou TRANS alto: panela estava presente
            if i > lb and (
                zones[i] == "PANELA"
                or (zones[i] == "TRANS" and w_s[i] > 20)
            ):
                ts_tara = pd.Timestamp(ts_arr[i + 1])  # PANELA -> ZERO

                # 5) Atravessa PANELA para achar onde chegou
                while i > lb and zones[i] in ("PANELA", "TRANS") and w_s[i] > 15:
                    i -= 1
                if i >= lb and zones[i] in ("ZERO", "NEG"):
                    ts_pan = pd.Timestamp(ts_arr[min(i + 1, cs)])

        return ts_vaz, ts_tara, ts_pan

    def _lookforward(ce, lf):
        """Do CHEIO end, busca: tara_reset (retorno a ZERO apos NEG)."""
        ts_reset = None
        found_neg = False
        for i in range(ce + 1, lf):
            if zones[i] == "NEG":
                found_neg = True
            elif zones[i] == "ZERO" and found_neg:
                ts_reset = pd.Timestamp(ts_arr[i])
                break
        return ts_reset

    def _detect_plato(cs, ce):
        """Dentro do CHEIO, encontra inicio do plato (taxa estabiliza)."""
        for i in range(cs, ce + 1):
            if abs(rate[i]) < PLATO_RATE_MAX:
                j = i
                while j <= ce and abs(rate[j]) < PLATO_RATE_MAX:
                    j += 1
                dur_c = (
                    (ts_arr[min(j, ce)] - ts_arr[i])
                    / np.timedelta64(1, "s")
                )
                if dur_c >= PLATO_CONFIRM_S:
                    return pd.Timestamp(ts_arr[i])
        return pd.Timestamp(ts_arr[cs])

    # =================================================================
    # 4) MONTAR CICLOS
    # =================================================================
    ciclos = []
    events_list = []

    for seq_idx, (cs, ce) in enumerate(seg_cheio):
        seq = seq_idx + 1
        ts_cs = pd.Timestamp(ts_arr[cs])
        ts_ce = pd.Timestamp(ts_arr[ce])
        w_max = float(w_s[cs : ce + 1].max())

        # Limites de busca
        lb = seg_cheio[seq_idx - 1][1] if seq_idx > 0 else 0
        lf = (
            seg_cheio[seq_idx + 1][0]
            if seq_idx < len(seg_cheio) - 1
            else n - 1
        )

        # Eventos
        ts_vaz, ts_tara, ts_pan = _lookback(cs, lb)
        ts_plato = _detect_plato(cs, ce)
        ts_saiu = ts_ce
        ts_reset = _lookforward(ce, lf)

        plato_dur = (ts_ce - ts_plato).total_seconds()
        cheio_dur = (ts_ce - ts_cs).total_seconds()

        ciclo = {
            "ciclo_seq": seq,
            "ts_panela_chegou": ts_pan,
            "ts_tara": ts_tara,
            "ts_vazamento_inicio": ts_vaz,
            "ts_plato_inicio": ts_plato,
            "ts_panela_saiu": ts_saiu,
            "ts_tara_reset": ts_reset,
            "peso_max": w_max,
            "plato_dur_s": plato_dur,
            "dur_cheio_total_s": cheio_dur,
        }

        # Duracoes derivadas
        if ts_pan and ts_tara:
            ciclo["dur_panela_espera_s"] = (
                (ts_tara - ts_pan).total_seconds()
            )
        else:
            ciclo["dur_panela_espera_s"] = None

        if ts_vaz and ts_plato:
            ciclo["dur_vazamento_s"] = (
                (ts_plato - ts_vaz).total_seconds()
            )
        else:
            ciclo["dur_vazamento_s"] = None

        if ts_saiu and ts_reset:
            ciclo["dur_negativo_s"] = (
                (ts_reset - ts_saiu).total_seconds()
            )
        else:
            ciclo["dur_negativo_s"] = None

        ciclos.append(ciclo)

        # Registrar eventos
        for etype, ets in [
            ("PANELA_CHEGOU", ts_pan),
            ("TARA", ts_tara),
            ("VAZ_INICIO", ts_vaz),
            ("PLATO_INICIO", ts_plato),
            ("PANELA_SAIU", ts_saiu),
            ("TARA_RESET", ts_reset),
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
    df_debug["peso_smooth"] = w_s
    df_debug["rate_ton_min"] = rate
    df_debug["zone"] = zones

    return df_sys, df_events, df_debug
