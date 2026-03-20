"""
validation.py

Plot de validacao de deteccao: sinal + eventos sobrepostos.
Substitui o codigo repetido nas celulas de validacao (FEA energia,
peso carro panela, inclinometro, etc.).

Uso:
    from src.viz.validation import plot_deteccao

    plot_deteccao(
        pdf=pdf_dia,
        df_events=ev_dia,
        col_ts="timestamp",
        col_signal="PESO_CARRO_PANELA",
        event_styles=EVENT_STYLES_PCP,
        titulo="Peso Carro Panela (14/02/2026)",
        hora_ini="15:30",
        hora_fim="17:10",
    )
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from typing import Dict, List, Optional, Tuple

from src.viz.theme import (
    EVCOMX_ORANGE, EVCOMX_GRAY,
    style_ax, style_title, style_xlabel, style_ylabel,
)


def plot_deteccao(
    pdf: pd.DataFrame,
    df_events: pd.DataFrame,
    col_ts: str = "timestamp",
    col_signal: str = "signal",
    event_styles: Optional[Dict[str, dict]] = None,
    titulo: str = "",
    hora_ini: str = "15:30",
    hora_fim: str = "17:10",
    dia: object = None,
    ref_lines: Optional[List[dict]] = None,
    shade_events: Optional[List[Tuple[str, str]]] = None,
    figsize: tuple = (22, 6),
    signal_label: str = "Sinal",
    tick_interval_min: int = 5,
) -> None:
    """Plota sinal + eventos detectados (linhas verticais) para validacao.

    Funcao generica que substitui os plots de validacao repetidos em
    cada celula de cada detector (FEA energia, peso carro panela,
    inclinometro, etc.).

    Args:
        pdf: DataFrame pandas com timestamps e sinal.
             Sera filtrado por dia e hora_ini/hora_fim.
        df_events: DataFrame com colunas:
             - event_ts (datetime): timestamp do evento
             - event_type (str): tipo do evento
             - ciclo_seq (int): numero do ciclo
        col_ts: nome da coluna de timestamp em pdf
        col_signal: nome da coluna do sinal em pdf
        event_styles: dict {event_type: {color, ls, lw, tag}}
             Cada event_type mapeia para estilo de linha vertical.
             Se None, usa estilo default.
        titulo: titulo do grafico (se vazio, gera automaticamente)
        hora_ini: hora inicio da janela (ex: "15:30")
        hora_fim: hora fim da janela (ex: "17:10")
        dia: data do dia (date ou Timestamp). Se None, usa dia com
             mais eventos no df_events.
        ref_lines: linhas de referencia horizontais.
             Lista de dicts {y, color, ls, lw, alpha, label}
        shade_events: pares de event_types para sombrear entre eles.
             Ex: [("INCLINANDO", "QUEDA")] sombrea a faixa.
        figsize: tamanho da figura
        signal_label: nome do sinal na legenda
        tick_interval_min: intervalo (min) dos ticks no eixo X
    """
    # ---- Determinar dia ----
    if dia is None:
        if len(df_events) > 0:
            ev_dias = df_events["event_ts"].dt.date.value_counts()
            dia = ev_dias.idxmax()
        else:
            dia = pdf[col_ts].dt.date.value_counts().idxmax()

    # ---- Filtrar dados pela janela ----
    pdf_f = pdf[pdf[col_ts].dt.date == dia].copy()
    pdf_f["_hora"] = pdf_f[col_ts].dt.strftime("%H:%M")
    pdf_f = pdf_f[
        (pdf_f["_hora"] >= hora_ini) & (pdf_f["_hora"] <= hora_fim)
    ].copy()

    ev_f = pd.DataFrame()
    if len(df_events) > 0:
        ev_f = df_events[df_events["event_ts"].dt.date == dia].copy()
        ev_f["_hora"] = ev_f["event_ts"].dt.strftime("%H:%M")
        ev_f = ev_f[
            (ev_f["_hora"] >= hora_ini) & (ev_f["_hora"] <= hora_fim)
        ].sort_values("event_ts")

    if len(pdf_f) == 0:
        print(f"Sem dados para dia={dia} janela={hora_ini}-{hora_fim}")
        return

    # ---- Titulo ----
    dia_str = (
        dia.strftime("%d/%m/%Y") if hasattr(dia, "strftime") else str(dia)
    )
    if not titulo:
        titulo = (
            f"{signal_label} - Deteccao FSM  ({dia_str})"
            f"  [{hora_ini} - {hora_fim}]"
        )

    # ---- Default event styles ----
    if event_styles is None:
        event_styles = {}

    default_style = {
        "color": EVCOMX_GRAY[3], "ls": "--", "lw": 1.0, "tag": "?"
    }

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=figsize)

    # Sinal
    ax.plot(
        pdf_f[col_ts], pdf_f[col_signal],
        linewidth=0.8, color=EVCOMX_GRAY[5], alpha=0.9,
    )

    # Linhas de referencia
    if ref_lines:
        for rl in ref_lines:
            ax.axhline(
                rl["y"],
                color=rl.get("color", EVCOMX_GRAY[2]),
                ls=rl.get("ls", ":"),
                lw=rl.get("lw", 0.8),
                alpha=rl.get("alpha", 0.5),
            )

    # Eventos (linhas verticais)
    if len(ev_f) > 0:
        for _, ev in ev_f.iterrows():
            st = event_styles.get(ev["event_type"], default_style)
            ax.axvline(
                ev["event_ts"],
                color=st["color"],
                linestyle=st["ls"],
                linewidth=st["lw"],
                alpha=0.70,
            )

        # Sombreamento entre pares de eventos
        if shade_events:
            for ev_start, ev_end in shade_events:
                starts = ev_f[ev_f["event_type"] == ev_start]
                ends = ev_f[ev_f["event_type"] == ev_end]
                for _, se in starts.iterrows():
                    match = ends[
                        (ends["ciclo_seq"] == se["ciclo_seq"])
                        & (ends["event_ts"] > se["event_ts"])
                    ]
                    if len(match) > 0:
                        ax.axvspan(
                            se["event_ts"],
                            match.iloc[0]["event_ts"],
                            alpha=0.12,
                            color=EVCOMX_ORANGE[3],
                        )

    # Legenda compacta
    legend_handles = [
        Line2D([0], [0], color=EVCOMX_GRAY[5], lw=1.0, label=signal_label),
    ]
    if ref_lines:
        for rl in ref_lines:
            if rl.get("label"):
                legend_handles.append(
                    Line2D(
                        [0], [0],
                        color=rl.get("color", EVCOMX_GRAY[2]),
                        ls=rl.get("ls", ":"),
                        lw=rl.get("lw", 0.8),
                        label=rl["label"],
                    )
                )
    for etype, st in event_styles.items():
        legend_handles.append(
            Line2D(
                [0], [0],
                color=st["color"],
                ls=st["ls"],
                lw=st["lw"],
                label=st["tag"],
            )
        )

    # Eixos
    style_ax(ax)
    style_title(ax, titulo)
    style_ylabel(ax, col_signal, fontsize=10)
    style_xlabel(ax, "Hora", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=tick_interval_min))
    ax.tick_params(axis="x", rotation=45, labelsize=8, colors=EVCOMX_GRAY[5])
    ax.legend(
        handles=legend_handles, loc="lower left",
        fontsize=8, ncol=5, frameon=False,
    )
    plt.tight_layout()
    plt.show()

    # ---- Print dos ciclos na janela ----
    if len(ev_f) > 0:
        print(f"\nJanela: {hora_ini} -> {hora_fim}  |  Dia: {dia_str}")
        print(f"Registros: {len(pdf_f):,}  |  Eventos: {len(ev_f)}")
        ciclos_na_janela = ev_f["ciclo_seq"].unique()
        for cseq in ciclos_na_janela:
            ev_c = ev_f[ev_f["ciclo_seq"] == cseq].sort_values("event_ts")
            parts = " | ".join(
                f"{r['event_type'].lower()}={r['event_ts'].strftime('%H:%M:%S')}"
                for _, r in ev_c.iterrows()
            )
            print(f"  C{cseq:>3} | {parts}")
