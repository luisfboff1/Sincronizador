"""plots.py

Funcoes de visualizacao reutilizaveis para todas as etapas (FEA, FP, VD, LC).

Uso:
    from src.plots import plot_sinais_eventos, plot_janela_lc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from typing import List, Dict, Optional, Any


# ================================================================
# FUNCAO GENERICA: plot de sinais + eventos detectados
# ================================================================

def plot_sinais_eventos(
    panels: List[Dict[str, Any]],
    title: str = "",
    figsize: tuple = (20, 10),
) -> None:
    """Plota paineis de sinais com eventos detectados sobrepostos.

    Funcao generica reutilizavel por qualquer etapa (FEA, FP, VD, LC).

    Args:
        panels: lista de dicts, cada um com:
            - ts:     array de timestamps do sinal
            - values: array de valores do sinal
            - label:  nome do sinal (ex: 'PESOBRA1')
            - color:  cor da linha (ex: '#1f77b4')
            - ylabel: rotulo eixo Y (ex: 'Peso Braco 1 (ton)')
            - vlines: lista de dicts com {ts, label, color, ls}
                      linhas verticais de eventos pontuais
            - spans:  (opcional) lista de dicts com {t0, t1, label, color}
                      faixas sombreadas com anotacao de duracao
        title:   titulo geral do grafico
        figsize: tamanho da figura
    """
    n = len(panels)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True, squeeze=False)
    axes = axes.flatten()

    for idx, panel in enumerate(panels):
        ax = axes[idx]
        ts = panel["ts"]
        vals = panel["values"]
        sig_label = panel.get("label", "")
        sig_color = panel.get("color", "#52595E")
        ylabel = panel.get("ylabel", "")

        # Sinal principal
        ax.plot(ts, vals, color=sig_color, lw=0.8, alpha=0.9)

        # Linhas verticais (eventos pontuais)
        legend_handles = [Line2D([0], [0], color=sig_color, lw=1)]
        legend_labels = [sig_label]

        vlines = panel.get("vlines", [])
        vline_legend_added = set()
        ymax = np.nanquantile(vals, 0.95) if len(vals) > 0 else 100

        for vl in vlines:
            vl_color = vl.get("color", "red")
            vl_ls = vl.get("ls", "--")
            vl_label = vl.get("label", "")
            ax.axvline(vl["ts"], color=vl_color, ls=vl_ls, lw=1.5, alpha=0.8)
            ax.annotate(
                vl_label,
                xy=(vl["ts"], ymax * 0.9),
                fontsize=7, color=vl_color, rotation=90, ha="right",
            )
            # Legenda unica por tipo
            leg_key = (vl_color, vl_ls)
            if leg_key not in vline_legend_added:
                vline_legend_added.add(leg_key)
                legend_handles.append(
                    Line2D([0], [0], color=vl_color, ls=vl_ls, lw=1.5)
                )
                legend_labels.append(vl_label.split(" ")[0] if vl_label else "evento")

        # Faixas sombreadas (ciclos com duracao)
        spans = panel.get("spans", [])
        span_legend_added = False
        for sp in spans:
            sp_color = sp.get("color", "#FFC7B2")
            t0_sp = sp["t0"]
            t1_sp = sp["t1"]
            ax.axvline(t0_sp, color="#FF4000", ls="-", lw=1.5, alpha=0.7)
            ax.axvline(t1_sp, color="#172129", ls="-", lw=1.5, alpha=0.7)
            ax.axvspan(t0_sp, t1_sp, alpha=0.06, color=sp_color)
            if sp.get("label"):
                mid = t0_sp + (t1_sp - t0_sp) / 2
                ax.annotate(
                    sp["label"],
                    xy=(mid, ymax * 0.05 if ymax > 0 else 5),
                    fontsize=8, color=sig_color,
                    ha="center", fontweight="bold",
                )
            if not span_legend_added:
                legend_handles.extend([
                    Line2D([0], [0], color="#FF4000", lw=1.5),
                    Line2D([0], [0], color="#172129", lw=1.5),
                ])
                legend_labels.extend(["Inicio", "Fim"])
                span_legend_added = True

        ax.legend(handles=legend_handles, labels=legend_labels,
                  loc="upper right", fontsize=8, frameon=False)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#BABDBF")
        ax.spines["bottom"].set_color("#BABDBF")

        if idx == 0 and title:
            ax.set_title(title, fontsize=13, fontweight="bold", color="#172129")

    # Eixo X
    axes[-1].set_xlabel("Horario", fontsize=10)
    for a in axes:
        a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        a.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
        a.tick_params(axis="x", rotation=0, labelsize=8)

    plt.tight_layout()
    plt.show()


# ================================================================
# WRAPPER LC: monta panels automaticamente a partir dos DataFrames
# ================================================================

def plot_janela_lc(
    pdf_csv: pd.DataFrame,
    pdf_opc: pd.DataFrame,
    df_bra_sys: pd.DataFrame,
    df_real_sys: pd.DataFrame,
    dia: str,
    h_ini: str = "06:00",
    h_fim: str = "13:00",
    escala: float = 10.0,
) -> None:
    """Plota janela de operacao LC com sinais + eventos detectados.

    Args:
        pdf_csv:     DataFrame pandas com timestamp, PESOBRA1, PESOBRA2
        pdf_opc:     DataFrame pandas com timestamp, PESOREAL
        df_bra_sys:  Eventos chegada torre (braco, chegada_torre_ts, ...)
        df_real_sys: Ciclos peso real (inicio/fim_pesoreal_ts, duracao_s, ...)
        dia:         Data no formato 'YYYY-MM-DD'
        h_ini:       Hora inicio (ex: '06:00')
        h_fim:       Hora fim (ex: '13:00')
        escala:      Divisor para converter para toneladas (default: 10)
    """
    t0 = pd.Timestamp(f"{dia} {h_ini}")
    t1 = pd.Timestamp(f"{dia} {h_fim}")

    # Recorta janela
    sl_csv = pdf_csv[
        (pdf_csv["timestamp"] >= t0) & (pdf_csv["timestamp"] <= t1)
    ].copy()
    sl_opc = pdf_opc[
        (pdf_opc["timestamp"] >= t0) & (pdf_opc["timestamp"] <= t1)
    ].copy()

    # Toneladas
    col_b1 = "ACI@LC_TORRE_PESOBRA1"
    col_b2 = "ACI@LC_TORRE_PESOBRA2"
    col_pr = "ACI@LC_TORRE_PESOREAL"
    sl_csv["B1_t"] = sl_csv[col_b1] / escala
    sl_csv["B2_t"] = sl_csv[col_b2] / escala
    sl_opc["PR_t"] = sl_opc[col_pr] / escala

    # Eventos na janela
    ev_bra = df_bra_sys[
        (df_bra_sys["chegada_torre_ts"] >= t0)
        & (df_bra_sys["chegada_torre_ts"] <= t1)
    ]
    ev_real = df_real_sys[
        (df_real_sys["inicio_pesoreal_ts"] >= t0)
        & (df_real_sys["inicio_pesoreal_ts"] <= t1)
    ]

    # Monta panels
    def _vlines_braco(braco):
        subset = ev_bra[ev_bra["braco"] == braco]
        tag = f"chegada B{braco}"
        return [
            {"ts": r["chegada_torre_ts"], "label": tag, "color": "#FF4000", "ls": "--"}
            for _, r in subset.iterrows()
        ]

    def _spans_real():
        out = []
        for _, r in ev_real.iterrows():
            dur = r["duracao_s"] / 60
            out.append({
                "t0": r["inicio_pesoreal_ts"],
                "t1": r["fim_pesoreal_ts"],
                "label": f"{dur:.0f}min",
                "color": "green",
            })
        return out

    panels = [
        {
            "ts": sl_csv["timestamp"].values,
            "values": sl_csv["B1_t"].values,
            "label": "PESOBRA1",
            "color": "#52595E",
            "ylabel": "Peso Braco 1 (ton)",
            "vlines": _vlines_braco(1),
        },
        {
            "ts": sl_csv["timestamp"].values,
            "values": sl_csv["B2_t"].values,
            "label": "PESOBRA2",
            "color": "#FF7040",
            "ylabel": "Peso Braco 2 (ton)",
            "vlines": _vlines_braco(2),
        },
        {
            "ts": sl_opc["timestamp"].values,
            "values": sl_opc["PR_t"].values,
            "label": "PESOREAL",
            "color": "#2E363D",
            "ylabel": "Peso Real Torre (ton)",
            "vlines": [],
            "spans": _spans_real(),
        },
    ]

    title = f"Sinais LC + Eventos Detectados (FSM) -- {dia} {h_ini} a {h_fim}"
    plot_sinais_eventos(panels, title=title)

    # Resumo
    n_b1 = (ev_bra["braco"] == 1).sum()
    n_b2 = (ev_bra["braco"] == 2).sum()
    print(f"\nJanela: {dia} {h_ini}-{h_fim}")
    print(f"  Chegadas torre: {len(ev_bra)} (B1: {n_b1}, B2: {n_b2})")
    print(f"  Ciclos peso real: {len(ev_real)}")
    if len(ev_real) > 0:
        print(f"  Duracao media: {ev_real['duracao_s'].mean()/60:.1f} min")
