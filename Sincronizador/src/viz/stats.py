"""
stats.py

Filtragem de ciclos, tabela de estatisticas, histogramas, boxplots
e tendencias temporais.  Reutilizavel para qualquer detector
(FEA energia, peso carro panela, inclinometro, etc.).

Uso:
    from src.viz.stats import (
        filtrar_ciclos_iqr,
        print_stats,
        plot_histogramas,
        plot_boxplot_antes_depois,
        plot_tendencia,
    )
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple

from src.viz.theme import (
    EVCOMX_ORANGE, EVCOMX_GRAY,
    style_ax, style_title, style_suptitle,
    style_xlabel, style_ylabel,
)


# ================================================================
# 1) FILTRAGEM MULTI-ETAPA
# ================================================================

def filtrar_ciclos_iqr(
    df: pd.DataFrame,
    metric_cols: List[str],
    metric_labels: Optional[Dict[str, str]] = None,
    # Etapa 1: anomalia (ex: peso_max < 55, angulo_max < 5)
    anomaly_col: Optional[str] = None,
    anomaly_op: str = "<",          # "<" ou ">"
    anomaly_val: Optional[float] = None,
    anomaly_label: str = "Anomalia",
    # Etapa 2: timeout (ex: plato_espera_min > 30)
    timeout_col: Optional[str] = None,
    timeout_max_min: Optional[float] = None,
    timeout_label: str = "Timeout",
    # Etapa 3: IQR
    iqr_factor: float = 1.5,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filtragem em 3 etapas: anomalia + timeout + IQR.

    Retorna:
        df_std:   DataFrame filtrado (ciclos padrao)
        df_antes: DataFrame original (para boxplot comparativo)
    """
    df_antes = df.copy()
    df_work = df.copy()
    if metric_labels is None:
        metric_labels = {c: c for c in metric_cols}

    if verbose:
        print(f"Ciclos antes da filtragem: {len(df_work)}")

    # --- Etapa 1: anomalia ---
    n_anom = 0
    if anomaly_col and anomaly_val is not None:
        if anomaly_op == "<":
            mask_anom = df_work[anomaly_col] < anomaly_val
        else:
            mask_anom = df_work[anomaly_col] > anomaly_val
        n_anom = mask_anom.sum()
        df_work = df_work[~mask_anom].copy()
        if verbose:
            print(f"\nEtapa 1 - {anomaly_label} "
                  f"({anomaly_col} {anomaly_op} {anomaly_val}):")
            print(f"  Removidos: {n_anom}")
            print(f"  Restantes: {len(df_work)}")

    # --- Etapa 2: timeout ---
    n_timeout = 0
    if timeout_col and timeout_max_min is not None:
        mask_to = df_work[timeout_col] > timeout_max_min
        n_timeout = mask_to.sum()
        df_work = df_work[~mask_to].copy()
        if verbose:
            print(f"\nEtapa 2 - {timeout_label} "
                  f"({timeout_col} > {timeout_max_min}):")
            print(f"  Removidos: {n_timeout}")
            print(f"  Restantes: {len(df_work)}")

    # --- Etapa 3: IQR ---
    outlier_mask = pd.Series(False, index=df_work.index)
    if verbose:
        print(f"\nEtapa 3 - Outliers IQR {iqr_factor}x:")
    for col in metric_cols:
        data = df_work[col].dropna()
        if len(data) < 10:
            continue
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        mask = (df_work[col] < lower) | (df_work[col] > upper)
        n_out = mask.sum()
        if n_out > 0 and verbose:
            label = metric_labels.get(col, col)
            print(f"  {label}: {n_out} outliers "
                  f"(range: [{lower:.2f}, {upper:.2f}])")
        outlier_mask = outlier_mask | mask

    n_iqr = outlier_mask.sum()
    df_std = df_work[~outlier_mask].copy()

    if verbose:
        print(f"\nResultado: {len(df_std)} ciclos padrao "
              f"de {len(df_antes)} originais")
        print(f"  Removidos: {n_anom} anomalias + "
              f"{n_timeout} timeouts + {n_iqr} IQR")

    return df_std, df_antes


# ================================================================
# 2) TABELA DE ESTATISTICAS
# ================================================================

def print_stats(
    df: pd.DataFrame,
    metric_cols: List[str],
    metric_labels: Dict[str, str],
    titulo: str = "ESTATISTICAS",
    metric_intervals: Optional[Dict[str, str]] = None,
) -> None:
    """Imprime tabela formatada de estatisticas por metrica."""
    if metric_intervals is None:
        metric_intervals = {}

    has_intervals = any(metric_intervals.values())

    print(f"\n{'=' * 130}")
    print(titulo)
    print(f"{'=' * 130}")

    if has_intervals:
        print(f"{'Metrica':<35} | {'Intervalo':<25} | {'N':>4} | "
              f"{'Media':>6} | {'Std':>5} | {'P05':>5} | {'P25':>5} | "
              f"{'P50':>5} | {'P75':>5} | {'P95':>5} | "
              f"{'Min':>5} | {'Max':>5}")
    else:
        print(f"{'Metrica':<35} | {'N':>4} | "
              f"{'Media':>6} | {'Std':>5} | {'P05':>5} | {'P25':>5} | "
              f"{'P50':>5} | {'P75':>5} | {'P95':>5} | "
              f"{'Min':>5} | {'Max':>5}")
    print("-" * 130)

    for col in metric_cols:
        data = df[col].dropna()
        if len(data) == 0:
            continue
        label = metric_labels.get(col, col)
        interval = metric_intervals.get(col, "")

        parts = [f"{label:<35}"]
        if has_intervals:
            parts.append(f"{interval:<25}")
        parts.extend([
            f"{len(data):>4}",
            f"{data.mean():>6.2f}",
            f"{data.std():>5.2f}",
            f"{data.quantile(0.05):>5.2f}",
            f"{data.quantile(0.25):>5.2f}",
            f"{data.quantile(0.50):>5.2f}",
            f"{data.quantile(0.75):>5.2f}",
            f"{data.quantile(0.95):>5.2f}",
            f"{data.min():>5.2f}",
            f"{data.max():>5.2f}",
        ])
        print(" | ".join(parts))


# ================================================================
# 3) HISTOGRAMAS
# ================================================================

def plot_histogramas(
    df: pd.DataFrame,
    hist_specs: List[Tuple[str, str, str]],
    titulo: str = "Distribuicao",
    grid_shape: Optional[Tuple[int, int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """Plota grade de histogramas com estilo EvcomX.

    Args:
        df: DataFrame com as colunas de metricas
        hist_specs: lista de (col_name, titulo_subplot, cor_hex).
            Ex: [("duracao_min", "Duracao (min)", "#FF4000"), ...]
        titulo: suptitle do grafico
        grid_shape: (nrows, ncols). Se None, calcula automaticamente.
        figsize: tamanho da figura. Se None, calcula automaticamente.
    """
    n = len(hist_specs)
    if n == 0:
        return

    if grid_shape is None:
        if n <= 3:
            grid_shape = (1, n)
        elif n <= 6:
            grid_shape = (2, 3)
        else:
            ncols = 3
            nrows = (n + ncols - 1) // ncols
            grid_shape = (nrows, ncols)

    if figsize is None:
        figsize = (grid_shape[1] * 6.5, grid_shape[0] * 5)

    fig, axes = plt.subplots(*grid_shape, figsize=figsize)
    if n == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (col, col_title, color) in enumerate(hist_specs):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        data = df[col].dropna()
        if len(data) == 0:
            ax.set_visible(False)
            continue

        p05 = data.quantile(0.05)
        p95 = data.quantile(0.95)
        mu = data.mean()
        med = data.median()
        sigma = data.std()
        n_bins = min(30, max(5, len(data) // 3))

        ax.hist(data, bins=n_bins, color=color, edgecolor="white", alpha=0.75)
        ax.axvline(mu, color=EVCOMX_GRAY[7], ls="-", lw=1.5, alpha=0.8,
                   label=f"media={mu:.2f}")
        ax.axvline(med, color=EVCOMX_GRAY[5], ls="--", lw=1.2, alpha=0.6,
                   label=f"mediana={med:.2f}")
        ax.axvline(p05, color=EVCOMX_GRAY[3], ls=":", lw=1.2, alpha=0.7,
                   label=f"P05={p05:.2f}")
        ax.axvline(p95, color=EVCOMX_GRAY[3], ls=":", lw=1.2, alpha=0.7,
                   label=f"P95={p95:.2f}")

        style_ax(ax)
        ax.set_title(col_title, fontsize=10, color=EVCOMX_GRAY[7])
        style_xlabel(ax, "minutos", fontsize=9)
        style_ylabel(ax, "frequencia", fontsize=9)
        ax.legend(fontsize=7, loc="upper right", frameon=False)
        ax.text(
            0.02, 0.95, f"n={len(data)}  sigma={sigma:.2f}",
            transform=ax.transAxes, fontsize=7, va="top",
            color=EVCOMX_GRAY[4],
        )

    # Esconde subplots vazios
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    style_suptitle(fig, titulo)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


# ================================================================
# 4) BOXPLOT ANTES vs DEPOIS
# ================================================================

def plot_boxplot_antes_depois(
    df_antes: pd.DataFrame,
    df_depois: pd.DataFrame,
    metric_cols: List[str],
    metric_labels: Dict[str, str],
    box_colors: Optional[List[str]] = None,
    titulo: str = "Antes vs Depois da filtragem",
    figsize: tuple = (18, 6),
) -> None:
    """Boxplot lado-a-lado: todos os ciclos vs ciclos padrao."""
    if box_colors is None:
        # Cores default alternando orange e gray
        pool = [
            EVCOMX_ORANGE[7], EVCOMX_ORANGE[5], EVCOMX_ORANGE[3],
            EVCOMX_GRAY[5], EVCOMX_GRAY[3], EVCOMX_ORANGE[4],
        ]
        box_colors = [pool[i % len(pool)] for i in range(len(metric_cols))]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    labels_short = [
        metric_labels.get(c, c).replace(" ", "\n").replace("(", "\n(")
        for c in metric_cols
    ]

    for ax_idx, (ax, df_src, subtitle) in enumerate([
        (axes[0], df_antes, f"ANTES -- {len(df_antes)} ciclos"),
        (axes[1], df_depois, f"DEPOIS -- {len(df_depois)} ciclos padrao"),
    ]):
        box_data = [df_src[c].dropna().values for c in metric_cols]
        bp = ax.boxplot(
            box_data,
            labels=labels_short,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            meanprops={
                "color": EVCOMX_ORANGE[7], "ls": "-", "lw": 1.5,
            },
            medianprops={"color": EVCOMX_GRAY[7], "lw": 1.5},
            flierprops={
                "marker": "o", "markersize": 4, "alpha": 0.5,
            },
        )
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.4)

        style_ax(ax)
        style_ylabel(ax, "minutos", fontsize=9)
        style_title(ax, subtitle, fontsize=11)
        ax.tick_params(axis="x", rotation=15, labelsize=7,
                       colors=EVCOMX_GRAY[5])

    style_suptitle(fig, titulo)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


# ================================================================
# 5) TENDENCIA TEMPORAL
# ================================================================

def plot_tendencia(
    df: pd.DataFrame,
    col_ts: str,
    metrics: List[Tuple[str, str, str, str]],
    titulo: str = "Tendencia temporal",
    date_fmt: str = "%d/%m",
    tick_locator: object = None,
    figsize: tuple = (20, 5),
) -> None:
    """Plota scatter+line para metricas ao longo do tempo.

    Args:
        df: DataFrame com timestamp e metricas
        col_ts: coluna de timestamp para eixo X
        metrics: lista de (col, label, color_line, color_dot).
            Ex: [("dur_min", "Duracao", "#FF7040", "#FF4000"), ...]
        titulo: titulo do grafico
        date_fmt: formato do eixo X
        tick_locator: matplotlib locator (ex: mdates.DayLocator)
        figsize: tamanho da figura
    """
    fig, ax = plt.subplots(figsize=figsize)

    for col, label, color_line, color_dot in metrics:
        data = df[[col_ts, col]].dropna()
        if len(data) == 0:
            continue
        ax.plot(
            data[col_ts], data[col],
            color=color_line, lw=0.8, alpha=0.5, zorder=2,
        )
        ax.scatter(
            data[col_ts], data[col],
            s=18, color=color_dot, zorder=4,
            edgecolors="white", linewidths=0.3,
            label=f"{label} (n={len(data)})",
        )

    style_ax(ax)
    style_title(ax, titulo)
    style_xlabel(ax, "Data / Hora")
    style_ylabel(ax, "Minutos")
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    if tick_locator:
        ax.xaxis.set_major_locator(tick_locator)
    ax.tick_params(axis="x", rotation=45, labelsize=8,
                   colors=EVCOMX_GRAY[5])
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_tendencia_3zooms(
    df: pd.DataFrame,
    col_ts: str,
    metrics: List[Tuple[str, str, str, str]],
    titulo_base: str = "Tendencia",
    week_start: object = None,
    week_end: object = None,
) -> None:
    """Plota tendencia em 3 niveis: mes, semana, dia.

    Conveniencia que chama plot_tendencia 3 vezes com
    recortes temporais automaticos.

    Args:
        df: DataFrame com col_ts e metricas
        col_ts: coluna de timestamp
        metrics: ver plot_tendencia
        titulo_base: prefixo do titulo
        week_start: inicio da semana (Timestamp). Se None, usa
                    semana central do dataset.
        week_end:   fim da semana (Timestamp). Se None, calcula.
    """
    df = df.sort_values(col_ts).reset_index(drop=True)
    df["_dia"] = df[col_ts].dt.date

    # -- Mes completo --
    plot_tendencia(
        df, col_ts, metrics,
        titulo=f"{titulo_base} -- Mes completo ({len(df)} ciclos)",
        date_fmt="%d/%m",
        tick_locator=mdates.DayLocator(interval=2),
    )

    # -- Semana --
    if week_start is None:
        mid = df[col_ts].median()
        week_start = mid - pd.Timedelta(days=3)
        week_end = mid + pd.Timedelta(days=4)
    df_week = df[
        (df[col_ts] >= week_start) & (df[col_ts] < week_end)
    ]
    ws = week_start.strftime("%d/%m") if hasattr(week_start, "strftime") else str(week_start)
    we = week_end.strftime("%d/%m") if hasattr(week_end, "strftime") else str(week_end)
    plot_tendencia(
        df_week, col_ts, metrics,
        titulo=f"{titulo_base} -- {ws} a {we} ({len(df_week)} ciclos)",
        date_fmt="%d/%m %H:%M",
        tick_locator=mdates.HourLocator(interval=12),
    )

    # -- Dia (o com mais ciclos) --
    top_dia = df.groupby("_dia").size().idxmax()
    df_day = df[df["_dia"] == top_dia]
    dia_str = (
        top_dia.strftime("%d/%m/%Y") if hasattr(top_dia, "strftime")
        else str(top_dia)
    )
    plot_tendencia(
        df_day, col_ts, metrics,
        titulo=f"{titulo_base} -- {dia_str} ({len(df_day)} ciclos)",
        date_fmt="%H:%M",
        tick_locator=mdates.HourLocator(interval=2),
        figsize=(18, 5),
    )
