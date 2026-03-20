"""lc_stats.py

Filtragem de corridas (timeout + IQR), estatisticas descritivas
e visualizacao de distribuicoes de tempos do LC.

Uso:
    from src.analysis.lc_stats import (
        filtrar_corridas_padrao,
        print_stats_tempos,
        plot_distribuicao_tempos,
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict


# ================================================================
# CONSTANTES
# ================================================================

METRIC_COLS = [
    "chegada_to_inicio_min",
    "duracao_vazamento_min",
    "ciclo_total_min",
    "fim_to_chegada_min",
    "fim_to_inicio_min",
]

METRIC_LABELS: Dict[str, str] = {
    "chegada_to_inicio_min": "Chegada -> Inicio real",
    "duracao_vazamento_min": "Duracao vazamento",
    "ciclo_total_min": "Ciclo total",
    "fim_to_chegada_min": "Fim ant. -> Chegada prox.",
    "fim_to_inicio_min": "Fim ant. -> Inicio prox.",
}


# ================================================================
# FILTRAGEM
# ================================================================

def _iqr_mask(series: pd.Series) -> pd.Series:
    """Retorna mask True = dentro do intervalo IQR (nao-outlier)."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return (series >= lo) & (series <= hi)


def filtrar_corridas_padrao(
    df_corridas: pd.DataFrame,
    timeout_min: float = 90,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Filtra corridas: remove paradas (timeout) e outliers (IQR 1.5x).

    Args:
        df_corridas: DataFrame com corridas emparelhadas.
        timeout_min: Duracao maxima aceita (min). Acima = parada.

    Returns:
        (df_std, df_paradas, resumo)
        - df_std: corridas padrao (filtradas)
        - df_paradas: corridas classificadas como parada
        - resumo: dict com contagens e ranges
    """
    df = df_corridas.copy()

    # 1) Classifica paradas
    df["tipo"] = np.where(
        df["duracao_vazamento_min"] > timeout_min, "parada", "normal"
    )
    df_paradas = df[df["tipo"] == "parada"].copy()
    n_parada = len(df_paradas)
    n_normal_ini = (df["tipo"] == "normal").sum()

    print(f"Timeout vazamento: {timeout_min} min")
    print(f"   Paradas / troca distribuidor: {n_parada}")
    print(f"   Operacao normal: {n_normal_ini}")

    if n_parada > 0:
        print("\nCorridas classificadas como PARADA:")
        for _, r in df_paradas.iterrows():
            bra = int(r["braco"]) if pd.notna(r.get("braco")) else "?"
            print(
                f"  B{bra} | {r['inicio_pesoreal_ts']} -> "
                f"{r['fim_pesoreal_ts']} | "
                f"duracao: {r['duracao_vazamento_min']:.1f} min"
            )

    # 2) Filtra somente normais
    df_normal = df[df["tipo"] == "normal"].copy().reset_index(drop=True)

    # Recalcula metricas entre corridas consecutivas normais
    df_normal["fim_to_chegada_min"] = np.nan
    df_normal["fim_to_inicio_min"] = np.nan
    for i in range(1, len(df_normal)):
        prev_fim = df_normal.at[i - 1, "fim_pesoreal_ts"]
        curr_cheg = df_normal.at[i, "chegada_torre_ts"]
        curr_ini = df_normal.at[i, "inicio_pesoreal_ts"]
        if pd.notna(prev_fim) and pd.notna(curr_cheg):
            df_normal.at[i, "fim_to_chegada_min"] = (
                (curr_cheg - prev_fim).total_seconds() / 60
            )
        if pd.notna(prev_fim) and pd.notna(curr_ini):
            df_normal.at[i, "fim_to_inicio_min"] = (
                (curr_ini - prev_fim).total_seconds() / 60
            )

    # 3) Remove outliers IQR
    print("\nRemocao de outliers residuais (IQR 1.5x):")
    outlier_mask = pd.Series(True, index=df_normal.index)
    iqr_ranges = {}

    for col in METRIC_COLS:
        data = df_normal[col].dropna()
        if len(data) < 5:
            continue
        m = _iqr_mask(df_normal[col])
        m = m | df_normal[col].isna()
        n_out = (~m).sum()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_ranges[col] = (lo, hi)
        if n_out > 0:
            print(f"  {col}: {n_out} outliers (range: [{lo:.1f}, {hi:.1f}])")
        outlier_mask &= m

    df_std = df_normal[outlier_mask].copy().reset_index(drop=True)
    n_iqr = len(df_normal) - len(df_std)

    print(f"\nCorridas padrao finais: {len(df_std)} de {len(df_corridas)} originais")
    print(f"   Removidos: {n_parada} paradas + {n_iqr} outliers IQR")

    resumo = {
        "total": len(df_corridas),
        "paradas": n_parada,
        "outliers_iqr": n_iqr,
        "padrao": len(df_std),
        "timeout_min": timeout_min,
        "iqr_ranges": iqr_ranges,
    }

    return df_std, df_paradas, resumo


# ================================================================
# ESTATISTICAS
# ================================================================

def print_stats_tempos(
    df_std: pd.DataFrame,
    title: Optional[str] = None,
) -> None:
    """Imprime tabela de estatisticas descritivas das metricas de tempo."""
    if title is None:
        title = f"{len(df_std)} corridas padrao"

    sep = "=" * 100
    print(f"\n{sep}")
    print(f"ESTATISTICAS -- {title}")
    print(sep)

    header_metric = "Metrica"
    header = (
        f"{header_metric:<28} | {'N':>4} | {'Media':>7} | {'Std':>6} | "
        f"{'P05':>6} | {'P25':>6} | {'P50':>6} | {'P75':>6} | "
        f"{'P95':>6} | {'Min':>6} | {'Max':>6}"
    )
    print(header)
    print("-" * 120)

    for col in METRIC_COLS:
        data = df_std[col].dropna()
        if len(data) == 0:
            continue
        label = METRIC_LABELS[col]
        row = (
            f"{label:<28} | {len(data):>4} | "
            f"{data.mean():>7.1f} | {data.std():>6.1f} | "
            f"{data.quantile(0.05):>6.1f} | {data.quantile(0.25):>6.1f} | "
            f"{data.median():>6.1f} | {data.quantile(0.75):>6.1f} | "
            f"{data.quantile(0.95):>6.1f} | {data.min():>6.1f} | "
            f"{data.max():>6.1f}"
        )
        print(row)

    # Atraso vs antecipacao
    ftc = df_std["fim_to_chegada_min"].dropna()
    if len(ftc) > 0:
        n_esp = (ftc < 0).sum()
        n_atr = (ftc >= 0).sum()
        print(f"\nAtraso vs Antecipacao (corridas padrao):")
        print(f"   Ja esperando: {n_esp} ({n_esp / len(ftc) * 100:.0f}%)")
        print(f"   Atrasou:      {n_atr} ({n_atr / len(ftc) * 100:.0f}%)")


# ================================================================
# PLOTS
# ================================================================

HIST_SPECS = [
    ("chegada_to_inicio_min",
     "Chegada torre -> Inicio peso real\n(giro torre, min)", "#FF4000"),
    ("duracao_vazamento_min",
     "Duracao vazamento\n(inicio -> fim peso real, min)", "#FF8C66"),
    ("ciclo_total_min",
     "Ciclo total\n(chegada -> fim, min)", "#73787D"),
    ("fim_to_chegada_min",
     "Fim ant. -> Chegada prox.\n(min, negativo = esperando)", "#FF541A"),
    ("fim_to_inicio_min",
     "Fim ant. -> Inicio prox.\n(intervalo entre corridas, min)", "#2E363D"),
]

BOX_COLORS = ["#FF4000", "#FF8C66", "#73787D", "#FF541A", "#2E363D"]


def plot_distribuicao_tempos(
    df_std: pd.DataFrame,
    df_all: Optional[pd.DataFrame] = None,
) -> None:
    """Plota histogramas e boxplots (antes vs depois) das metricas.

    Args:
        df_std: Corridas padrao filtradas.
        df_all: Todas as corridas (para boxplot comparativo).
    """
    # --- Histogramas ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (col, col_title, color) in enumerate(HIST_SPECS):
        ax = axes[idx]
        data = df_std[col].dropna()
        if len(data) == 0:
            ax.set_visible(False)
            continue

        p05, p95 = data.quantile(0.05), data.quantile(0.95)
        mu, med, sigma = data.mean(), data.median(), data.std()

        n_bins = min(30, max(5, len(data) // 3))
        ax.hist(data, bins=n_bins, color=color,
                edgecolor="white", alpha=0.75)

        ax.axvline(mu, color="#172129", ls="-", lw=1.5, alpha=0.8,
                   label=f"media={mu:.1f}")
        ax.axvline(med, color="#52595E", ls="--", lw=1.2, alpha=0.6,
                   label=f"mediana={med:.1f}")
        ax.axvline(p05, color="gray", ls=":", lw=1.2, alpha=0.7,
                   label=f"P05={p05:.1f}")
        ax.axvline(p95, color="gray", ls=":", lw=1.2, alpha=0.7,
                   label=f"P95={p95:.1f}")

        if (data < 0).any():
            ax.axvspan(ax.get_xlim()[0], 0, alpha=0.08, color="green")

        ax.set_title(col_title, fontsize=9)
        ax.set_xlabel("minutos", fontsize=8)
        ax.set_ylabel("frequencia", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(False)
        ax.legend(fontsize=7, loc="upper right", frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#BABDBF")
        ax.spines["bottom"].set_color("#BABDBF")
        ax.text(
            0.02, 0.95, f"n={len(data)}  sigma={sigma:.1f}",
            transform=ax.transAxes, fontsize=7, va="top", color="gray",
        )

    for idx in range(len(HIST_SPECS), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Distribuicao Tempos Padrao LC -- {len(df_std)} corridas",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

    # --- Boxplots comparativos ---
    if df_all is None:
        return

    box_cols = [c for c in METRIC_COLS if df_std[c].notna().sum() >= 3]
    box_labels = [
        METRIC_LABELS[c].replace(" -> ", "->\n").replace(" ant. ", " ant.\n")
        for c in box_cols
    ]

    if not box_cols:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax_idx, (ax, df_src, label) in enumerate([
        (axes[0], df_all, f"ANTES -- Todas {len(df_all)} corridas"),
        (axes[1], df_std, f"DEPOIS -- {len(df_std)} corridas padrao"),
    ]):
        box_data = [df_src[c].dropna().values for c in box_cols]
        bp = ax.boxplot(
            box_data, labels=box_labels, patch_artist=True,
            showmeans=True, meanline=True,
            meanprops={"color": "#FF4000", "ls": "-", "lw": 1.5},
            medianprops={"color": "#172129", "lw": 1.5},
            flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
        )
        for patch, c in zip(bp["boxes"], BOX_COLORS[:len(box_cols)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.4)
        ax.set_ylabel("minutos")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#BABDBF")
        ax.spines["bottom"].set_color("#BABDBF")
        ax.tick_params(axis="x", rotation=15, labelsize=8)

    fig.suptitle(
        "Comparacao: Antes vs Depois da filtragem (paradas + IQR)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()
