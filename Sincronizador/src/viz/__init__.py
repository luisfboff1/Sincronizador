"""
src.viz

Pacote de visualizacao reutilizavel com estilo EvcomX.

Modulos:
    theme       - Paleta EvcomX, helpers de estilo
    validation  - Plot de validacao de deteccao (sinal + eventos)
    stats       - Filtragem, estatisticas, histogramas, boxplots, tendencias

Uso rapido:
    from src.viz import (
        # theme
        EVCOMX_ORANGE, EVCOMX_GRAY,
        # validation
        plot_deteccao,
        # stats
        filtrar_ciclos_iqr, print_stats,
        plot_histogramas, plot_boxplot_antes_depois,
        plot_tendencia, plot_tendencia_3zooms,
    )
"""

from src.viz.theme import (
    EVCOMX_ORANGE,
    EVCOMX_GRAY,
    style_ax,
    style_title,
    style_suptitle,
    style_xlabel,
    style_ylabel,
)

from src.viz.validation import plot_deteccao

from src.viz.stats import (
    filtrar_ciclos_iqr,
    print_stats,
    plot_histogramas,
    plot_boxplot_antes_depois,
    plot_tendencia,
    plot_tendencia_3zooms,
)
