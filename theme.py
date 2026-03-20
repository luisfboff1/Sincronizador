"""
theme.py

Paleta EvcomX e helpers de estilo para graficos matplotlib.
Todas as funcoes de visualizacao do projeto usam estas constantes.

Uso:
    from src.viz.theme import EVCOMX_ORANGE, EVCOMX_GRAY, style_ax
"""

# ================================================================
# PALETA EVCOMX
# ================================================================
#  0=mais-claro .. 7=mais-escuro

EVCOMX_ORANGE = [
    "#FFF5F2",  # 0  fundo
    "#FFE3D9",  # 1  fundo leve
    "#FFC7B2",  # 2  referencia suave
    "#FFA88C",  # 3  referencia media
    "#FF8C66",  # 4  destaque suave
    "#FF7040",  # 5  destaque medio
    "#FF541A",  # 6  destaque forte
    "#FF4000",  # 7  destaque maximo / primario
]

EVCOMX_GRAY = [
    "#F2F5F5",  # 0  fundo
    "#DBDEDE",  # 1  grid suave
    "#BABDBF",  # 2  bordas, spines
    "#969C9E",  # 3  texto terciario
    "#73787D",  # 4  texto secundario
    "#52595E",  # 5  sinal principal
    "#2E363D",  # 6  texto forte
    "#172129",  # 7  titulo / destaque
]


# ================================================================
# HELPERS
# ================================================================

def style_ax(ax, grid=False):
    """Aplica estilo EvcomX padrao a um eixo matplotlib.

    - Remove spines top/right
    - Pinta spines left/bottom com GRAY[2]
    - Configura ticks
    - Opcional: grid sutil
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EVCOMX_GRAY[2])
    ax.spines["bottom"].set_color(EVCOMX_GRAY[2])
    ax.tick_params(labelsize=8, colors=EVCOMX_GRAY[5])
    if grid:
        ax.grid(True, color=EVCOMX_GRAY[1], lw=0.5, alpha=0.5)
    else:
        ax.grid(False)


def style_title(ax, titulo, fontsize=13):
    """Titulo bold com cor EvcomX GRAY[7]."""
    ax.set_title(
        titulo, fontsize=fontsize, fontweight="bold",
        color=EVCOMX_GRAY[7],
    )


def style_suptitle(fig, titulo, fontsize=13):
    """Suptitle bold com cor EvcomX GRAY[7]."""
    fig.suptitle(
        titulo, fontsize=fontsize, fontweight="bold",
        color=EVCOMX_GRAY[7],
    )


def style_xlabel(ax, label, fontsize=10):
    ax.set_xlabel(label, fontsize=fontsize, color=EVCOMX_GRAY[5])


def style_ylabel(ax, label, fontsize=10):
    ax.set_ylabel(label, fontsize=fontsize, color=EVCOMX_GRAY[5])
