## Diretrizes para visualização de dados

Ao gerar plots, **não usar grid** em nenhuma hipótese, salvo se o usuário pedir explicitamente. Priorizar gráficos limpos, com baixo ruído visual e alta legibilidade.

### Princípios visuais

* Respeitar a relação **ink/data**: usar apenas os elementos visuais estritamente necessários para comunicar os dados.
* Evitar excesso de contornos, linhas auxiliares, sombras, preenchimentos pesados e ornamentos.
* Remover elementos decorativos que não agreguem interpretação.
* Priorizar contraste, hierarquia visual clara e leitura rápida.
* Eixos, ticks e rótulos devem existir apenas quando ajudarem a compreensão.
* Preferir títulos curtos, legendas objetivas e textos diretamente informativos.
* Em gráficos com muitas categorias, reduzir competição visual entre séries secundárias e destacar apenas o que for analiticamente relevante.

### Regras gerais de estilo para plots

* Não exibir grid (`grid=False` ou equivalente).
* Usar espessuras de linha moderadas.
* Evitar marcadores excessivos em séries longas.
* Evitar bordas desnecessárias em barras, áreas e legendas.
* Usar transparência com moderação.
* Manter consistência de cores entre gráficos do mesmo contexto analítico.
* Revisar o gráfico final para garantir clareza, economia visual e boa interpretação.

---

## Paleta de cores e colormap

Definir uma lista fixa de cores prioritárias para uso nos gráficos. Essa lista deve ser usada antes de recorrer a qualquer colormap.

```python
DEFAULT_COLORS = [
    "#FFF5F2",  # laranja 1 (mais fraco)
    "#FFE3D9",  # laranja 2
    "#FFC7B2",  # laranja 3
    "#FFA88C",  # laranja 4
    "#FF8C66",  # laranja 5
    "#FF7040",  # laranja 6
    "#FF541A",  # laranja 7
    "#FF4000",  # laranja 8 (mais forte)

    "#F2F5F5",  # cinza 1 (mais fraco)
    "#DBDEDE",  # cinza 2
    "#BABDBF",  # cinza 3
    "#969C9E",  # cinza 4
    "#73787D",  # cinza 5
    "#52595E",  # cinza 6
    "#2E363D",  # cinza 7
    "#172129",  # cinza 8 (mais forte)
]
```

### Regra de uso das cores

* Se o número de categorias for menor ou igual ao tamanho de `DEFAULT_COLORS`, usar `DEFAULT_COLORS`.
* Se o número de categorias exceder essa quantidade, gerar cores a partir de `DEFAULT_COLORMAP`.
* Ao amostrar cores do colormap, distribuir os pontos de forma uniforme para maximizar diferenciação visual entre categorias.
* Evitar cores excessivamente saturadas ou combinações que prejudiquem acessibilidade.
* Sempre que houver destaque analítico, usar contraste por intenção, não por excesso.

### Colormap para casos com muitas categorias

Quando a quantidade de séries, grupos ou categorias exceder o número de cores disponíveis em `DEFAULT_COLORS`, usar um colormap customizado gerado pela interpolação entre as cores `#E02F1A` e `#161C23`.

Usar o seguinte padrão de implementação:

```python
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def create_two_color_colormap(color1, color2, name="custom_cmap", n=256):
    """
    Cria um colormap interpolando entre duas cores.

    Parameters
    ----------
    color1 : str
        Cor inicial (hex, nome ou RGB)
    color2 : str
        Cor final
    name : str
        Nome do colormap
    n : int
        Número de níveis do colormap

    Returns
    -------
    cmap : LinearSegmentedColormap
    """

    cmap = LinearSegmentedColormap.from_list(
        name,
        [color1, color2],
        N=n
    )

    return cmap


DEFAULT_COLORMAP = create_two_color_colormap("#E02F1A", "#161C23")
```

Exemplo de amostragem de `n` cores a partir do colormap:

```python
colors = [DEFAULT_COLORMAP(x) for x in np.linspace(0, 1, n)]
```

### Implementação esperada para cores

Ao escrever código de visualização:

1. Desabilitar grid explicitamente.
2. Aplicar `DEFAULT_COLORS` como primeira opção.
3. Se necessário, criar e usar `DEFAULT_COLORMAP` com as cores `#E02F1A` e `#161C23`.
4. Revisar o gráfico final para garantir clareza, economia visual e boa interpretação.

---

## Diretriz exclusiva para histogramas de comparação

Todo histograma cujo objetivo seja **comparar duas distribuições** deve seguir um padrão único e obrigatório: usar `sns.histplot` com o parâmetro `hue` para separar os grupos comparados.

### Regra obrigatória para histogramas comparativos

* Sempre usar `sns.histplot` com `hue` quando o objetivo for comparar distribuições.
* Se os dados já estiverem em formato adequado, usar diretamente `sns.histplot(..., hue=...)`.
* Se os dados não estiverem em formato adequado, reestruturar a tabela para formato longo antes de gerar o gráfico.
* Não gerar histogramas comparativos com múltiplas chamadas independentes para cada grupo.
* O objetivo é garantir que todas as distribuições comparadas usem exatamente o mesmo esquema de bins.

### Justificativa técnica

O uso de `sns.histplot` com `hue` garante consistência entre os grupos comparados, especialmente em:

* largura dos bins;
* limites dos bins;
* alinhamento visual;
* comparabilidade entre distribuições.

Isso evita discrepâncias causadas por construções separadas, nas quais cada subconjunto poderia receber discretizações ligeiramente diferentes.

### Padrão de implementação

Estruturar os dados para que exista:

* uma coluna numérica com os valores;
* uma coluna categórica identificando o grupo/distribuição.

Exemplo recomendado:

```python
import seaborn as sns

sns.histplot(
    data=df_long,
    x="valor",
    hue="grupo",
    bins=30
)
```

### Reestruturação dos dados quando necessário

Quando as distribuições estiverem em colunas separadas, converter para formato longo antes de plotar:

```python
import pandas as pd
import seaborn as sns

df_long = df.melt(
    value_vars=["grupo_a", "grupo_b"],
    var_name="grupo",
    value_name="valor"
).dropna()

sns.histplot(
    data=df_long,
    x="valor",
    hue="grupo",
    bins=30
)
```

### Regras adicionais para histogramas comparativos

* Não usar grid.
* Definir `bins` explicitamente sempre que possível.
* Garantir que `hue` represente apenas a separação entre distribuições.
* Em histogramas comparativos, considerar `sns.histplot + hue` como abordagem padrão obrigatória.
* Se o histograma for de comparação, o uso de `sns.histplot` com `hue` não é opcional; caso os dados não permitam uso direto, reestruturar os dados até permitir.

---

## Resumo operacional obrigatório

Ao gerar qualquer visualização:

1. Não usar grid.
2. Priorizar clareza e boa relação ink/data.
3. Usar `DEFAULT_COLORS` como primeira escolha.
4. Se faltar cor para o número de categorias, usar `DEFAULT_COLORMAP`.
5. Em histogramas comparativos, usar obrigatoriamente `sns.histplot` com `hue`.
6. Se necessário, reestruturar os dados para formato longo antes de plotar histogramas comparativos.
7. Garantir consistência visual, estatística e interpretativa entre gráficos do mesmo contexto.
