# Sincronizador de Tempos — Aciaria

## Visão Geral

O **Sincronizador** é um sistema para detectar e validar automaticamente os timestamps de início/fim de cada etapa do processo de aciaria, comparando os registros manuais do operador com detecções automáticas baseadas em sensores PIMS/OPC.

**Objetivo:** Substituir ou validar os tempos registrados manualmente, usando sinais de processo (peso, temperatura, vazão de gás, etc.) para detectar eventos com precisão de segundos.

---

## Arquitetura Geral

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         FLUXO DE PRODUÇÃO                                  │
│                                                                           │
│   ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐                              │
│   │ FEA │ ───► │ FP  │ ───► │ VD  │ ───► │ LC  │                              │
│   └─────┘     └─────┘     └─────┘     └─────┘                              │
│   Forno       Forno       Desgasei-   Lingota-                            │
│   Elétrico    Panela      ficação     mento                               │
│   a Arco                  a Vácuo     Contínuo                            │
└───────────────────────────────────────────────────────────────────────────┘
```

Cada etapa possui:
* **Dados do Operador:** Timestamps registrados manualmente (tabelas MES)
* **Dados de Sensores:** Séries temporais PIMS/OPC (segundo a segundo)
* **Intra-etapas:** Algoritmos de detecção baseados em tags específicas

---

## Status Atual do Sistema

O Sincronizador opera em **modo de desenvolvimento/validação**, com diferentes níveis de maturidade por etapa:

| Etapa | Status | Detecção | Comparação c/ Operador | Módulo Extraído |
| --- | --- | --- | --- | --- |
| **LC** | 🟢 Mais maduro | FSM por PESOBRA1/2 | ✅ Implementada | `lc_pesobra_logic.py` + `comparacao_operador.py` |
| **VD** | 🟢 Em desenvolvimento | Volume acumulado de gás (Tank 1/2) | ✅ Implementada | `vd_logic.py` |
| **FP** | 🟡 Em desenvolvimento | Gás total (Argônio + Nitrogênio) | ✅ Implementada | Inline nos notebooks |
| **FEA** | 🟡 Em desenvolvimento | Energia elétrica acumulada | ✅ Implementada | Inline nos notebooks |
| **Unificado** | 🟡 Em desenvolvimento | Pipeline com filas (FEA→FP→VD→LC) | Em progresso | `MVP v3` |

**Fluxo de trabalho atual:**
1. Os notebooks `Tags *` exploram os dados brutos de cada etapa (EDA, estatísticas, plots)
2. As funções de detecção são desenvolvidas e calibradas nos notebooks
3. Funções maduras são extraídas para módulos `.py` importáveis (`lc_pesobra_logic.py`, `vd_logic.py`)
4. O notebook `MVP v3` unifica todas as detecções em uma pipeline com simulação de filas
5. A comparação com o operador valida os resultados por proximidade temporal

---

## Estrutura de Arquivos

O projeto segue o padrão [Reproducible Data Science](https://khuyentran1401.github.io/reproducible-data-science/structure_project/introduction.html), com código reutilizável em `src/` e notebooks de exploração na raiz.

```
Sincronizador/
│
├── README.md                                       # Esta documentação
│
├── ─── src/ (MÓDULOS PYTHON IMPORTÁVEIS) ───────────
│
├── src/
│   ├── __init__.py                                 # Package raiz (versão)
│   ├── config.py                                   # Nomes de tabelas, constantes, parâmetros
│   │
│   ├── data/                                       # Leitura e pré-processamento
│   │   ├── __init__.py                             # Re-exporta funções principais
│   │   ├── read.py                                 # Leitura centralizada de tabelas
│   │   └── process.py                              # Pré-processamento e df_final
│   │
│   ├── detection/                                  # Algoritmos de detecção por etapa
│   │   ├── __init__.py                             # Re-exporta detectores
│   │   ├── lc_logic.py                             # FSM LC por PESOBRA1/2
│   │   └── vd_logic.py                             # Detecção VD por volume de gás
│   │
│   └── comparison/                                 # Comparação sistema vs operador
│       ├── __init__.py                             # Re-exporta comparacao_operador
│       └── operador.py                             # Match temporal sistema vs operador
│
├── ─── MÓDULOS LEGADOS (.py na raiz) ───────────────
│
├── lc_pesobra_logic.py                             # (Legado) → use src/detection/lc_logic.py
├── comparacao_operador.py                          # (Legado) → use src/comparison/operador.py
├── vd_logic.py                                     # (Legado) → use src/detection/vd_logic.py
│
├── ─── NOTEBOOKS PRINCIPAIS ────────────────────────
│
├── MVP v3                                          # Pipeline unificada (FEA+FP+VD+LC) com filas
├── MVP v1                                          # Primeira versão da pipeline unificada
├── Função Unificada                                # Versão intermediária com funções FP/VD/LC
│
├── ─── NOTEBOOKS DE EXPLORAÇÃO POR ETAPA ───────────
│
├── Tags Intra - LC                                 # Notebook principal do LC (PESOBRA FSM)
├── Tags LC                                         # Exploração inicial de tags LC (PESOREAL)
├── Tags FP                                         # Detecção FP por gás + comparação
├── Tags VD                                         # Detecção VD por volume + comparação
├── Tags FEA                                        # Análise de tags FEA (IBA/energia)
├── Tags IBA                                        # Ingestão de dados IBA (CSVs → Unity Catalog)
│
├── ─── NOTEBOOKS AUXILIARES ────────────────────────
│
├── Tabelas e Tags                                  # Referência: exploração de todas as tabelas
├── Obtenção de estatísticas de tempo de cada etapa  # Estatísticas de duração por etapa
├── TAG FEA                                         # (Reservado — vazio)
```

---

## Descrição dos Arquivos

### src/ — Módulos Reutilizáveis

A pasta `src/` contém todo o código reutilizável, organizado por responsabilidade:

| Módulo | Arquivo | Funções Principais | Descrição |
| --- | --- | --- | --- |
| **config** | `src/config.py` | Constantes | Nomes de tabelas, timezone, delays, colunas PIMS |
| **data.read** | `src/data/read.py` | `read_mes_tables()`, `read_opc_table()`, `read_pims_hist()`, `read_pims_with_opc()` | Leitura centralizada de todas as tabelas (MES, OPC, PIMS) |
| **data.process** | `src/data/process.py` | `build_df_final()`, `preprocess_pims_data()`, `prepare_lc_plot_data()`, `convert_datetime()` | Pré-processamento e construção do DataFrame unificado |
| **detection.lc_logic** | `src/detection/lc_logic.py` | `detect_lc_events_fsm_pesobra()` | FSM de detecção LC por PESOBRA1/2 |
| **detection.vd_logic** | `src/detection/vd_logic.py` | `detect_vd_events()` | Detecção de ciclos VD por volume de gás |
| **comparison.operador** | `src/comparison/operador.py` | `comparacao_operador()` | Match temporal entre sistema e operador |

#### Como usar nos notebooks

```python
import sys
sys.path.insert(0, "/Workspace/Users/execs1@ex.gerdau.com/Sincronizador")

# Leitura e processamento
from src.data.read import read_mes_tables, read_opc_table, read_pims_hist
from src.data.process import build_df_final, preprocess_pims_data, prepare_lc_plot_data

# Detecção
from src.detection.lc_logic import detect_lc_events_fsm_pesobra
from src.detection.vd_logic import detect_vd_events

# Comparação
from src.comparison.operador import comparacao_operador

# Exemplo completo
df_final = build_df_final(spark)                          # tabela unificada MES
df_pims = read_pims_hist(spark)                           # PIMS histórico
df_plot, lc_cols = prepare_lc_plot_data(df_pims, START, END)  # dados prontos p/ plot
df_sys, events, debug, feat = detect_lc_events_fsm_pesobra(df_plot, ...)
df_cmp = comparacao_operador(df_sys, df_final)
```

### Módulos Legados (raiz — serão removidos)

> Os arquivos `.py` na raiz do projeto são versões legadas. Use os módulos em `src/` para novos desenvolvimentos.

### Módulos Python (importáveis)

| Arquivo | Descrição | Entradas | Saídas |
| --- | --- | --- | --- |
| `lc_pesobra_logic.py` | FSM que detecta início/fim de corridas de LC usando peso dos braços da torre (PESOBRA1/2). Alterna automaticamente entre braços. | DataFrame Pandas com `timestamp`, `ACI@LC_TORRE_PESOBRA1`, `ACI@LC_TORRE_PESOBRA2` | `df_sys_lc`, `df_events`, `df_debug`, `df_features` |
| `comparacao_operador.py` | Match por proximidade temporal entre corridas detectadas e registros do operador. Genérico para qualquer etapa (FEA, FP, VD, LC). | `df_sys` (sistema), `df_operador` (MES) | DataFrame com deltas (segundos) de início e fim |
| `vd_logic.py` | Detecta ciclos de uso dos tanques de vácuo (Tank 1/2) com base no volume acumulado de gás. Inclui forward-fill streaming-safe e pré-processamento PIMS. | Spark DataFrame com colunas de volume N2/Ar para vasos 1/2 | `df_vd_final`, `df_vd_events`, `df_vd_debug` |

### Notebooks Principais (Pipeline)

| Notebook | Descrição |
| --- | --- |
| **MVP v3** | Versão mais recente da pipeline unificada. Contém as 4 funções de detecção (FEA por energia, FP por gás, VD por volume, LC por PESOBRA) + sistema de filas (`Event`, `Corrida`) que simula o fluxo FEA→FP→VD→LC com delays configuráveis. Função central: `run_unified_detector()`. |
| **MVP v1** | Primeira versão da pipeline. Mesma estrutura do MVP v3, mas com versões anteriores dos detectores (VD antigo, LC por peso torre simples). Usado como referência histórica. |
| **Função Unificada** | Versão intermediária. Contém `detect_fp_events_gas_based()`, `detect_fp_events_fsm_gas()` (streaming-ready), `detect_vd_events_gas_based()`, `detect_lc_events()`, e `build_status_corrida_table()` que gera a tabela única. Inclui funções de comparação FP old vs FSM. |

### Notebooks de Exploração por Etapa

| Notebook | Etapa | Descrição |
| --- | --- | --- |
| **Tags Intra - LC** | LC | Notebook principal do LC. Carrega dados OPC/PIMS, aplica pré-processamento, executa a FSM `detect_lc_events_fsm_pesobra()`, plota cada variável LC com marcadores de corrida, e compara com o operador via `comparacao_operador()`. Período configurável via `START_TS`/`END_TS`. |
| **Tags LC** | LC | Exploração inicial de tags do LC usando `ACI@LC_TORRE_PESOREAL`. Contém protótipos de detecção por threshold/histerese e slope, antes da lógica FSM por PESOBRA. Serviu como base para o `Tags Intra - LC`. |
| **Tags FP** | FP | Análise do Forno Panela. Contém `detect_fp_events_gas_based()` (detecção por gás total: Argônio + Nitrogênio dos carros 1/2), `prepare_fp_gas_series()`, `debug_fp_period()` (visualização), `build_fp_comparison_table()`, e comparação com dados oficiais do MES. |
| **Tags VD** | VD | Análise da Desgaseificação a Vácuo. Contém `detect_tank_usage_events()`, `debug_vd_period()` (visualização com marcadores oficiais), `get_vd_events_comparison()`. Usa tags de volume N2/Ar dos vasos 1/2 da `tb_aciaria_vd`. |
| **Tags FEA** | FEA | Análise das tags do Forno Elétrico. Carrega dados IBA (`hist_tags_IBA_novembro`) com 45 tags (correntes, energia, potência, inclinação, etc.), faz EDA (histogramas, séries temporais), e plota com marcadores de corrida do `df_final`. |
| **Tags IBA** | FEA | Ingestão de dados brutos IBA extraídos do sistema SCADA. Lê CSVs diários (1.csv a 29.csv), concatena, mapeia códigos de tag para nomes reais usando `equivalenciaTags.xlsx`, e salva como tabela Unity Catalog (`hist_tags_IBA_novembro`). |

### Notebooks Auxiliares

| Notebook | Descrição |
| --- | --- |
| **Tabelas e Tags** | Notebook de referência que explora todas as tabelas do projeto: `tb_aciaria_vd`, `tb_aciaria_fp`, `tb_admaci_lc_corridalingotamento`, `tb_admaci_fea_corridas`, `tb_aciaria_lc`. Para cada tabela: lista colunas, conta linhas, mostra range de datas, plota séries temporais. Também constrói o `df_final` juntando FEA+FP+VD+LC e documenta as variáveis construídas. |
| **Obtenção de estatísticas de tempo de cada etapa** | Análise estatística das durações de cada etapa do processo. Lê múltiplas tabelas MES (`tb_admaci_aci_retorno`, `tb_admaci_aci_corrida`, etc.), filtra por período, e gera sumários estatísticos (percentis, distribuições) dos tempos de processamento. |
| **TAG FEA** | Reservado para futuro desenvolvimento de detecção FEA. Atualmente vazio. |

---

## Fontes de Dados

### Tabelas do Operador (MES)

| Etapa | Tabela | Campos Principais |
| --- | --- | --- |
| FEA | `industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fea_corridas` | HRVAZAMENTO, TTT, DATA |
| FP | `industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fp_corrida` | HORACHEGADAFP, HORASAIDAFP, TTTFP |
| VD | `industrial_iatemperaturascha_refined_prd.computed_features.tb_corridatempos` | datahorasaidavd |
| LC | `industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_lc_corridalingotamento` | INICIOLINGOTAMENTO, FINALLINGOTAMENTO |

### Tabelas de Sensores (PIMS/OPC)

| Fonte | Tabela | Frequência | Uso |
| --- | --- | --- | --- |
| PIMS | `industrial_iatemperaturascha_refined_dev.computed_features.hist_tags_novembro_2025` | 1s | Tags FEA, LC, VD, FP (54 colunas) |
| PIMS IBA | `industrial_iatemperaturascha_refined_dev.computed_features.hist_tags_IBA_novembro` | 1min | Tags FEA específicas (45 colunas SCADA) |
| OPC LC | `industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_lc` | 1s | Peso braços torre, velocidade, NUMCORR |
| OPC VD | `industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_vd` | 1s | Volume gás, pressão vácuo |
| OPC FP | `industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_fp` | 1s | Volume gás Argônio/Nitrogênio carros 1/2 |

### Tabelas Auxiliares

| Tabela | Uso |
| --- | --- |
| `industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_aci_retorno` | Corridas com retorno (filtro) |
| `industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_aci_corrida` | Dados gerais de corrida |

---

## Etapas do Sincronizador

### 1. FEA — Forno Elétrico a Arco

**Status:** 🟡 Em Desenvolvimento

**Descrição:** Detectar início/fim do tratamento no forno elétrico.

**Algoritmo:** `detect_fea_events_fsm_energia()` (em MVP v1/v3) — FSM baseada na curva de energia elétrica acumulada (`ACI@FEA_ELET_ENERGIA`). Detecta início quando energia sai de zero (>200 por 10s) e fim quando atinge plateau alto.

**Intra-etapas:**
* [x] Detecção por energia elétrica acumulada
* [ ] Detecção por potência elétrica
* [ ] Detecção por temperatura
* [ ] Detecção por vazamento

**Campos gerados:**
* `inicio_fea` — Calculado como `hr_vaz_fea - TTT`
* `final_fea` — Igual a `hr_vaz_fea` (hora do vazamento)

---

### 2. FP — Forno Panela

**Status:** 🟡 Em Desenvolvimento

**Descrição:** Detectar chegada e saída da panela no forno panela.

**Algoritmo:** `detect_fp_events_gas_based()` e `detect_fp_events_fsm_gas()` (streaming-ready) — detecta ciclos de uso do FP pelo volume total de gás (Argônio + Nitrogênio) dos carros 1 e 2. Identifica picos de gás acima de 800 unidades e detecta início/fim por threshold relativo.

**Tags utilizadas:** `ACI@FP_Volume_Argonio_Carro_1`, `ACI@FP_Volume_Nitrogenio_Carro_1`, `ACI@FP_Volume_Argonio_Carro_2`, `ACI@FP_Volume_Nitrogenio_Carro_2`

**Intra-etapas:**
* [x] Detecção por gás total (Argônio + Nitrogênio)
* [ ] Detecção por presença de panela
* [ ] Detecção por temperatura
* [ ] Detecção por potência de aquecimento

**Campos gerados:**
* `inicio_fp` — Hora de chegada
* `final_fp` — Hora de saída
* `inicio_fp_tttfp` — Calculado como `saida_fp - TTTFP`

---

### 3. VD — Desgaseificação a Vácuo

**Status:** 🟢 Em Desenvolvimento

**Descrição:** Detectar ciclos de uso dos tanques de vácuo (Tank 1 e Tank 2).

**Algoritmo:** `detect_vd_events()` (em `vd_logic.py`) — FSM baseada no volume acumulado de gás N2/Ar dos vasos 1 e 2. Detecta reset (queda >99%) para identificar fim de ciclo, e atividade de fluxo (>0.24) para início. Inclui forward-fill streaming-safe.

**Tags utilizadas:** `ACI@VD_VASO1_VOLUMEN2`, `ACI@VD_VASO2_VOLUMEN2`, `ACI@VD_VASO1_VOLUMEAR`, `ACI@VD_VASO2_VOLUMEAR`

**Intra-etapas:**
* [x] Detecção por volume acumulado de gás (`vd_logic.py`)
* [ ] Detecção por pressão de vácuo

**Campos gerados:**
* `inicio_vd` — Calculado como `final_vd - TTTVD`
* `final_vd` — `datahorasaidavd`

---

### 4. LC — Lingotamento Contínuo

**Status:** 🟢 Mais Maduro

**Descrição:** Detectar início e fim de corridas de lingotamento usando dados dos braços da torre.

**Notebook principal:** `Tags Intra - LC`

#### Intra-etapas:

| Intra-etapa | Status | Tag/Sensor | Descrição |
| --- | --- | --- | --- |
| **PESOBRA1/2** | ✅ Implementado | `ACI@LC_TORRE_PESOBRA1`, `ACI@LC_TORRE_PESOBRA2` | Peso dos braços da torre |
| NUMCORR | 🟡 Planejado | `ACI@LC_GERAL_NUMCORR` | Número da corrida no sistema |
| VELOCIDADE | 🟡 Planejado | `ACI@LC_*_VELOCIDADE` | Velocidade de lingotamento |

**Campos gerados:**
* `inicio_lc_sys` — Detectado automaticamente
* `final_lc_sys` — Detectado automaticamente

---

## LC: Detecção por PESOBRA1/2 (FSM)

### Visão Geral

A função `detect_lc_events_fsm_pesobra` detecta automaticamente **início e fim de corridas** usando a curva de peso dos braços da torre, que alterna entre ~110t (cheio) e ~3t (vazio).

### Máquina de Estados (FSM)

```
┌──────┐    edge detectado     ┌──────────────┐   gap > 180s    ┌─────────┐
│ IDLE │ ──────────────────►   │ START_ARMED  │ ──────────────► │ RUNNING │
└──────┘                       └──────────────┘                 └─────────┘
    ▲                                                               │
    │                     peso sobe 2t após cruzar 60t              │
    └───────────────────────────────────────────────────────────────┘
```

| Estado | Descrição |
| --- | --- |
| **IDLE** | Aguardando candidato de início no braço atual |
| **START_ARMED** | Acumulando candidatos; fecha cluster após 180s sem novos |
| **RUNNING** | Corrida em andamento; monitorando para detectar fim |

### Pipeline de Features

| # | Feature | Descrição |
| --- | --- | --- |
| 1 | **Smoothing** | Mediana móvel 5s sobre PESOBRA |
| 2 | **R1 (Queda Brusca)** | `p[i] - p[i-8]` <= -2t, se esteve >100t nos últimos 60s |
| 3 | **R2 (Sequência)** | 5s consecutivos com `diff(p) < -0.05` |
| 4 | **Refractory** | Bloqueia 80s após salto de carga (+6t em 5s) |
| 5 | **Faixa Válida** | Candidato entre 80t e 115t |
| 6 | **Peak Gate** | (Opcional) Exige pico >108t nos últimos 60s |
| 7 | **Edge Detection** | Detecta transição False→True (flanco) |
| 8 | **Score** | Queda futura 10s * 2.0 + bônus R1(3.0) + bônus R2(2.0) |

### Lógica de Transições

**IDLE → START_ARMED:**
* Edge detectado no braço atual → abre cluster

**START_ARMED → RUNNING:**
* Gap 180s sem novos candidatos → fecha cluster
* Seleciona candidato com maior score
* Timestamp vira `inicio_lc_sys`

**RUNNING → IDLE:**
* Fase 1: Aguarda peso cruzar 60t para baixo
* Fase 2: Sliding reference — se sobe 2t em 25s → fim detectado
* Registra `final_lc_sys`, incrementa corrida, alterna braço

### Parâmetros Principais

| Parâmetro | Default | Descrição |
| --- | --- | --- |
| `DROP_TON` | 2.0 | Queda mínima (t) para R1 |
| `DROP_WINDOW_S` | 8 | Janela (s) para calcular queda R1 |
| `HIGH_WEIGHT` | 100.0 | Peso mínimo para "esteve carregado" |
| `REFRACTORY_AFTER_LOAD_S` | 80 | Bloqueio pós-carga (s) |
| `CLUSTER_GAP_S` | 180 | Gap para fechar cluster |
| `END_ARM_WEIGHT` | 60.0 | Peso que arma detecção de fim |
| `END_RISE_TON` | 2.0 | Subida mínima para confirmar fim |
| `initial_corrida` | - | Número da primeira corrida |
| `initial_braco` | 1 | Braço inicial (1 ou 2) |

### Retornos

| DataFrame | Descrição |
| --- | --- |
| `df_sys_lc` | Corridas detectadas (corrida, inicio_lc_sys, final_lc_sys, braco) |
| `df_debug` | Trace completo da FSM para debug |
| `df_features` | Features calculadas (r1, r2, edge, score, etc.) |

---

## Pipeline Unificada (MVP v3)

O notebook `MVP v3` integra todos os detectores em uma pipeline baseada em **simulação de filas**:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ detect_fea  │ ──► │ detect_fp   │ ──► │ detect_vd   │ ──► │ detect_lc   │
│ (energia)   │     │ (gás total) │     │ (volume N2) │     │ (PESOBRA)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
        │                  │                  │                  │
        └──────────────────┴──────────────────┴──────────────────┘
                                    │
                          ┌─────────────────┐
                          │ simulate_corridas│
                          │ (sistema filas)  │
                          └─────────────────┘
                                    │
                          ┌─────────────────┐
                          │   df_corridas   │
                          │ (tabela única)  │
                          └─────────────────┘
```

**Classes principais:**
* `Event` — Evento temporal (`ts`, `kind`: FEA_START, FP_END, etc.)
* `Corrida` — Representa uma corrida com timestamps de cada etapa

**Delays configuráveis:**
* FEA → FP: 3 minutos
* FP → VD: 5 minutos
* VD → LC: 2 minutos

**Função central:** `run_unified_detector(df_fea_raw, spark_fp_raw, spark_vd_raw, spark_lc_raw, start_ts, end_ts, initial_heat)`

---

## Comparação com Operador

A função `comparacao_operador()` faz **match por proximidade temporal** (não por número de corrida):

1. Ordena corridas do operador por `inicio_lc`
2. Para cada corrida, encontra a do sistema com `inicio_lc_sys` mais próximo
3. Aceita match se delta < 15 minutos
4. Calcula `delta_inicio_s` e `delta_fim_s` (negativo = sistema antecipou)

**Nota:** Os números de corrida diferem entre sistema e operador porque vêm de fontes independentes.

---

## Uso Típico

```python
# 1. Definir período
START_TS = "2025-11-20 10:00:00"
END_TS   = "2025-11-20 15:59:59"

# 2. Rodar FSM do LC
df_sys_lc, df_events, df_debug, df_feat = detect_lc_events_fsm_pesobra(
    df=df_lc_plot,
    initial_corrida=130268,
    initial_braco=1,
)

# 3. Comparar com operador
df_cmp_final = comparacao_operador(df_sys_lc, df_final)
```

Para alterar o período, modifique `START_TS` e `END_TS` na célula 6 do notebook `Tags Intra - LC`.

---

## Roadmap

* [x] LC: Detecção por PESOBRA1/2
* [x] LC: Comparação temporal com operador
* [x] LC: Plot de validação visual
* [x] FP: Detecção por gás total (Argônio + Nitrogênio)
* [x] FP: Comparação com dados oficiais MES
* [x] VD: Detecção por volume acumulado de gás
* [x] VD: Comparação com dados oficiais MES
* [x] FEA: Detecção por energia elétrica acumulada
* [x] Pipeline unificada MVP v3 com filas
* [ ] VD: Integrar vd_logic.py no fluxo principal
* [ ] FP: Extrair módulo .py dedicado
* [ ] FEA: Extrair módulo .py dedicado
* [ ] FEA: Detecção por potência/vazamento
* [ ] Pipeline unificado: df_final com todos os timestamps sincronizados
* [ ] App Streamlit: Dashboard de monitoramento