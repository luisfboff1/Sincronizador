"""
config.py

Nomes de tabelas, constantes e parâmetros centralizados do Sincronizador.
Toda referência a tabelas ou constantes globais deve vir deste módulo.
"""

# ======================================================================
# TABELAS MES (Registros do Operador)
# ======================================================================
TABLE_FEA_CORRIDAS = (
    "industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fea_corridas"
)
TABLE_FP_CORRIDA = (
    "industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_fp_corrida"
)
TABLE_VD_CORRIDA_TEMPOS = (
    "industrial_iatemperaturascha_refined_prd.computed_features.tb_corridatempos"
)
TABLE_LC_CORRIDALINGOTAMENTO = (
    "industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_lc_corridalingotamento"
)

# ======================================================================
# TABELAS OPC (Sensores segundo-a-segundo)
# ======================================================================
TABLE_OPC_LC = "industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_lc"
TABLE_OPC_VD = "industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_vd"
TABLE_OPC_FP = "industrial_trusted_prd.gsb_cha_opc_aciaria.tb_aciaria_fp"

# ======================================================================
# TABELA CSV (Tags PIMS exportadas)
# ======================================================================
TABLE_CSV = "industrial_composicaoquimicacha_refined_dev.csv_valid.csv_concat"

# ======================================================================
# TABELAS PIMS (Histórico de Tags)
# ======================================================================
TABLE_PIMS_HIST_TAGS = (
    "industrial_iatemperaturascha_refined_dev"
    ".computed_features.hist_tags_novembro_2025"
)
TABLE_PIMS_HIST_IBA = (
    "industrial_iatemperaturascha_refined_dev"
    ".computed_features.hist_tags_IBA_novembro"
)

# ======================================================================
# TABELAS AUXILIARES
# ======================================================================
TABLE_ACI_RETORNO = (
    "industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_aci_retorno"
)
TABLE_ACI_CORRIDA = (
    "industrial_trusted_prd.gsb_cha_mes_aciaria.tb_admaci_aci_corrida"
)

# ======================================================================
# TIMEZONE
# ======================================================================
TIMEZONE = "America/Sao_Paulo"
UTC_OFFSET_HOURS = 3

# ======================================================================
# DELAYS ENTRE ETAPAS (minutos) — usado no simulador de filas
# ======================================================================
DELAY_FEA_FP_MIN = 3
DELAY_FP_VD_MIN = 5
DELAY_VD_LC_MIN = 2

# ======================================================================
# TEMPOS MÍNIMOS DE TRANSPORTE (minutos)
# ======================================================================
MIN_TIME_TRANSPORT_FEA_FP = 3
MIN_TIME_FP = 19
MIN_TIME_TRANSPORT_FP_VD = 3
MIN_TIME_VD = 2

# ======================================================================
# COLUNAS PIMS POR ETAPA
# ======================================================================
VD_COLUMNS_TO_FILL = [
    "ACI@VD_VASO1_VOLUMEN2",
    "ACI@VD_VASO2_VOLUMEN2",
    "ACI@VD_VASO1_VOLUMEAR",
    "ACI@VD_VASO2_VOLUMEAR",
]

PIMS_TIMESTAMP_COL = "timestamp"

# ======================================================================
# COLUNAS LC (tags de interesse)
# ======================================================================
LC_COLS_CSV = [
    "ACI@LC_TORRE_PESOBRA1",
    "ACI@LC_TORRE_PESOBRA2",
]

LC_COLS_OPC = [
    "ACI@LC_TORRE_PESOREAL",
    "ACI@LC_DISTRIB_PESO",
    "ACI@LC_DISTRIB_TEMP",
    "ACI@LC_GERAL_NUMCORR",
    "ACI@LC_EXTEND_VEL2V1",
    "ACI@LC_EXTEND_VEL2V2",
    "ACI@LC_EXTEND_VEL2V3",
]
