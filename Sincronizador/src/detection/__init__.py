"""Modulo de deteccao automatica de eventos por etapa."""

from src.detection.lc_logic import detect_lc_events_fsm_pesobra
from src.detection.lc_peso_bra import detect_lc_chegada_torre
from src.detection.lc_peso_real import detect_lc_peso_real
from src.detection.vd_logic import detect_vd_events
from src.detection.fea_energia import detect_fea_energia
from src.detection.fea_peso_carro_panela import detect_peso_carro_panela
from src.detection.fea_inclinom import detect_fea_inclinom

__all__ = [
    "detect_lc_events_fsm_pesobra",
    "detect_lc_chegada_torre",
    "detect_lc_peso_real",
    "detect_vd_events",
    "detect_fea_energia",
    "detect_peso_carro_panela",
    "detect_fea_inclinom",
]
