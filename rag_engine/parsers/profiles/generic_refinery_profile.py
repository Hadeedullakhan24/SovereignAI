"""Generic refinery profile adhering to international hydrocarbon and petrochemical standards."""

from rag_engine.parsers.profiles.base_profile import RefineryProfile
from rag_engine.schemas.parsed_document import EquipmentType


class GenericRefineryProfile(RefineryProfile):
    """Generic refinery profile matching industry standard tags and units."""

    def __init__(self, **kwargs) -> None:
        defaults = {
            "profile_name": "generic",
            "refinery_name": "Generic Hydrocarbon Refinery",
            "plant_units": [
                "CDU", "VDU", "FCCU", "HCU", "CRU", "SRU", "DHDS", "MSU",
                "ATU", "SWS", "ARU", "PSA", "ETP", "CPP", "OFFSITES", "UTILITIES"
            ],
            "equipment_tag_patterns": [
                r"\b[A-Z]{1,4}-\d{2,5}[A-Z]?\b",                # Standard P-101A, MOV-2001
                r"\b[A-Z]{1,3}\d{3,5}[A-Z]?\b",                  # P101A, V201
                r"\b[A-Z]{1,4}_\d{2,5}[A-Z]?\b",                # Pump_101
                r"\bPIPE-\d{2,4}-[A-Z0-9-]+\b",                 # PIPE-101-CS
                r"\bLINE-\d{2,4}-[A-Z0-9-]+\b",                 # LINE-202
            ],
            "equipment_type_prefixes": {
                "P-": EquipmentType.PUMP,
                "P": EquipmentType.PUMP,
                "V-": EquipmentType.VALVE,
                "V": EquipmentType.VALVE,
                "MOV-": EquipmentType.VALVE,
                "PRV-": EquipmentType.VALVE,
                "PSV-": EquipmentType.VALVE,
                "CV-": EquipmentType.VALVE,
                "FCV-": EquipmentType.VALVE,
                "PCV-": EquipmentType.VALVE,
                "M-": EquipmentType.MOTOR,
                "M": EquipmentType.MOTOR,
                "K-": EquipmentType.COMPRESSOR,
                "C-": EquipmentType.COMPRESSOR,
                "HX-": EquipmentType.HEAT_EXCHANGER,
                "E-": EquipmentType.HEAT_EXCHANGER,
                "HE-": EquipmentType.HEAT_EXCHANGER,
                "D-": EquipmentType.PRESSURE_VESSEL,
                "TK-": EquipmentType.TANK,
                "T-": EquipmentType.TANK,
                "F-": EquipmentType.FURNACE,
                "H-": EquipmentType.FURNACE,
                "L-": EquipmentType.PIPELINE,
                "PL-": EquipmentType.PIPELINE,
                "PIT-": EquipmentType.INSTRUMENT,
                "TIT-": EquipmentType.INSTRUMENT,
                "FIT-": EquipmentType.INSTRUMENT,
                "LIT-": EquipmentType.INSTRUMENT,
                "PT-": EquipmentType.INSTRUMENT,
                "TT-": EquipmentType.INSTRUMENT,
                "FT-": EquipmentType.INSTRUMENT,
                "LT-": EquipmentType.INSTRUMENT,
            },
            "equipment_type_keywords": {
                "pump": EquipmentType.PUMP,
                "centrifugal pump": EquipmentType.PUMP,
                "valve": EquipmentType.VALVE,
                "gate valve": EquipmentType.VALVE,
                "globe valve": EquipmentType.VALVE,
                "motor": EquipmentType.MOTOR,
                "induction motor": EquipmentType.MOTOR,
                "compressor": EquipmentType.COMPRESSOR,
                "heat exchanger": EquipmentType.HEAT_EXCHANGER,
                "condenser": EquipmentType.HEAT_EXCHANGER,
                "cooler": EquipmentType.HEAT_EXCHANGER,
                "reboiler": EquipmentType.HEAT_EXCHANGER,
                "pipeline": EquipmentType.PIPELINE,
                "pipe": EquipmentType.PIPELINE,
                "vessel": EquipmentType.PRESSURE_VESSEL,
                "drum": EquipmentType.PRESSURE_VESSEL,
                "column": EquipmentType.PRESSURE_VESSEL,
                "tower": EquipmentType.PRESSURE_VESSEL,
                "reactor": EquipmentType.PRESSURE_VESSEL,
                "furnace": EquipmentType.FURNACE,
                "heater": EquipmentType.FURNACE,
                "boiler": EquipmentType.FURNACE,
                "tank": EquipmentType.TANK,
                "storage tank": EquipmentType.TANK,
                "transmitter": EquipmentType.INSTRUMENT,
                "indicator": EquipmentType.INSTRUMENT,
                "gauge": EquipmentType.INSTRUMENT,
            },
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
