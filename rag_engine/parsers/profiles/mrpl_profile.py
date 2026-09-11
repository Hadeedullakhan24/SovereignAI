"""Mangalore Refinery and Petrochemicals Limited (MRPL) domain profile."""

from rag_engine.parsers.profiles.generic_refinery_profile import GenericRefineryProfile
from rag_engine.schemas.parsed_document import EquipmentType


class MRPLProfile(GenericRefineryProfile):
    """MRPL-specific refinery profile for Mangalore Refinery complex (Phases I, II, III)."""

    def __init__(self, **kwargs) -> None:
        mrpl_kwargs = {
            "profile_name": "mrpl",
            "refinery_name": "Mangalore Refinery and Petrochemicals Limited (MRPL)",
            "plant_units": [
                "CDU-1", "CDU-2", "CDU-3", "VDU-1", "VDU-2", "VDU-3",
                "DCU", "PFCC", "NHT", "CCR", "OHCU", "HGU", "DHDT",
                "ISOM", "SPM", "ALKY", "GDS", "SRU-1", "SRU-2", "SRU-3",
                "CPP", "ETP", "OM&S", "PHASE-1", "PHASE-2", "PHASE-3"
            ],
            "standard_names": [
                "OISD", "OISD-105", "OISD-116", "OISD-117", "OISD-155", "OISD-166",
                "PNGRB", "PNGRB T4S", "API 610", "API 650", "API 520", "API 526",
                "API 510", "API 570", "ASME B31.3", "ASME SEC VIII", "ASME",
                "ISO 9001", "ISO 14001", "ISO 45001", "IBR", "PESO", "FACTORY ACT"
            ],
        }
        # Update with MRPL specific prefix additions
        extra_prefixes = {
            "PSV-": EquipmentType.VALVE,
            "MOV-": EquipmentType.VALVE,
            "SDV-": EquipmentType.VALVE,
            "BDV-": EquipmentType.VALVE,
            "XV-": EquipmentType.VALVE,
            "P-": EquipmentType.PUMP,
            "K-": EquipmentType.COMPRESSOR,
            "C-": EquipmentType.COMPRESSOR,
            "E-": EquipmentType.HEAT_EXCHANGER,
            "HX-": EquipmentType.HEAT_EXCHANGER,
            "V-": EquipmentType.PRESSURE_VESSEL,
            "D-": EquipmentType.PRESSURE_VESSEL,
            "TK-": EquipmentType.TANK,
            "F-": EquipmentType.FURNACE,
            "H-": EquipmentType.FURNACE,
            "B-": EquipmentType.FURNACE,
            "LIC-": EquipmentType.INSTRUMENT,
            "PIC-": EquipmentType.INSTRUMENT,
            "TIC-": EquipmentType.INSTRUMENT,
            "FIC-": EquipmentType.INSTRUMENT,
            "PIT-": EquipmentType.INSTRUMENT,
            "TIT-": EquipmentType.INSTRUMENT,
            "LIT-": EquipmentType.INSTRUMENT,
            "FIT-": EquipmentType.INSTRUMENT,
        }
        mrpl_kwargs.update(kwargs)
        super().__init__(**mrpl_kwargs)
        # Combine prefixes
        combined_prefixes = dict(self.equipment_type_prefixes)
        combined_prefixes.update(extra_prefixes)
        object.__setattr__(self, "equipment_type_prefixes", combined_prefixes)
