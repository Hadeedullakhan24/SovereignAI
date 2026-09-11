"""Refinery profile module for multi-tenant and multi-refinery deployment."""

from typing import Dict, Type

from rag_engine.parsers.profiles.base_profile import RefineryProfile
from rag_engine.parsers.profiles.generic_refinery_profile import GenericRefineryProfile
from rag_engine.parsers.profiles.mrpl_profile import MRPLProfile
from rag_engine.parsers.exceptions import ProfileError

_PROFILE_REGISTRY: Dict[str, Type[RefineryProfile]] = {
    "generic": GenericRefineryProfile,
    "mrpl": MRPLProfile,
}


def get_profile(name: str = "mrpl") -> RefineryProfile:
    """Retrieve an instantiated refinery profile by key ('generic', 'mrpl', etc.)."""
    key = name.lower().strip()
    profile_cls = _PROFILE_REGISTRY.get(key)
    if not profile_cls:
        raise ProfileError(f"Refinery profile '{name}' not found. Available: {list(_PROFILE_REGISTRY.keys())}")
    return profile_cls()


def register_profile(key: str, profile_cls: Type[RefineryProfile]) -> None:
    """Register a new refinery profile (e.g. HPCL, IOCL, ONGC, BPCL, Reliance)."""
    _PROFILE_REGISTRY[key.lower().strip()] = profile_cls


__all__ = [
    "RefineryProfile",
    "GenericRefineryProfile",
    "MRPLProfile",
    "get_profile",
    "register_profile",
]
