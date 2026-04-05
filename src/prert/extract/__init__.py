"""Regulation-specific extractors for Phase 1."""

from .gdpr_parser import parse_gdpr_controls, parse_gdpr_controls_from_text
from .iso_parser import parse_iso_controls, parse_iso_controls_from_text
from .nist_parser import parse_nist_controls, parse_nist_controls_from_text

__all__ = [
    "parse_gdpr_controls",
    "parse_gdpr_controls_from_text",
    "parse_iso_controls",
    "parse_iso_controls_from_text",
    "parse_nist_controls",
    "parse_nist_controls_from_text",
]
