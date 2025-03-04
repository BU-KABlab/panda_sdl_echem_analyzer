"""
Electrochemical Analysis Library

A library for analyzing and visualizing data from electrochemical experiments.
"""

from .analyzer import ElectrochemicalAnalyzer
from .colormaps import ColorMapGenerator
from .plotters import CAPlotter, CVPlotter, OCPPlotter
from .utils import modify_current_density

__all__ = [
    "ElectrochemicalAnalyzer",
    "ColorMapGenerator",
    "CVPlotter",
    "CAPlotter",
    "OCPPlotter",
    "modify_current_density",
]
