"""Multi-objective Pareto Sustainability Optimization for European Soil-Water Nexus."""

from src.optimization.sustainability_allocator import (
    ParetoAllocator,
    OptimizationResult,
    NovelCrop,
    NOVEL_CROPS,
    LAND_USE_CLASSES,
    MANAGEMENT_PRACTICES,
)

__all__ = [
    "ParetoAllocator",
    "OptimizationResult",
    "NovelCrop",
    "NOVEL_CROPS",
    "LAND_USE_CLASSES",
    "MANAGEMENT_PRACTICES",
]
