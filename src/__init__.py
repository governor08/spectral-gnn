"""
src/__init__.py
Exposition des modules principaux du package spectral-gnn.
"""

from src.data_loader import load_cora, CoraData
from src.graph_math import (
    build_adjacency,
    build_degree,
    normalized_laplacian,
    estimate_lambda_max,
    rescale_laplacian,
    to_dense,
)
from src.layers import StrictSpectralConv, ChebConvFromScratch
from src.model import ChebGNN
from src.train import train

__all__ = [
    "load_cora",
    "CoraData",
    "build_adjacency",
    "build_degree",
    "normalized_laplacian",
    "estimate_lambda_max",
    "rescale_laplacian",
    "to_dense",
    "StrictSpectralConv",
    "ChebConvFromScratch",
    "ChebGNN",
    "train",
]
