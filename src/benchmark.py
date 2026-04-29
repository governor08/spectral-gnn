"""
Benchmark comparatif : StrictSpectralConv vs ChebConvFromScratch.
Graphes Erdős-Rényi aléatoires, 3 forward passes par méthode (médiane).

Auteur : S. Oussama
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # pas de display requis
import matplotlib.pyplot as plt

from src.layers import StrictSpectralConv, ChebConvFromScratch
from src.graph_math import (
    build_adjacency,
    build_degree,
    normalized_laplacian,
    estimate_lambda_max,
    rescale_laplacian,
    to_dense,
)


NODE_SIZES: List[int] = [100, 500, 1000, 2000, 5000]
N_REPEATS: int = 3
IN_FEATURES: int = 32
OUT_FEATURES: int = 16
K: int = 3
EDGE_PROB: float = 0.05


def _random_edges(num_nodes: int, p: float) -> torch.Tensor:
    # triangulaire sup pour éviter les doublons, puis symétrie
    upper = torch.rand(num_nodes, num_nodes) < p
    upper = upper.triu(diagonal=1)
    adj_bool = upper | upper.T
    row, col = adj_bool.nonzero(as_tuple=True)
    edges = torch.stack([row, col], dim=0).long()
    return edges


def _build_operators(
    num_nodes: int,
    edges: torch.Tensor,
    dense: bool,
) -> torch.Tensor:
    A = build_adjacency(edges, num_nodes, add_self_loops=True)
    D = build_degree(A)
    L = normalized_laplacian(A, D, dense=dense)
    lam_max = estimate_lambda_max(L, num_iter=20)
    L_tilde = rescale_laplacian(L, lam_max)
    if dense and L_tilde.is_sparse:
        L_tilde = to_dense(L_tilde)
    return L_tilde


def _time_forward(
    layer: torch.nn.Module,
    x: torch.Tensor,
    op: torch.Tensor,
    n_repeats: int,
) -> float:
    times: List[float] = []
    layer.eval()

    with torch.no_grad():
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _ = layer(x, op)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    return float(np.median(times))


def run_benchmark(report_dir: str | Path = "report") -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    times_strict: List[float] = []
    times_cheb: List[float] = []

    print(f"\n{'='*60}")
    print(f" Benchmark : StrictSpectralConv vs ChebConvFromScratch")
    print(f" Tailles N = {NODE_SIZES} | K={K} | {N_REPEATS} répétitions")
    print(f"{'='*60}")

    for N in NODE_SIZES:
        print(f"\n N = {N} noeuds …")
        torch.manual_seed(0)

        edges = _random_edges(N, EDGE_PROB)
        x = torch.randn(N, IN_FEATURES)

        L_tilde_dense = _build_operators(N, edges, dense=True)
        strict_layer = StrictSpectralConv(IN_FEATURES, OUT_FEATURES)

        t_strict = _time_forward(strict_layer, x, L_tilde_dense, N_REPEATS)
        times_strict.append(t_strict)
        print(f"   StrictSpectral : {t_strict:.2f} ms (médiane)")

        # même opérateur pour que la comparaison soit équitable
        cheb_layer = ChebConvFromScratch(IN_FEATURES, OUT_FEATURES, K=K)
        t_cheb = _time_forward(cheb_layer, x, L_tilde_dense, N_REPEATS)
        times_cheb.append(t_cheb)
        print(f"   ChebConv(K={K}) : {t_cheb:.2f} ms (médiane)")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.loglog(
        NODE_SIZES, times_strict,
        marker="o", linewidth=2, markersize=7, color="#e74c3c",
        label="StrictSpectralConv (diagonalisation exacte, O(N^3))",
    )
    ax.loglog(
        NODE_SIZES, times_cheb,
        marker="s", linewidth=2, markersize=7, color="#2980b9",
        label=f"ChebConvFromScratch (K={K}, O(K·N·F))",
    )

    N_arr = np.array(NODE_SIZES, dtype=float)
    ref_cubic = (N_arr / NODE_SIZES[0]) ** 3 * times_strict[0]
    ref_linear = (N_arr / NODE_SIZES[0]) ** 1 * times_cheb[0]

    ax.loglog(NODE_SIZES, ref_cubic, linestyle="--", color="#c0392b", alpha=0.4, label="O(N^3) ref")
    ax.loglog(NODE_SIZES, ref_linear, linestyle="--", color="#1a6696", alpha=0.4, label="O(N) ref")

    ax.set_xlabel("Nombre de noeuds N", fontsize=12)
    ax.set_ylabel("Temps median de forward pass (ms)", fontsize=12)
    ax.set_title(
        "Benchmark spectral\nDiagonalisation exacte vs Approximation Tchebychev",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    out_path = report_dir / "benchmark.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n Graphique sauvegardé : {out_path.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_benchmark()
