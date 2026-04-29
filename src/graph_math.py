"""
Opérateurs spectraux sur graphes : adjacence, degrés, Laplacien normalisé, rescaling.
Toutes les fonctions supportent sparse et dense.

Auteur : S. Oussama
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_adjacency(
    edges: torch.Tensor,
    num_nodes: int,
    add_self_loops: bool = True,
) -> torch.Tensor:
    row, col = edges[0], edges[1]

    if add_self_loops:
        self_loop_idx = torch.arange(num_nodes, dtype=edges.dtype, device=edges.device)
        row = torch.cat([row, self_loop_idx], dim=0)
        col = torch.cat([col, self_loop_idx], dim=0)

    values = torch.ones(row.size(0), dtype=torch.float32, device=edges.device)
    indices = torch.stack([row, col], dim=0)

    A = torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_nodes, num_nodes),
        dtype=torch.float32,
        device=edges.device,
    ).coalesce()

    return A


def build_degree(A: torch.Tensor) -> torch.Tensor:
    """d_i = sum_j A_{ij}"""
    if A.is_sparse:
        # torch.sparse COO ne supporte pas sum(dim=1) directement
        A_coo = A.coalesce()
        row_indices = A_coo.indices()[0]
        values = A_coo.values()
        D = torch.zeros(A.shape[0], dtype=torch.float32, device=A.device)
        D.scatter_add_(0, row_indices, values)
    else:
        D = A.sum(dim=1)

    return D


def normalized_laplacian(
    A: torch.Tensor,
    D: torch.Tensor,
    dense: bool = True,
) -> torch.Tensor:
    """L_norm = I - D^{-1/2} A D^{-1/2}"""
    N = D.shape[0]
    device = D.device

    # clamp pour éviter la division par zéro sur les noeuds isolés
    D_inv_sqrt = torch.where(D > 0, D.pow(-0.5), torch.zeros_like(D))

    if A.is_sparse:
        A_coo = A.coalesce()
        row, col = A_coo.indices()
        vals = A_coo.values()

        norm_vals = D_inv_sqrt[row] * vals * D_inv_sqrt[col]

        A_norm_sparse = torch.sparse_coo_tensor(
            torch.stack([row, col], dim=0),
            norm_vals,
            size=(N, N),
            dtype=torch.float32,
            device=device,
        ).coalesce()

        if dense:
            A_norm_dense = A_norm_sparse.to_dense()
            L = torch.eye(N, dtype=torch.float32, device=device) - A_norm_dense
        else:
            diag_idx = torch.arange(N, device=device)
            diag_vals = torch.ones(N, dtype=torch.float32, device=device)

            i_row = torch.cat([row, diag_idx], dim=0)
            i_col = torch.cat([col, diag_idx], dim=0)
            i_vals = torch.cat([-norm_vals, diag_vals], dim=0)

            L = torch.sparse_coo_tensor(
                torch.stack([i_row, i_col], dim=0),
                i_vals,
                size=(N, N),
                dtype=torch.float32,
                device=device,
            ).coalesce()
    else:
        A_norm = D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)
        L = torch.eye(N, dtype=torch.float32, device=device) - A_norm

    return L


def estimate_lambda_max(
    L: torch.Tensor,
    num_iter: int = 50,
    tol: float = 1e-6,
) -> float:
    """Itération de puissance pour estimer la valeur propre dominante de L."""
    N = L.shape[0]
    device = L.device if not L.is_sparse else L.coalesce().values().device

    torch.manual_seed(42)
    v = torch.randn(N, 1, dtype=torch.float32, device=device)
    v = v / v.norm()

    lambda_max = 0.0

    for _ in range(num_iter):
        if L.is_sparse:
            w = torch.sparse.mm(L, v)
        else:
            w = L @ v

        lambda_new = float((v.T @ w).item())
        v = w / (w.norm() + 1e-12)

        if abs(lambda_new - lambda_max) / (abs(lambda_max) + 1e-12) < tol:
            lambda_max = lambda_new
            break
        lambda_max = lambda_new

    return float(lambda_max)


def rescale_laplacian(
    L: torch.Tensor,
    lambda_max: float,
) -> torch.Tensor:
    """L_tilde = (2 / lambda_max) * L - I  =>  valeurs propres dans [-1, 1]"""
    N = L.shape[0]
    device = L.device if not L.is_sparse else L.coalesce().values().device

    scale = 2.0 / (lambda_max + 1e-12)

    if L.is_sparse:
        L_coo = L.coalesce()
        row, col = L_coo.indices()
        vals = L_coo.values()

        scaled_vals = scale * vals

        diag_idx = torch.arange(N, device=device)
        diag_vals = -torch.ones(N, dtype=torch.float32, device=device)

        i_row = torch.cat([row, diag_idx], dim=0)
        i_col = torch.cat([col, diag_idx], dim=0)
        i_vals = torch.cat([scaled_vals, diag_vals], dim=0)

        L_tilde = torch.sparse_coo_tensor(
            torch.stack([i_row, i_col], dim=0),
            i_vals,
            size=(N, N),
            dtype=torch.float32,
            device=device,
        ).coalesce()
    else:
        L_tilde = scale * L - torch.eye(N, dtype=torch.float32, device=device)

    return L_tilde


def to_dense(M: torch.Tensor) -> torch.Tensor:
    if M.is_sparse:
        return M.to_dense()
    return M
