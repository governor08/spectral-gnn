"""
Deux couches de convolution spectrale from scratch :
  StrictSpectralConv  — filtre exact par diagonalisation
  ChebConvFromScratch — approximation polynomiale de Tchebychev

Auteur : S. Oussama
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class StrictSpectralConv(nn.Module):
    """y = U * diag(g_theta(lambda)) * U^T * x  — diagonalisation exacte O(N^3)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.theta = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.theta)

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        # L doit être dense — eigh ne supporte pas le format sparse
        eigenvalues, U = torch.linalg.eigh(L)  # O(N^3)

        x_hat = U.T @ x

        # filtre spectral appris : sigma(theta) module chaque fréquence
        x_filtered = x_hat * torch.sigmoid(eigenvalues.unsqueeze(1))
        x_proj = x_filtered @ self.theta

        out = U @ x_proj + self.bias
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class ChebConvFromScratch(nn.Module):
    """y = sum_{k=0}^{K-1} theta_k * T_k(L_tilde) * x  — récurrence Tchebychev."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        K: int = 3,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K

        # K matrices de projection empilées dans un seul paramètre 3D
        self.weight = nn.Parameter(torch.empty(K, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for k in range(self.K):
            nn.init.kaiming_uniform_(self.weight[k], a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def _cheb_mult(L_tilde: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if L_tilde.is_sparse:
            return torch.sparse.mm(L_tilde, v)
        return L_tilde @ v

    def forward(self, x: torch.Tensor, L_tilde: torch.Tensor) -> torch.Tensor:
        # T_0(L)x = x,  T_1(L)x = Lx,  T_k(L)x = 2L*T_{k-1} - T_{k-2}
        T_prev = x
        T_curr = self._cheb_mult(L_tilde, x)

        out = T_prev @ self.weight[0]

        if self.K > 1:
            out = out + T_curr @ self.weight[1]

        for k in range(2, self.K):
            T_next = 2.0 * self._cheb_mult(L_tilde, T_curr) - T_prev
            out = out + T_next @ self.weight[k]
            # fenêtre glissante : on ne garde que les 2 derniers termes
            T_prev = T_curr
            T_curr = T_next

        out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"K={self.K}"
        )
