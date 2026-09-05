"""
Two graph convolution layers, built from scratch.

StrictSpectralConv  — the theoretically exact spectral filter via full eigendecomposition.
                      O(N³). Included so you can see exactly what ChebConv approximates.

ChebConvFromScratch — the practical version from Defferrard et al. (2016).
                      Approximates the spectral filter with Chebyshev polynomials.
                      O(K·|E|). This is what actually trains on Cora.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class StrictSpectralConv(nn.Module):
    """Exact spectral convolution: y = U · diag(g_θ(λ)) · Uᵀ · x.

    Computes the full eigendecomposition of L at every forward pass — O(N³).
    Impractical at scale, but gives a clean reference to verify ChebConv against.
    """

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
        # eigh requires a dense matrix — pass L as dense when using this layer
        eigenvalues, U = torch.linalg.eigh(L)  # O(N³)

        x_hat = U.T @ x  # project signal into graph Fourier basis

        # learned filter: sigmoid gates each frequency component
        x_filtered = x_hat * torch.sigmoid(eigenvalues.unsqueeze(1))
        x_proj = x_filtered @ self.theta

        out = U @ x_proj + self.bias  # back to node domain
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class ChebConvFromScratch(nn.Module):
    """Chebyshev spectral convolution: y = Σ θ_k · T_k(L̃) · x.

    Instead of computing U explicitly, we apply Chebyshev polynomials of L̃ directly.
    T_0 captures the node itself, T_1 its immediate neighbors, T_k neighbors k hops away.
    The model learns how much weight to give each neighborhood depth via θ_k.
    """

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

        # one projection matrix per polynomial order, stacked into a single 3D parameter
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
        # Chebyshev recursion: T_0(L̃)x = x, T_1(L̃)x = L̃x, T_k = 2L̃·T_{k-1} - T_{k-2}
        T_prev = x
        T_curr = self._cheb_mult(L_tilde, x)

        out = T_prev @ self.weight[0]

        if self.K > 1:
            out = out + T_curr @ self.weight[1]

        for k in range(2, self.K):
            T_next = 2.0 * self._cheb_mult(L_tilde, T_curr) - T_prev
            out = out + T_next @ self.weight[k]
            # sliding window — only the last two terms are needed at each step
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
