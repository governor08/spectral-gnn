"""
ChebGNN : deux couches ChebConvFromScratch pour la classification de noeuds sur Cora.
Architecture : ChebConv -> ReLU -> Dropout -> ChebConv -> LogSoftmax

Auteur : S. Oussama
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import ChebConvFromScratch


class ChebGNN(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        num_classes: int = 7,
        K: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.K = K
        self.dropout_rate = dropout

        self.conv1 = ChebConvFromScratch(in_features, hidden_dim, K=K)
        self.conv2 = ChebConvFromScratch(hidden_dim, num_classes, K=K)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        L_tilde: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv1(x, L_tilde)
        h = F.relu(h)
        embeddings = h.detach()  # snapshot pour visualisation, sans graph autograd
        h = self.dropout(h)

        out = self.conv2(h, L_tilde)
        log_probs = F.log_softmax(out, dim=1)

        return log_probs, embeddings

    def predict(self, x: torch.Tensor, L_tilde: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            log_probs, _ = self.forward(x, L_tilde)
            preds = log_probs.argmax(dim=1)
        return preds

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, "
            f"hidden={self.hidden_dim}, "
            f"classes={self.num_classes}, "
            f"K={self.K}, "
            f"dropout={self.dropout_rate}"
        )
