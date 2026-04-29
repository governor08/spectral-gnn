"""
Boucle d'entraînement pour ChebGNN sur Cora.
Adam, NLLLoss, 200 époques, snapshots d'embeddings toutes les log_every époques.

Auteur : S. Oussama
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import time

import torch
import torch.nn as nn
import torch.optim as optim

from src.model import ChebGNN
from src.data_loader import CoraData


EmbeddingHistory = List[Tuple[int, torch.Tensor]]
TrainingHistory = Dict[str, List[float]]


def _accuracy(
    log_probs: torch.Tensor,
    y: torch.Tensor,
    mask: torch.BoolTensor,
) -> float:
    preds = log_probs.argmax(dim=1)
    correct = (preds[mask] == y[mask]).sum()
    return float(correct.item()) / float(mask.sum().item())


def train(
    model: ChebGNN,
    data: CoraData,
    L_tilde: torch.Tensor,
    num_epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    log_every: int = 10,
    device: str = "cpu",
) -> Tuple[ChebGNN, TrainingHistory, EmbeddingHistory]:
    torch_device = torch.device(device)
    model = model.to(torch_device)

    X = data.X.to(torch_device)
    y = data.y.to(torch_device)
    train_mask = data.train_mask.to(torch_device)
    val_mask = data.val_mask.to(torch_device)
    test_mask = data.test_mask.to(torch_device)
    L_tilde = L_tilde.to(torch_device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.NLLLoss()

    history: TrainingHistory = {"train_loss": [], "val_acc": [], "test_acc": []}
    emb_hist: EmbeddingHistory = []

    print(f"\n{'='*60}")
    print(f" Entraînement ChebGNN — {model.count_parameters()} paramètres")
    print(f" Device : {torch_device} | Epoques : {num_epochs} | lr={lr}")
    print(f"{'='*60}")

    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        log_probs, embeddings = model(X, L_tilde)
        loss = criterion(log_probs[train_mask], y[train_mask])

        loss.backward()
        optimizer.step()

        train_loss = float(loss.item())

        model.eval()
        with torch.no_grad():
            log_probs_eval, embeddings_eval = model(X, L_tilde)
            val_acc = _accuracy(log_probs_eval, y, val_mask)
            test_acc = _accuracy(log_probs_eval, y, test_mask)

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

        if epoch % log_every == 0:
            emb_hist.append((epoch, embeddings_eval.cpu().clone()))
            elapsed = time.time() - t_start
            print(
                f"Epoch {epoch:>3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Acc: {val_acc*100:.2f}% | "
                f"Test Acc: {test_acc*100:.2f}% | "
                f"[{elapsed:.1f}s]"
            )

    total_time = time.time() - t_start
    final_test_acc = history["test_acc"][-1]
    print(f"\n Entraînement termine en {total_time:.1f}s")
    print(f" Test Accuracy finale : {final_test_acc*100:.2f}%")
    print(f"{'='*60}\n")

    return model, history, emb_hist
