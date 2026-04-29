"""
Point d'entrée du projet spectral-gnn.

Usage :
    python main.py                         # entraînement seul
    python main.py --benchmark             # + benchmark comparatif
    python main.py --visualize             # + GIF t-SNE
    python main.py --benchmark --visualize # tout

Auteur : S. Oussama
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.data_loader import load_cora
from src.graph_math import (
    build_adjacency,
    build_degree,
    normalized_laplacian,
    estimate_lambda_max,
    rescale_laplacian,
    to_dense,
)
from src.model import ChebGNN
from src.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ChebGNN from scratch — classification de noeuds sur Cora",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--report-dir", type=str, default="report")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = ROOT_DIR / args.data_dir
    report_dir = ROOT_DIR / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Chargement du dataset Cora …")
    data = load_cora(data_dir)
    print(f"      {data}")

    print("\n[2/4] Construction de A, D, L normalise et L_tilde …")
    A = build_adjacency(data.edges, data.num_nodes, add_self_loops=True)
    D = build_degree(A)

    # dense : N=2708 reste gérable (~58 Mo RAM)
    L = normalized_laplacian(A, D, dense=True)
    lambda_max = estimate_lambda_max(L, num_iter=50)
    L_tilde = rescale_laplacian(L, lambda_max)

    print(f"      lambda_max estimé : {lambda_max:.4f}")
    print(f"      L_tilde shape     : {L_tilde.shape}")

    print("\n[3/4] Initialisation du modèle ChebGNN …")
    model = ChebGNN(
        in_features=data.num_features,
        hidden_dim=args.hidden_dim,
        num_classes=data.num_classes,
        K=args.K,
        dropout=args.dropout,
    )
    print(f"      {model}")
    print(f"      Paramètres entraînables : {model.count_parameters():,}")

    print("\n[4/4] Entraînement …")
    model, history, emb_hist = train(
        model=model,
        data=data,
        L_tilde=L_tilde,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=5e-4,
        log_every=10,
        device=args.device,
    )

    if args.benchmark:
        print("\n[opt] Lancement du benchmark …")
        from src.benchmark import run_benchmark
        run_benchmark(report_dir=report_dir)

    if args.visualize:
        print("\n[opt] Génération du GIF t-SNE …")
        if not emb_hist:
            print("      Aucun embedding sauvegardé — activez log_every dans train().")
        else:
            from src.visualize import generate_tsne_animation
            generate_tsne_animation(
                emb_hist=emb_hist,
                labels=data.y,
                report_dir=report_dir,
                perplexity=30.0,
                fps=2,
            )

    print("\n Projet spectral-gnn — execution terminee.")
    print(f" Figures disponibles dans : {report_dir.resolve()}\n")


if __name__ == "__main__":
    main()
