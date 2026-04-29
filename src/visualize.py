"""
Visualisation des embeddings intermédiaires via t-SNE animé.
Prend les snapshots produits par train.py et génère un GIF frame par frame.

Auteur : S. Oussama
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE


CORA_CLASS_NAMES = [
    "Case Based",
    "Genetic Alg.",
    "Neural Nets",
    "Probabilistic",
    "Reinf. Learning",
    "Rule Learning",
    "Theory",
]

PALETTE = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
]


def _compute_tsne(
    embeddings: torch.Tensor,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    emb_np = embeddings.numpy().astype(np.float32)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=300,       # rapide pour l'animation
        init="pca",         # plus stable que l'init random
        learning_rate="auto",
    )
    coords = tsne.fit_transform(emb_np)
    return coords


def _render_frame(
    coords: np.ndarray,
    labels: np.ndarray,
    epoch: int,
    num_epochs: int,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))

    for cls_id, cls_name in enumerate(CORA_CLASS_NAMES):
        mask = labels == cls_id
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=PALETTE[cls_id], label=cls_name,
            s=8, alpha=0.7, linewidths=0,
        )

    ax.set_title(
        f"t-SNE des embeddings ChebGNN — Epoch {epoch}/{num_epochs}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=7, markerscale=2.0, framealpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE dim 1", fontsize=9)
    ax.set_ylabel("t-SNE dim 2", fontsize=9)

    fig.tight_layout()
    return fig


def generate_tsne_animation(
    emb_hist: List[Tuple[int, torch.Tensor]],
    labels: torch.Tensor,
    report_dir: str | Path = "report",
    perplexity: float = 30.0,
    fps: int = 2,
) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    labels_np = labels.numpy()
    num_epochs_total = emb_hist[-1][0] if emb_hist else 0

    print(f"\n{'='*60}")
    print(f" Génération du GIF t-SNE — {len(emb_hist)} frames")
    print(f"{'='*60}")

    frames: List[np.ndarray] = []

    for i, (epoch, embeddings) in enumerate(emb_hist):
        print(f"   Frame {i+1}/{len(emb_hist)} — époque {epoch} …")

        # graine différente par frame pour éviter les minima locaux répétés
        coords = _compute_tsne(embeddings, perplexity=perplexity, random_state=i)
        fig = _render_frame(coords, labels_np, epoch, num_epochs_total)

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame_arr = np.asarray(buf)
        frames.append(frame_arr[..., :3])  # RGBA -> RGB
        plt.close(fig)

    if not frames:
        print("[visualize] Aucune frame a générer — historique vide.")
        return

    _save_gif(frames, report_dir / "tsne_animation.gif", fps=fps)

    print(f"\n GIF sauvegardé : {(report_dir / 'tsne_animation.gif').resolve()}")
    print(f"{'='*60}\n")


def _save_gif(
    frames: List[np.ndarray],
    output_path: Path,
    fps: int = 2,
) -> None:
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow est requis pour générer le GIF. "
            "Installez-le avec : pip install Pillow"
        )

    duration_ms = int(1000 / fps)
    pil_frames = [Image.fromarray(f) for f in frames]

    pil_frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


if __name__ == "__main__":
    print("Ce module s'utilise via main.py --visualize")
