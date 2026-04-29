"""
Téléchargement et parsing du dataset Cora depuis l'URL officielle LINQS.
Retourne les tenseurs PyTorch prêts pour l'entraînement.

Auteur : S. Oussama
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import Tuple, Dict
import urllib.request

import numpy as np
import torch


CORA_URL = (
    "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
)
CORA_URL_MIRROR = (
    "https://github.com/kimiyoung/planetoid/raw/master/data/"
    "ind.cora.x"
)
CORA_ALT_URL = (
    "https://people.cs.umass.edu/~mccallum/data/cora.tgz"
)

LABEL_MAP: Dict[str, int] = {
    "Case_Based": 0,
    "Genetic_Algorithms": 1,
    "Neural_Networks": 2,
    "Probabilistic_Methods": 3,
    "Reinforcement_Learning": 4,
    "Rule_Learning": 5,
    "Theory": 6,
}

NUM_TRAIN = 140
NUM_VAL = 500
NUM_TEST = 1000


def _download_cora(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    cora_dir = data_dir / "cora"

    content_file = cora_dir / "cora.content"
    cites_file = cora_dir / "cora.cites"

    if content_file.exists() and cites_file.exists():
        print("[data_loader] Cora déjà présent — skip téléchargement.")
        return cora_dir

    archive_path = data_dir / "cora.tgz"
    urls = [CORA_URL, CORA_ALT_URL]

    for url in urls:
        try:
            print(f"[data_loader] Téléchargement depuis {url} …")
            urllib.request.urlretrieve(url, archive_path)
            print("[data_loader] Téléchargement terminé.")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[data_loader] Echec ({exc}), tentative suivante …")
    else:
        raise RuntimeError(
            "Impossible de télécharger Cora. Vérifiez votre connexion."
        )

    import tarfile

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(data_dir)
    print(f"[data_loader] Archive extraite dans {data_dir}.")

    if not content_file.exists():
        candidates = list(data_dir.rglob("cora.content"))
        if candidates:
            actual_dir = candidates[0].parent
            return actual_dir

    return cora_dir


def _parse_content(content_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    raw_ids = []
    feat_rows = []
    label_list = []

    with open(content_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            raw_ids.append(parts[0])
            feat_rows.append([float(v) for v in parts[1:-1]])
            label_list.append(LABEL_MAP[parts[-1]])

    node_id_to_idx: Dict[str, int] = {nid: i for i, nid in enumerate(raw_ids)}
    features = np.array(feat_rows, dtype=np.float32)
    labels = np.array(label_list, dtype=np.int64)
    return features, labels, node_id_to_idx  # type: ignore[return-value]


def _parse_cites(
    cites_path: Path,
    node_id_to_idx: Dict[str, int],
) -> np.ndarray:
    src_list = []
    dst_list = []

    with open(cites_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            src_str, dst_str = parts[0], parts[1]
            if src_str not in node_id_to_idx or dst_str not in node_id_to_idx:
                continue
            src_list.append(node_id_to_idx[src_str])
            dst_list.append(node_id_to_idx[dst_str])

    edges = np.array([src_list, dst_list], dtype=np.int64)
    return edges


def _build_masks(
    num_nodes: int,
    labels: np.ndarray,
) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
    # stratégie Planetoid : 20 noeuds par classe pour train
    train_indices = []
    per_class = NUM_TRAIN // len(LABEL_MAP)

    for cls in range(len(LABEL_MAP)):
        cls_indices = np.where(labels == cls)[0]
        train_indices.extend(cls_indices[:per_class].tolist())

    train_set = set(train_indices)

    remaining = [i for i in range(num_nodes) if i not in train_set]
    val_indices = remaining[:NUM_VAL]
    test_indices = remaining[NUM_VAL: NUM_VAL + NUM_TEST]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return (
        train_mask,  # type: ignore[return-value]
        val_mask,    # type: ignore[return-value]
        test_mask,   # type: ignore[return-value]
    )


class CoraData:
    """Conteneur immutable pour les données Cora."""

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        edges: torch.Tensor,
        train_mask: torch.BoolTensor,
        val_mask: torch.BoolTensor,
        test_mask: torch.BoolTensor,
    ) -> None:
        self.X = X
        self.y = y
        self.edges = edges
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        self.num_nodes: int = int(X.shape[0])
        self.num_edges: int = int(edges.shape[1])
        self.num_classes: int = int(y.max().item()) + 1
        self.num_features: int = int(X.shape[1])

    def __repr__(self) -> str:
        return (
            f"CoraData("
            f"nodes={self.num_nodes}, "
            f"edges={self.num_edges}, "
            f"features={self.num_features}, "
            f"classes={self.num_classes}, "
            f"train={self.train_mask.sum().item()}, "
            f"val={self.val_mask.sum().item()}, "
            f"test={self.test_mask.sum().item()}"
            f")"
        )


def load_cora(data_dir: str | Path = "data") -> CoraData:
    """Point d'entrée principal — charge Cora depuis le disque ou internet."""
    data_dir = Path(data_dir)
    cora_dir = _download_cora(data_dir)

    content_path = cora_dir / "cora.content"
    cites_path = cora_dir / "cora.cites"

    print("[data_loader] Parsing cora.content …")
    features_np, labels_np, node_id_to_idx = _parse_content(content_path)

    print("[data_loader] Parsing cora.cites …")
    edges_np = _parse_cites(cites_path, node_id_to_idx)  # type: ignore[arg-type]

    num_nodes = features_np.shape[0]

    X = torch.from_numpy(features_np)
    y = torch.from_numpy(labels_np)
    edges = torch.from_numpy(edges_np)

    edges_rev = edges.flip(0)
    edges_full = torch.cat([edges, edges_rev], dim=1)
    edges_full = torch.unique(edges_full, dim=1)

    train_mask, val_mask, test_mask = _build_masks(num_nodes, labels_np)

    print(f"[data_loader] Dataset chargé : {num_nodes} noeuds, "
          f"{edges_full.shape[1]} aretes (non-dirigees).")

    return CoraData(X, y, edges_full, train_mask, val_mask, test_mask)
