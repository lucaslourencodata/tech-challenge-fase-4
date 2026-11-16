"""Índice persistente: matriz vetorial normalizada + documentos SQLite."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .config import INDEX_DIR, MODEL_NAME, PROCESSED_DIR


class Encoder(Protocol):
    def encode(self, sentences, **kwargs): ...


def get_encoder(model_name: str = MODEL_NAME):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def unit_vectors(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    denominator = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return (x / denominator).astype(np.float32)


def create_index(
    frame: pd.DataFrame,
    encoder: Encoder,
    output_dir: str | Path = INDEX_DIR,
    model_name: str = MODEL_NAME,
    batch_size: int = 64,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        raise ValueError("Não há avaliações textuais para indexar.")
    if frame["doc_id"].duplicated().any():
        raise ValueError("IDs de documentos duplicados.")
    # Preparação em lote evita matrizes temporárias gigantes.
    chunks = []
    texts = frame["texto"].astype(str).tolist()
    for i in range(0, len(texts), batch_size):
        batch = encoder.encode(texts[i:i+batch_size], convert_to_numpy=True, show_progress_bar=False)
        chunks.append(unit_vectors(batch))
        if (i // batch_size) % 30 == 0:
            print(f"Vetorizados: {min(i + batch_size, len(texts)):,}/{len(texts):,}")
    vectors = np.concatenate(chunks, axis=0)
    if vectors.shape[0] != len(frame) or not np.isfinite(vectors).all():
        raise RuntimeError("Índice inconsistente.")
    np.save(output_dir / "embeddings.npy", vectors, allow_pickle=False)
    with sqlite3.connect(output_dir / "documents.sqlite") as connection:
        frame.to_sql("documents", connection, if_exists="replace", index=False)
    manifest = {
        "model_name": model_name, "documents": len(frame), "dimensions": int(vectors.shape[1]),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "one review = one document; cosine similarity; no chunking",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_index(folder: str | Path = INDEX_DIR) -> tuple[pd.DataFrame, np.ndarray, dict]:
    folder = Path(folder)
    with (folder / "manifest.json").open(encoding="utf-8") as f:
        manifest = json.load(f)
    with sqlite3.connect(folder / "documents.sqlite") as connection:
        documents = pd.read_sql_query("SELECT * FROM documents", connection)
    vectors = np.load(folder / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    if len(documents) != vectors.shape[0] or vectors.shape[1] != manifest["dimensions"]:
        raise RuntimeError("Índice inconsistente. Reexecute scripts/build_index.py.")
    return documents, vectors, manifest
