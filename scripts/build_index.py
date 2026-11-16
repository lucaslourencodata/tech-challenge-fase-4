"""Uso: python scripts/build_index.py [--limit N] [--batch-size N]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voc_olist.config import INDEX_DIR, PROCESSED_DIR, MODEL_NAME
from voc_olist.data import build_documents
from voc_olist.download import ensure_dataset
from voc_olist.store import create_index, get_encoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Indexar só N textos para testar")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.limit < 0 or args.batch_size <= 0:
        parser.error("--limit deve ser >= 0 e --batch-size > 0")
    paths = ensure_dataset()
    print(f"CSVs disponíveis: {', '.join(p.name for p in paths)}")
    docs = build_documents()
    if args.limit:
        docs = docs.head(args.limit).reset_index(drop=True)
    print(f"Preparadas {len(docs)} avaliações com texto.")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    docs.to_csv(PROCESSED_DIR / "reviews_clean.csv.gz", index=False, compression="gzip")
    print(f"Carregando modelo: {MODEL_NAME}")
    encoder = get_encoder(MODEL_NAME)
    manifest = create_index(docs, encoder, INDEX_DIR, MODEL_NAME, args.batch_size)
    print(f"Índice pronto! {manifest['documents']} documentos, {manifest['dimensions']} dimensões.")


if __name__ == "__main__":
    main()
