"""Download do dataset público na origem, sem distribuir dados no repositório."""
from __future__ import annotations

from pathlib import Path
import shutil

from .config import RAW_DIR


EXPECTED = ["olist_order_reviews_dataset.csv", "olist_orders_dataset.csv",
            "olist_order_items_dataset.csv", "olist_products_dataset.csv"]


def ensure_dataset(folder: str | Path = RAW_DIR) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / EXPECTED[0]).exists():
        return sorted(folder.glob("*.csv"))
    print("Baixando dataset Olist diretamente do Kaggle...")
    try:
        import kagglehub
        source = Path(kagglehub.dataset_download("olistbr/brazilian-ecommerce"))
    except Exception as exc:
        raise RuntimeError(
            "Não foi possível baixar a base do Kaggle automaticamente. "
            "Baixe https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce "
            "e extraia os CSVs em data/raw/ antes de tentar novamente. "
            f"Detalhe: {exc}"
        ) from exc
    paths = []
    for name in EXPECTED + ["product_category_name_translation.csv"]:
        matches = list(source.rglob(name))
        if matches:
            target = folder / name
            shutil.copy2(matches[0], target)
            paths.append(target)
    if not (folder / EXPECTED[0]).exists():
        raise RuntimeError("O download não continha o CSV de avaliações esperado.")
    return paths
