"""Preparação documentada das avaliações Olist e metadados de pedidos."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from .config import RAW_DIR

REQUIRED_REVIEW_COLUMNS = {
    "review_id", "order_id", "review_score", "review_comment_title",
    "review_comment_message", "review_creation_date",
}

# Heurísticas transparentes (não substituem avaliação semântica ou anotação humana).
THEME_RULES = {
    "Entrega e prazo": r"\b(?:entreg\w*|atras\w*|demor\w*|prazo\w*|transportador\w*|frete\w*|receb\w*)\b",
    "Produto e qualidade": r"\b(?:produto\w*|defeit\w*|quebrad\w*|danificad\w*|qualidad\w*|funcion\w*|material\w*)\b",
    "Atendimento": r"\b(?:atendiment\w*|suport\w*|contato\w*|respost\w*|vendedor\w*|loja\w*)\b",
    "Embalagem": r"\b(?:embalag\w*|empacot\w*|caixa\w*|protegid\w*)\b",
    "Troca e devolução": r"\b(?:troc\w*|devolu\w*|reembols\w*|estorn\w*|cancel\w*)\b",
    "Pagamento e preço": r"\b(?:pagament\w*|preco\w*|preç\w*|valor\w*|caro\w*|cobr\w*|parcel\w*)\b",
}


def normalize_text(value: object) -> str:
    """Preserva o texto sem corrigir/alterar seu sentido; limpa HTML e espaços."""
    if pd.isna(value) if not isinstance(value, (list, dict)) else False:
        return ""
    text = str(value)
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _optional_csv(name: str, folder: Path) -> pd.DataFrame | None:
    file = folder / name
    return pd.read_csv(file, low_memory=False) if file.exists() else None


def _order_metadata(folder: Path) -> pd.DataFrame | None:
    """Agrega categorias por pedido ANTES do join: nunca multiplica avaliações."""
    orders = _optional_csv("olist_orders_dataset.csv", folder)
    if orders is None:
        return None
    wanted = [
        "order_id", "order_purchase_timestamp", "order_delivered_customer_date",
        "order_estimated_delivery_date", "order_status",
    ]
    orders = orders[[c for c in wanted if c in orders.columns]].drop_duplicates("order_id")
    for col in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        if col not in orders.columns:
            orders[col] = pd.NaT
        orders[col] = pd.to_datetime(orders[col], errors="coerce")
    delivered = orders["order_delivered_customer_date"]
    planned = orders["order_estimated_delivery_date"]
    # NA desconhecido: não classificar como "no prazo" indevidamente.
    orders["atrasou_entrega"] = (delivered > planned).where(delivered.notna() & planned.notna())
    orders["mes_compra"] = orders["order_purchase_timestamp"].dt.strftime("%Y-%m")

    items = _optional_csv("olist_order_items_dataset.csv", folder)
    products = _optional_csv("olist_products_dataset.csv", folder)
    if items is not None and products is not None:
        if {"order_id", "product_id"}.issubset(items) and {"product_id", "product_category_name"}.issubset(products):
            merge = items[["order_id", "product_id"]].merge(
                products[["product_id", "product_category_name"]].drop_duplicates("product_id"),
                on="product_id", how="left", validate="many_to_one",
            )
            grouped = merge.groupby("order_id")["product_category_name"].agg(
                lambda s: " | ".join(sorted(set(s.dropna().astype(str))))
            ).rename("categorias")
            orders = orders.merge(grouped, on="order_id", how="left", validate="one_to_one")
    if "categorias" not in orders:
        orders["categorias"] = ""
    return orders[["order_id", "mes_compra", "atrasou_entrega", "categorias", "order_status"]] if "order_status" in orders else orders[["order_id", "mes_compra", "atrasou_entrega", "categorias"]]


def build_documents(folder: str | Path = RAW_DIR) -> pd.DataFrame:
    """Uma linha por avaliação com texto; ids estáveis e metadados úteis."""
    folder = Path(folder)
    review_file = folder / "olist_order_reviews_dataset.csv"
    if not review_file.exists():
        raise FileNotFoundError(f"CSV não encontrado: {review_file}")
    reviews = pd.read_csv(review_file, low_memory=False)
    missing = REQUIRED_REVIEW_COLUMNS - set(reviews.columns)
    if missing:
        raise ValueError(f"O CSV de avaliações não tem as colunas: {', '.join(sorted(missing))}")
    reviews = reviews.copy()
    for col in ("review_comment_title", "review_comment_message"):
        reviews[col] = reviews[col].map(normalize_text)
    reviews["texto"] = (
        reviews["review_comment_title"] + ". " + reviews["review_comment_message"]
    ).str.strip(" .")
    reviews = reviews.loc[reviews["texto"].str.len() >= 5].copy()
    reviews["review_score"] = pd.to_numeric(reviews["review_score"], errors="coerce")
    reviews = reviews.dropna(subset=["review_score", "review_id", "order_id"])
    reviews = reviews[reviews["review_score"].between(1, 5)].copy()
    reviews["review_score"] = reviews["review_score"].astype("int8")
    reviews = reviews.drop_duplicates(subset=["review_id", "order_id"], keep="first")
    reviews["doc_id"] = [
        hashlib.sha1(f"{rid}|{oid}".encode("utf-8")).hexdigest()[:16]
        for rid, oid in zip(reviews["review_id"], reviews["order_id"])
    ]
    reviews["data_avaliacao"] = pd.to_datetime(reviews["review_creation_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    meta = _order_metadata(folder)
    if meta is not None:
        reviews = reviews.merge(meta, on="order_id", how="left", validate="many_to_one")
    else:
        reviews["mes_compra"] = None
        reviews["atrasou_entrega"] = None
        reviews["categorias"] = ""
    for col in ("categorias", "mes_compra"):
        reviews[col] = reviews[col].fillna("").astype(str)
    reviews["tem_reclamacao"] = reviews["review_score"] <= 2
    for label, rule in THEME_RULES.items():
        reviews[f"tema__{label}"] = reviews["texto"].str.contains(rule, flags=re.I, regex=True, na=False)
    cols = ["doc_id", "review_id", "order_id", "review_score", "data_avaliacao",
            "texto", "mes_compra", "atrasou_entrega", "categorias", "tem_reclamacao"] + [
                f"tema__{key}" for key in THEME_RULES
            ]
    return reviews[cols].reset_index(drop=True)
