import pandas as pd
import pytest

from voc_olist.data import normalize_text, build_documents, THEME_RULES
from voc_olist.metrics import summarize


def test_normalize_text():
    assert normalize_text(None) == ""
    assert normalize_text("  <b>bom</b>   produto\n") == "bom produto"
    assert normalize_text(float("nan")) == ""


def create_csvs(folder):
    pd.DataFrame([
        {"review_id": "r1", "order_id": "o1", "review_score": 1,
         "review_comment_title": "Entrega", "review_comment_message": "Chegou atrasado", "review_creation_date": "2018-01-10"},
        {"review_id": "r1", "order_id": "o1", "review_score": 1,
         "review_comment_title": "Entrega", "review_comment_message": "Chegou atrasado", "review_creation_date": "2018-01-10"},
        {"review_id": "r2", "order_id": "o2", "review_score": 5,
         "review_comment_title": None, "review_comment_message": "Adorei o produto!", "review_creation_date": "2018-02-02"},
        {"review_id": "r3", "order_id": "o3", "review_score": 3,
         "review_comment_title": None, "review_comment_message": None, "review_creation_date": "2018-02-02"},
        {"review_id": "r4", "order_id": "o4", "review_score": 9,
         "review_comment_title": "Ótimo", "review_comment_message": "Excelente compra", "review_creation_date": "2018-02-02"},
    ]).to_csv(folder / "olist_order_reviews_dataset.csv", index=False)
    pd.DataFrame([
        {"order_id": "o1", "order_purchase_timestamp": "2018-01-01", "order_delivered_customer_date": "2018-01-20", "order_estimated_delivery_date": "2018-01-16", "order_status": "delivered"},
        {"order_id": "o2", "order_purchase_timestamp": "2018-01-02", "order_delivered_customer_date": None, "order_estimated_delivery_date": "2018-01-16", "order_status": "shipped"},
    ]).to_csv(folder / "olist_orders_dataset.csv", index=False)
    pd.DataFrame([
        {"order_id": "o1", "product_id": "p1"},
        {"order_id": "o1", "product_id": "p2"},
    ]).to_csv(folder / "olist_order_items_dataset.csv", index=False)
    pd.DataFrame([
        {"product_id": "p1", "product_category_name": "livros"},
        {"product_id": "p2", "product_category_name": "eletronicos"},
    ]).to_csv(folder / "olist_products_dataset.csv", index=False)


def test_build_documents_avoids_join_multiplication_and_removes_empty(tmp_path):
    create_csvs(tmp_path)
    docs = build_documents(tmp_path)
    assert len(docs) == 2
    assert docs["doc_id"].is_unique
    by_order = docs.set_index("order_id")
    assert by_order.loc["o1", "categorias"] == "eletronicos | livros"
    assert by_order.loc["o1", "atrasou_entrega"] == True
    assert pd.isna(by_order.loc["o2", "atrasou_entrega"])
    assert by_order.loc["o2", "texto"] == "Adorei o produto!"
    assert by_order.loc["o1", "tema__Entrega e prazo"] == True


def test_missing_required_columns(tmp_path):
    pd.DataFrame({"x": [1]}).to_csv(tmp_path / "olist_order_reviews_dataset.csv", index=False)
    with pytest.raises(ValueError, match="colunas"):
        build_documents(tmp_path)


def test_empty_dataset_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_documents(tmp_path)


def test_summary_counts_are_global_not_topk(tmp_path):
    create_csvs(tmp_path)
    docs = build_documents(tmp_path)
    stats = summarize(docs)
    assert stats["total_textos"] == 2
    assert stats["notas_baixas"] == 1
    assert stats["notas_altas"] == 1
    assert next(t for t in stats["temas"] if t["tema"] == "Entrega e prazo")["reclamacoes"] == 1
