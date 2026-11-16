import numpy as np
import pandas as pd
import pytest

from voc_olist.store import create_index, load_index, unit_vectors
from voc_olist.retrieval import search, content_words, has_textual_support
from voc_olist.pipeline import filtered_docs, answer_question


class TinyEncoder:
    def encode(self, sentences, **kwargs):
        out = []
        for sentence in sentences:
            text = sentence.lower()
            if "entrega" in text or "atraso" in text:
                out.append([1.0, 0.0, 0.0])
            elif "produto" in text or "qualidade" in text:
                out.append([0.0, 1.0, 0.0])
            elif "atendimento" in text:
                out.append([0.0, 0.0, 1.0])
            else:
                out.append([0.5, 0.5, 0.5])
        return np.asarray(out, dtype=np.float32)


@pytest.fixture
def resources(tmp_path):
    docs = pd.DataFrame([
        {"doc_id": "d1", "review_id": "r1", "order_id": "o1", "texto": "Minha entrega atrasou dez dias", "review_score": 1,
         "categorias": "livros", "atrasou_entrega": True, "data_avaliacao": "2018-01-01",
         "tema__Entrega e prazo": True},
        {"doc_id": "d2", "review_id": "r2", "order_id": "o2", "texto": "Produto excelente e bonito", "review_score": 5,
         "categorias": "brinquedos", "atrasou_entrega": False, "data_avaliacao": "2018-01-03",
         "tema__Entrega e prazo": False},
        {"doc_id": "d3", "review_id": "r3", "order_id": "o3", "texto": "Atendimento nunca respondeu", "review_score": 2,
         "categorias": "livros", "atrasou_entrega": False, "data_avaliacao": "2018-01-04",
         "tema__Entrega e prazo": False},
    ])
    encoder = TinyEncoder()
    create_index(docs, encoder, tmp_path, "TinyEncoder", batch_size=2)
    loaded_docs, vectors, manifest = load_index(tmp_path)
    return loaded_docs, vectors, manifest, encoder


def test_index_is_persistent_and_normalized(resources):
    docs, vectors, manifest, _ = resources
    assert manifest["documents"] == 3
    assert manifest["dimensions"] == 3
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    assert docs.iloc[0]["review_id"] == "r1"


def test_unit_vectors_zero():
    assert np.allclose(unit_vectors(np.array([0., 0., 0.])), 0)


def test_retrieval_by_semantics(resources):
    docs, vectors, _, encoder = resources
    hits, reason = search("Houve atraso na entrega?", docs, vectors, encoder, min_similarity=0.4)
    assert reason is None
    assert hits[0].doc_id == "d1"
    assert hits[0].similarity == 1.0


def test_retrieval_by_rating_category_and_delivery(resources):
    docs, vectors, _, encoder = resources
    hits, reason = search("Houve atraso na entrega?", docs, vectors, encoder,
                          max_score=2, category="livros", delivery_late=True)
    assert reason is None
    assert [hit.doc_id for hit in hits] == ["d1"]
    assert len(filtered_docs(docs, max_score=2, category="livros", delivery_late=True)) == 1


def test_empty_filter_refuses(resources):
    docs, vectors, _, encoder = resources
    hits, reason = search("Atraso na entrega", docs, vectors, encoder, category="inexistente")
    assert hits == []
    assert "filtros" in reason


def test_out_of_scope_refuses(resources):
    docs, vectors, _, encoder = resources
    hits, reason = search("Qual é a capital do Canadá?", docs, vectors, encoder, min_similarity=0.0)
    assert not hits
    assert "evidências textuais" in reason


def test_high_similarity_threshold_refuses(resources):
    docs, vectors, _, encoder = resources
    hits, reason = search("Atraso na entrega", docs, vectors, encoder, min_similarity=1.1)
    assert hits == []
    assert "similaridade" in reason


def test_pipeline_refuses_without_calling_llm(resources):
    docs, vectors, _, encoder = resources
    result = answer_question("Capital do Canadá?", docs, vectors, encoder, provider="Gemini")
    assert result.refusal
    assert "Evidências insuficientes" in result.answer


def test_question_content_words():
    assert "entrega" in content_words("Quais as reclamações sobre entrega?")
    assert "quais" not in content_words("Quais as reclamações sobre entrega?")


def test_generic_voc_question_is_supported(resources):
    docs, vectors, _, encoder = resources
    hits, reason = search("Quais aspectos da compra são mais elogiados?", docs, vectors, encoder, min_similarity=0.0)
    assert hits
    assert reason is None
