from unittest.mock import Mock, patch

from voc_olist.llm import extractive_fallback, generate, has_valid_citations
from voc_olist.retrieval import Hit


def _hits():
    return [
        Hit(doc_id="1", review_id="r1", order_id="o1", texto="Recebi um produto diferente do pedido",
            score=1, similarity=0.81, categorias="livros", data_avaliacao="2018-01-01", atrasou_entrega=False),
        Hit(doc_id="2", review_id="r2", order_id="o2", texto="Entrega demorou bastante",
            score=2, similarity=0.73, categorias="livros", data_avaliacao="2018-02-01", atrasou_entrega=True),
    ]


def test_citations_require_real_refs():
    assert has_valid_citations("Divergência [E1]", 2)
    assert has_valid_citations("Atraso [e2]", 2)
    assert not has_valid_citations("Problemas sem fonte", 2)
    assert not has_valid_citations("Divergência [E3]", 2)
    assert not has_valid_citations("Atraso [E2] e [E99]", 2)


def test_extractive_fallback_is_source_backed():
    result = extractive_fallback(_hits(), None)
    assert "contingência" in result
    assert "[E1]" in result and "[E2]" in result
    assert "Recebi um produto diferente do pedido" in result
    assert "não uma amostra" in result


def test_retry_missing_citations_with_ollama():
    with patch("voc_olist.llm._ollama_generate", side_effect=["Resposta sem fonte", "Reclamação por atraso [E2]."]) as fn:
        answer = generate("O que falam da entrega?", _hits(), {}, "Ollama", model="fake")
    assert answer == "Reclamação por atraso [E2]."
    assert fn.call_count == 2
    assert "FORMATO DA SAÍDA" in fn.call_args.args[0]


def test_fallback_when_llm_never_cites():
    with patch("voc_olist.llm._ollama_generate", return_value="Sem fontes") as fn:
        answer = generate("Quais os principais problemas?", _hits(), {}, "Ollama", model="fake")
    assert "Resposta de contingência" in answer
    assert "[E1]" in answer
    assert fn.call_count == 2


def test_valid_first_answer_skips_second_call():
    with patch("voc_olist.llm._ollama_generate", return_value="Produto errado [E1].") as fn:
        answer = generate("Problemas?", _hits(), {}, "Ollama", model="fake")
    assert answer == "Produto errado [E1]."
    assert fn.call_count == 1


def test_retry_failure_gracefully_falls_back():
    with patch("voc_olist.llm._ollama_generate", side_effect=["Resposta sem fonte", RuntimeError("429")]):
        answer = generate("Problemas?", _hits(), {}, "Ollama", model="fake")
    assert "Resposta de contingência" in answer
