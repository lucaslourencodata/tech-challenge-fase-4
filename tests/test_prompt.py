from voc_olist.llm import make_prompt
from voc_olist.retrieval import Hit


def test_prompt_cites_original_ids_and_not_top_k_frequencies():
    h = Hit(doc_id="d", review_id="review123", order_id="order123", texto="A entrega foi demorada",
            score=1, similarity=0.91, categorias="livros", data_avaliacao="2018-01-01", atrasou_entrega=True)
    prompt = make_prompt("O que reclamam da entrega?", [h], {
        "total_textos": 100,
        "notas_baixas": 20,
        "notas_altas": 50,
        "temas": [{"tema":"Entrega e prazo", "mencoes":25, "reclamacoes": 14, "elogios_ou_notas_altas": 5}],
    })
    assert "[E1]" in prompt
    assert "review123" in prompt
    assert "100 avaliações" in prompt
    assert "amostra" in prompt
    assert "não confiáveis" in prompt
