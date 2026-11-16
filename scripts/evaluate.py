"""Avaliação reprodutível de busca temática e abstenção (não substitui anotação humana)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voc_olist.store import load_index, get_encoder
from voc_olist.retrieval import search

CASES = [
    ("Os clientes reclamam de atrasos na entrega?", "Entrega e prazo"),
    ("Existem avaliações sobre defeitos dos produtos?", "Produto e qualidade"),
    ("Como os clientes avaliam o atendimento?", "Atendimento"),
    ("Há problemas com embalagens danificadas?", "Embalagem"),
    ("O que é comentado sobre troca e devolução?", "Troca e devolução"),
]


def main():
    docs, vectors, manifest = load_index()
    model = get_encoder(manifest["model_name"])
    count = 0
    print("AVALIAÇÃO EXPLORATÓRIA DE RECUPERAÇÃO (rótulos heurísticos)")
    for query, target in CASES:
        hits, error = search(query, docs, vectors, model, top_k=10, min_similarity=0.25)
        ids = {h.doc_id for h in hits}
        tagged = docs.loc[docs["doc_id"].isin(ids), f"tema__{target}"] if hits else []
        match = int(sum(tagged))
        precision_proxy = match / len(hits) if hits else 0
        count += precision_proxy
        print(f"{target:<22} | {len(hits):2d} resultados | {match:2d} com palavras-chave | proxy P@10={precision_proxy:.2f}")
        if error:
            print("  Sem evidência:", error)
    print(f"Média das proporções de correspondência temática: {count/len(CASES):.2f}")
    print("Atenção: não é precision@k com anotação humana, nem métrica de veracidade do LLM.")


if __name__ == "__main__":
    main()
