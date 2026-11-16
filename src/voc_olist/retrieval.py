"""Busca semântica, filtros e controle explícito de evidência."""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

import numpy as np
import pandas as pd

from .data import THEME_RULES
from .store import unit_vectors


@dataclass(frozen=True)
class Hit:
    doc_id: str
    review_id: str
    order_id: str
    texto: str
    score: int
    similarity: float
    categorias: str
    data_avaliacao: str
    atrasou_entrega: bool | None


STOPWORDS = set("a as ao aos o os e em no na nos nas de da do dos das para por com um uma uns umas suas que qual quais como quanto quantos oque os seus sua sobre tem ha existe existem sao foi foram mais menos muito muitos principal principais clientes cliente consumidor consumidores avaliacao avaliacoes comentario comentarios relato relatos experiencia experiencias compra compras pedido pedidos dizer dizem posso pode gostaria saber me voce voces".split())
BROAD_TERMS = set("problemas reclamacoes reclamacao reclamam reclama reclamações elogios elogio elogiam elogia padroes insatisfacao satisfacao negativos positivas positivos ruins bons temas pontos frequentes recorrentes".split())


def _fold(txt: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", txt.lower()) if not unicodedata.combining(ch))


def content_words(txt: str) -> set[str]:
    return {word for word in re.findall(r"[a-z]{4,}", _fold(txt)) if word not in STOPWORDS}


def has_textual_support(question: str, hits: list[Hit]) -> bool:
    """Filtro conservador: impede respostas a perguntas fora do assunto do corpus."""
    if not hits:
        return False
    query_words = content_words(question)
    if not query_words:
        return False
    # Perguntas gerais sobre VoC (sem produto/etapa específica) são válidas
    # mesmo quando os comentários não contêm a palavra "elogio" ou "problema".
    if query_words.issubset(BROAD_TERMS):
        return True
    normalized_question = _fold(question)
    generic_intent = re.search(r"\b(?:reclam\w*|elog\w*|insatisf\w*|satisf\w*|problema\w*|aspecto\w*|padro\w*|ponto\w*)\b", normalized_question)
    specific_topic = any(re.search(pattern, normalized_question, flags=re.I) for pattern in THEME_RULES.values())
    if generic_intent and not specific_topic:
        return True
    evidence = " ".join(h.texto for h in hits)
    evidence_words = content_words(evidence)
    if query_words & evidence_words:
        return True
    # Sinônimos por tema: entregas/atrasos/demora, produtos/defeitos etc.
    normalized_evidence = _fold(evidence)
    for regex in THEME_RULES.values():
        if re.search(regex, normalized_question, flags=re.I) and re.search(regex, normalized_evidence, flags=re.I):
            return True
    return False


def search(
    question: str,
    docs: pd.DataFrame,
    vectors: np.ndarray,
    encoder,
    top_k: int = 8,
    min_similarity: float = 0.28,
    max_score: int | None = None,
    min_score: int | None = None,
    category: str | None = None,
    delivery_late: bool | None = None,
) -> tuple[list[Hit], str | None]:
    if not question.strip():
        return [], "Digite uma pergunta."
    if vectors.ndim != 2 or len(docs) != len(vectors):
        raise ValueError("Índice e metadados fora de sincronia.")
    mask = np.ones(len(docs), dtype=bool)
    if max_score is not None:
        mask &= docs["review_score"].to_numpy() <= max_score
    if min_score is not None:
        mask &= docs["review_score"].to_numpy() >= min_score
    if category:
        mask &= docs["categorias"].fillna("").str.contains(re.escape(category), case=False, regex=True).to_numpy()
    if delivery_late is not None:
        mask &= docs["atrasou_entrega"].eq(delivery_late).fillna(False).to_numpy()
    candidates = np.flatnonzero(mask)
    if not len(candidates):
        return [], "Não existem avaliações que atendam aos filtros selecionados."
    query_vector = unit_vectors(encoder.encode([question], convert_to_numpy=True))[0]
    if query_vector.shape[0] != vectors.shape[1]:
        raise ValueError("Dimensões incompatíveis: use o modelo de embeddings do índice.")
    sim = np.asarray(vectors[candidates] @ query_vector, dtype=np.float32)
    n = min(max(top_k, 1), len(sim))
    local = np.argpartition(-sim, n-1)[:n]
    local = local[np.argsort(-sim[local])]
    hits = []
    for k in local:
        val = float(sim[k])
        if val < min_similarity:
            continue
        row = docs.iloc[int(candidates[k])]
        late = row.get("atrasou_entrega")
        if pd.isna(late):
            late = None
        else:
            late = bool(late)
        hits.append(Hit(
            doc_id=str(row["doc_id"]), review_id=str(row["review_id"]),
            order_id=str(row["order_id"]), texto=str(row["texto"]),
            score=int(row["review_score"]), similarity=round(val, 4),
            categorias=str(row.get("categorias", "") or ""),
            data_avaliacao=str(row.get("data_avaliacao", "") or ""),
            atrasou_entrega=late,
        ))
    if not hits:
        return [], "Não encontrei avaliações com similaridade suficiente para fundamentar uma resposta."
    if not has_textual_support(question, hits):
        return [], "Não encontrei evidências textuais suficientes sobre esse assunto nas avaliações recuperadas."
    return hits, None
