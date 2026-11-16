"""Orquestra RAG com métricas calculadas no subconjunto de análise."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np

from .llm import generate
from .metrics import summarize
from .retrieval import Hit, search


@dataclass
class Result:
    answer: str
    evidence: list[Hit]
    scope_metrics: dict
    refusal: str | None


def filtered_docs(docs: pd.DataFrame, min_score=None, max_score=None, category=None, delivery_late=None) -> pd.DataFrame:
    out = docs
    if min_score is not None:
        out = out.loc[out["review_score"] >= min_score]
    if max_score is not None:
        out = out.loc[out["review_score"] <= max_score]
    if category:
        out = out.loc[out["categorias"].fillna("").str.contains(category, case=False, regex=False)]
    if delivery_late is not None:
        out = out.loc[out["atrasou_entrega"].eq(delivery_late).fillna(False)]
    return out


def answer_question(question: str, docs: pd.DataFrame, vectors: np.ndarray, encoder,
                    provider: str, api_key: str = "", model: str = "", **search_args) -> Result:
    args = {key: search_args.get(key) for key in ("min_score", "max_score", "category", "delivery_late")}
    scope = filtered_docs(docs, **args)
    metrics = summarize(scope)
    hits, refusal = search(question, docs, vectors, encoder, **search_args)
    if refusal:
        return Result(f"**Evidências insuficientes.** {refusal}", hits, metrics, refusal)
    response = generate(question, hits, metrics, provider=provider, api_key=api_key, model=model)
    return Result(response, hits, metrics, None)
