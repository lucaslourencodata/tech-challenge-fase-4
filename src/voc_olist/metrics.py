"""Contagens de TODOS os documentos filtrados, separadas dos exemplos RAG."""
from __future__ import annotations

import pandas as pd
from .data import THEME_RULES


def summarize(frame: pd.DataFrame) -> dict:
    total = len(frame)
    low = frame["review_score"] <= 2 if total else pd.Series([], dtype=bool)
    high = frame["review_score"] >= 4 if total else pd.Series([], dtype=bool)
    topics = []
    for label in THEME_RULES:
        col = f"tema__{label}"
        if col not in frame:
            continue
        matched = frame[col].fillna(False).astype(bool)
        topics.append({
            "tema": label,
            "mencoes": int(matched.sum()),
            "reclamacoes": int((matched & low).sum()),
            "elogios_ou_notas_altas": int((matched & high).sum()),
        })
    return {
        "total_textos": total,
        "nota_media": round(float(frame["review_score"].mean()), 2) if total else None,
        "notas_baixas": int(low.sum()),
        "notas_altas": int(high.sum()),
        "temas": sorted(topics, key=lambda row: row["reclamacoes"], reverse=True),
    }


def summary_context(summary: dict) -> str:
    themes = "; ".join(
        f"{t['tema']}: {t['mencoes']} avaliações com palavras-chave ({t['reclamacoes']} com nota 1–2)"
        for t in summary["temas"]
    )
    return (
        f"Contagens sobre TODAS as {summary['total_textos']} avaliações textuais no recorte atual. "
        f"Notas 1–2: {summary['notas_baixas']}; notas 4–5: {summary['notas_altas']}. "
        f"Classificação de temas por expressão regular, com sobreposição permitida. {themes}. "
        "Não interpretar menções por palavras-chave como prevalência exata de problemas."
    )
