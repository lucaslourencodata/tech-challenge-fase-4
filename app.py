"""Interface Streamlit do Voice of Customer Intelligence com RAG."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from voc_olist.config import INDEX_DIR, DEFAULT_GEMINI_MODEL, DEFAULT_OLLAMA_MODEL
from voc_olist.metrics import summarize
from voc_olist.pipeline import answer_question, filtered_docs
from voc_olist.retrieval import search
from voc_olist.store import load_index, get_encoder

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Voice of Customer | RAG Olist", page_icon="🗣️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:2rem; max-width:1320px}
[data-testid="stMetric"] {background:rgba(124,124,124,.08); padding:16px; border-radius:12px}
h1 {letter-spacing:-.035em}
.smallnote {font-size:.88rem; opacity:.75}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Carregando índice e modelo de linguagem vetorial...")
def cached_resources():
    docs, vectors, manifest = load_index(INDEX_DIR)
    return docs, vectors, manifest, get_encoder(manifest["model_name"])

st.title("🗣️ Voice of Customer Intelligence")
st.caption("Tech Challenge — Fase 4 · FIAP Pós-Tech · Avaliações Olist · RAG com evidências verificáveis")

if not (INDEX_DIR / "manifest.json").exists():
    st.warning("O índice vetorial ainda não foi criado.")
    st.code("python scripts/build_index.py", language="powershell")
    st.write("No Windows, use **INICIAR_WINDOWS.bat** para instalar, baixar e indexar automaticamente.")
    st.stop()

try:
    docs, vectors, manifest, encoder = cached_resources()
except Exception as exc:
    st.error(f"Falha ao carregar os dados: {exc}")
    st.stop()

with st.sidebar:
    st.subheader("Configurações de análise")
    rating = st.selectbox("Nota das avaliações", ["Todas", "1 a 2 · insatisfação", "3 · neutras", "4 a 5 · satisfação"])
    min_score, max_score = {"Todas": (None, None), "1 a 2 · insatisfação": (1, 2),
                            "3 · neutras": (3, 3), "4 a 5 · satisfação": (4, 5)}[rating]
    delivery = st.selectbox("Entrega registrada", ["Todas", "Atrasada", "Não atrasada"])
    late = {"Todas": None, "Atrasada": True, "Não atrasada": False}[delivery]
    popular = (
        docs["categorias"].dropna().str.split(" | ", regex=False).explode().value_counts().head(30).index.tolist()
        if "categorias" in docs else []
    )
    popular = [s for s in popular if s]
    cat = st.selectbox("Categoria", ["Todas"] + popular)
    category = None if cat == "Todas" else cat
    st.divider()
    st.subheader("Geração da resposta")
    provider = st.radio("Modelo de linguagem", ["Gemini", "Ollama (local)"], index=0)
    if provider == "Gemini":
        api_key = st.text_input("Chave Gemini API", type="password", value="", help="Informe sua chave aqui ou no arquivo .env. Não é gravada pelo aplicativo.")
        model_name = st.text_input("Modelo", value=DEFAULT_GEMINI_MODEL)
    else:
        api_key = ""
        model_name = st.text_input("Modelo Ollama", value=DEFAULT_OLLAMA_MODEL)
    top_k = st.slider("Quantidade de evidências", 3, 15, 8)
    threshold = st.slider("Similaridade mínima", 0.15, 0.65, 0.28, 0.01)
    st.caption(f"Índice: {manifest['documents']:,} textos · embeddings: {manifest['model_name']}")

args = dict(min_score=min_score, max_score=max_score, category=category,
            delivery_late=late, top_k=top_k, min_similarity=threshold)
scope = filtered_docs(docs, min_score, max_score, category, late)
summary = summarize(scope)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avaliações com texto", f"{summary['total_textos']:,}".replace(",", "."))
col2.metric("Nota média", f"{summary['nota_media']:.2f}/5" if summary["nota_media"] is not None else "—")
col3.metric("Notas 1–2", f"{summary['notas_baixas']:,}".replace(",", "."))
col4.metric("Notas 4–5", f"{summary['notas_altas']:,}".replace(",", "."))

tab1, tab2, tab3 = st.tabs(["💬 Perguntas com RAG", "📊 Exploração do corpus", "🧪 Sobre as evidências"])
with tab1:
    presets = [
        "Quais são os principais problemas relatados pelos clientes?",
        "O que os consumidores reclamam sobre atrasos na entrega?",
        "Quais aspectos da compra são mais elogiados?",
        "Quais relatos mencionam defeitos ou baixa qualidade dos produtos?",
        "Quais experiências foram relatadas sobre atendimento?",
    ]
    preset = st.selectbox("Pergunta sugerida", ["Escrever minha pergunta"] + presets)
    typed = st.text_area("Pergunta ao assistente", placeholder="Ex.: Quais avaliações indicam problemas com o prazo de entrega?", height=95)
    question = typed.strip() or (preset if preset != "Escrever minha pergunta" else "")
    if st.button("Buscar evidências e responder", type="primary", use_container_width=True):
        if not question:
            st.warning("Digite ou selecione uma pergunta.")
        else:
            result = None
            failure = None
            try:
                if provider == "Gemini" and not (api_key or os.environ.get("GEMINI_API_KEY")):
                    failure = "Informe a chave Gemini na barra lateral para ativar a geração. A recuperação das evidências continua disponível."
                else:
                    with st.spinner("Recuperando avaliações e produzindo resposta fundamentada..."):
                        result = answer_question(question, docs, vectors, encoder,
                                                 provider="Ollama" if provider.startswith("Ollama") else "Gemini",
                                                 api_key=api_key, model=model_name, **args)
            except Exception as exc:
                failure = f"Não foi possível gerar o texto com o modelo: {exc}"
            if result is None:
                hits, no_evidence = search(question, docs, vectors, encoder, **args)
                st.session_state["result"] = {
                    "question": question, "answer": no_evidence or failure, "hits": hits, "failure": failure
                }
            else:
                st.session_state["result"] = {
                    "question": question, "answer": result.answer, "hits": result.evidence, "failure": None
                }
    saved = st.session_state.get("result")
    if saved:
        st.subheader("Resposta")
        if saved["failure"]:
            st.info(saved["answer"])
        else:
            st.markdown(saved["answer"])
        st.caption("As referências [E1], [E2] etc. correspondem aos cartões abaixo. Confirme a coerência entre cada afirmação e sua evidência.")
        st.subheader(f"Avaliações recuperadas ({len(saved['hits'])})")
        for i, hit in enumerate(saved["hits"], 1):
            with st.expander(f"[E{i}] Nota {hit.score}/5 · Similaridade {hit.similarity:.3f} · {hit.data_avaliacao}", expanded=i <= 3):
                st.write(hit.texto)
                st.caption(f"review_id: {hit.review_id} · order_id: {hit.order_id} · "
                           f"categoria(s): {hit.categorias or 'indisponível'} · "
                           f"entrega atrasada: {hit.atrasou_entrega}")
with tab2:
    st.caption("Classificação transparente por palavras-chave, não resultado de classificação supervisionada. Uma avaliação pode aparecer em mais de um tema.")
    rows = summary["temas"]
    if rows:
        chart_df = pd.DataFrame(rows).sort_values("reclamacoes", ascending=True)
        fig = px.bar(chart_df, x="reclamacoes", y="tema", orientation="h",
                     labels={"reclamacoes": "Menções nas avaliações com nota 1–2", "tema": "Tema"},
                     template="plotly_white")
        fig.update_layout(showlegend=False, margin=dict(l=0,r=15,t=15,b=0), height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame(rows).rename(columns={"tema":"Tema", "mencoes":"Menções totais",
                          "reclamacoes":"Menções com nota 1–2", "elogios_ou_notas_altas":"Menções com nota 4–5"}),
                          use_container_width=True, hide_index=True)
    st.info("As contagens são calculadas no conjunto filtrado inteiro. As evidências recuperadas pelo RAG são exemplos, não permitem estimar prevalência estatística.")
with tab3:
    st.markdown("""
    **Método:** uma avaliação textual por documento; modelo multilíngue para embeddings; normalização L2;
    similaridade cosseno sobre índice vetorial persistente; filtros de nota, categoria e entrega;
    contexto com avaliações identificadas; geração condicionada às fontes; recusa quando faltam evidências.

    **Limites:** o dataset é histórico (2016–2018); palavras-chave não equivalem a uma taxonomia validada;
    a similaridade mínima é ajustável e não é probabilidade; o LLM pode cometer erros, por isso os comentários
    usados como contexto aparecem na interface.
    """)
    st.json({"documentos_indexados": manifest["documents"], "dimensao": manifest["dimensions"],
             "modelo_embeddings": manifest["model_name"], "data_indice_utc": manifest["created_utc"]})
