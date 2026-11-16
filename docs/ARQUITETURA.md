# Arquitetura funcional

## 1. Ingestão

**Fonte:** Olist Brazilian E-Commerce Public Dataset (CSV). Fonte principal `olist_order_reviews_dataset.csv`, com campos `review_id`, `order_id`, `review_score`, título, comentário e data. Enriquecimento opcional: `olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, `olist_products_dataset.csv`.

**Regra de unidade de análise:** cada par (`review_id`, `order_id`) com texto útil gera exatamente um documento. O join por `order_id` é `many_to_one` porque a categoria de produto é agregada por pedido. Nenhuma linha é multiplicada por vários produtos.

## 2. Preparação

- Campos de título e comentário são concatenados, mantendo o texto original com limpeza de HTML/espaços.
- Campos de nota fora de 1–5, textos vazios e duplicatas são removidos.
- `doc_id` é SHA1 truncado do par original para ID estável.
- `atrasou_entrega` é `NA` se faltar data real ou estimada; caso contrário, booleano derivado da comparação de datas.
- Mês da compra é uma string YYYY-MM; categorias ficam concatenadas sem repetição.
- Rótulos de tema são *heurísticos* e não exclusivos; usados somente em análises exploratórias transparentes.

## 3. Embeddings

- Modelo: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Encoding em lotes; matriz normalizada (`float32`) de dimensão 384 por texto.
- Persistência: `artifacts/embeddings.npy`, `documents.sqlite`, `manifest.json`.
- O próprio modelo processa o texto até seu limite de tokens; textos muito longos podem ser truncados automaticamente.

## 4. Recuperação

- Query → embedding com mesmo modelo → normalização L2 → produto escalar (= similaridade cosseno).
- Filtros antes da ordenação: score min/max, categoria(s) e atraso de entrega conhecido.
- Ordenação dos maiores scores, `top_k` ajustável (3–15 no painel).
- Threshold conservador mas configurável e verificação de relevância textual.
- Sem resultados, filtros vazios ou texto incompatível → recusa explícita.

## 5. Geração

- Contexto limitado às avaliações top-k com rótulos `[E1..En]`, notas e IDs originais.
- Contexto agregado separado contém números globais do recorte, nunca calculados sobre top-k.
- LLM: Gemini (API) ou Ollama (local).
- Prompt exige citações, proíbe invenção de números e instrui a tratar comentários como dados não confiáveis, sem obedecer a instruções ali contidas.
- Evidências são exibidas integralmente em expanders no painel.

## 6. Falhas e controles

- Sem chave Gemini: permite recuperar evidências, sem apresentar resposta gerativa fictícia.
- Ollama offline: mostra erro de conexão, mas mantém os resultados de recuperação.
- Sem evidência: a aplicação responde insuficiência, sem enviar dados ao LLM.
- **Limitação residual:** o LLM pode citar erroneamente; o usuário deve conferir cada afirmação contra a evidência aberta. Seria desejável avaliação humana da fidelidade e uma checagem automática de citações em evolução futura.

## 7. Critérios de aceitação

- Construção e carregamento do índice sem inconsistência dimensional.
- Preservação dos identificadores das avaliações.
- Recuperação por similaridade, com filtros e threshold.
- Contagens que não confundam exemplos top-k com taxa de ocorrência.
- Caminho de abstenção quando faltam evidências.
- Resposta fundamentada quando um serviço LLM está disponível.
