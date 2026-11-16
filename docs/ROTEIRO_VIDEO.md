# Roteiro de gravação — máximo 7 minutos

> **Grave com o sistema aberto e a base REAL indexada.** O roteiro abaixo é texto pronto para narrar. Mostre resultados efetivos na tela; não invente dados ou métricas. Tempo total previsto: ~6min30.

## 00:00–00:40 · Problema de negócio

**Na tela:** painel inicial com o total de avaliações.

**Fala:** "Nosso Tech Challenge da Fase 4 desenvolve uma solução de Voice of Customer Intelligence para analisar avaliações de clientes da base histórica Olist. O problema é transformar milhares de comentários textuais em insights pesquisáveis para gestores, sem depender de leitura manual. Nossa solução aplica Retrieval-Augmented Generation, ou RAG, para recuperar comentários relevantes e embasar as respostas geradas por IA."

## 00:40–01:35 · Arquitetura

**Na tela:** abra `README.md` na seção Arquitetura e mostre o diagrama Mermaid no GitHub/preview.

**Fala:** "O fluxo começa nos CSVs Olist. Depois selecionamos avaliações com texto, tratamos valores ausentes, preparamos metadados e criamos um documento por avaliação. Vetorizamos os textos com um modelo multilíngue de embeddings. Os vetores são armazenados em NumPy e os documentos em SQLite. A pergunta passa pelo mesmo encoder; comparamos por similaridade cosseno, aplicamos filtros e enviamos os textos relevantes a um modelo de linguagem. A resposta mostra referências que apontam para comentários originais."

## 01:35–02:20 · Preparação da base

**Na tela:** mostre rapidamente `src/voc_olist/data.py`, `store.py` e o terminal com a execução real de `scripts/build_index.py` ou o `manifest.json`.

**Fala:** "Para manter rastreabilidade, preservamos review_id e order_id. Evitamos multiplicar avaliações ao relacionar itens e categorias agregando por pedido. O texto recebe apenas limpeza leve. Notas inválidas e avaliações sem texto são desconsideradas. Registramos também nota, mês da compra, categoria e atraso observado, sempre preservando informações ausentes quando não podem ser calculadas."

## 02:20–03:20 · Consulta 1: problemas recorrentes

**Na tela:** selecione "Quais são os principais problemas relatados pelos clientes?" e clique em Buscar. Role os cartões de evidência.

**Fala:** "Agora fazemos uma pergunta em português. O sistema transforma a consulta em vetor e recupera avaliações relevantes por similaridade semântica. Cada cartão inclui a nota, similaridade, data e identificadores. O modelo gera a resposta usando somente as fontes e cita referências como E1 e E2, que podem ser abertas para auditoria. O objetivo não é apenas responder bem, mas mostrar de onde veio cada conclusão."

## 03:20–04:20 · Consulta 2: prazo de entrega

**Na tela:** pergunte "O que os consumidores reclamam sobre atrasos na entrega?"; escolha notas 1–2 e mostre filtros. Abra uma evidência.

**Fala:** "A segunda investigação é focada em entrega. Filtramos avaliações com notas baixas para explorar insatisfação e consultamos relatos sobre atraso. Também poderíamos restringir por categoria ou por atraso de entrega observado nos dados de pedido. Os filtros afetam a população de busca e o conjunto usado nas contagens temáticas."

## 04:20–05:05 · Indicadores e cuidado estatístico

**Na tela:** aba "Exploração do corpus" com gráfico e tabela de temas.

**Fala:** "Além do RAG, há um painel de menções por temas, calculado sobre o conjunto de comentários filtrados. Como a busca semântica retorna exemplos mais relevantes, não seria correto usá-los para calcular a proporção de clientes com determinado problema. Por isso separamos as evidências recuperadas das contagens globais. A classificação por palavras-chave é uma heurística e uma mesma avaliação pode pertencer a mais de um tema."

## 05:05–05:50 · Perguntas sem evidências

**Na tela:** pergunte "Qual é a capital do Canadá?". Se o threshold estiver baixo e o controle não recusar, aumente o threshold; não prometa que todas as consultas fora do escopo serão automaticamente barradas.

**Fala:** "Uma exigência importante é reconhecer a ausência de evidências. Quando a recuperação não encontra suporte suficiente, a aplicação sinaliza que não pode responder com fundamento na base. Combinamos limiar de similaridade, verificação de pertinência textual e instruções de abstenção para reduzir alucinações. Nenhum desses métodos sozinho garante correção perfeita."

## 05:50–06:35 · Conclusão e decisões

**Na tela:** `README.md`, `docs/ARQUITETURA.md`, testes ou terminal `pytest -q` executado.

**Fala:** "O repositório contém código, instruções de instalação, arquitetura, justificativas, testes e roteiro de reprodução. As decisões centrais foram usar um documento por avaliação, embeddings multilíngues, índice simples e auditável, metadados para segmentação e fontes originais para rastrear insights. Como evoluções, sugerimos avaliação de recuperação com relevância anotada, reranking e checagem automática de fidelidade."

**Na tela antes de finalizar:** deixe o endereço do repositório público visível.

## Checklist antes da gravação

- [ ] Base real completa indexada (não use `--limit`).
- [ ] Chave Gemini ativa ou Ollama instalado e rodando.
- [ ] Pelo menos 2 perguntas respondidas de fato, com evidências coerentes.
- [ ] Uma pergunta fora de escopo demonstrada com recusa (ajuste threshold se necessário).
- [ ] Resultados e IDs legíveis na captura de tela.
- [ ] Testes executados e repositório publicado.
- [ ] Vídeo exportado com duração **até 7 minutos**.
