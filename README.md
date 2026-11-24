# Private Knowledge RAG — cited enterprise document assistant

**Business problem:** Employees waste time hunting for policy and contract answers; unsupported chatbot answers introduce risk. **Target outcome (not a measured claim):** Faster retrieval with visible evidence and explicit abstention.

## Capabilities

* PDF or text ingestion; 120-word sliding windows with 25-word overlap
* Qdrant embedded local vector store; OpenAI semantic embeddings (`EMBEDDING\_BACKEND=openai`) or deterministic hashing **for offline demo only**
* Lexical relevance reranker and minimum confidence gate
* Answers grounded to numbered retrieved passages via optional OpenAI API, with citation payloads
* Explicit "I don't know" when insufficient evidence
* Evaluation script covering answerability, citations and abstention

## Run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app:app --reload
# another terminal:
curl -X POST http://localhost:8000/documents/text -H 'Content-Type: application/json' -d '{"name":"policy","text":"Finance approves invoices above $5000."}'
curl -X POST http://localhost:8000/ask -H 'Content-Type: application/json' -d '{"question":"Who approves invoices?"}'
pytest -q
python evaluate.py
```

Open `/docs` for interactive API demo. To enable LLM summaries, copy `.env.example` to `.env`, set `OPENAI\_API\_KEY`, and export it into your environment (or use `uvicorn --env-file .env app:app`).

## Design tradeoffs

This dependency-light mode deliberately uses hashed bag-of-words vectors instead of true semantic embeddings; it works only for lexical overlap. Set `EMBEDDING\_BACKEND=openai` and `OPENAI\_API\_KEY` to use semantic OpenAI `text-embedding-3-small` (1536D); the app selects a distinct Qdrant collection by embedding mode. Evaluate a cross-encoder reranker before deployment. Indexed content is not encrypted and this demo has no tenant RBAC; **use synthetic files only**. PDF extraction skips scans/OCR. Retrieval scores are not calibrated truth probabilities. LLM responses may still hallucinate; require citation entailment checks for regulated use. Evaluation prints measurements when executed, not manufactured results.

## Showcase

Demo: upload a vendor policy, ask an answerable question, open citation passages, ask an out-of-scope question and show abstention. Add a deployed URL and 60–90 second Loom link after you create them.

## Cost and measurement

Measure retrieval recall@k, grounded-answer pass rate (manual rubric), unsupported answer rate, p50/p95 response latency and token cost per question. OpenAI billing varies by current model pricing; report measured token usage multiplied by the vendor's published rate.


