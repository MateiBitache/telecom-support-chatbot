# Telecom Customer Support Chatbot

A conversational AI assistant for a fictional mobile operator, **NetWave**. It answers customer questions about billing, roaming, mobile data, SIM/eSIM and plans, using **retrieval-augmented generation (RAG)**: it first finds the relevant sections of a knowledge base, then asks a large language model (Claude) to write an answer based only on those sections. When it cannot answer confidently, or the customer asks for a person, it **escalates to a human agent**.

## How it works

```
Customer question
      │
      ▼
Asked for a human? ──yes──► escalate to agent
      │ no
      ▼
Retriever (TF-IDF, word + character n-grams, cosine similarity)
finds the top 3 knowledge base sections
      │
      ▼
Best score below threshold? ──yes──► escalate to agent
      │ no
      ▼
LLM (Claude) writes the answer from those sections only
(offline mode without an API key: returns the best section directly)
      │
      ▼
Answer + sources + confidence score
```

- **Knowledge base**: Markdown files in `knowledge_base/`, split into one section per `##` heading. To add knowledge, just add a heading and an answer.
- **Retrieval**: scikit-learn TF-IDF on word n-grams (exact terms) and character n-grams (word forms such as "stole"/"stolen", small typos).
- **Grounding**: the system prompt tells the model to use only the retrieved excerpts, which limits made-up answers ("hallucinations").
- **Escalation**: low retrieval confidence or phrases like "real person", "agent" or "complaint" hand the conversation to a human.
- **Conversation memory**: the last 5 exchanges are sent with each request, so follow-up questions work.

## Project structure

```
chatbot/
  retriever.py   knowledge base loading and TF-IDF search
  bot.py         RAG logic, escalation rules, system prompt
  llm.py         Claude API wrapper
app.py           REST API (FastAPI)
cli.py           terminal chat
knowledge_base/  support articles (Markdown)
tests/           pytest test suite
```

## Run it

```bash
pip install -r requirements.txt

# Optional: use Claude to generate answers (otherwise it runs in offline mode)
export ANTHROPIC_API_KEY=your_key

# Chat in the terminal
python cli.py

# Or start the REST API, then open http://127.0.0.1:8000/docs
uvicorn app:app --reload
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "My mobile data is not working", "history": []}'
```

```json
{
  "answer": "First, check that mobile data is turned on and airplane mode is off...",
  "sources": ["internet.md > My mobile data is not working"],
  "escalated": false,
  "confidence": 0.42
}
```

## Tests

```bash
python -m pytest
```

The tests check knowledge base parsing, retrieval quality on sample questions, escalation, the prompt and history sent to the LLM (using a fake LLM, so no API key is needed) and the REST API, including input validation.

## Ideas for next steps

- Replace TF-IDF with embedding-based semantic search
- Add a simple web chat interface
- Log escalated questions to find gaps in the knowledge base
- Containerise with Docker

## Author

**Matei Bitache** - Computer Science graduate, University of Bucharest
[github.com/mateibitache](https://github.com/mateibitache)

## License

MIT - see [LICENSE](LICENSE).
