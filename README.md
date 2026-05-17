# ⚖️ NyayaEval

> **High-throughput, multilingual legal document evaluation pipeline** for Indian district court records.

NyayaEval ingests unstructured court documents in low-resource regional languages, adapts and translates them into standardized English legal concepts using [**Adaptive Data by Adaption**](https://adaptionlabs.ai), constructs a Knowledge Graph in Neo4j, runs a rigorous automated evaluation loop (LangGraph + LLM-as-Judge) to catch hallucinations, and exports verified data for open-source publication.

**Built for the [AI Agents Hackathon 2026](https://hackindia.org/2026/ai-agents-hackathon-2026) — Adaptive Data Track.**

---

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌───────────────┐    ┌─────────────┐    ┌──────────┐
│  Ingestion   │───▶│  Adaptation  │───▶│ Graph Builder  │───▶│  Evaluator   │───▶│  Export   │
│  (PDF/Text)  │    │ (Adaption    │    │ (Neo4j KG)     │    │ (LLM-Judge  │    │ (JSONL/  │
│              │    │  SDK)        │    │                │    │  + RAGAS)    │    │ CSV/HF)  │
└─────────────┘    └─────────────┘    └───────────────┘    └──────┬──────┘    └──────────┘
                                                                  │
                                                          ┌───────▼───────┐
                                                          │   Corrector    │
                                                          │ (Self-healing  │
                                                          │  feedback loop)│
                                                          └───────────────┘
```

### Module Layout

| Layer | Package | Purpose |
|-------|---------|---------|
| Core | `nyayaeval/core/` | Domain models, LangGraph state, evaluation schemas |
| Agents | `nyayaeval/agents/` | Pipeline node functions (ingestion, adaptation, etc.) |
| Connectors | `nyayaeval/connectors/` | External I/O adapters (Neo4j, Redis, Adaption SDK, LLMs) |
| Pipeline | `nyayaeval/pipeline/` | LangGraph graph construction and routing |
| Export | `nyayaeval/export/` | JSONL/CSV serialization |
| API | `nyayaeval/api/` | FastAPI HTTP interface |
| Config | `nyayaeval/config/` | Environment-based settings |

---

## 🔌 Powered by Adaptive Data

NyayaEval is built on the [**Adaptive Data**](https://adaptionlabs.ai) platform by **Adaption**, which provides the core multilingual data pipeline:

- **Ingest** — Upload legal document datasets to the Adaption platform
- **Adapt** — Run multilingual adaptation across 242 languages (Hindi, Tamil, Bengali, etc.)
- **Evaluate** — Retrieve quality metrics and grade improvements from Adaption
- **Export** — Download adapted datasets for HuggingFace/Kaggle publication

The Adaption SDK (`pip install adaption`) is integrated as a first-class connector in the NyayaEval pipeline, powering the adaptation agent's translation and quality evaluation workflow.

> **Credits:** This project uses the Adaptive Data platform and SDK. Adaption is a sponsor of the AI Agents Hackathon 2026.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- API keys: [Adaption](https://adaptionlabs.ai) + [Gemini](https://aistudio.google.com) / [OpenAI](https://platform.openai.com) / [Groq](https://console.groq.com)

### 1. Clone & Configure

```bash
git clone https://github.com/HackIndiaXYZ/ai-agents-hackathon-2026-the-last-minute-clutch.git
cd ai-agents-hackathon-2026-the-last-minute-clutch
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start Infrastructure

```bash
docker compose up -d
```

This provisions:
- **Neo4j 5.x** (bolt://localhost:7687, browser at http://localhost:7474)
- **Redis 7.x** (localhost:6379)

### 3. Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### 4. Run the API Server

```bash
uvicorn nyayaeval.main:app --reload
```

### 5. Submit a Document

```bash
curl -X POST http://localhost:8000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"raw_text": "Court document text...", "source_language": "hi"}'
```

### 6. Run Tests

```bash
pytest -m unit        # Unit tests only (30 tests)
pytest                # All tests
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | LLM backend: `openai`, `gemini`, or `groq` |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `ADAPTION_API_KEY` | — | Adaption platform API key |
| `NEO4J_PASSWORD` | — | Neo4j database password |
| `MAX_RETRIES` | `3` | Max correction loop attempts |

See [`.env.example`](.env.example) for all configuration options.

---

## 🧪 Development

### Project Principles

- **Async-first**: All I/O-bound operations use `async/await`
- **Strict typing**: Full type annotations, enforced via mypy
- **Hexagonal architecture**: Core domain has zero external dependencies
- **Immutable state updates**: LangGraph nodes return partial state diffs
- **Adaptive Data integration**: Meaningful SDK usage throughout the pipeline

---

## 👥 Team

**The Last Minute Clutch** — AI Agents Hackathon 2026

---

## 📄 License

MIT
