# BunkingTraveller

A small retrieval-augmented-generation (RAG) chatbot that answers questions about
places using a local knowledge base: it embeds a list of places, retrieves the
closest match with FAISS, and generates an answer with a local Hugging Face
seq2seq model.

## Project structure

```
app/
  main.py          CLI entry point
  rag_pipeline.py  Embedding + FAISS retrieval + generation (RAGPipeline)
data/
  sample_places.json  Knowledge base (name + description per place)
tests/
  test_rag_pipeline.py
requirements.txt      Runtime dependencies
requirements-dev.txt  + pytest, for running the test suite
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11 is what this was built and tested against — `torch`/`faiss-cpu`
wheel availability is the constraint if you change that.

## Running

```bash
python app/main.py
```

This downloads two models from Hugging Face on first run (~few hundred MB):
`sentence-transformers/all-MiniLM-L6-v2` for embeddings and
`declare-lab/flan-alpaca-base` for generation. Then ask it things like
"What is the Eiffel Tower?" and type `exit`/`quit` to stop.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The test suite includes real integration tests that load the actual models
(no mocking), so the first run takes ~10-15s and needs network access the
first time to download them.

## Known gotcha: import order matters

`faiss` and `torch` loaded in the same process **segfault on macOS** unless
`torch` (or anything that imports it, like `transformers`) is imported
*before* `faiss`. This is a known `libomp` conflict
(facebookresearch/faiss#2371), not a bug in this code — `rag_pipeline.py`
imports `torch` first specifically to avoid it. Keep that ordering if you
touch the imports there.

## Design decisions / what's intentionally not here yet

- **No web API.** `fastapi`/`uvicorn` aren't installed — this is CLI-only for
  now. `main.py` is structured so a FastAPI layer could call into
  `RAGPipeline` directly without changes to `rag_pipeline.py`.
- **No LangChain/LangGraph/Chroma.** Retrieval is plain FAISS over an
  in-memory `IndexFlatL2`, which is enough for a knowledge base this size.
  Worth revisiting if the place list grows large enough that flat L2 search
  becomes a bottleneck, or if multi-step/agentic behavior is needed.
- **No remote LLM (OpenAI, etc.).** Generation runs locally via
  `declare-lab/flan-alpaca-base`, which is small and CPU-friendly but limited
  in answer quality. Swapping in an API-based model would mean changing
  `RAGPipeline.__init__`/`ask()` in `rag_pipeline.py`.
- **Knowledge base is a static JSON file.** Fine for a sample/demo dataset;
  would need a real datastore to support adding/editing places at runtime.
