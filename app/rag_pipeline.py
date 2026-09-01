# app/rag_pipeline.py
#
# RAG (retrieval-augmented generation) logic for the BunkingTraveller chatbot.
# Split out of the original app/chatbot.py prototype so the retrieval/generation
# logic can be reused by a CLI (app/main.py) or, later, a web API.

import json
from pathlib import Path

import torch  # noqa: F401 - must be imported before faiss to avoid a libomp
# segfault when both are loaded in the same process on macOS (see
# https://github.com/facebookresearch/faiss/issues/2371). Keep this import first.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_places.json"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Other LLMs considered/tried during prototyping (kept for reference):
#   "openchat/openchat_3.5"
#   "tiiuae/falcon-7b-instruct"
#   "google/flan-t5-base"
LLM_MODEL_NAME = "declare-lab/flan-alpaca-base"


def load_places(data_path: Path = DATA_PATH) -> list[dict]:
    """Load the knowledge base of places from a JSON file."""
    with open(data_path, "r", encoding="utf-8") as f:
        places = json.load(f)
    if not places:
        raise ValueError(f"No places found in {data_path}")
    return places


class RAGPipeline:
    """Embeds a knowledge base of places, retrieves the closest match for a
    question via FAISS, and generates an answer with a local seq2seq LLM."""

    def __init__(
        self,
        data_path: Path = DATA_PATH,
        embed_model_name: str = EMBED_MODEL_NAME,
        llm_model_name: str = LLM_MODEL_NAME,
    ):
        self.places = load_places(data_path)
        self.corpus = [f"{p['name']}: {p['desc']}" for p in self.places]

        self.embed_model = SentenceTransformer(embed_model_name)
        self.index = self._build_index(self.corpus)

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)

    def _build_index(self, corpus: list[str]) -> faiss.IndexFlatL2:
        embeddings = self.embed_model.encode(corpus)
        dimension = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        return index

    def retrieve(self, question: str, k: int = 1) -> list[str]:
        """Return the top-k corpus entries closest to the question."""
        question_embedding = self.embed_model.encode([question])
        _, top_k = self.index.search(np.array(question_embedding), k=k)
        return [self.corpus[i] for i in top_k[0]]

    def ask(self, question: str) -> str:
        top_doc = self.retrieve(question, k=1)[0]
        prompt = f"Answer the question based on the context below:\nContext: {top_doc}\nQuestion: {question}"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        output_ids = self.model.generate(**inputs, max_length=150)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
