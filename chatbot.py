# app/chatbot.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load LLM from Hugging Face
# llm_model_name = "openchat/openchat_3.5"
# llm_model_name = "tiiuae/falcon-7b-instruct"
# llm_model_name = "google/flan-t5-base"
llm_model_name = "declare-lab/flan-alpaca-base"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

# Sample knowledge base (3 places only for now)
knowledge = [
    {"name": "Big Ben", "desc": "A famous clock tower in London."},
    {"name": "Tower of London", "desc": "Historic castle on the north bank of the Thames."},
    {"name": "London Eye", "desc": "A giant Ferris wheel on the South Bank of the River Thames."}
]

# Step 1: Convert to vector embeddings
corpus = [f"{item['name']}: {item['desc']}" for item in knowledge]
corpus_embeddings = embed_model.encode(corpus)

# Step 2: Build FAISS index
dimension = corpus_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(corpus_embeddings))

# Step 3: Handle a query
def ask_bot(user_question):
    question_embedding = embed_model.encode([user_question])
    _, top_k = index.search(np.array(question_embedding), k=1)
    top_doc = corpus[top_k[0][0]]

    prompt = f"Answer the question based on the context below:\nContext: {top_doc}\nQuestion: {user_question}"
    output = llm(prompt, max_length=150)

    # Support both response formats
    if isinstance(output[0], dict):
        return output[0].get("generated_text", "No output.")
    elif isinstance(output[0], str):
        return output[0]
    else:
        return "Unknown output format."

# Try it!
if __name__ == "__main__":
    while True:
        q = input("Ask me something: ")
        if q.lower() in ["exit", "quit"]: break
        print(ask_bot(q))
