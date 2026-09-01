# app/main.py
#
# CLI entry point for the BunkingTraveller chatbot.
# Retrieval/generation logic lives in app/rag_pipeline.py.

from rag_pipeline import RAGPipeline


def main():
    print("Loading BunkingTraveller chatbot (this downloads models on first run)...")
    bot = RAGPipeline()
    print("Ready. Ask about a place, or type 'exit'/'quit' to stop.")

    while True:
        question = input("Ask me something: ")
        if question.lower() in ("exit", "quit"):
            break
        print(bot.ask(question))


if __name__ == "__main__":
    main()
