from app.api.routes import build_retriever


def main() -> None:
    retriever = build_retriever()
    retriever.similarity_search("RAG 系统包含哪些核心步骤？", top_k=1)
    print(
        {
            "retriever": type(retriever).__name__,
            "reranker": type(retriever.reranker).__name__,
            "status": "ok",
        }
    )


if __name__ == "__main__":
    main()
