from pathlib import Path

from app.core.config import get_settings
from app.evaluation.dataset import load_eval_cases
from app.evaluation.metrics import hit_rate_at_k
from app.rag.bm25 import BM25Retriever
from app.rag.documents import chunk_to_retrieved_document
from app.rag.embeddings import build_embeddings
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.vector_store import ChromaVectorStore


def main() -> None:
    settings = get_settings()
    eval_path = Path("data/eval/sample_eval.jsonl")
    cases = load_eval_cases(eval_path)
    dense = ChromaVectorStore(
        persist_dir=settings.chroma_dir,
        collection_name=settings.chroma_collection,
        embeddings=build_embeddings(settings),
    )
    sparse = BM25Retriever.from_jsonl(settings.bm25_corpus_path)

    all_retrieved = []
    for case in cases:
        dense_docs = [
            chunk_to_retrieved_document(chunk)
            for chunk in dense.similarity_search(case.question, top_k=settings.dense_retrieval_top_k)
        ]
        sparse_docs = sparse.search(case.question, top_k=settings.bm25_retrieval_top_k)
        fused = reciprocal_rank_fusion([dense_docs, sparse_docs], top_k=settings.retrieval_top_k, k=settings.rrf_k)
        all_retrieved.append(fused)

    expected_sources = [case.expected_sources for case in cases]
    print({"cases": len(cases), "hit_rate_at_k": hit_rate_at_k(all_retrieved, expected_sources)})


if __name__ == "__main__":
    main()
