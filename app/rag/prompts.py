from app.ingestion.chunker import Chunk


def build_rag_prompt(question: str, chunks: list[Chunk]) -> str:
    context_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        page = chunk.metadata.get("page")
        page_text = f", page={page}" if page is not None else ""
        context_blocks.append(
            f"[{index}] source={chunk.source}{page_text}, chunk={chunk.metadata.get('chunk_index')}\n{chunk.content}"
        )

    context = "\n\n".join(context_blocks)
    return (
        "你是一个企业知识库问答助手。请只基于给定上下文回答问题。\n"
        "如果上下文不足以回答，请直接说知识库中没有找到相关内容。\n"
        "回答要简洁、准确，并尽量指出依据来自哪些引用编号。\n\n"
        f"上下文：\n{context}\n\n"
        f"问题：{question}\n\n"
        "答案："
    )
