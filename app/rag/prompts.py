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
        "涉及金额、比例、日期、数量等精确数值时，优先保留上下文中的原始数值、单位和格式，不要只给换算后的表达。\n"
        "如果需要补充换算结果，请放在原始表达之后。\n\n"
        "不要使用“根据上下文信息”、“根据给定上下文”、“根据提供的资料”、“根据文档内容”等开场白。\n"
        "直接给出结论或要点。\n\n"
        f"上下文：\n{context}\n\n"
        f"问题：{question}\n\n"
        "答案："
    )
