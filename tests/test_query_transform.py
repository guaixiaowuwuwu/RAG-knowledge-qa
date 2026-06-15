from app.rag.query_transform import QueryTransformer


class FakeLLM:
    def complete(self, prompt: str) -> str:
        if "不同角度改写" in prompt:
            return "RAG 核心步骤有哪些\n知识库问答系统流程"
        if "假设性回答" in prompt:
            return "RAG 系统通常包含文档解析、分块、向量化、检索和生成。"
        return ""


def test_query_transformer_generates_rewrite_and_hyde_variants():
    transformer = QueryTransformer(
        llm=FakeLLM(),
        rewrite_enabled=True,
        hyde_enabled=True,
        max_variants=4,
    )

    variants = transformer.expand("RAG 怎么做？")

    assert variants == [
        "RAG 怎么做？",
        "RAG 核心步骤有哪些",
        "知识库问答系统流程",
        "RAG 系统通常包含文档解析、分块、向量化、检索和生成。",
    ]


def test_query_transformer_can_disable_llm_expansion():
    transformer = QueryTransformer(
        llm=FakeLLM(),
        rewrite_enabled=False,
        hyde_enabled=False,
        max_variants=4,
    )

    assert transformer.expand("RAG 怎么做？") == ["RAG 怎么做？"]


class FailingLLM:
    def complete(self, prompt: str) -> str:
        raise RuntimeError("LLM unavailable")


def test_query_transformer_falls_back_to_original_query_on_failure():
    transformer = QueryTransformer(
        llm=FailingLLM(),
        rewrite_enabled=True,
        hyde_enabled=True,
        max_variants=4,
    )

    assert transformer.expand("RAG 怎么做？") == ["RAG 怎么做？"]


def test_query_transformer_runtime_toggles_override_defaults():
    transformer = QueryTransformer(
        llm=FakeLLM(),
        rewrite_enabled=True,
        hyde_enabled=True,
        max_variants=4,
    )

    variants = transformer.expand(
        "RAG 怎么做？",
        rewrite_enabled=False,
        hyde_enabled=True,
        max_variants=2,
    )

    assert variants == [
        "RAG 怎么做？",
        "RAG 系统通常包含文档解析、分块、向量化、检索和生成。",
    ]
