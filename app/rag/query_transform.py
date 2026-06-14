class QueryTransformer:
    def __init__(self, llm, rewrite_enabled: bool, hyde_enabled: bool, max_variants: int):
        self.llm = llm
        self.rewrite_enabled = rewrite_enabled
        self.hyde_enabled = hyde_enabled
        self.max_variants = max(1, max_variants)

    def expand(self, query: str) -> list[str]:
        variants = [query]

        if self.rewrite_enabled and len(variants) < self.max_variants:
            variants.extend(self._rewrite(query))

        if self.hyde_enabled and len(variants) < self.max_variants:
            variants.append(self._hyde(query))

        cleaned: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            normalized = variant.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(normalized)
            if len(cleaned) >= self.max_variants:
                break
        return cleaned

    def _rewrite(self, query: str) -> list[str]:
        prompt = (
            "请将以下用户问题从不同角度改写，生成2个语义相近但表述不同的查询。\n"
            "每行一个查询，不要编号。\n\n"
            f"原始问题：{query}"
        )
        response = self.llm.complete(prompt)
        return [line.strip(" -0123456789.、") for line in response.splitlines() if line.strip()]

    def _hyde(self, query: str) -> str:
        prompt = (
            "请针对以下问题，写一段简短的假设性回答。"
            "不需要保证准确性，只需要包含相关术语和概念，用于检索。\n\n"
            f"问题：{query}"
        )
        return self.llm.complete(prompt).strip()
